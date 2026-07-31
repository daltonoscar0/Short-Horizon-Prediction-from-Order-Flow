"""Hand-computed checks for the pure data-quality primitives in edge.io.quality."""
import math
import unittest

import numpy as np
import pyarrow as pa

from edge.io.lobster import ASK_SENTINEL_PRICE, BID_SENTINEL_PRICE
from edge.io.quality import (
    ALIGNMENT_HOLD_RATE_THRESHOLD,
    BIN_NS,
    DAY_START_NS,
    TICK_SIZE_UNITS,
    alignment_effect_expected,
    build_intraday_profile,
    check_monotonic_ts,
    compute_alignment_hold_rate,
    confirm_u_shape,
    crossed_locked_counts,
    gaps_over_threshold,
    level_size_at,
    median_mid_dollars,
    ns_to_hhmmss,
    sample_alignment_indices,
    sentinel_row_count,
    top_k_gaps,
)


def _toy_alignment_table(event_type, direction, price, size, bid_px, bid_sz, ask_px, ask_sz):
    """Minimal event-indexed table (2 levels/side) in the columns compute_alignment_hold_rate
    reads -- built inline, not via tests/toybook.py, to keep this suite's fixtures local."""
    num_levels = len(bid_px[0])
    columns = {
        "event_type": pa.array(event_type, type=pa.int8()),
        "direction": pa.array(direction, type=pa.int8()),
        "price": pa.array(price, type=pa.int64()),
        "size": pa.array(size, type=pa.int64()),
    }
    for side, kind, rows in (
        ("bid", "px", bid_px),
        ("bid", "sz", bid_sz),
        ("ask", "px", ask_px),
        ("ask", "sz", ask_sz),
    ):
        for lvl in range(num_levels):
            columns[f"{side}_{kind}_{lvl + 1}"] = pa.array([r[lvl] for r in rows], type=pa.int64())
    return pa.table(columns), num_levels


class TestCrossedLocked(unittest.TestCase):
    def test_crossed_includes_locked(self):
        bid = np.array([100, 100, 101, 102])
        ask = np.array([101, 100, 100, 103])
        # locked at index 1 (100 == 100); crossed at index 2 (101 > 100); locked counts as
        # crossed under the >= definition, so crossed = 2.
        self.assertEqual(crossed_locked_counts(bid, ask), (2, 1))

    def test_clean_book_is_zero(self):
        bid = np.array([100, 101])
        ask = np.array([101, 102])
        self.assertEqual(crossed_locked_counts(bid, ask), (0, 0))


class TestSentinelRows(unittest.TestCase):
    def test_counts_either_side(self):
        bid = np.array([100, BID_SENTINEL_PRICE, 100])
        ask = np.array([101, 102, ASK_SENTINEL_PRICE])
        self.assertEqual(sentinel_row_count(bid, ask), 2)


class TestAlignmentEffectExpected(unittest.TestCase):
    def test_new_limit_adds(self):
        self.assertEqual(alignment_effect_expected(1, 7), 7)

    def test_partial_cancel_full_delete_execution_remove(self):
        for event_type in (2, 3, 4):
            self.assertEqual(alignment_effect_expected(event_type, 7), -7)


class TestLevelSizeAt(unittest.TestCase):
    def test_found_returns_size(self):
        self.assertEqual(level_size_at([100, 90], [10, 50], 100), 10)
        self.assertEqual(level_size_at([100, 90], [10, 50], 90), 50)

    def test_not_displayed_returns_none_not_zero(self):
        self.assertIsNone(level_size_at([100, 90], [10, 50], 500))

    def test_sentinel_price_never_matches(self):
        # A defensive guard: even if a message price coincided with the sentinel value
        # (never happens for real LOBSTER prices), a sentinel level must not be read as real.
        self.assertIsNone(level_size_at([ASK_SENTINEL_PRICE, 90], [0, 50], ASK_SENTINEL_PRICE))
        self.assertIsNone(level_size_at([BID_SENTINEL_PRICE, 90], [0, 50], BID_SENTINEL_PRICE))


class TestSampleAlignmentIndices(unittest.TestCase):
    def test_excludes_index_zero_and_non_scoreable_types(self):
        # types: 1(scoreable, but index 0 -> excluded anyway), 5(hidden, unaffected -> excluded),
        # 3(scoreable), 6(cross, unaffected -> excluded), 4(scoreable)
        event_type = np.array([1, 5, 3, 6, 4])
        idx = sample_alignment_indices(event_type, n_rows=5, sample_size=100, seed=0)
        self.assertEqual(sorted(idx), [2, 4])

    def test_deterministic_given_seed(self):
        event_type = np.array([1] + [1, 2, 3, 4] * 10)  # 40 scoreable rows after index 0
        first = sample_alignment_indices(event_type, n_rows=len(event_type), sample_size=5, seed=0)
        second = sample_alignment_indices(event_type, n_rows=len(event_type), sample_size=5, seed=0)
        self.assertEqual(first, second)
        self.assertEqual(len(first), 5)


class TestAlignmentHoldRate(unittest.TestCase):
    """Toy 5-event, 2-level book (built inline, see _toy_alignment_table):

    k=0: type=1 dir=+1 price=100 size=10 -> bid_sz_1 (from absent) 10
    k=1: type=1 dir=+1 price=100 size=5  -> bid_sz_1 10->15 (+5 new limit)
    k=2: type=2 dir=+1 price=100 size=5  -> bid_sz_1 15->10 (-5 partial cancel)
    k=3: type=3 dir=+1 price=100 size=10 -> bid level empties entirely (level-1 becomes 90)
    k=4: type=4 dir=-1 price=200 size=3  -> ask_sz_1 20->17 (visible execution)
    """

    EVENT_TYPE = [1, 1, 2, 3, 4]
    DIRECTION = [1, 1, 1, 1, -1]
    PRICE = [100, 100, 100, 100, 200]
    SIZE = [10, 5, 5, 10, 3]
    BID_PX = [[100, 90], [100, 90], [100, 90], [90, 80], [90, 80]]
    BID_SZ = [[10, 50], [15, 50], [10, 50], [50, 20], [50, 20]]
    ASK_PX = [[200, 210], [200, 210], [200, 210], [200, 210], [200, 210]]
    ASK_SZ = [[20, 5], [20, 5], [20, 5], [20, 5], [17, 5]]

    def test_identity_holds_on_aligned_toy_book(self):
        table, num_levels = _toy_alignment_table(
            self.EVENT_TYPE, self.DIRECTION, self.PRICE, self.SIZE,
            self.BID_PX, self.BID_SZ, self.ASK_PX, self.ASK_SZ,
        )
        result = compute_alignment_hold_rate(table, num_levels, seed=0, sample_size=10)
        self.assertEqual(result["n_sampled"], 4)  # k=1,2,3,4 (k=0 has no k-1)
        self.assertEqual(result["n_checked"], 4)
        self.assertEqual(result["n_holds"], 4)
        self.assertEqual(result["hold_rate"], 1.0)
        self.assertGreaterEqual(result["hold_rate"], ALIGNMENT_HOLD_RATE_THRESHOLD)

    def test_identity_fails_on_shifted_toy_book(self):
        # Same messages, book rows shifted +1 (row k gets the ORIGINAL row k+1's state; last
        # row duplicated so counts still match) -- the exact mutation used against the real
        # INTC/AMZN pair to prove make data catches an off-by-one.
        shifted_bid_px = self.BID_PX[1:] + [self.BID_PX[-1]]
        shifted_bid_sz = self.BID_SZ[1:] + [self.BID_SZ[-1]]
        shifted_ask_px = self.ASK_PX[1:] + [self.ASK_PX[-1]]
        shifted_ask_sz = self.ASK_SZ[1:] + [self.ASK_SZ[-1]]
        table, num_levels = _toy_alignment_table(
            self.EVENT_TYPE, self.DIRECTION, self.PRICE, self.SIZE,
            shifted_bid_px, shifted_bid_sz, shifted_ask_px, shifted_ask_sz,
        )
        result = compute_alignment_hold_rate(table, num_levels, seed=0, sample_size=10)
        self.assertEqual(result["n_holds"], 0)
        self.assertLess(result["hold_rate"], ALIGNMENT_HOLD_RATE_THRESHOLD)

    def test_level_absent_from_both_rows_excluded_not_scored_as_pass(self):
        # A single type-1 event at a price that never appears in either row's visible
        # levels (e.g. a deep order beyond the top-of-book depth carried in this toy book).
        # 0 - 0 == 0 would falsely "pass" a zero-effect read if not explicitly excluded.
        event_type = [1, 1]
        direction = [1, 1]
        price = [100, 500]  # k=1's price (500) is never a displayed level below
        size = [10, 7]
        bid_px = [[100, 90], [100, 90]]
        bid_sz = [[10, 50], [10, 50]]
        ask_px = [[200, 210], [200, 210]]
        ask_sz = [[20, 5], [20, 5]]
        table, num_levels = _toy_alignment_table(
            event_type, direction, price, size, bid_px, bid_sz, ask_px, ask_sz
        )
        result = compute_alignment_hold_rate(table, num_levels, seed=0, sample_size=10)
        self.assertEqual(result["n_sampled"], 1)  # only k=1 is eligible (k=0 has no k-1)
        self.assertEqual(result["n_checked"], 0)  # undecidable, excluded from the denominator
        self.assertEqual(result["n_holds"], 0)
        self.assertTrue(math.isnan(result["hold_rate"]))


class TestMonotonicTs(unittest.TestCase):
    def test_ties_are_not_violations(self):
        got = check_monotonic_ts(np.array([1, 1, 2, 2, 2, 3]))
        self.assertTrue(got["is_monotonic"])
        self.assertEqual(got["num_ties"], 3)  # (1,1), (2,2), (2,2)
        self.assertEqual(got["num_violations"], 0)

    def test_strict_decrease_flagged_at_the_later_index(self):
        got = check_monotonic_ts(np.array([1, 5, 4, 9]))
        self.assertFalse(got["is_monotonic"])
        self.assertEqual(got["num_violations"], 1)
        self.assertEqual(got["first_violation_index"], 2)  # index of the 4, not the 5

    def test_short_input(self):
        got = check_monotonic_ts(np.array([7]))
        self.assertTrue(got["is_monotonic"])
        self.assertIsNone(got["first_violation_index"])


class TestGaps(unittest.TestCase):
    def test_top_k_sorted_descending(self):
        ts = np.array([0, 10, 40, 45, 145])
        gaps = top_k_gaps(ts, k=2)
        self.assertEqual([g["gap_ns"] for g in gaps], [100, 30])
        self.assertEqual(gaps[0]["before_idx"], 3)
        self.assertEqual(gaps[0]["after_ts_ns"], 145)

    def test_threshold_is_strict(self):
        ts = np.array([0, 10, 20, 40])
        self.assertEqual(gaps_over_threshold(ts, 10), 1)  # only the 20 -> 40 gap
        self.assertEqual(gaps_over_threshold(ts, 20), 0)


class TestMedianMid(unittest.TestCase):
    def test_scaling_is_ten_thousand_units_per_dollar(self):
        bid = np.array([270_100, 270_200])
        ask = np.array([270_300, 270_400])
        # mids 270200 and 270300 -> median 270250 raw units -> $27.025
        self.assertAlmostEqual(median_mid_dollars(bid, ask), 27.025, places=6)


class TestNsToHhmmss(unittest.TestCase):
    def test_market_open(self):
        self.assertEqual(ns_to_hhmmss(DAY_START_NS), "09:30:00")

    def test_truncates_sub_second(self):
        self.assertEqual(ns_to_hhmmss(DAY_START_NS + 1_999_999_999), "09:30:01")


class TestIntradayProfile(unittest.TestCase):
    def test_binning_and_means(self):
        # Two events in bin 0 (09:30-10:00), one in bin 1 (10:00-10:30).
        ts = np.array([DAY_START_NS, DAY_START_NS + 60 * 10**9, DAY_START_NS + BIN_NS + 1])
        bid_px = np.array([100, 100, 200])
        ask_px = np.array([200, 300, 400])  # spreads 100, 200, 200 raw units
        bid_sz = np.array([1, 2, 3])
        ask_sz = np.array([4, 5, 6])  # depths 5, 7, 9

        profile = build_intraday_profile(ts, bid_px, bid_sz, ask_px, ask_sz)
        self.assertEqual(len(profile), 13)  # 09:30-16:00 in 30-minute bins
        self.assertEqual(profile[0]["event_count"], 2)
        self.assertEqual(profile[1]["event_count"], 1)
        self.assertEqual(profile[0]["bin_start"], "09:30:00")
        # bin 0 mean spread = (100 + 200)/2 = 150 raw units = 1.5 ticks
        self.assertAlmostEqual(profile[0]["mean_spread_ticks"], 150 / TICK_SIZE_UNITS)
        self.assertAlmostEqual(profile[0]["mean_l1_depth"], 6.0)  # (5 + 7)/2
        self.assertEqual(profile[5]["event_count"], 0)
        self.assertNotEqual(profile[5]["mean_spread_ticks"], profile[5]["mean_spread_ticks"])  # NaN

    def test_out_of_range_stamps_clip_into_the_end_bins(self):
        ts = np.array([DAY_START_NS - 10**9, DAY_START_NS + 100 * BIN_NS])
        one = np.array([1, 1])
        profile = build_intraday_profile(ts, one, one, one * 2, one)
        self.assertEqual(profile[0]["event_count"], 1)
        self.assertEqual(profile[-1]["event_count"], 1)


class TestUShape(unittest.TestCase):
    def test_open_close_heavy_profile_passes(self):
        counts = [100, 90, 20, 10, 10, 10, 10, 20, 80, 100, 30, 40]
        # edge_n = 12//4 = 3: edges mean = (100+90+20+80+100+30... ) -- computed by hand below
        edges = counts[:3] + counts[-3:]
        middle = counts[3:-3]
        self.assertGreater(sum(edges) / len(edges), sum(middle) / len(middle))
        self.assertTrue(confirm_u_shape(counts))

    def test_flat_profile_fails(self):
        self.assertFalse(confirm_u_shape([10] * 12))

    def test_too_few_bins(self):
        self.assertFalse(confirm_u_shape([1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
