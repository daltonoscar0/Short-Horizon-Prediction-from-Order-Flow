"""Hand-computed checks for the pure data-quality primitives in edge.io.quality."""
import unittest

import numpy as np

from edge.io.lobster import ASK_SENTINEL_PRICE, BID_SENTINEL_PRICE
from edge.io.quality import (
    BIN_NS,
    DAY_START_NS,
    TICK_SIZE_UNITS,
    build_intraday_profile,
    check_monotonic_ts,
    confirm_u_shape,
    crossed_locked_counts,
    gaps_over_threshold,
    median_mid_dollars,
    ns_to_hhmmss,
    sentinel_row_count,
    top_k_gaps,
)


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
