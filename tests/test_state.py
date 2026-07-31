"""Hand-computed tests for state.py (src/edge/features/state.py) against tests/toybook.py.

Every expected value below is derived arithmetically in a comment above the assertion,
from the toy fixture's documented raw values (tests/toybook.py's module docstring), never
by calling the function under test a second way.
"""
import unittest

import numpy as np

from edge.features import NS_PER_SEC, SESSION_END_NS, SESSION_START_NS, TOD_BUCKET_NS
from edge.features import state as fam
from edge.features.book import book_at_stamps
from edge.features.clocks import Stamps, stamps_for_clock
from edge.features.frame import event_frame_from_table
from edge.io.lobster import ASK_SENTINEL_PRICE, BID_SENTINEL_PRICE

from tests.toybook import (
    TOY_TS_NS,
    build_toy_frame,
    make_event_frame,
    toy_arrays,
    toy_event_frame,
)

# The 22 columns state.py must emit, exactly (FEATURE_WINDOWS = ('100ms', '1s', '5s')).
EXPECTED_COLUMNS = {
    "st_spread_ticks",
    "st_spread_ticks_int",
    "st_depth_bid_touch",
    "st_depth_ask_touch",
    "st_depth_touch",
    "st_depth_total",
    "st_slope_bid",
    "st_slope_ask",
    "st_tod_bucket",
    "st_tod_frac",
    "st_rv_100ms",
    "st_rv_1s",
    "st_rv_5s",
    "st_event_rate_100ms",
    "st_event_rate_1s",
    "st_event_rate_5s",
    "st_cancel_rate_100ms",
    "st_cancel_rate_1s",
    "st_cancel_rate_5s",
    "st_avg_order_size_100ms",
    "st_avg_order_size_1s",
    "st_avg_order_size_5s",
}


def _stamps_at(indices):
    """Stamps() landing exactly on the given toy-frame event indices."""
    idx = np.asarray(indices, dtype=np.int64)
    return Stamps(index=idx, ts_ns=np.array(TOY_TS_NS, dtype=np.int64)[idx])


class TestOutputShape(unittest.TestCase):
    def test_exactly_the_22_expected_columns(self):
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([0, 3]))
        self.assertEqual(set(out), EXPECTED_COLUMNS)
        for key, arr in out.items():
            self.assertTrue(key.startswith("st_"))
            self.assertNotIn("_sadj", key)
            self.assertEqual(arr.shape, (2,))
            self.assertEqual(arr.dtype, np.float64)

    def test_book_none_matches_passed_in_book_bit_identical(self):
        frame = toy_event_frame()
        stamps = _stamps_at(range(8))
        book = book_at_stamps(frame, stamps)
        out_with_book = fam.compute(frame, stamps, book)
        out_without_book = fam.compute(frame, stamps)
        for key in EXPECTED_COLUMNS:
            np.testing.assert_array_equal(out_with_book[key], out_without_book[key])


class TestSpreadTicks(unittest.TestCase):
    def test_hand_computed_spread_regime(self):
        # toybook.py: spread = [500,...,1000,...] raw -> [5,...,10,...] ticks (rows 0-2
        # vs 3-7). Row 0: ask 101000 - bid 100500 = 500 raw = 5 ticks.
        # Row 3: ask 101000 - bid 100000 = 1000 raw = 10 ticks.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([0, 3]))
        self.assertEqual(out["st_spread_ticks"].tolist(), [5.0, 10.0])
        # Both already integral, so np.round is a no-op here.
        self.assertEqual(out["st_spread_ticks_int"].tolist(), [5.0, 10.0])


class TestDepth(unittest.TestCase):
    def test_hand_computed_touch_and_total_depth(self):
        # Row 0: bid_sz_l=[10,40], ask_sz_l=[50,30] (toybook.py TOY_BID_SZ/TOY_ASK_SZ[0]).
        #   touch: bid=10, ask=50, touch=10+50=60.
        #   total (both levels valid): (10+50) + (40+30) = 60 + 70 = 130.
        # Row 3: bid_sz_l=[40,15], ask_sz_l=[50,30] (bid touch deleted -> L1 40, L2 15;
        #   ask untouched: 50,30 -- the ask delete/depletion happens at events 4 and 7).
        #   touch: bid=40, ask=50, touch=90.
        #   total: (40+50) + (15+30) = 90 + 45 = 135.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([0, 3]))
        self.assertEqual(out["st_depth_bid_touch"].tolist(), [10.0, 40.0])
        self.assertEqual(out["st_depth_ask_touch"].tolist(), [50.0, 50.0])
        self.assertEqual(out["st_depth_touch"].tolist(), [60.0, 90.0])
        self.assertEqual(out["st_depth_total"].tolist(), [130.0, 135.0])


class TestSlope(unittest.TestCase):
    def test_hand_computed_two_point_regression_row0(self):
        # Row 0: mid=100750. bid_px_l=[100500,100000], bid_sz_l=[10,40].
        #   distance_bid = (mid-px)/100 -> [2.5, 7.5]; cumulative size -> [10, 50].
        #   Two points (2.5,10),(7.5,50): slope = (50-10)/(7.5-2.5) = 40/5 = 8.0.
        # ask_px_l=[101000,102000], ask_sz_l=[50,30].
        #   distance_ask = (px-mid)/100 -> [2.5, 12.5]; cumulative size -> [50, 80].
        #   slope = (80-50)/(12.5-2.5) = 30/10 = 3.0.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([0]))
        self.assertAlmostEqual(out["st_slope_bid"][0], 8.0, places=12)
        self.assertAlmostEqual(out["st_slope_ask"][0], 3.0, places=12)

    def test_hand_computed_two_point_regression_row3(self):
        # Row 3: mid=100500. bid_px_l=[100000,98000], bid_sz_l=[40,15].
        #   distance_bid -> [(100500-100000)/100, (100500-98000)/100] = [5.0, 25.0].
        #   cumulative size -> [40, 55]. slope = (55-40)/(25-5) = 15/20 = 0.75.
        # ask_px_l=[101000,102000], ask_sz_l=[50,30] (unchanged from row 0).
        #   distance_ask -> [(101000-100500)/100, (102000-100500)/100] = [5.0, 15.0].
        #   cumulative size -> [50, 80]. slope = (80-50)/(15-5) = 30/10 = 3.0.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([3]))
        self.assertAlmostEqual(out["st_slope_bid"][0], 0.75, places=12)
        self.assertAlmostEqual(out["st_slope_ask"][0], 3.0, places=12)


class TestSlopeNaNRules(unittest.TestCase):
    def test_fewer_than_two_valid_bid_levels_is_nan(self):
        # Variant: row 0's bid L2 becomes the LOBSTER sentinel (no displayed level) ->
        # only 1 valid bid level -> slope_bid must be NaN. The ask side is untouched, so
        # slope_ask must still be the row-0 value (3.0) computed above -- proving the two
        # sides are masked independently, not gated by a single shared "book valid" flag.
        d = toy_arrays()
        d["bid_px"][0][1] = BID_SENTINEL_PRICE
        d["bid_sz"][0][1] = 0
        frame = make_event_frame(**d)
        out = fam.compute(frame, _stamps_at([0]))
        self.assertTrue(np.isnan(out["st_slope_bid"][0]))
        self.assertAlmostEqual(out["st_slope_ask"][0], 3.0, places=12)

    def test_zero_x_variance_is_nan(self):
        # Variant: row 0's bid L2 price is forced equal to L1's (100500), so both valid
        # levels sit at the SAME distance from mid (2.5 ticks each) -> zero x-variance.
        d = toy_arrays()
        d["bid_px"][0][1] = d["bid_px"][0][0]
        frame = make_event_frame(**d)
        out = fam.compute(frame, _stamps_at([0]))
        self.assertTrue(np.isnan(out["st_slope_bid"][0]))


class TestBookInvalidNaN(unittest.TestCase):
    def test_touch_features_are_nan_when_book_invalid_but_depth_total_is_not(self):
        # Variant: row 0's ask touch becomes the sentinel (no displayed level) -> the
        # TOUCH is invalid (book.valid False), so spread/touch-depth/slope must all be
        # NaN. Level 2 (bid 100000/40, ask 102000/30, untouched) is still a valid level,
        # so depth_total is a real number, not NaN: 0 (masked L1) + (40+30) = 70 -- the
        # sec.7 "sum over an empty/partial window is legitimately real, not missing" rule
        # applied per-level, not gated by the touch.
        d = toy_arrays()
        d["ask_px"][0][0] = ASK_SENTINEL_PRICE
        d["ask_sz"][0][0] = 0
        frame = make_event_frame(**d)
        out = fam.compute(frame, _stamps_at([0]))
        self.assertTrue(np.isnan(out["st_spread_ticks"][0]))
        self.assertTrue(np.isnan(out["st_spread_ticks_int"][0]))
        self.assertTrue(np.isnan(out["st_depth_bid_touch"][0]))
        self.assertTrue(np.isnan(out["st_depth_ask_touch"][0]))
        self.assertTrue(np.isnan(out["st_depth_touch"][0]))
        self.assertTrue(np.isnan(out["st_slope_bid"][0]))
        self.assertTrue(np.isnan(out["st_slope_ask"][0]))
        self.assertEqual(out["st_depth_total"][0], 70.0)


class TestTimeOfDay(unittest.TestCase):
    def test_hand_computed_bucket_and_fraction(self):
        # Row 0 ts_ns = 34200000000001 = SESSION_START_NS + 1.
        # bucket = floor(1 / TOD_BUCKET_NS) = floor(1 / 1_800_000_000_000) = 0.
        # frac = 1 / (SESSION_END_NS - SESSION_START_NS) = 1 / 23_400_000_000_000.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([0]))
        self.assertEqual(out["st_tod_bucket"][0], 0.0)
        # 1 / 23_400_000_000_000 = 4.273504273504274e-14, asserted as a LITERAL rather than
        # re-derived from the same constants the module uses (INTERFACE.md sec. 10).
        self.assertAlmostEqual(out["st_tod_frac"][0], 4.273504273504274e-14, places=18)
        self.assertGreater(TOD_BUCKET_NS, 0)  # guards the imported constant's sanity

    def test_buckets_advance_across_the_session(self):
        # Every stamp on the 8-event toy fixture falls inside the first half-hour, so
        # bucket 0 is the ONLY value it can ever produce -- st_tod_bucket could be
        # hardcoded to zero (or TOD_BUCKET_NS scaled by 10) and the fixture would not
        # notice. These stamps are placed by hand, deep into the session, so the bucketing
        # arithmetic is actually exercised.
        #   09:30:00 + 0        -> floor(0 / 1800s)     = bucket 0,  frac 0.0
        #   09:30:00 + 1800s    -> floor(1800 / 1800)   = bucket 1   (first boundary, exact)
        #   09:30:00 + 1799.999s-> still bucket 0       (just inside the boundary)
        #   09:30:00 + 12600s   -> floor(12600 / 1800)  = bucket 7   (13:00:00)
        #   09:30:00 + 23400s   -> floor(23400 / 1800)  = bucket 13  (16:00:00, the close)
        offsets_s = [0, 1800, 12600, 23400]
        expected_buckets = [0.0, 1.0, 7.0, 13.0]
        ts = np.array(
            [SESSION_START_NS + int(s) * 1_000_000_000 for s in offsets_s], dtype=np.int64
        )
        frame = toy_event_frame()
        # Every stamp resolves to the last event <= t, which is the final toy row (row 7)
        # for all of these; the time-of-day columns depend only on stamps.ts_ns.
        stamps = Stamps(index=np.full(len(offsets_s), 7, dtype=np.int64), ts_ns=ts)
        out = fam.compute(frame, stamps)
        self.assertEqual(out["st_tod_bucket"].tolist(), expected_buckets)
        # frac at 13:00:00 = 12600 / 23400 = 0.5384615384615384 (a literal, not a re-derivation).
        self.assertAlmostEqual(out["st_tod_frac"][2], 0.5384615384615384, places=12)
        # frac at the close is exactly 1.0.
        self.assertAlmostEqual(out["st_tod_frac"][3], 1.0, places=12)


class TestRealizedVolRightEdge(unittest.TestCase):
    """Pins st_rv_1s's right edge: a window ending at event 2 must NOT see event 3's jump.

    toybook.py's mids are [100750,100750,100750,100500,100500,100500,100500,100500] raw,
    i.e. per-event changes [-, 0, 0, -250, 0, 0, 0, 0] raw = [-, 0, 0, -2.5, 0, 0, 0, 0]
    ticks. Contribution[n] = (mid(n)-mid(n-1))^2 in ticks^2, attached to event n:
    contrib = [NaN, 0, 0, 6.25, 0, 0, 0, 0]. A 1s window easily spans events 0-3 from
    either stamp (all 8 toy events sit within 400 microseconds of each other).
    """

    def test_window_at_event_2_excludes_event_3s_jump_event_3_includes_it(self):
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([2, 3]))
        rv = out["st_rv_1s"]
        # Stamp at event 2: window (t-1s, t] = events 0-2, contributions [NaN,0,0] ->
        # sum (NaN excluded) = 0 -> rv = sqrt(0) = 0.0. A right-edge off-by-one would
        # spuriously pull in event 3's contribution and give 2.5 here instead.
        self.assertEqual(rv[0], 0.0)
        # Stamp at event 3: window = events 0-3, contributions [NaN,0,0,6.25] ->
        # sum = 6.25 -> rv = sqrt(6.25) = 2.5.
        self.assertEqual(rv[1], 2.5)


class TestRealizedVolWarmupNaN(unittest.TestCase):
    def test_first_event_alone_has_no_prior_transition_so_rv_is_nan(self):
        # Stamp at event 0: the only contribution slot reachable is contribution[0],
        # which is NaN by definition (no event -1 to difference mid(0) against) -> the
        # window holds zero FINITE contributions -> rv is NaN, not 0.0.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([0]))
        self.assertTrue(np.isnan(out["st_rv_1s"][0]))


class TestEventRate(unittest.TestCase):
    def test_hand_computed_rate_at_event_3(self):
        # Stamp at event 3 (ts=34200100000500).
        # 1s window: lower_bound way before event 0 -> start=0, end_exclusive=4 ->
        #   4 events / 1.0s = 4.0 events/sec.
        # 100ms window: lower_bound = 34200100000500 - 100_000_000 = 34200000000500.
        #   Frame ts: idx0=...000001, idx1=...000002 (both < lower_bound -> excluded);
        #   idx2=34200100000000 (> lower_bound -> included); idx3=...000500 (included).
        #   -> 2 events / 0.1s = 20.0 events/sec.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([3]))
        self.assertEqual(out["st_event_rate_1s"][0], 4.0)
        self.assertEqual(out["st_event_rate_100ms"][0], 20.0)


class TestCancelRate(unittest.TestCase):
    def test_hand_computed_rate_at_event_3(self):
        # 1s window at event 3 covers events 0-3, types [1,1,2,3]. All four are
        # book-changing (types 1-4). Cancel types (2,3) count = 2 (events 2,3).
        # rate = 2/4 = 0.5.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([3]))
        self.assertEqual(out["st_cancel_rate_1s"][0], 0.5)

    def test_zero_book_changing_denominator_is_nan(self):
        # Variant: push event 5 (type 5, hidden execution -- NOT book-changing) 6
        # seconds after event 4, so the largest FEATURE_WINDOWS lookback (5s) at that
        # stamp reaches back only to event 5 itself, never to any type 1-4 event.
        # window (t-5s, t] = {event 5} only; book-changing count = 0 -> NaN.
        d = toy_arrays()
        d["ts_ns"][5] = TOY_TS_NS[4] + 6 * NS_PER_SEC
        d["ts_ns"][6] = d["ts_ns"][5] + 1 * NS_PER_SEC
        d["ts_ns"][7] = d["ts_ns"][6] + 1 * NS_PER_SEC
        frame = make_event_frame(**d)
        stamps = Stamps(index=np.array([5], dtype=np.int64), ts_ns=np.array([d["ts_ns"][5]], dtype=np.int64))
        out = fam.compute(frame, stamps)
        self.assertTrue(np.isnan(out["st_cancel_rate_5s"][0]))


class TestAvgOrderSize(unittest.TestCase):
    def test_hand_computed_average_over_two_new_limit_orders(self):
        # Stamp at event 1, 1s window -> covers events 0,1, both type-1 (new limit),
        # sizes 10 and 20 (toybook.py TOY_SIZE). avg = (10+20)/2 = 15.0.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([1]))
        self.assertEqual(out["st_avg_order_size_1s"][0], 15.0)

    def test_hand_computed_average_over_single_new_limit_order(self):
        # Stamp at event 0 alone, 1s window -> covers only event 0, type-1, size 10.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([0]))
        self.assertEqual(out["st_avg_order_size_1s"][0], 10.0)

    def test_zero_type1_events_in_window_is_nan(self):
        # Stamp at event 4, 100ms window: lower_bound = 34200200000000 - 100_000_000 =
        # 34200100000000, which EQUALS event 2's ts exactly -> excluded (side="right").
        # Window = events 3,4 (types 3,4) -- no type-1 event in range -> NaN.
        frame = toy_event_frame()
        out = fam.compute(frame, _stamps_at([4]))
        self.assertTrue(np.isnan(out["st_avg_order_size_100ms"][0]))


class TestBothClocks(unittest.TestCase):
    def test_runs_on_calendar_and_event_clocks_with_right_shape_and_dtype(self):
        frame = toy_event_frame()
        for clock in ("calendar", "event"):
            stamps = stamps_for_clock(frame, clock)
            out = fam.compute(frame, stamps)
            n = len(stamps.index)
            self.assertEqual(set(out), EXPECTED_COLUMNS)
            for arr in out.values():
                self.assertEqual(arr.shape, (n,))
                self.assertEqual(arr.dtype, np.float64)

    def test_event_clock_is_legitimately_empty_on_this_toy_frame(self):
        # EVENT_CLOCK_N=10 book-changing events; the toy frame has only 7 -> empty
        # Stamps, and compute() must return length-0 float64 arrays, not raise.
        frame = toy_event_frame()
        stamps = stamps_for_clock(frame, "event")
        self.assertEqual(len(stamps.index), 0)
        out = fam.compute(frame, stamps)
        for key in EXPECTED_COLUMNS:
            self.assertEqual(out[key].shape, (0,))
            self.assertEqual(out[key].dtype, np.float64)


class TestNoLookahead(unittest.TestCase):
    """The violation detector: future events must not move a single value at an earlier
    stamp. Exercises the per-event windowed features (rv/event_rate/cancel_rate/
    avg_order_size), which read `frame` directly rather than only `book`."""

    def test_appending_future_events_leaves_earlier_stamps_bit_identical(self):
        full_table = build_toy_frame()
        past_table = full_table.slice(0, 4)  # events 0-3 only; events 4-7 are "the future"
        full = event_frame_from_table(full_table, num_levels=2)
        past = event_frame_from_table(past_table, num_levels=2)

        stamps = _stamps_at([0, 1, 2, 3])
        out_full = fam.compute(full, stamps)
        out_past = fam.compute(past, stamps)

        self.assertEqual(set(out_full), set(out_past))
        for key in out_full:
            np.testing.assert_array_equal(
                out_full[key],
                out_past[key],
                err_msg=f"{key} at an earlier stamp changed when future events were appended",
            )


if __name__ == "__main__":
    unittest.main()
