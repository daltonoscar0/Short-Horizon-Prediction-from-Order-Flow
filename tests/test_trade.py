"""Hand-computed tests for trade.py (src/edge/features/trade.py) against tests/toybook.py.

Toy fixture recap (0-indexed, full derivation in tests/toybook.py):
  ts_ns  = [...001, ...002, 100000000, 100000500, 200000000, 200000001, 300000000, 400000000]
           (all offsets from 34200000000000)
  type   = [1, 1, 2, 3, 4, 5, 1, 3]
  dir    = [1, 1, 1, 1, -1, -1, -1, -1]
  price  = [100500, 100500, 100500, 100500, 101000, 101000, 103000, 102000]
  size   = [10, 20, 5, 25, 10, 5, 15, 30]
  mid    = [100750, 100750, 100750, 100500, 100500, 100500, 100500, 100500]  (per-event, from
           book_at_events / frame.mid: rows 0-2 bid 100500/ask 101000, rows 3-7 bid 100000/ask 101000)

Only events 4 (type 4, dir -1, price 101000, size 10) and 5 (type 5, dir -1, price 101000,
size 5) are trades (TRADE_EVENT_TYPES = (4, 5)). Both have direction -1 (resting ASK order
hit), so aggressor sign = -direction = +1 for both: BOTH are buyer-initiated.
"""
import unittest

import numpy as np

import edge.features.trade as trade
from edge.features.book import book_at_stamps
from edge.features.clocks import Stamps, stamps_for_clock
from edge.features.frame import event_frame_from_table

from tests.toybook import TOY_TS_NS, build_toy_frame, make_event_frame, toy_arrays, toy_event_frame

ALL_INDEX_STAMPS = Stamps(
    index=np.arange(8, dtype=np.int64), ts_ns=np.array(TOY_TS_NS, dtype=np.int64)
)


def _stamp(i):
    """A single-stamp Stamps object anchored exactly at toy-fixture event i."""
    return Stamps(index=np.array([i], dtype=np.int64), ts_ns=np.array([TOY_TS_NS[i]], dtype=np.int64))


class TestColumnManifest(unittest.TestCase):
    def test_emits_exactly_the_23_documented_columns(self):
        out = trade.compute(toy_event_frame(), ALL_INDEX_STAMPS)

        expected = set()
        for w in ("100ms", "1s", "5s"):
            expected.update(
                {
                    f"tr_signed_vol_{w}",
                    f"tr_buy_vol_{w}",
                    f"tr_sell_vol_{w}",
                    f"tr_count_{w}",
                    f"tr_aggr_imb_{w}",
                    f"tr_avg_size_{w}",
                    f"tr_quote_ratio_{w}",
                }
            )
        expected.update({"tr_eff_spread", "tr_last_trade_sign"})

        self.assertEqual(len(expected), 23)
        self.assertEqual(set(out.keys()), expected)
        for key, arr in out.items():
            self.assertTrue(key.startswith("tr_"), key)
            self.assertEqual(arr.shape, (8,), key)
            self.assertEqual(arr.dtype, np.float64, key)

    def test_result_identical_with_or_without_explicit_book(self):
        # book is accepted but never read (see module docstring); confirm the None-vs-
        # explicit paths are bit-identical, as INTERFACE.md sec. 5 requires of every family.
        frame = toy_event_frame()
        out_default = trade.compute(frame, ALL_INDEX_STAMPS)
        out_explicit = trade.compute(frame, ALL_INDEX_STAMPS, book_at_stamps(frame, ALL_INDEX_STAMPS))
        for key in out_default:
            np.testing.assert_array_equal(out_default[key], out_explicit[key], err_msg=key)


class TestBeforeFirstTradeQuietWindow(unittest.TestCase):
    """Stamp at event 3 (34200100000500), strictly before the first trade (event 4)."""

    def setUp(self):
        self.out = trade.compute(toy_event_frame(), _stamp(3))

    def test_windowed_sums_are_zero_not_nan_and_ratios_are_nan(self):
        # 100ms window at t=34200100000500: lower_bound = t - 100ms = 34200000000500.
        # trailing_start_index picks the first frame ts strictly > lower_bound:
        # ts[1]=...000002 < lower_bound < ts[2]=100000000 -> start=2. end_exclusive=3+1=4.
        # Window = events {2, 3} (types 2, 3) -- neither is a trade.
        # tr_signed_vol_100ms: a SUM over a window with zero trades is legitimately 0.0.
        self.assertEqual(self.out["tr_signed_vol_100ms"][0], 0.0)
        self.assertEqual(self.out["tr_buy_vol_100ms"][0], 0.0)
        self.assertEqual(self.out["tr_sell_vol_100ms"][0], 0.0)
        self.assertEqual(self.out["tr_count_100ms"][0], 0.0)
        # tr_aggr_imb: total volume (buy+sell) is 0 -> the ratio's own denominator is 0 -> NaN.
        self.assertTrue(np.isnan(self.out["tr_aggr_imb_100ms"][0]))
        # tr_avg_size: mean over 0 trades -> NaN.
        self.assertTrue(np.isnan(self.out["tr_avg_size_100ms"][0]))
        # tr_quote_ratio: denominator is book-changing-event count, NOT trade count.
        # Events 2 (type 2) and 3 (type 3) are BOTH book-changing -> bc_count = 2.
        # ratio = trade_count / bc_count = 0 / 2 = 0.0 -- a legitimate ratio, NOT NaN,
        # even though the numerator is 0. This is the case that distinguishes
        # tr_quote_ratio's zero-denominator rule from tr_aggr_imb's.
        self.assertEqual(self.out["tr_quote_ratio_100ms"][0], 0.0)

    def test_last_trade_fields_are_nan_before_any_trade(self):
        self.assertTrue(np.isnan(self.out["tr_eff_spread"][0]))
        self.assertTrue(np.isnan(self.out["tr_last_trade_sign"][0]))


class TestVisibleTradeExactlyAtStampIsIncluded(unittest.TestCase):
    """Stamp at event 4 itself: type 4, dir -1, price 101000, size 10 -- the TARGETED
    boundary test that a trade stamped exactly at t is included, both in the trailing
    window sums and as "the last trade at or before t"."""

    def setUp(self):
        self.out = trade.compute(toy_event_frame(), _stamp(4))

    def test_windowed_features_include_the_boundary_trade(self):
        # 100ms window at t=34200200000000: lower_bound = 34200100000000 = ts[2] exactly.
        # side="right" excludes an exact tie at the left edge -> start=3. end_exclusive=5.
        # Window = events {3, 4} (types 3, 4). Only event 4 is a trade: aggressor sign
        # = -direction = -(-1) = +1 (buy), size 10 -> signed contribution +10.0.
        self.assertEqual(self.out["tr_signed_vol_100ms"][0], 10.0)
        self.assertEqual(self.out["tr_buy_vol_100ms"][0], 10.0)
        self.assertEqual(self.out["tr_sell_vol_100ms"][0], 0.0)  # sum over 0 sells: 0.0, not NaN
        self.assertEqual(self.out["tr_count_100ms"][0], 1.0)
        # aggr_imb = (buy - sell) / (buy + sell) = (10 - 0) / (10 + 0) = 1.0.
        self.assertEqual(self.out["tr_aggr_imb_100ms"][0], 1.0)
        # avg_size = 10 / 1 = 10.0.
        self.assertEqual(self.out["tr_avg_size_100ms"][0], 10.0)
        # bc_count over {3, 4}: types 3 and 4 are both book-changing -> bc_count = 2.
        # quote_ratio = trade_count / bc_count = 1 / 2 = 0.5.
        self.assertEqual(self.out["tr_quote_ratio_100ms"][0], 0.5)

    def test_effective_spread_reads_mid_immediately_before_the_trade(self):
        # The mid immediately before event 4 is the mid AT EVENT 3 (the row preceding the
        # trade row), not event 4's own post-trade mid: mid[3] = 100500 raw (bid 100000 /
        # ask 101000). Effective spread = 2 * |P_trade - mid_before|
        #   = 2 * |101000 - 100500| = 2 * 500 = 1000 raw = 1000 / 100 = 10.0 ticks.
        self.assertEqual(self.out["tr_eff_spread"][0], 10.0)
        # aggressor sign of event 4 = -direction = -(-1) = +1.0 (buyer-initiated).
        self.assertEqual(self.out["tr_last_trade_sign"][0], 1.0)


class TestHiddenTradeExactlyAtStampIsIncluded(unittest.TestCase):
    """Stamp at event 5 itself: type 5 (HIDDEN execution), dir -1, price 101000, size 5.
    Pins that a type-5 trade both contributes to the windows and is picked up as "the
    last trade", exactly like a type-4 trade."""

    def setUp(self):
        self.out = trade.compute(toy_event_frame(), _stamp(5))

    def test_windowed_features_over_a_1s_lookback(self):
        # 1s window at t=34200200000001: lower_bound = t - 1e9 = 34199200000001, which is
        # below every toy timestamp -> start=0. end_exclusive=5+1=6. Window = events 0-5.
        # Trades in {0..5}: event 4 (+10, buy) and event 5 (+5, buy; -direction=-(-1)=+1).
        self.assertEqual(self.out["tr_signed_vol_1s"][0], 15.0)
        self.assertEqual(self.out["tr_buy_vol_1s"][0], 15.0)
        self.assertEqual(self.out["tr_sell_vol_1s"][0], 0.0)
        self.assertEqual(self.out["tr_count_1s"][0], 2.0)
        # aggr_imb = (15 - 0) / (15 + 0) = 1.0.
        self.assertEqual(self.out["tr_aggr_imb_1s"][0], 1.0)
        # avg_size = (10 + 5) / 2 = 7.5.
        self.assertEqual(self.out["tr_avg_size_1s"][0], 7.5)
        # bc_count over events 0-5: types [1,1,2,3,4,5] -- only the type-5 event (5) is
        # NOT book-changing -> bc_count = 5. quote_ratio = 2 / 5 = 0.4.
        self.assertEqual(self.out["tr_quote_ratio_1s"][0], 0.4)

    def test_hidden_trade_is_the_last_trade_and_mid_before_reads_row_4(self):
        # Last trade at/before t is event 5 (the "<=" tie at t includes it). mid_before is
        # the mid at the row IMMEDIATELY PRECEDING event 5, i.e. event 4's post-state:
        # mid[4] = 100500 raw (unchanged from event 3/4 -- event 5 is hidden, book
        # unaffected, so mid[4] == mid[3] here, but the code must still read row 4, not
        # row 5 or row 3; see TestOffByOnePinAggressiveSellClearsBid for a case where the
        # three differ).
        # eff_spread = 2 * |101000 - 100500| = 1000 raw = 10.0 ticks.
        self.assertEqual(self.out["tr_eff_spread"][0], 10.0)
        # aggressor sign of event 5 = -(-1) = +1.0.
        self.assertEqual(self.out["tr_last_trade_sign"][0], 1.0)


class TestLastTradePersistsPastItsOwnWindow(unittest.TestCase):
    """Stamp at the final event (7): the last trade (event 5) is long out of a 100ms
    window, but tr_eff_spread / tr_last_trade_sign must still report it -- they are not
    windowed."""

    def setUp(self):
        self.out = trade.compute(toy_event_frame(), _stamp(7))

    def test_100ms_window_is_quiet_but_last_trade_fields_persist(self):
        # 100ms window at t=34200400000000: lower_bound = 34200300000000 = ts[6] exactly.
        # side="right" excludes the tie at the left edge -> start=7, end_exclusive=8.
        # Window = {event 7} only (type 3, a delete, not a trade).
        self.assertEqual(self.out["tr_signed_vol_100ms"][0], 0.0)
        self.assertEqual(self.out["tr_count_100ms"][0], 0.0)
        self.assertTrue(np.isnan(self.out["tr_aggr_imb_100ms"][0]))
        self.assertTrue(np.isnan(self.out["tr_avg_size_100ms"][0]))
        # bc_count = 1 (event 7, type 3, is book-changing); trade count 0 -> ratio 0/1=0.0.
        self.assertEqual(self.out["tr_quote_ratio_100ms"][0], 0.0)
        # tr_eff_spread / tr_last_trade_sign are NOT windowed: they still report event 5,
        # the last trade at or before t, identically to TestHiddenTradeExactlyAtStamp.
        self.assertEqual(self.out["tr_eff_spread"][0], 10.0)
        self.assertEqual(self.out["tr_last_trade_sign"][0], 1.0)


class TestOffByOnePinAggressiveSellClearsBid(unittest.TestCase):
    """A variant built from toy_arrays() (tests/toybook.py: "Build the variant, do not
    build a new fixture") where event 4 becomes an aggressive SELL that clears the bid
    touch entirely, moving the book AT that very row. Unlike the stock fixture (where
    neither trade moves the touch price, so mid_before == mid_after coincidentally), this
    variant makes mid_before and mid_after clearly different numbers, so reading the wrong
    row is caught by value, not just by construction.
    """

    def setUp(self):
        d = toy_arrays()
        # Flip event 4 from "ask depleted" (dir -1) to "bid touch cleared by an aggressive
        # sell" (dir +1: the resting BID order is the one executed). Trade price 100000
        # matches the bid touch being hit; size 40 exactly consumes it.
        d["direction"][4] = 1
        d["price"][4] = 100000
        d["size"][4] = 40
        # Post-trade book: bid touch drops from (100000, 40) [event 3's state] to a much
        # lower level; ask side is untouched by a bid-side trade.
        d["bid_px"][4] = [95000, 90000]
        d["bid_sz"][4] = [10, 5]
        d["ask_px"][4] = [101000, 102000]  # unchanged from event 3
        d["ask_sz"][4] = [50, 30]  # unchanged from event 3
        self.frame = make_event_frame(**d)
        self.out = trade.compute(self.frame, _stamp(4))

    def test_aggressor_sign_flips_for_a_bid_side_hit(self):
        # direction=+1 (resting BID hit) -> aggressor sign = -direction = -1.0 (SELLER
        # initiated) -- the opposite sign from every other trade in this test file, so a
        # sign-flip bug that happens to work on direction=-1 trades cannot hide here.
        self.assertEqual(self.out["tr_last_trade_sign"][0], -1.0)

    def test_effective_spread_uses_the_pre_trade_mid_not_the_post_trade_mid(self):
        # mid_before = mid at event 3 (UNCHANGED by this edit) = (100000 + 101000) / 2
        #            = 100500 raw.
        # mid_after (event 4's own post-state, must NOT be used) = (95000 + 101000) / 2
        #            = 98000 raw.
        # Correct: eff_spread = 2 * |P_trade - mid_before| = 2 * |100000 - 100500|
        #        = 2 * 500 = 1000 raw = 10.0 ticks.
        # An off-by-one that reads the trade's own post-state instead would give
        # 2 * |100000 - 98000| = 4000 raw = 40.0 ticks -- a different, plausible-looking
        # number, which is exactly why this variant exists.
        self.assertEqual(self.out["tr_eff_spread"][0], 10.0)
        self.assertNotEqual(self.out["tr_eff_spread"][0], 40.0)

    def test_windowed_features_reflect_the_sell(self):
        # 1s window at t=event4's ts covers events 0-4 (as in TestHiddenTradeExactlyAtStamp
        # but with event 4's direction/size changed). Only event 4 is a trade here:
        # aggressor sign -1.0, size 40 -> signed contribution -40.0.
        out = trade.compute(self.frame, _stamp(4))
        self.assertEqual(out["tr_signed_vol_1s"][0], -40.0)
        self.assertEqual(out["tr_buy_vol_1s"][0], 0.0)
        self.assertEqual(out["tr_sell_vol_1s"][0], 40.0)
        self.assertEqual(out["tr_count_1s"][0], 1.0)
        # aggr_imb = (0 - 40) / (0 + 40) = -1.0.
        self.assertEqual(out["tr_aggr_imb_1s"][0], -1.0)
        self.assertEqual(out["tr_avg_size_1s"][0], 40.0)
        # bc_count over events 0-4: types [1,1,2,3,4] -- all book-changing -> 5.
        # quote_ratio = 1 / 5 = 0.2.
        self.assertEqual(out["tr_quote_ratio_1s"][0], 0.2)


class TestQuoteRatioNaNWhenNoBookChangingEvents(unittest.TestCase):
    """A 2-event variant (tests/toybook.py's "build a variant inline" route, since this
    scenario -- a trade isolated from any book-changing event in its own window -- is not
    reachable from the 8-event fixture) isolating tr_quote_ratio's OWN zero-denominator
    rule (book-changing count), as distinct from tr_aggr_imb's (trade volume)."""

    def setUp(self):
        ts_ns = [0, 1_000_000_000]  # 1s apart
        event_type = [1, 5]  # event 0: new bid limit (book-changing); event 1: hidden trade
        direction = [1, -1]
        price = [100000, 101000]
        size = [10, 5]
        bid_px = [[100000, 99900], [100000, 99900]]
        bid_sz = [[10, 5], [10, 5]]
        ask_px = [[101000, 101100], [101000, 101100]]
        ask_sz = [[20, 8], [20, 8]]
        self.frame = make_event_frame(
            ts_ns, event_type, direction, price, size, bid_px, bid_sz, ask_px, ask_sz
        )
        stamp = Stamps(index=np.array([1], dtype=np.int64), ts_ns=np.array([1_000_000_000], dtype=np.int64))
        self.out = trade.compute(self.frame, stamp)

    def test_quote_ratio_is_nan_but_aggr_imb_is_defined_in_the_same_window(self):
        # 100ms window at t=1e9: lower_bound = 1e9 - 1e8 = 9e8. Event 0 (ts=0) is way
        # below lower_bound -> excluded. Event 1 (ts=1e9=t) included. Window = {event 1}
        # only, a type-5 (hidden) trade -- NOT a book-changing event.
        # tr_count_100ms = 1 (one trade); bc_count_100ms = 0 (no book-changing event in
        # the window) -> quote_ratio's OWN denominator is 0 -> NaN, even though a trade
        # happened.
        self.assertEqual(self.out["tr_count_100ms"][0], 1.0)
        self.assertTrue(np.isnan(self.out["tr_quote_ratio_100ms"][0]))
        # By contrast, tr_aggr_imb's denominator is total trade VOLUME, which is 5 (> 0)
        # here, so it IS defined: aggressor sign = -direction = -(-1) = +1 (buy), size 5
        # -> aggr_imb = (5 - 0) / (5 + 0) = 1.0. avg_size = 5 / 1 = 5.0.
        self.assertEqual(self.out["tr_aggr_imb_100ms"][0], 1.0)
        self.assertEqual(self.out["tr_avg_size_100ms"][0], 5.0)
        self.assertEqual(self.out["tr_signed_vol_100ms"][0], 5.0)


class TestEffSpreadNaNWhenTradeIsTheVeryFirstEvent(unittest.TestCase):
    """A 1-event frame whose sole event is itself a trade: there is no preceding row to
    read mid_before from, so tr_eff_spread / tr_last_trade_sign must be NaN even though
    the trade is real and the windowed features see it fine."""

    def setUp(self):
        ts_ns = [0]
        event_type = [4]
        direction = [-1]
        price = [101000]
        size = [10]
        bid_px = [[100000, 99000]]
        bid_sz = [[10, 5]]
        ask_px = [[101000, 102000]]
        ask_sz = [[20, 8]]
        self.frame = make_event_frame(
            ts_ns, event_type, direction, price, size, bid_px, bid_sz, ask_px, ask_sz
        )
        stamp = Stamps(index=np.array([0], dtype=np.int64), ts_ns=np.array([0], dtype=np.int64))
        self.out = trade.compute(self.frame, stamp)

    def test_last_trade_fields_are_nan_with_no_preceding_row(self):
        self.assertTrue(np.isnan(self.out["tr_eff_spread"][0]))
        self.assertTrue(np.isnan(self.out["tr_last_trade_sign"][0]))

    def test_windowed_features_are_unaffected_by_the_missing_preceding_row(self):
        # aggressor sign = -direction = -(-1) = +1 (buy), size 10 -> signed_vol = 10.0.
        # The window-based features have no "look one row back" requirement, so they are
        # well-defined even though the last-trade fields are not.
        self.assertEqual(self.out["tr_signed_vol_100ms"][0], 10.0)
        self.assertEqual(self.out["tr_count_100ms"][0], 1.0)


class TestNoLookahead(unittest.TestCase):
    """The violation detector: appending future events (including a future TRADE) must
    not move a single value at an earlier stamp."""

    def test_appending_future_events_leaves_earlier_stamps_bit_identical(self):
        full_table = build_toy_frame()
        past_table = full_table.slice(0, 5)  # events 0-4 only; events 5-7 (a trade among
        # them, event 5) are "the future"
        full = event_frame_from_table(full_table)
        past = event_frame_from_table(past_table)

        stamps = Stamps(index=np.arange(5, dtype=np.int64), ts_ns=np.array(TOY_TS_NS[:5], dtype=np.int64))
        out_full = trade.compute(full, stamps)
        out_past = trade.compute(past, stamps)

        self.assertEqual(set(out_full.keys()), set(out_past.keys()))
        for key in out_full:
            np.testing.assert_array_equal(
                out_full[key],
                out_past[key],
                err_msg=f"{key} at an earlier stamp changed when future events were appended",
            )


class TestBothClocks(unittest.TestCase):
    """INTERFACE.md sec. 10: compute() must run on both stamps_for_clock outputs and
    return the right shapes/dtypes. The toy fixture has only 7 book-changing events, fewer
    than EVENT_CLOCK_N=10, so the event clock legitimately yields an EMPTY Stamps here --
    doubling as the "empty stamps must not raise" check (INTERFACE.md sec. 3)."""

    def test_runs_on_calendar_and_event_clocks(self):
        frame = toy_event_frame()
        expected_keys = None
        for clock in ("calendar", "event"):
            stamps = stamps_for_clock(frame, clock)
            out = trade.compute(frame, stamps)
            n = stamps.index.shape[0]
            if expected_keys is None:
                expected_keys = set(out.keys())
                self.assertEqual(len(expected_keys), 23)
            self.assertEqual(set(out.keys()), expected_keys)
            for key, arr in out.items():
                self.assertEqual(arr.shape, (n,), key)
                self.assertEqual(arr.dtype, np.float64, key)

    def test_event_clock_is_empty_on_the_toy_fixture(self):
        # 7 book-changing events (indices 0,1,2,3,4,6,7; event 5 is type 5, hidden, not
        # book-changing), EVENT_CLOCK_N=10 -> event_stamps returns zero stamps.
        stamps = stamps_for_clock(toy_event_frame(), "event")
        self.assertEqual(stamps.index.shape[0], 0)
        out = trade.compute(toy_event_frame(), stamps)
        for key, arr in out.items():
            self.assertEqual(arr.shape, (0,), key)
            self.assertEqual(arr.dtype, np.float64, key)


if __name__ == "__main__":
    unittest.main()
