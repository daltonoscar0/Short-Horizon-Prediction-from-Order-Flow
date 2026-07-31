"""Hand-computed tests for edge.costs.pnl (EVAL_PROTOCOL.md sec. 3).

Every expected value is derived arithmetic-by-arithmetic in a comment. The costs module is
where optimistic backtest bugs live, so the tests here are deliberately about the ways a PnL
number gets flattered: a round trip that forgets to cross the spread twice, a fill deeper than
the displayed book, a latency shift that reads the price it was supposed to wait for.
"""
import unittest

import numpy as np

from edge.costs import TAKE_FEE_PER_SHARE, TRADE_SIZE_SHARES
from edge.costs.pnl import (
    apply_latency,
    break_even_cost_per_share,
    break_even_cost_ticks,
    passive_round_trip_pnl,
    round_trip_pnl,
    turnover_and_holding,
    walk_the_book,
)


class TestLatencyReadsTheLaterRow(unittest.TestCase):
    def test_hand_computed_indices(self):
        # ts = [0, 10, 20, 30] ms. With delta = 10ms, the action row for t=0 is the last row
        # with ts <= 10ms, i.e. row 1; for t=10 it is row 2; for t=20 it is row 3.
        ts = np.array([0, 10, 20, 30], dtype=np.int64) * 1_000_000
        idx = apply_latency(ts, 10 * 1_000_000)
        self.assertEqual(idx.tolist(), [1, 2, 3, -1])

    def test_zero_latency_is_the_identity(self):
        ts = np.array([0, 10, 20, 30], dtype=np.int64) * 1_000_000
        np.testing.assert_array_equal(apply_latency(ts, 0), np.arange(4))

    def test_past_the_session_end_is_minus_one_not_clamped(self):
        # Clamping would invent a fill at a price that never existed. -1 forces the caller
        # to drop the row.
        ts = np.array([0, 10, 20], dtype=np.int64) * 1_000_000
        idx = apply_latency(ts, 50 * 1_000_000)
        self.assertTrue(np.all(idx == -1))

    def test_latency_lands_between_events_on_the_earlier_one(self):
        # delta puts the action time at 15ms, strictly between rows at 10 and 20 -> row 1,
        # the same "<= t resolves to the earlier event" convention the clocks use.
        ts = np.array([0, 10, 20, 30], dtype=np.int64) * 1_000_000
        self.assertEqual(apply_latency(ts, 15 * 1_000_000)[0], 1)


class TestWalkTheBook(unittest.TestCase):
    def test_fills_inside_the_top_level_at_the_top_price(self):
        px = np.array([100100.0, 100200.0, 100300.0])
        sz = np.array([500.0, 500.0, 500.0])
        vwap, filled = walk_the_book(px, sz, shares=100, side=1)
        self.assertEqual(vwap, 100100.0)
        self.assertEqual(filled, 100.0)

    def test_hand_computed_walk_across_two_levels(self):
        # Want 700 shares. 500 at 100100 + 200 at 100200:
        #   notional = 500*100100 + 200*100200 = 50,050,000 + 20,040,000 = 70,090,000
        #   vwap = 70,090,000 / 700 = 100,128.571428...
        px = np.array([100100.0, 100200.0, 100300.0])
        sz = np.array([500.0, 500.0, 500.0])
        vwap, filled = walk_the_book(px, sz, shares=700, side=1)
        self.assertAlmostEqual(vwap, 70_090_000.0 / 700.0, places=6)
        self.assertEqual(filled, 700.0)

    def test_shortfall_is_reported_not_filled_at_the_last_price(self):
        # The displayed book holds 300; asking for 1000 must report filled=300, NOT invent
        # 700 more at the deepest price. This is the classic optimistic backtest bug.
        px = np.array([100100.0, 100200.0])
        sz = np.array([100.0, 200.0])
        vwap, filled = walk_the_book(px, sz, shares=1000, side=1)
        self.assertEqual(filled, 300.0)
        # vwap = (100*100100 + 200*100200)/300 = (10,010,000 + 20,040,000)/300 = 100,166.66...
        self.assertAlmostEqual(vwap, 30_050_000.0 / 300.0, places=6)

    def test_empty_book_gives_nan_and_zero_fill(self):
        vwap, filled = walk_the_book(np.array([np.nan]), np.array([0.0]), 100, side=1)
        self.assertTrue(np.isnan(vwap))
        self.assertEqual(filled, 0.0)


class TestRoundTripPaysTheSpreadTwice(unittest.TestCase):
    def test_hand_computed_long_that_loses_the_spread(self):
        # Book unchanged: bid 100000 (=$10.00), ask 100100 (=$10.01), 1 tick wide.
        # Buy at the ask 10.01, sell back at the bid 10.00 -> -0.01/share gross.
        # 100 shares -> gross = -$1.00. Fees = 2 * 0.0030 * 100 = $0.60.
        # PnL = -1.00 - 0.60 = -$1.60. A signal must beat this to be worth anything.
        pnl = round_trip_pnl(100000, 100100, 100000, 100100, direction=1, shares=100)
        self.assertAlmostEqual(float(pnl), -1.60, places=10)

    def test_short_is_symmetric_on_an_unchanged_book(self):
        pnl = round_trip_pnl(100000, 100100, 100000, 100100, direction=-1, shares=100)
        self.assertAlmostEqual(float(pnl), -1.60, places=10)

    def test_a_favourable_two_tick_move_hand_computed(self):
        # Entry book bid 100000 / ask 100100; exit book bid 100200 / ask 100300.
        # Long: buy at 10.01, sell at 10.02 -> +0.01/share -> +$1.00 gross on 100 shares.
        # Fees $0.60 -> net +$0.40.
        pnl = round_trip_pnl(100000, 100100, 100200, 100300, direction=1, shares=100)
        self.assertAlmostEqual(float(pnl), 0.40, places=10)

    def test_zero_direction_is_exactly_zero_pnl_and_no_fees(self):
        pnl = round_trip_pnl(100000, 100100, 100000, 100100, direction=0, shares=100)
        self.assertEqual(float(pnl), 0.0)

    def test_extra_cost_is_charged_on_both_legs(self):
        # extra 0.001/share on 100 shares over two legs = $0.20 on top of the -$1.60 base.
        pnl = round_trip_pnl(
            100000, 100100, 100000, 100100, direction=1, shares=100,
            extra_cost_per_share=0.001,
        )
        self.assertAlmostEqual(float(pnl), -1.80, places=10)

    def test_vectorises_over_an_array_of_directions(self):
        d = np.array([1.0, -1.0, 0.0])
        pnl = round_trip_pnl(100000, 100100, 100000, 100100, direction=d, shares=100)
        np.testing.assert_allclose(pnl, np.array([-1.60, -1.60, 0.0]), atol=1e-10)


class TestBreakEvenCost(unittest.TestCase):
    def test_hand_computed_break_even(self):
        # Total PnL $12.00 over 10 round trips of 100 shares. Each trip pays the extra cost
        # twice, so total extra = 2 * c * 100 * 10 = 2000c. Break-even: 12 = 2000c ->
        # c = $0.006/share.
        self.assertAlmostEqual(
            break_even_cost_per_share(np.full(10, 1.20), n_trades=10, shares=100),
            0.006, places=12,
        )

    def test_break_even_in_ticks(self):
        # $0.006/share / $0.01 per tick = 0.6 ticks.
        self.assertAlmostEqual(
            break_even_cost_ticks(np.full(10, 1.20), n_trades=10, shares=100),
            0.6, places=12,
        )

    def test_a_losing_strategy_reports_a_negative_break_even_not_zero(self):
        # Already underwater before any extra cost. Clipping this to 0 would hide the
        # result; the negative number IS the finding.
        c = break_even_cost_per_share(np.full(10, -1.20), n_trades=10, shares=100)
        self.assertLess(c, 0.0)
        self.assertAlmostEqual(c, -0.006, places=12)

    def test_no_trades_is_nan(self):
        self.assertTrue(np.isnan(break_even_cost_per_share(np.array([]), n_trades=0)))

    def test_nan_pnl_rows_are_ignored_not_propagated(self):
        pnl = np.array([1.20] * 9 + [np.nan])
        c = break_even_cost_per_share(pnl, n_trades=9, shares=100)
        self.assertAlmostEqual(c, (9 * 1.20) / (2 * 100 * 9), places=12)


class TestPassiveIsASensitivityNotTheHeadline(unittest.TestCase):
    def test_unfilled_rows_contribute_zero(self):
        pnl = passive_round_trip_pnl(
            100050, 100000, 100100, direction=np.array([1.0, 1.0]),
            traded_through=np.array([True, False]), shares=100,
        )
        self.assertNotEqual(float(pnl[0]), 0.0)
        self.assertEqual(float(pnl[1]), 0.0)

    def test_rebate_makes_passive_cheaper_than_aggressive_on_the_same_move(self):
        # Same unchanged book. Aggressive pays the full spread plus two take fees; passive
        # posts at the mid, earns a rebate, and pays one take fee on the exit.
        aggressive = float(round_trip_pnl(100000, 100100, 100000, 100100, 1, shares=100))
        passive = float(passive_round_trip_pnl(
            100050, 100000, 100100, direction=1, traded_through=True, shares=100,
        ))
        self.assertGreater(passive, aggressive)


class TestTurnover(unittest.TestCase):
    def test_counts_only_nonzero_directions(self):
        t = turnover_and_holding(np.array([1.0, 0.0, -1.0, 0.0, 1.0]), horizon_ns=1_000_000_000)
        self.assertEqual(t["n_trades"], 3)
        self.assertAlmostEqual(t["trade_rate"], 0.6, places=12)
        self.assertAlmostEqual(t["holding_seconds"], 1.0, places=12)


if __name__ == "__main__":
    unittest.main()
