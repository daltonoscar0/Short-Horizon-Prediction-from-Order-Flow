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


class TestTotalBreakEvenIsNotTheIncrementalOne(unittest.TestCase):
    """The 2026-08-02 red team's headline correction, as an executable check."""

    def test_hand_computed_total_break_even(self):
        from edge.costs.pnl import break_even_total_ticks

        # Gross $12.00 over 10 round trips of 100 shares -> c = 12/(2*100*10) = $0.006/share
        # = 0.6 ticks. Same arithmetic as the incremental version; the DIFFERENCE is which
        # series it is fed.
        self.assertAlmostEqual(
            break_even_total_ticks(np.full(10, 1.20), n_trades=10, shares=100),
            0.6, places=12,
        )

    def test_the_two_columns_differ_by_the_costs_already_paid(self):
        from edge.costs.pnl import break_even_total_ticks

        # A signal earning 0.02 ticks/trade gross that pays a 1.61-tick round trip: gross
        # $0.02, net $0.02 - $1.61 (per 100 shares, in dollars: 0.02 and -1.59).
        gross = np.full(100, 0.02)
        net = gross - 1.61
        total = break_even_total_ticks(gross, n_trades=100, shares=100)
        incr = break_even_cost_ticks(net, n_trades=100, shares=100)
        self.assertGreater(total, 0.0)   # the signal can pay SOMETHING
        self.assertLess(incr, 0.0)       # but not what it is already being charged
        self.assertAlmostEqual(total - incr, 1.61 / (2 * 100) / 0.01, places=9)

    def test_incremental_break_even_cannot_separate_a_constant_from_a_signal(self):
        """Why the column was demoted, at the magnitudes actually measured on this data.

        INTC calendar 1s: L1 earns 0.0213 gross ticks/trade, a constant always-short earns
        0.0027 on the same trades, and the round trip costs 1.61 ticks. The signal is ~8x the
        constant, and the incremental column puts them 1% apart — inside its own rounding.
        """
        from edge.costs.pnl import break_even_total_ticks

        # Dollars per 100-share round trip: ticks * $0.01 * 100 shares.
        def dollars(ticks):
            return ticks * 0.01 * 100

        signal_gross = np.full(1000, dollars(0.0213))
        const_gross = np.full(1000, dollars(0.0027))
        spread_cost = dollars(1.61)

        incr_signal = break_even_cost_ticks(signal_gross - spread_cost, 1000, shares=100)
        incr_const = break_even_cost_ticks(const_gross - spread_cost, 1000, shares=100)
        self.assertLess(abs(incr_signal - incr_const) / abs(incr_const), 0.02)

        tot_signal = break_even_total_ticks(signal_gross, 1000, shares=100)
        tot_const = break_even_total_ticks(const_gross, 1000, shares=100)
        self.assertGreater(tot_signal / tot_const, 7.0)


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


class TestNonOverlappingDecisions(unittest.TestCase):
    """EVAL_PROTOCOL.md sec. 3: single unit, no inventory accumulation, flat at horizon."""

    def test_positions_never_overlap_on_a_regular_grid(self):
        from edge.costs.run import _non_overlapping

        # Stamps 100ms apart, horizon 1s, latency 0 -> a position opened at t closes at
        # t+1s, so the next eligible decision is 10 stamps later.
        ts = np.arange(50, dtype=np.int64) * 100_000_000
        keep, _ = _non_overlapping(ts, horizon_ns=1_000_000_000, delta_ns=0)
        kept = np.flatnonzero(keep)
        self.assertEqual(kept.tolist(), list(range(0, 50, 10)))

    def test_latency_widens_the_spacing(self):
        from edge.costs.run import _non_overlapping

        # delta 100ms + horizon 1s = 1.1s span -> 11 stamps apart on a 100ms grid.
        ts = np.arange(50, dtype=np.int64) * 100_000_000
        keep, _ = _non_overlapping(ts, horizon_ns=1_000_000_000, delta_ns=100_000_000)
        kept = np.flatnonzero(keep)
        self.assertEqual(kept.tolist(), [0, 11, 22, 33, 44])

    def test_irregular_stamps_still_never_overlap(self):
        from edge.costs.run import _non_overlapping

        rng = np.random.default_rng(0)
        ts = np.cumsum(rng.integers(1, 500_000_000, size=2000)).astype(np.int64)
        horizon = 1_000_000_000
        keep, _ = _non_overlapping(ts, horizon_ns=horizon, delta_ns=0)
        chosen = ts[keep]
        # Every consecutive pair must be at least the holding span apart.
        self.assertTrue(np.all(np.diff(chosen) >= horizon))

    def test_first_stamp_is_always_taken(self):
        from edge.costs.run import _non_overlapping

        ts = np.array([5, 6, 7], dtype=np.int64) * 1_000_000_000
        keep, _ = _non_overlapping(ts, horizon_ns=1_000_000_000, delta_ns=0)
        self.assertTrue(bool(keep[0]))

    def test_spacing_holds_ACROSS_fold_boundaries(self):
        """The invariant is about positions, and positions do not know about folds.

        Regression test for the 2026-08-02 red team: `next_free` used to reset per fold, so a
        position opened at the end of one test block could still be open when the next block
        opened one. Measured on the shipped table as 4 gaps of 800-1000ms against a required
        1001ms, one per fold boundary.
        """
        from edge.costs.run import _non_overlapping

        span = 1_000_000_000
        # Fold 1's last kept decision is at 1.5s and is still open until 2.5s. Fold 2 opens
        # at 2.0s, inside that window -- the exact shape of the 4 boundary violations the
        # red team measured.
        fold1 = np.array([0, 1_500], dtype=np.int64) * 1_000_000
        fold2 = np.arange(2_000, 3_000, 100, dtype=np.int64) * 1_000_000

        keep1, watermark = _non_overlapping(fold1, horizon_ns=span, delta_ns=0)
        keep2, _ = _non_overlapping(fold2, horizon_ns=span, delta_ns=0, next_free=watermark)

        chosen = np.concatenate([fold1[keep1], fold2[keep2]])
        self.assertTrue(
            np.all(np.diff(chosen) >= span),
            f"positions overlap across the boundary: gaps {np.diff(chosen).tolist()}",
        )

    def test_resetting_per_fold_is_what_the_regression_test_catches(self):
        """The broken version must actually fail the check above, or the test is a comment."""
        from edge.costs.run import _non_overlapping

        span = 1_000_000_000
        fold1 = np.array([0, 1_500], dtype=np.int64) * 1_000_000
        fold2 = np.arange(2_000, 3_000, 100, dtype=np.int64) * 1_000_000

        keep1, _ = _non_overlapping(fold1, horizon_ns=span, delta_ns=0)
        keep2, _ = _non_overlapping(fold2, horizon_ns=span, delta_ns=0)  # reset: the bug
        chosen = np.concatenate([fold1[keep1], fold2[keep2]])
        self.assertTrue(np.any(np.diff(chosen) < span))


class TestCostControls(unittest.TestCase):
    """The controls exist to make the headline column falsifiable. They must not be fitted."""

    def test_constant_direction_never_flips_with_the_data(self):
        from edge.costs.run import _ConstantDirection

        long_ctrl = _ConstantDirection(+1.0, "CTRL always long")
        X = np.zeros((100, 0))
        # A strongly negative label would flip a fitted through-origin rescale. A control
        # that flips is not a control.
        long_ctrl.fit(X, np.full(100, -5.0))
        self.assertTrue(np.all(long_ctrl.predict(X) > 0))

        short_ctrl = _ConstantDirection(-1.0, "CTRL always short")
        short_ctrl.fit(X, np.full(100, +5.0))
        self.assertTrue(np.all(short_ctrl.predict(X) < 0))

    def test_oracle_reproduces_the_sign_of_the_realised_move(self):
        from edge.costs.run import _Oracle

        oracle = _Oracle("label_y_mid_1s")
        y = np.array([0.5, -0.5, 0.0, 2.0], dtype=np.float64)
        X = y.reshape(-1, 1)
        oracle.fit(X, y)
        np.testing.assert_array_equal(oracle.predict(X), np.array([1.0, -1.0, 0.0, 1.0]))

    def test_oracle_declares_the_label_as_its_feature(self):
        """It reads the future by construction; the feature list is where that is visible."""
        from edge.costs.run import _Oracle

        self.assertEqual(_Oracle("label_y_mid_5s").features, ("label_y_mid_5s",))

    def test_simulate_calibrates_causally_like_the_ladder(self):
        """Phase 3 kept the in-sample rescale for a day after the ladder dropped it.

        The 2026-08-02 red team killed that calibration for the harness; `costs/run.py` went
        on calling `rung.fit(X, y)` with no holdout until the third pass found it on
        2026-08-03, and reverting it passes every other test in this suite. A spy rung records
        what `simulate` actually passes, because the difference is invisible in the output for
        most cells and worth ~40% of a break-even in a few.
        """
        from edge.costs.run import simulate
        from edge.harness.splits import Fold

        seen = {}

        class SpyRung:
            name, features = "spy", ("f",)
            scale_ = 1.0

            def fit(self, X, y, scale_holdout=None):
                seen["holdout"] = scale_holdout
                return self

            def predict(self, X):
                return np.ones(X.shape[0])

        n = 400
        ts = np.arange(n, dtype=np.int64) * 100_000_000
        cols = {"f": np.ones(n)}
        touch = {"ts_ns": ts,
                 "bid_px_1": np.full(n, 100.0), "ask_px_1": np.full(n, 101.0)}
        fold = Fold(fold_id=0, protocol="C",
                    train_idx=np.arange(0, 300), test_idx=np.arange(300, n))
        simulate(cols, touch, np.zeros(n), ts, 1_000_000_000, 1_000_000,
                 SpyRung(), [fold], np.ones(n, dtype=bool))

        self.assertIn("holdout", seen)
        self.assertIsNotNone(seen["holdout"],
                             "simulate fitted in-sample -- the calibration killed 2026-08-02")
        hold = seen["holdout"]
        self.assertTrue(hold[-1], "protocol C's holdout must be the training TAIL")
        self.assertFalse(hold[0])
        self.assertAlmostEqual(hold.mean(), 0.2, places=2)

    def test_every_cost_predictor_accepts_the_harness_calling_convention(self):
        """The controls override `fit`, so they can silently fall out of step with `Rung.fit`.

        They did: when `simulate` moved to the causal calibration on 2026-08-03 it began
        passing `scale_holdout=`, and both controls raised `TypeError` at runtime — a whole
        cost run lost to a signature drift no unit test covered. Every predictor the cost
        runner builds must accept the keyword, and the controls must still IGNORE it.
        """
        from edge.costs.run import build_controls
        from edge.harness.models import build_ladder

        y = np.full(100, -5.0)  # sign chosen to flip anything that actually calibrates
        hold = np.zeros(100, dtype=bool)
        hold[80:] = True

        for rung in build_controls("label_y_mid_1s") + build_ladder(seed=0):
            X = np.zeros((100, len(rung.features)))
            with self.subTest(rung=rung.name):
                rung.fit(X, y, scale_holdout=hold)  # must not raise
                if rung.name.startswith("CTRL"):
                    self.assertEqual(rung.scale_, 1.0)
