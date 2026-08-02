"""Tests for edge.costs.stats (EVAL_PROTOCOL.md sec. 2, "Inferential").

Hand-checkable properties rather than hand-computed literals where the closed form is a
rational approximation: the normal quantile function is pinned against known table values,
and the three estimators are pinned against cases whose ANSWER is known by construction
(identical models -> DM 0; independent noise -> CI covers 0; more trials -> higher bar).
"""
import math
import unittest

import numpy as np

from edge.costs.stats import (
    _norm_cdf,
    _norm_ppf,
    block_bootstrap_ci,
    deflated_sharpe,
    diebold_mariano,
    expected_max_sharpe,
)


class TestNormalHelpers(unittest.TestCase):
    def test_cdf_at_known_points(self):
        self.assertAlmostEqual(_norm_cdf(0.0), 0.5, places=12)
        self.assertAlmostEqual(_norm_cdf(1.959963985), 0.975, places=8)
        self.assertAlmostEqual(_norm_cdf(-1.959963985), 0.025, places=8)

    def test_ppf_at_known_points(self):
        self.assertAlmostEqual(_norm_ppf(0.975), 1.959963985, places=6)
        self.assertAlmostEqual(_norm_ppf(0.5), 0.0, places=8)
        self.assertAlmostEqual(_norm_ppf(0.025), -1.959963985, places=6)

    def test_ppf_inverts_cdf(self):
        for p in (0.01, 0.1, 0.3, 0.5, 0.7, 0.9, 0.99):
            self.assertAlmostEqual(_norm_cdf(_norm_ppf(p)), p, places=7)


class TestDieboldMariano(unittest.TestCase):
    def test_identical_models_give_zero_loss_differential(self):
        rng = np.random.default_rng(0)
        e = rng.normal(size=500)
        r = diebold_mariano(e, e.copy())
        self.assertAlmostEqual(r["mean_loss_diff"], 0.0, places=12)
        self.assertTrue(np.isnan(r["dm_stat"]) or abs(r["dm_stat"]) < 1e-8)

    def test_a_clearly_better_model_is_detected_with_the_right_sign(self):
        # A has half the error magnitude of B, so A's squared loss is lower and the
        # statistic must be NEGATIVE (the convention: negative means A wins).
        rng = np.random.default_rng(1)
        eb = rng.normal(size=2000)
        ea = 0.5 * eb
        r = diebold_mariano(ea, eb)
        self.assertLess(r["dm_stat"], 0.0)
        self.assertLess(r["p_value"], 0.01)

    def test_hac_lag_widens_the_standard_error_on_autocorrelated_errors(self):
        # Build a strongly autocorrelated loss differential. A longer HAC lag must not
        # produce a LARGER statistic -- that is the whole reason the correction exists.
        rng = np.random.default_rng(2)
        base = rng.normal(size=3000)
        smooth = np.convolve(base, np.ones(20) / 20.0, mode="same")
        ea = smooth + 0.05
        eb = smooth
        naive = diebold_mariano(ea, eb, lag=1)
        corrected = diebold_mariano(ea, eb, lag=50)
        self.assertLess(abs(corrected["dm_stat"]), abs(naive["dm_stat"]))

    def test_too_few_observations_is_nan_not_a_confident_answer(self):
        r = diebold_mariano(np.ones(4), np.zeros(4))
        self.assertTrue(np.isnan(r["dm_stat"]))

    def test_absolute_loss_is_supported(self):
        rng = np.random.default_rng(3)
        eb = rng.normal(size=1000)
        r = diebold_mariano(0.5 * eb, eb, loss="absolute")
        self.assertLess(r["dm_stat"], 0.0)

    def test_unknown_loss_raises(self):
        with self.assertRaises(ValueError):
            diebold_mariano(np.zeros(20), np.ones(20), loss="huber")


class TestBlockBootstrap(unittest.TestCase):
    def test_ci_brackets_the_sample_mean(self):
        rng = np.random.default_rng(4)
        v = rng.normal(loc=0.5, scale=1.0, size=500)
        ci = block_bootstrap_ci(v, block_len=10, n_boot=500, seed=0)
        self.assertLessEqual(ci["lo"], ci["mean"])
        self.assertLessEqual(ci["mean"], ci["hi"])

    def test_zero_mean_noise_ci_covers_zero(self):
        rng = np.random.default_rng(5)
        v = rng.normal(loc=0.0, scale=1.0, size=1000)
        ci = block_bootstrap_ci(v, block_len=10, n_boot=500, seed=0)
        self.assertLess(ci["lo"], 0.0)
        self.assertGreater(ci["hi"], 0.0)

    def test_a_strong_signal_ci_excludes_zero(self):
        rng = np.random.default_rng(6)
        v = rng.normal(loc=2.0, scale=0.5, size=1000)
        ci = block_bootstrap_ci(v, block_len=10, n_boot=500, seed=0)
        self.assertGreater(ci["lo"], 0.0)

    def test_deterministic_under_a_fixed_seed(self):
        v = np.random.default_rng(7).normal(size=300)
        a = block_bootstrap_ci(v, block_len=5, n_boot=200, seed=0)
        b = block_bootstrap_ci(v, block_len=5, n_boot=200, seed=0)
        self.assertEqual(a["lo"], b["lo"])
        self.assertEqual(a["hi"], b["hi"])

    def test_block_length_is_clamped_to_the_sample(self):
        v = np.arange(10, dtype=np.float64)
        ci = block_bootstrap_ci(v, block_len=1000, n_boot=50, seed=0)
        self.assertEqual(ci["block_len"], 10)


class TestDeflatedSharpe(unittest.TestCase):
    def test_the_bar_rises_with_the_number_of_trials(self):
        # The whole point: the more configurations you tried, the higher the Sharpe you
        # need to have found something.
        bars = [expected_max_sharpe(n, var_sharpe=0.01) for n in (2, 10, 100, 1000)]
        for a, b in zip(bars, bars[1:]):
            self.assertLess(a, b)

    def test_one_trial_sets_no_bar(self):
        self.assertEqual(expected_max_sharpe(1, var_sharpe=0.01), 0.0)

    def test_pure_noise_does_not_clear_the_bar_after_many_trials(self):
        rng = np.random.default_rng(8)
        r = rng.normal(loc=0.0, scale=1.0, size=1000)
        d = deflated_sharpe(r, n_trials=500)
        self.assertLess(d["dsr"], 0.95)

    def test_a_strong_genuine_signal_survives_deflation(self):
        rng = np.random.default_rng(9)
        r = rng.normal(loc=0.3, scale=1.0, size=2000)  # Sharpe ~0.3 per period
        d = deflated_sharpe(r, n_trials=10)
        self.assertGreater(d["dsr"], 0.95)

    def test_more_trials_lowers_the_dsr_for_the_same_series(self):
        rng = np.random.default_rng(10)
        r = rng.normal(loc=0.08, scale=1.0, size=1000)
        few = deflated_sharpe(r, n_trials=2)
        many = deflated_sharpe(r, n_trials=5000)
        self.assertGreater(few["dsr"], many["dsr"])

    def test_reports_skew_and_kurtosis(self):
        rng = np.random.default_rng(11)
        d = deflated_sharpe(rng.normal(size=500), n_trials=5)
        self.assertAlmostEqual(d["skew"], 0.0, places=0)
        self.assertAlmostEqual(d["kurtosis"], 3.0, places=0)  # non-excess

    def test_too_few_points_is_nan(self):
        self.assertTrue(np.isnan(deflated_sharpe(np.ones(4), n_trials=5)["dsr"]))

    def test_zero_variance_returns_nan_not_infinite_sharpe(self):
        self.assertTrue(np.isnan(deflated_sharpe(np.ones(50), n_trials=5)["dsr"]))


class TestLinearLossComparesPnLNotMagnitude(unittest.TestCase):
    """`loss="linear"` treats inputs as losses already. Squaring PnL tests the wrong thing."""

    def test_negative_stat_means_the_first_series_has_the_lower_loss(self):
        rng = np.random.default_rng(3)
        # A earns 0.05 ticks/trade more than B, on the same trades.
        b = rng.normal(0.0, 0.5, size=800)
        a = b + 0.05
        dm = diebold_mariano(-a, -b, lag=10, loss="linear")  # loss = -PnL
        self.assertLess(dm["dm_stat"], 0.0)
        self.assertAlmostEqual(dm["mean_loss_diff"], -0.05, places=2)

    def test_squared_loss_would_get_the_sign_wrong_on_pnl(self):
        """The reason the linear mode exists, stated as a test.

        A loses money, B makes money, same magnitude. Squared loss calls them equal; linear
        loss correctly prefers B.
        """
        a = np.full(500, -0.10)   # A loses 0.10/trade
        b = np.full(500, +0.10)   # B makes 0.10/trade
        sq = diebold_mariano(-a, -b, lag=5, loss="squared")
        self.assertAlmostEqual(sq["mean_loss_diff"], 0.0, places=12)  # blind to the sign
        lin = diebold_mariano(-a, -b, lag=5, loss="linear")
        self.assertAlmostEqual(lin["mean_loss_diff"], 0.20, places=12)  # A's loss is higher

    def test_unknown_loss_is_rejected(self):
        with self.assertRaises(ValueError):
            diebold_mariano(np.ones(20), np.zeros(20), loss="hinge")


if __name__ == "__main__":
    unittest.main()
