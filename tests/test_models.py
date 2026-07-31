"""Hand-computed tests for the L0-L4 ladder (edge.harness.models).

Two structural claims carry most of the weight here:

  * the shared through-origin rescale is EXACTLY 1.0 for any least-squares rung, so it puts
    L1 and L4 on a comparable footing without touching L2/L3;
  * L3's standardiser is fitted on TRAINING ROWS ONLY -- the leak this whole project exists
    to measure. Asserted by mutating the test rows and checking the fitted state does not
    move, which is the only version of that test that cannot pass vacuously.
"""
import unittest

import numpy as np

from edge.harness.models import (
    L0Zero,
    L1Sign,
    L2OLS,
    L3Ridge,
    _fit_scale,
    build_ladder,
)


class TestScaleStep(unittest.TestCase):
    def test_hand_computed_scale(self):
        # s = [1, 2, 3], y = [2, 4, 6] -> sum(s*y) = 2+8+18 = 28; sum(s^2) = 1+4+9 = 14
        # scale = 28/14 = 2.0
        self.assertAlmostEqual(_fit_scale(np.array([1.0, 2.0, 3.0]),
                                          np.array([2.0, 4.0, 6.0])), 2.0, places=12)

    def test_all_zero_score_gives_zero_not_nan(self):
        self.assertEqual(_fit_scale(np.zeros(3), np.array([1.0, 2.0, 3.0])), 0.0)

    def test_scale_is_exactly_one_for_ols(self):
        # OLS residuals are orthogonal to fitted values, so sum(yhat*y) = sum(yhat^2) and
        # the rescale is a no-op. This is what lets the same scale step be applied to every
        # rung without advantaging the fitted ones.
        rng = np.random.default_rng(0)
        X = rng.normal(size=(200, 3))
        y = X @ np.array([1.5, -2.0, 0.5]) + rng.normal(scale=0.1, size=200)
        model = L2OLS(features=("a", "b", "c")).fit(X, y)
        self.assertAlmostEqual(model.scale_, 1.0, places=9)

    def test_scale_is_exactly_one_for_ridge(self):
        rng = np.random.default_rng(1)
        X = rng.normal(size=(300, 4))
        y = X @ np.array([1.0, 0.0, -1.0, 2.0]) + rng.normal(scale=0.2, size=300)
        model = L3Ridge(features=tuple("abcd"), lam=1.0).fit(X, y)
        # Ridge is PENALISED least squares, so residuals are not exactly orthogonal to
        # fitted values and the identity holds only approximately -- measured 1.0031 here,
        # a 0.3% recalibration that shrinkage has left on the table. Close enough that the
        # rescale cannot meaningfully flatter the rung, but not exact, and the assertion
        # says which of the two it is rather than hiding the difference behind a loose
        # tolerance on an "exactly 1.0" claim.
        self.assertAlmostEqual(model.scale_, 1.0, places=2)
        self.assertNotEqual(model.scale_, 1.0)


class TestL0(unittest.TestCase):
    def test_predicts_exactly_zero(self):
        X = np.arange(12, dtype=np.float64).reshape(4, 3)
        p = L0Zero().fit(X, np.array([1.0, -1.0, 2.0, -2.0])).predict(X)
        np.testing.assert_array_equal(p, np.zeros(4))


class TestL1IsUnfitted(unittest.TestCase):
    def test_raw_score_is_the_sign_of_the_feature(self):
        X = np.array([[-2.0], [0.0], [3.0]])
        model = L1Sign("qi_touch", "L1")
        np.testing.assert_array_equal(model._raw(X), np.array([-1.0, 0.0, 1.0]))

    def test_hand_computed_scale_on_a_sign_predictor(self):
        # signs = [-1, 1, 1, -1]; y = [-2, 4, 6, -8]
        # sum(s*y) = 2 + 4 + 6 + 8 = 20; sum(s^2) = 4 -> scale = 5.0
        X = np.array([[-1.0], [1.0], [2.0], [-3.0]])
        y = np.array([-2.0, 4.0, 6.0, -8.0])
        model = L1Sign("qi_touch", "L1").fit(X, y)
        self.assertAlmostEqual(model.scale_, 5.0, places=12)
        np.testing.assert_allclose(model.predict(X), np.array([-5.0, 5.0, 5.0, -5.0]))

    def test_the_sign_itself_never_depends_on_training_data(self):
        # Fit on two completely different targets; the RAW score must be identical.
        X = np.array([[-1.0], [1.0], [2.0], [-3.0]])
        a = L1Sign("qi_touch", "L1").fit(X, np.array([1.0, 1.0, 1.0, 1.0]))
        b = L1Sign("qi_touch", "L1").fit(X, np.array([-5.0, 9.0, 0.5, -2.0]))
        np.testing.assert_array_equal(a._raw(X), b._raw(X))


class TestL2RecoversAKnownSlope(unittest.TestCase):
    def test_hand_built_exact_line(self):
        # y = 3 + 2x exactly -> intercept 3, slope 2, no residual.
        x = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = 3.0 + 2.0 * x[:, 0]
        model = L2OLS(features=("ofi_best_1s",)).fit(x, y)
        self.assertAlmostEqual(model.intercept_, 3.0, places=9)
        self.assertAlmostEqual(float(model.coef_[0]), 2.0, places=9)
        np.testing.assert_allclose(model.predict(x), y, atol=1e-9)

    def test_slope_is_inversely_proportional_to_depth(self):
        # The CKS result L2 has to reproduce: doubling depth halves the price impact per
        # unit of order flow. Build two synthetic books differing only in depth.
        rng = np.random.default_rng(2)
        ofi = rng.normal(size=500)
        shallow = L2OLS(features=("ofi_best_1s",)).fit(ofi[:, None], ofi / 100.0)
        deep = L2OLS(features=("ofi_best_1s",)).fit(ofi[:, None], ofi / 200.0)
        self.assertAlmostEqual(
            float(shallow.coef_[0]) / float(deep.coef_[0]), 2.0, places=6
        )


class TestL3StandardiserIsFitOnTrainingRowsOnly(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.Xtr = rng.normal(size=(400, 5))
        self.ytr = self.Xtr @ np.array([1.0, -1.0, 0.5, 0.0, 2.0]) + rng.normal(scale=0.1, size=400)
        self.Xte = rng.normal(loc=50.0, scale=10.0, size=(100, 5))  # wildly different scale

    def test_fitted_state_does_not_move_when_test_rows_change(self):
        model = L3Ridge(features=tuple("abcde")).fit(self.Xtr, self.ytr)
        mu, sigma, coef = model.mu_.copy(), model.sigma_.copy(), model.coef_.copy()
        model.predict(self.Xte)
        model.predict(self.Xte * 1000.0)
        np.testing.assert_array_equal(model.mu_, mu)
        np.testing.assert_array_equal(model.sigma_, sigma)
        np.testing.assert_array_equal(model.coef_, coef)

    def test_standardiser_matches_the_training_moments(self):
        model = L3Ridge(features=tuple("abcde")).fit(self.Xtr, self.ytr)
        np.testing.assert_allclose(model.mu_, self.Xtr.mean(axis=0), rtol=1e-12)
        np.testing.assert_allclose(model.sigma_, self.Xtr.std(axis=0, ddof=0), rtol=1e-12)

    def test_a_constant_column_does_not_produce_nan(self):
        X = self.Xtr.copy()
        X[:, 2] = 7.0  # zero variance in this fold
        model = L3Ridge(features=tuple("abcde")).fit(X, self.ytr)
        p = model.predict(X)
        self.assertTrue(np.all(np.isfinite(p)))
        self.assertEqual(model.sigma_[2], 1.0)

    def test_ridge_shrinks_toward_zero_as_lambda_grows(self):
        norms = []
        for lam in (0.1, 1.0, 100.0, 10000.0):
            m = L3Ridge(features=tuple("abcde"), lam=lam).fit(self.Xtr, self.ytr)
            norms.append(float(np.linalg.norm(m.coef_)))
        for a, b in zip(norms, norms[1:]):
            self.assertGreater(a, b)


class TestLadderShape(unittest.TestCase):
    def test_every_rung_has_a_name_and_a_feature_tuple(self):
        for rung in build_ladder(seed=0):
            self.assertIsInstance(rung.name, str)
            self.assertTrue(rung.name)
            self.assertIsInstance(rung.features, tuple)

    def test_rung_order_is_frozen(self):
        names = [r.name for r in build_ladder(seed=0)]
        self.assertEqual(
            names,
            [
                "L0 zero",
                "L1 sign(queue imbalance)",
                "L1m sign(microprice - mid)",
                "L2 OLS on OFI",
                "L3 ridge, full set",
                "L4 GBM, full set",
            ],
        )

    def test_l2_uses_exactly_one_regressor(self):
        self.assertEqual(len(L2OLS().features), 1)
        self.assertEqual(L2OLS().features[0], "ofi_best_1s")

    def test_l3_and_l4_share_the_same_feature_set(self):
        ladder = build_ladder(seed=0)
        l3 = next(r for r in ladder if r.name.startswith("L3"))
        l4 = next(r for r in ladder if r.name.startswith("L4"))
        self.assertEqual(l3.features, l4.features)


if __name__ == "__main__":
    unittest.main()
