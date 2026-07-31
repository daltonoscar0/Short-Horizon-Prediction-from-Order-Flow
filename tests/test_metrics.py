"""Hand-computed tests for edge.harness.metrics (EVAL_PROTOCOL.md sec. 2).

House style: every expected value is a literal derived arithmetic-by-arithmetic in a comment,
never a second call into the formula under test.

The load-bearing test in this file is TestR2IsAgainstZeroNotTheTestMean. EVAL_PROTOCOL.md
sec. 2 names using the test mean as "a subtle leak and the most common one in this
literature", and on a near-zero-mean return series the two definitions sit within a few
thousandths of each other -- so only a case built to separate them can tell them apart.
"""
import unittest

import numpy as np

from edge.harness.metrics import (
    auc_binary,
    calibration_slope,
    classification_metrics,
    fold_summary,
    r2_vs_test_mean,
    r2_vs_zero,
    spearman_ic,
)


class TestR2IsAgainstZeroNotTheTestMean(unittest.TestCase):
    def test_zero_forecast_scores_exactly_zero(self):
        # By construction: SSE = sum(y^2), so 1 - sum(y^2)/sum(y^2) = 0.
        y = np.array([1.0, -2.0, 3.0, -4.0])
        self.assertEqual(r2_vs_zero(y, np.zeros(4)), 0.0)

    def test_perfect_forecast_scores_one(self):
        y = np.array([1.0, -2.0, 3.0, -4.0])
        self.assertEqual(r2_vs_zero(y, y.copy()), 1.0)

    def test_the_two_definitions_disagree_on_an_offset_series(self):
        # y = [10, 11, 12, 13]: mean 11.5, sum(y^2) = 100+121+144+169 = 534.
        # A constant forecast of 11.5 gives SSE = 2.25+0.25+0.25+2.25 = 5.0.
        #   vs zero:      1 - 5/534   = 0.990636704...
        #   vs test mean: 1 - 5/5     = 0.0     (sum((y-ybar)^2) = 5.0 exactly)
        # The mean-based number correctly says "no better than knowing the mean"; the
        # zero-based number says "explains 99% of the raw second moment". For RETURNS,
        # where the honest benchmark is predicting nothing, the zero-based one is the
        # right question -- and a model handed the test mean has been handed the answer.
        y = np.array([10.0, 11.0, 12.0, 13.0])
        p = np.full(4, 11.5)
        self.assertAlmostEqual(r2_vs_zero(y, p), 1.0 - 5.0 / 534.0, places=12)
        self.assertAlmostEqual(r2_vs_test_mean(y, p), 0.0, places=12)

    def test_a_worse_than_zero_forecast_is_negative(self):
        # y = [1, -1], yhat = [-1, 1]: SSE = 4 + 4 = 8, sum(y^2) = 2 -> 1 - 8/2 = -3.
        self.assertEqual(r2_vs_zero(np.array([1.0, -1.0]), np.array([-1.0, 1.0])), -3.0)

    def test_all_zero_truth_is_nan_not_zero(self):
        self.assertTrue(np.isnan(r2_vs_zero(np.zeros(4), np.zeros(4))))


class TestSpearmanIC(unittest.TestCase):
    def test_perfect_monotone_agreement_is_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(spearman_ic(y, np.array([10.0, 20.0, 30.0, 40.0])), 1.0, places=12)

    def test_perfect_inversion_is_minus_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(spearman_ic(y, np.array([40.0, 30.0, 20.0, 10.0])), -1.0, places=12)

    def test_hand_computed_with_ties(self):
        # y    = [1, 2, 2, 3]  -> ranks [0, 1.5, 1.5, 3]
        # yhat = [5, 6, 7, 8]  -> ranks [0, 1,   2,   3]
        # centred: ry = [-1.5, 0, 0, 1.5]; rp = [-1.5, -0.5, 0.5, 1.5]
        # num = 2.25 + 0 + 0 + 2.25 = 4.5
        # |ry| = sqrt(2.25+2.25) = sqrt(4.5); |rp| = sqrt(2.25+0.25+0.25+2.25) = sqrt(5)
        # rho = 4.5 / (sqrt(4.5)*sqrt(5)) = 4.5 / sqrt(22.5) = 0.9486832980505138
        y = np.array([1.0, 2.0, 2.0, 3.0])
        p = np.array([5.0, 6.0, 7.0, 8.0])
        self.assertAlmostEqual(spearman_ic(y, p), 0.9486832980505138, places=12)

    def test_constant_predictor_is_nan_not_zero(self):
        # A constant forecast has no rank variance. Reporting 0.0 would let L0 look like a
        # neutral predictor rather than an undefined one.
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(np.isnan(spearman_ic(y, np.ones(4))))


class TestCalibrationSlope(unittest.TestCase):
    def test_perfect_calibration_is_one(self):
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(calibration_slope(y, y.copy()), 1.0, places=12)

    def test_hand_computed_half_scale_prediction(self):
        # yhat = 2*y means realized moves half as much as predicted -> slope 0.5.
        # p = [2,4,6,8], pbar = 5, pc = [-3,-1,1,3]; y = [1,2,3,4], ybar = 2.5,
        # yc = [-1.5,-0.5,0.5,1.5]. num = 4.5+0.5+0.5+4.5 = 10; var = 9+1+1+9 = 20.
        y = np.array([1.0, 2.0, 3.0, 4.0])
        self.assertAlmostEqual(calibration_slope(y, 2.0 * y), 10.0 / 20.0, places=12)


class TestClassificationAgainstTheBaseRate(unittest.TestCase):
    def test_always_flat_predictor_on_an_imbalanced_block(self):
        # 8 flat, 1 up, 1 down -- the 1s-horizon situation. Predicting flat always:
        #   accuracy       = 8/10 = 0.80          <- meaningless on its own
        #   base rate      = 8/10 = 0.80          <- and it beats nothing
        #   balanced acc   = mean(recall) over the 3 present classes = (0 + 1 + 0)/3
        y = np.array([0.0] * 8 + [1.0, -1.0])
        p = np.zeros(10)
        m = classification_metrics(y, p)
        self.assertAlmostEqual(m["accuracy"], 0.8, places=12)
        self.assertAlmostEqual(m["base_rate"], 0.8, places=12)
        self.assertAlmostEqual(m["balanced_accuracy"], 1.0 / 3.0, places=12)
        # MCC of a CONSTANT predictor is 0/0: every prediction lands in one column, so
        # s^2 - dot(p_k, p_k) = 0. Some libraries return 0.0 here; this module returns NaN,
        # matching spearman_ic's rule that an undefined statistic is never reported as a
        # neutral one -- an L0 constant predictor scoring "MCC 0.0" reads as a real, if
        # unimpressive, measurement, when in fact nothing was measurable.
        self.assertTrue(np.isnan(m["mcc"]))

    def test_perfect_prediction(self):
        y = np.array([1.0, -1.0, 0.0, 1.0, -1.0, 0.0])
        m = classification_metrics(y, y.copy())
        self.assertAlmostEqual(m["balanced_accuracy"], 1.0, places=12)
        self.assertAlmostEqual(m["mcc"], 1.0, places=12)

    def test_absent_class_does_not_drag_balanced_accuracy(self):
        # No -1 in the truth: balanced accuracy averages over the two PRESENT classes only.
        y = np.array([0.0, 0.0, 1.0, 1.0])
        m = classification_metrics(y, y.copy())
        self.assertAlmostEqual(m["balanced_accuracy"], 1.0, places=12)


class TestAUC(unittest.TestCase):
    def test_perfect_separation(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(auc_binary(y, np.array([1.0, 2.0, 3.0, 4.0])), 1.0, places=12)

    def test_inverted_separation(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(auc_binary(y, np.array([4.0, 3.0, 2.0, 1.0])), 0.0, places=12)

    def test_all_ties_is_one_half(self):
        y = np.array([0.0, 0.0, 1.0, 1.0])
        self.assertAlmostEqual(auc_binary(y, np.ones(4)), 0.5, places=12)

    def test_single_class_is_nan(self):
        self.assertTrue(np.isnan(auc_binary(np.ones(4), np.array([1.0, 2.0, 3.0, 4.0]))))


class TestFoldSummaryReportsTheWorstFold(unittest.TestCase):
    def test_the_motivating_example_from_the_protocol(self):
        # EVAL_PROTOCOL.md sec. 1: "A mean IC of 0.03 built from folds of
        # {0.14, 0.02, -0.01, 0.00, 0.00} is a story about one day, not a signal."
        # mean = 0.15/5 = 0.03, median = 0.00, worst = -0.01.
        s = fold_summary([0.14, 0.02, -0.01, 0.00, 0.00])
        self.assertAlmostEqual(s["mean"], 0.03, places=12)
        self.assertAlmostEqual(s["median"], 0.0, places=12)
        self.assertAlmostEqual(s["worst"], -0.01, places=12)
        self.assertEqual(s["n_folds"], 5)

    def test_nan_folds_are_dropped_not_counted(self):
        s = fold_summary([0.1, np.nan, 0.3])
        self.assertEqual(s["n_folds"], 2)
        self.assertAlmostEqual(s["mean"], 0.2, places=12)

    def test_empty_is_nan_everywhere(self):
        s = fold_summary([])
        self.assertEqual(s["n_folds"], 0)
        self.assertTrue(np.isnan(s["mean"]))


if __name__ == "__main__":
    unittest.main()
