"""Tests for the per-fold metric wiring in `edge.harness.run.score_fold`.

`metrics.py`'s functions are covered by hand-computed literals in `test_metrics.py`. What is
covered HERE is the wiring, which is where the two spec metrics (EVAL_PROTOCOL.md sec. 2)
could be right in isolation and wrong in the report:

  * AUC must be computed on the DIRECTIONAL subset -- the rows the deadband calls up or down.
    Pooling the flats into the negative class turns AUC into a measure of tie mass, and this
    project's phase-1 red team already caught one number reported on that axis. The test
    constructs a case where the two definitions disagree in the FIRST DECIMAL, so a regression
    to the pooled form cannot pass;
  * the calibration slope must be measured on the prediction the harness actually reports,
    i.e. AFTER the through-origin rescale, not on the rung's raw score.

A stub rung is used rather than a real one so the prediction vector is known exactly; the real
rungs are covered in `test_models.py`.
"""
import unittest

import numpy as np

from edge.harness.metrics import auc_binary
from edge.harness.run import scale_holdout_mask, score_fold
from edge.harness.splits import Fold


class StubRung:
    """A rung whose raw score is fixed in advance, so the metrics are hand-checkable.

    It mirrors the real `Rung` split between `_raw` (the model's own output) and `predict`
    (that output times the through-origin scale), because two of the metrics under test are
    defined on opposite sides of that line: the calibration slope on the rescaled prediction,
    `auc_raw` on the raw score.
    """

    def __init__(self, raw_by_row, scale=1.0):
        self.name = "stub"
        self.features = ("f",)
        self._raw_by_row = np.asarray(raw_by_row, dtype=np.float64)
        self.scale_ = scale
        self.fitted_on = None

    def fit(self, X, y, scale_holdout=None):
        self.fitted_on = (X.shape[0], None if scale_holdout is None else int(scale_holdout.sum()))

    def _raw(self, X):
        # X's first column carries the row index, so the stub can return a per-row constant.
        return self._raw_by_row[X[:, 0].astype(int)]

    def predict(self, X):
        return self.scale_ * self._raw(X)


def _harness(pred, y, scale=1.0, n=200):
    """One fold over `n` rows, train on the first half, test on the second."""
    cols = {"f": np.arange(n, dtype=np.float64)}
    fold = Fold(fold_id=0, protocol="C",
                train_idx=np.arange(0, n // 2), test_idx=np.arange(n // 2, n))
    mask = np.ones(n, dtype=bool)
    return score_fold(StubRung(pred, scale=scale), cols, y, fold, mask)


class TestAUCIsDirectionalNotPooled(unittest.TestCase):
    """The flats must be excluded, and the two definitions must be made to disagree."""

    def setUp(self):
        # 100 test rows: 20 up (+2 ticks), 20 down (-2 ticks), 60 flat (0). The deadband is
        # 0.5 ticks, so the flats classify to 0 and must not enter the AUC at all.
        n = 200
        self.n = n
        y = np.zeros(n, dtype=np.float64)
        te = np.arange(100, 200)
        y[te[:20]] = 2.0
        y[te[20:40]] = -2.0
        # The prediction ranks every DOWN row above every UP row -> directional AUC = 0.0.
        # But it parks all 60 flats at the BOTTOM, so pooling the flats into the negative
        # class scores the up rows above 60 of the 80 negatives and reports 0.75. The two
        # definitions land on opposite sides of chance, not within rounding of each other.
        pred = np.zeros(n, dtype=np.float64)
        pred[te[:20]] = 1.0    # up rows, middle score
        pred[te[20:40]] = 2.0  # down rows, HIGHER score
        pred[te[40:]] = 0.0    # flats, lowest of all
        self.y, self.pred, self.te = y, pred, te

    def test_auc_uses_only_the_directional_rows(self):
        got = _harness(self.pred, self.y, n=self.n)
        self.assertAlmostEqual(got["auc"], 0.0, places=12)

    def test_the_pooled_definition_would_give_a_materially_different_number(self):
        # Guards the test itself: if these two ever coincide, the test above is vacuous.
        y_cls = np.zeros(100)
        y_cls[:20] = 1.0
        y_cls[20:40] = -1.0
        pooled = auc_binary((y_cls > 0).astype(np.float64), self.pred[self.te])
        # 20 up rows x 80 negatives; the ups beat the 60 flats and lose to the 20 downs.
        self.assertAlmostEqual(pooled, (20 * 60) / (20 * 80), places=12)
        self.assertGreater(pooled, 0.5)  # opposite side of chance from the directional 0.0

    def test_a_correctly_ordered_predictor_scores_one(self):
        pred = -self.pred  # invert: up rows now rank above down rows
        got = _harness(pred, self.y, n=self.n)
        self.assertAlmostEqual(got["auc"], 1.0, places=12)

    def test_a_one_sided_test_block_gives_nan_not_a_half(self):
        # Every directional row is an up move: AUC is undefined, and reporting 0.5 for it
        # would silently add a chance-level row to the table.
        y = np.zeros(self.n, dtype=np.float64)
        y[np.arange(100, 120)] = 2.0
        got = _harness(self.pred, y, n=self.n)
        self.assertTrue(np.isnan(got["auc"]))

    def test_an_all_flat_test_block_gives_nan(self):
        got = _harness(self.pred, np.zeros(self.n, dtype=np.float64), n=self.n)
        self.assertTrue(np.isnan(got["auc"]))


class TestTheCalibrationHoldoutRuleIsProtocolDependent(unittest.TestCase):
    """The asymmetry that killed the 2026-08-03 calibration-slope reading.

    Under A the holdout is a random 20% of the training rows — exchangeable with A's random
    test fold. Under B/C/D it is the last 20% in time, which is not exchangeable with a later
    test block. That difference is a channel in every `A - C` gap this report quotes, so it
    has to be overridable to be measurable.
    """

    def test_protocol_a_gets_a_scattered_holdout_and_c_gets_a_contiguous_tail(self):
        a = scale_holdout_mask(1000, "A", 0)
        c = scale_holdout_mask(1000, "C", 0)
        self.assertTrue(c[800:].all())
        self.assertFalse(c[:800].any())
        # A's holdout must reach into the early rows; a positional one cannot.
        self.assertTrue(a[:800].any())
        self.assertAlmostEqual(a.mean(), 0.2, places=2)

    def test_the_rule_can_be_forced_so_the_channel_is_measurable(self):
        forced = scale_holdout_mask(1000, "A", 0, holdout_rule="positional")
        np.testing.assert_array_equal(forced, scale_holdout_mask(1000, "C", 0))

    def test_a_is_seeded_per_fold_so_the_headline_is_reproducible(self):
        np.testing.assert_array_equal(scale_holdout_mask(500, "A", 3),
                                      scale_holdout_mask(500, "A", 3))
        self.assertFalse(np.array_equal(scale_holdout_mask(500, "A", 3),
                                        scale_holdout_mask(500, "A", 4)))

    def test_an_unknown_rule_is_rejected_rather_than_silently_defaulted(self):
        with self.assertRaises(ValueError):
            scale_holdout_mask(100, "C", 0, holdout_rule="whatever")


class TestCalibrationSlopeIsOnTheReportedPrediction(unittest.TestCase):
    def test_a_perfectly_calibrated_prediction_is_one(self):
        n = 200
        rng = np.random.default_rng(0)
        y = rng.normal(size=n)
        got = _harness(y, y, n=n)
        self.assertAlmostEqual(got["calibration_slope"], 1.0, places=10)

    def test_a_prediction_at_half_amplitude_has_slope_two(self):
        n = 200
        rng = np.random.default_rng(1)
        y = rng.normal(size=n)
        got = _harness(0.5 * y, y, n=n)
        self.assertAlmostEqual(got["calibration_slope"], 2.0, places=10)

    def test_the_slope_follows_the_rescale_not_the_raw_score(self):
        # The whole point of the 2026-08-02 correction is that the through-origin rescale
        # changes the reported amplitude. If the slope were measured on the raw score it
        # would be blind to exactly the failure it exists to detect.
        n = 200
        rng = np.random.default_rng(2)
        y = rng.normal(size=n)
        raw = _harness(y, y, scale=1.0, n=n)["calibration_slope"]
        loud = _harness(y, y, scale=4.0, n=n)["calibration_slope"]
        self.assertAlmostEqual(raw, 1.0, places=10)
        self.assertAlmostEqual(loud, 0.25, places=10)

    def test_a_directionally_inverted_prediction_has_a_negative_slope(self):
        n = 200
        rng = np.random.default_rng(3)
        y = rng.normal(size=n)
        got = _harness(-y, y, n=n)
        self.assertLess(got["calibration_slope"], 0.0)


class TestANegativeScaleFlipsAUCButNotAUCRaw(unittest.TestCase):
    """The channel the 2026-08-03 red team found in the published AUC ladder.

    The through-origin rescale is rank-preserving only UP TO SIGN. The fitted scale came out
    negative in 27 of 400 protocol-C folds and 0 of 400 protocol-A folds, and a negative scale
    sends AUC to 1-AUC — which is what manufactured the entire apparent A−C gap on the
    unfitted rungs. `auc_raw` is scored before the rescale and cannot be moved by it.
    """

    def setUp(self):
        n = 200
        self.n = n
        y = np.zeros(n, dtype=np.float64)
        te = np.arange(100, 200)
        y[te[:20]] = 2.0
        y[te[20:40]] = -2.0
        raw = np.zeros(n, dtype=np.float64)
        raw[te[:20]] = 2.0    # up rows ranked above
        raw[te[20:40]] = 1.0  # down rows below -> raw AUC = 1.0
        self.y, self.raw = y, raw

    def test_a_positive_scale_leaves_both_at_one(self):
        got = _harness(self.raw, self.y, scale=1.0, n=self.n)
        self.assertAlmostEqual(got["auc"], 1.0, places=12)
        self.assertAlmostEqual(got["auc_raw"], 1.0, places=12)

    def test_a_negative_scale_inverts_auc_and_leaves_auc_raw_alone(self):
        got = _harness(self.raw, self.y, scale=-1.0, n=self.n)
        self.assertAlmostEqual(got["auc"], 0.0, places=12)      # 1 - AUC
        self.assertAlmostEqual(got["auc_raw"], 1.0, places=12)  # immune by construction
        self.assertLess(got["scale"], 0.0)

    def test_the_negative_scale_is_reported_so_it_can_be_counted(self):
        # run_cell counts these per cell; a silent sign flip is what made the artifact
        # invisible in the first place.
        self.assertLess(_harness(self.raw, self.y, scale=-0.5, n=self.n)["scale"], 0.0)
        self.assertGreater(_harness(self.raw, self.y, scale=0.5, n=self.n)["scale"], 0.0)


class TestBothMetricsReachTheFoldSummary(unittest.TestCase):
    def test_the_keys_the_report_writer_reads_are_present(self):
        n = 200
        rng = np.random.default_rng(4)
        y = rng.normal(size=n)
        got = _harness(y, y, n=n)
        for key in ("calibration_slope", "auc", "auc_raw", "scale", "r2", "ic",
                    "balanced_accuracy", "mcc", "base_rate", "n_train", "n_test"):
            self.assertIn(key, got)


if __name__ == "__main__":
    unittest.main()
