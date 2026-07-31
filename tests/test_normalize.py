"""Hand-computed test for rolling_zscore, plus the leak-detector the Phase-1 plan requires.

test_no_full_sample_fit is the mandatory adversarial check: (a) z-scores at stamp t must
be bit-identical whether or not future data has been appended yet (proves no lookahead,
i.e. nothing is fit on the full array), and (b) grep src/edge/features/ for the forbidden
patterns a full-sample scaler would leave behind ('.fit(', 'StandardScaler', 'sklearn').
"""
import os
import unittest

import numpy as np

from edge.features.normalize import rolling_zscore

FEATURES_DIR = os.path.join(os.path.dirname(__file__), "..", "src", "edge", "features")
FORBIDDEN_PATTERNS = [".fit(", "StandardScaler", "sklearn"]


class TestRollingZscore(unittest.TestCase):
    def test_hand_computed_values(self):
        # values = [10, 12, 11, 13, 20, 15], window=3, min_periods=2.
        # t=0: window=[] (empty)                          -> NaN (below min_periods)
        # t=1: window=[10]                                -> NaN (below min_periods)
        # t=2: window=[10,12], mean=11, var=((10-11)^2+(12-11)^2)/1=2, std=sqrt(2)
        #      z = (11-11)/sqrt(2) = 0.0
        # t=3: window=[10,12,11], mean=11, var=(1+1+0)/2=1, std=1
        #      z = (13-11)/1 = 2.0
        # t=4: window=[12,11,13], mean=12, var=(0+1+1)/2=1, std=1
        #      z = (20-12)/1 = 8.0
        # t=5: window=[11,13,20], mean=44/3, var=(sum sq resid)/2 = 67/3, std=sqrt(67/3)
        #      z = (15-44/3)/sqrt(67/3) = 0.07053456158585995
        values = [10.0, 12.0, 11.0, 13.0, 20.0, 15.0]
        z = rolling_zscore(values, window=3, min_periods=2)

        self.assertTrue(np.isnan(z[0]))
        self.assertTrue(np.isnan(z[1]))
        self.assertAlmostEqual(z[2], 0.0, places=12)
        self.assertAlmostEqual(z[3], 2.0, places=12)
        self.assertAlmostEqual(z[4], 8.0, places=12)
        self.assertAlmostEqual(z[5], 0.07053456158585995, places=12)

    def test_zero_variance_window_is_nan_not_divide_by_zero(self):
        # A constant window (no variance) must yield NaN, not a ZeroDivisionError/inf.
        values = [5.0, 5.0, 5.0, 5.0]
        z = rolling_zscore(values, window=3, min_periods=2)
        self.assertTrue(np.isnan(z[2]))
        self.assertTrue(np.isnan(z[3]))

    def test_nan_input_excluded_from_baseline(self):
        # A NaN feature draw (warmup elsewhere in the pipeline) must not corrupt the
        # baseline mean/std for later, well-defined points.
        values = [10.0, np.nan, 12.0, 14.0]
        z = rolling_zscore(values, window=3, min_periods=2)
        # t=3 baseline window is [10, nan, 12] -> valid count is 2 (10, 12), mean=11,
        # var=((10-11)^2+(12-11)^2)/1=2, std=sqrt(2); z=(14-11)/sqrt(2)=2.1213203435596424
        self.assertAlmostEqual(z[3], 2.1213203435596424, places=12)


class TestNoFullSampleFit(unittest.TestCase):
    def test_z_scores_unchanged_by_appended_future_data(self):
        rng = np.random.default_rng(0)
        past = rng.normal(size=20)
        future = rng.normal(size=10)

        z_past_only = rolling_zscore(past, window=5, min_periods=2)
        z_with_future = rolling_zscore(np.concatenate([past, future]), window=5, min_periods=2)

        # Bit-identical: a rolling estimator can only ever look backward, so appending
        # rows the past values don't yet know about must not move a single earlier z.
        np.testing.assert_array_equal(z_past_only, z_with_future[: len(past)])

    def test_no_forbidden_patterns_under_features_package(self):
        offenders = []
        for root, _, files in os.walk(FEATURES_DIR):
            for fname in files:
                if not fname.endswith(".py"):
                    continue
                path = os.path.join(root, fname)
                with open(path, encoding="utf-8") as fh:
                    text = fh.read()
                for pattern in FORBIDDEN_PATTERNS:
                    if pattern in text:
                        offenders.append((path, pattern))
        self.assertEqual(offenders, [], f"forbidden full-sample-fit pattern(s) found: {offenders}")


if __name__ == "__main__":
    unittest.main()
