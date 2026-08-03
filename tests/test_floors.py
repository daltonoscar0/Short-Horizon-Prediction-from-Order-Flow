"""Tests for the artifact floor (`edge.harness.frozen_floor`) and the calibration controls.

The floor is the number every gap in this project is judged against, and until 2026-08-03 it
had **no test coverage at all**. The fourth red-team pass mutated it six ways — flipping
`refit_scale`'s default, fitting the floor's scale in-sample, giving every protocol A's holdout
rule, collapsing `_floor_key` to L1, measuring only one magnitude, and making `score_fold`
ignore `holdout_rule` — and all six passed a 400-test suite. Two of those are verbatim
reversions of corrections that earlier passes killed the report over.

The tests here are built on a synthetic cell with a KNOWN answer rather than on real data, so
they run in milliseconds and assert mechanism rather than a memorised number.
"""
import unittest

import numpy as np

from edge.harness import N_FOLDS
from edge.harness.models import _fit_scale
from edge.harness.run import (
    _floor_key,
    common_usable_mask,
    frozen_floor,
    frozen_floor_multi,
    run_cell,
    scale_holdout_mask,
)
from edge.harness.splits import make_folds


def _synthetic_cell(n=6000, seed=0):
    """A cell with real signal, on a 100ms grid, big enough for five folds of either shape."""
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.int64) * 100_000_000
    cols = {}
    from edge.harness.models import build_ladder

    for name in sorted({f for r in build_ladder(seed=seed) for f in r.features}):
        cols[name] = rng.normal(size=n)
    y = 0.4 * cols["qi_touch"] + 0.3 * cols["ofi_best_1s"] + rng.normal(size=n)
    return cols, y, ts


class TestTheFloorRefitsTheScale(unittest.TestCase):
    """`refit_scale=True` is the whole 2026-08-03 correction. It must be observable."""

    @classmethod
    def setUpClass(cls):
        cls.cols, cls.y, cls.ts = _synthetic_cell()
        cls.h = 1_000_000_000

    def test_refit_and_frozen_scale_are_not_the_same_measurement(self):
        refit = frozen_floor(self.cols, self.y, self.ts, self.h, 0,
                             protocols=("A", "C"), refit_scale=True)
        frozen = frozen_floor(self.cols, self.y, self.ts, self.h, 0,
                              protocols=("A", "C"), refit_scale=False)
        a = refit["A"]["r2"]["mean"] - refit["C"]["r2"]["mean"]
        b = frozen["A"]["r2"]["mean"] - frozen["C"]["r2"]["mean"]
        # If these ever coincide, the refit is not happening and the floor is blind to the
        # calibration channel again -- which is exactly what shipped and was killed.
        self.assertNotAlmostEqual(a, b, places=6)

    def test_the_default_is_the_like_for_like_floor(self):
        """A flipped default silently republishes the killed floor. It survived six mutations."""
        default = frozen_floor(self.cols, self.y, self.ts, self.h, 0, protocols=("A", "C"))
        explicit = frozen_floor(self.cols, self.y, self.ts, self.h, 0,
                                protocols=("A", "C"), refit_scale=True)
        self.assertEqual(default["A"]["r2"]["mean"], explicit["A"]["r2"]["mean"])
        self.assertEqual(default["C"]["r2"]["mean"], explicit["C"]["r2"]["mean"])

    def test_the_scale_is_fitted_on_the_HOLDOUT_not_in_sample(self):
        """Re-derive every fitted scale from the mask. In-sample was the killed calibration.

        Asserted by exact equality against an independent recomputation, because an in-sample
        scale is numerically close to a held-out one -- close enough that any inequality-based
        test passes against the mutation.
        """
        out = frozen_floor(self.cols, self.y, self.ts, self.h, 0, protocols=("A", "C"))
        mask = common_usable_mask(self.cols, self.y, seed=0)
        for protocol in ("A", "C"):
            raw = out[protocol]["frozen_raw"]
            folds = make_folds(self.ts, horizon_ns=self.h, protocol=protocol,
                               n_folds=N_FOLDS, seed=0)
            expected = []
            for f in folds:
                tr = f.train_idx[mask[f.train_idx]]
                te = f.test_idx[mask[f.test_idx]]
                if te.size < 20 or tr.size < 50:
                    continue
                hold = scale_holdout_mask(tr.size, f.protocol, f.fold_id)
                expected.append(_fit_scale(raw[tr][hold], self.y[tr][hold]))
                # The in-sample alternative must be a DIFFERENT number, or the test is vacuous.
                in_sample = _fit_scale(raw[tr], self.y[tr])
                self.assertNotAlmostEqual(expected[-1], in_sample, places=9)
            np.testing.assert_allclose(out[protocol]["scales"], expected, rtol=0, atol=0)

    def test_the_floor_uses_each_protocols_OWN_holdout_rule(self):
        """A floor that hands every protocol A's rule cannot see the rule's effect at all."""
        out = frozen_floor(self.cols, self.y, self.ts, self.h, 0, protocols=("A", "C"))
        mask = common_usable_mask(self.cols, self.y, seed=0)
        raw = out["C"]["frozen_raw"]
        folds = make_folds(self.ts, horizon_ns=self.h, protocol="C",
                           n_folds=N_FOLDS, seed=0)
        under_c_rule, under_a_rule = [], []
        for f in folds:
            tr = f.train_idx[mask[f.train_idx]]
            te = f.test_idx[mask[f.test_idx]]
            if te.size < 20 or tr.size < 50:
                continue
            under_c_rule.append(_fit_scale(
                raw[tr][scale_holdout_mask(tr.size, "C", f.fold_id)],
                self.y[tr][scale_holdout_mask(tr.size, "C", f.fold_id)]))
            a_hold = scale_holdout_mask(tr.size, "A", f.fold_id)
            under_a_rule.append(_fit_scale(raw[tr][a_hold], self.y[tr][a_hold]))
        # The two rules must give different scales on protocol C's folds...
        self.assertFalse(np.allclose(under_c_rule, under_a_rule, atol=1e-9))
        # ...and the floor must have used C's.
        np.testing.assert_allclose(out["C"]["scales"], under_c_rule, rtol=0, atol=0)

    def test_the_frozen_model_is_identical_across_protocols(self):
        """The point of a floor: nothing about the MODEL may differ, or it is not a floor."""
        out = frozen_floor(self.cols, self.y, self.ts, self.h, 0, protocols=("A", "C"))
        # `auc_raw` is scored on the frozen raw model with no rescale at all, so any protocol
        # difference in it is pure partition. It must be small; a large value means the model
        # is being refitted somewhere it should not be.
        gap = abs(out["A"]["auc_raw"]["mean"] - out["C"]["auc_raw"]["mean"])
        self.assertLess(gap, 0.05)

    def test_negative_scales_are_counted_not_swallowed(self):
        out = frozen_floor(self.cols, self.y, self.ts, self.h, 0, protocols=("A", "C"))
        self.assertIn("n_negative_scale", out["C"])
        self.assertIsInstance(out["C"]["n_negative_scale"], int)


class TestTheFloorIsMeasuredAtEveryRungMagnitude(unittest.TestCase):
    """One magnitude does not bound another — established 2026-08-02, still untested until now."""

    @classmethod
    def setUpClass(cls):
        cls.cols, cls.y, cls.ts = _synthetic_cell()

    def test_multi_returns_one_floor_per_requested_magnitude(self):
        out = frozen_floor_multi(self.cols, self.y, self.ts, 1_000_000_000, 0,
                                 protocols=("A", "C"))
        self.assertEqual(set(out), {"L1", "L3", "L4"})

    def test_the_magnitudes_do_not_collapse_to_one_number(self):
        out = frozen_floor_multi(self.cols, self.y, self.ts, 1_000_000_000, 0,
                                 protocols=("A", "C"))
        gaps = {k: v["A"]["r2"]["mean"] - v["C"]["r2"]["mean"] for k, v in out.items()}
        self.assertNotAlmostEqual(gaps["L1"], gaps["L4"], places=6)


class TestFloorKeyMatchesEachRungToItsOwnMagnitude(unittest.TestCase):
    """A `_floor_key` that collapses to L1 reprints the L1 floor beside an L4 gap."""

    def test_l3_and_l4_get_their_own(self):
        self.assertEqual(_floor_key("L3 ridge, full set"), "L3")
        self.assertEqual(_floor_key("L4 GBM, full set"), "L4")

    def test_the_unfitted_rungs_share_l1s(self):
        for name in ("L0 zero", "L1 sign(queue imbalance)",
                     "L1m sign(microprice - mid)", "L2 OLS on OFI"):
            self.assertEqual(_floor_key(name), "L1")

    def test_it_does_not_map_everything_to_one_magnitude(self):
        keys = {_floor_key(n) for n in
                ("L0 zero", "L3 ridge, full set", "L4 GBM, full set")}
        self.assertEqual(len(keys), 3)


class TestTheHoldoutRuleReachesTheFit(unittest.TestCase):
    """`score_fold` ignoring `holdout_rule` would make the `A' - C` control a copy of `A`.

    `test_score_fold.py` covers `scale_holdout_mask` in isolation. This covers the WIRING,
    which is the distinction that file's own docstring says it exists to make.
    """

    @classmethod
    def setUpClass(cls):
        cls.cols, cls.y, cls.ts = _synthetic_cell()

    def test_forcing_a_to_the_positional_rule_changes_its_numbers(self):
        default = run_cell(self.cols, self.y, self.ts, 1_000_000_000, "A", 0)
        forced = run_cell(self.cols, self.y, self.ts, 1_000_000_000, "A", 0,
                          holdout_rule="positional")
        moved = [n for n in default
                 if not n.startswith("_") and n in forced
                 and default[n]["r2"]["mean"] != forced[n]["r2"]["mean"]]
        self.assertTrue(moved, "holdout_rule reached nothing -- the control is a duplicate of A")

    def test_the_walk_forward_protocols_already_use_the_positional_rule(self):
        """Forcing C to 'positional' must be a NO-OP, or the two arms are not comparable."""
        default = run_cell(self.cols, self.y, self.ts, 1_000_000_000, "C", 0)
        forced = run_cell(self.cols, self.y, self.ts, 1_000_000_000, "C", 0,
                          holdout_rule="positional")
        for name in default:
            if name.startswith("_"):
                continue
            self.assertAlmostEqual(default[name]["r2"]["mean"], forced[name]["r2"]["mean"],
                                   places=12, msg=name)

    def test_the_mask_itself_is_what_the_wiring_passes(self):
        n = 1000
        self.assertTrue(scale_holdout_mask(n, "A", 0, "positional")[int(n * 0.8):].all())
        self.assertFalse(scale_holdout_mask(n, "A", 0, "positional")[:int(n * 0.8)].any())


class TestTheCommonMaskIsProtocolIndependent(unittest.TestCase):
    """Every protocol must be scored on the same rows, or a gap is partly a sample difference."""

    def test_the_mask_does_not_depend_on_the_protocol(self):
        cols, y, _ = _synthetic_cell()
        m = common_usable_mask(cols, y, seed=0)
        self.assertEqual(m.shape, y.shape)
        self.assertTrue(m.any())


if __name__ == "__main__":
    unittest.main()
