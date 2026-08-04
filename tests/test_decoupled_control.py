"""The clean calibration control: hold the MODEL's fit rows fixed, move only the scale rows.

Written before the code it constrains (RULES.md rule 2).

WHY IT EXISTS. `A' - C` was supposed to isolate the calibration rule's contribution to the
A-versus-C gap. It cannot, and the 2026-08-03 fourth red team showed why: `Rung.fit` fits the
raw model on the COMPLEMENT of the calibration holdout, so changing the rule also changes which
rows the model trains on. Under `A'` the GBM sees the first 80% of the day in time and never
sees its tail -- a temporally handicapped model, not the same model calibrated differently.
Standardising on the other rule instead points the opposite way (+0.5223 against `A'`'s
+0.1379), which is why the published figure is a bracket rather than a share.

THE FIX under test. Fit the raw model on the same seeded random 80% of each fold's training
rows regardless of protocol or rule, and fit the scale on the rule's own slice, ACCEPTING that
the scale rows now overlap the model's. The overlap costs some shrinkage -- it is a partly
in-sample scale -- but it is the *same* overlap under every arm, so a difference between arms
can no longer be a difference in the model's training window.

The tests below pin the two properties that make it a control at all:
  1. the model's fit rows do not move when the calibration rule moves;
  2. the scale's fit rows DO move.
Either one failing turns the control back into the confounded comparison it replaces.
"""
import inspect
import unittest

import numpy as np

from edge.harness.models import L3Ridge, _fit_scale
from edge.harness.run import (
    DECOUPLED_ARMS,
    model_fit_mask,
    run_cell,
    scale_holdout_mask,
    score_fold,
)
from edge.harness.splits import Fold


def _cell(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    ts = np.arange(n, dtype=np.int64) * 100_000_000
    from edge.harness.models import build_ladder

    cols = {name: rng.normal(size=n)
            for name in sorted({f for r in build_ladder(seed=seed) for f in r.features})}
    y = 0.4 * cols["qi_touch"] + 0.3 * cols["ofi_best_1s"] + rng.normal(size=n)
    return cols, y, ts


class TestTheModelFitRowsCannotDependOnProtocolOrRule(unittest.TestCase):
    """Property 1. This is the whole point; if it fails there is no control.

    An earlier version of this class asserted it by calling `model_fit_mask` twice with
    IDENTICAL arguments and comparing the result to itself — tautological, and the fifth red
    team showed it could not fail for any deterministic implementation. The property is
    structural, so it is asserted structurally: the protocol and the rule must not be
    reachable from the mask at all.
    """

    def test_the_mask_signature_cannot_see_the_protocol_or_the_rule(self):
        params = set(inspect.signature(model_fit_mask).parameters)
        self.assertEqual(params, {"n_train", "fold_id", "seed"})
        for forbidden in ("protocol", "rule", "holdout_rule", "fold"):
            self.assertNotIn(forbidden, params)

    def test_the_call_site_passes_neither(self):
        """Guards the WIRING, which is where every surviving mutation lived.

        `model_fit_mask` being protocol-blind is worthless if `score_fold` reintroduces the
        dependence at the call site — e.g. `model_fit_mask(tr.size, fold.fold_id, seed=hash(
        fold.protocol))`. Four separate mutations of exactly this shape passed a 429-test
        suite before this test existed.
        """
        src = inspect.getsource(score_fold)
        call = [ln for ln in src.splitlines() if "model_fit_mask(" in ln]
        self.assertTrue(call, "score_fold no longer calls model_fit_mask")
        for line in call:
            self.assertNotIn("protocol", line)
            self.assertNotIn("holdout_rule", line)

    def test_it_keeps_roughly_eighty_percent(self):
        m = model_fit_mask(5000, fold_id=0, seed=0)
        self.assertAlmostEqual(m.mean(), 0.8, places=2)

    def test_it_is_seeded_and_varies_by_fold(self):
        a = model_fit_mask(1000, fold_id=0, seed=0)
        np.testing.assert_array_equal(a, model_fit_mask(1000, fold_id=0, seed=0))
        self.assertFalse(np.array_equal(a, model_fit_mask(1000, fold_id=1, seed=0)))

    def test_it_is_scattered_not_a_temporal_block(self):
        """A positional model mask would reintroduce the lost-training-window confound."""
        m = model_fit_mask(1000, fold_id=0, seed=0)
        self.assertTrue(m[-50:].any(), "the model must still see the tail of its training set")
        self.assertTrue(m[:50].any())

    def test_the_call_site_mask_is_scattered_too(self):
        """The mask being scattered is worthless if the CALL SITE substitutes a positional one.

        This mutation — `mm = np.zeros(...); mm[:int(.8*n)] = True` at the call site — restores
        the exact lost-training-window confound the control exists to remove, and it survived
        the whole suite before this test.
        """
        n = 4000
        cols, y, ts = _cell(n=n)
        from edge.harness.models import build_ladder

        rung = next(r for r in build_ladder(seed=0) if r.name.startswith("L3 "))
        seen = {}
        real_fit = type(rung).fit

        def spy(self, X, yy, scale_holdout=None, model_mask=None):
            seen["mask"] = None if model_mask is None else np.asarray(model_mask).copy()
            return real_fit(self, X, yy, scale_holdout=scale_holdout, model_mask=model_mask)

        fold = Fold(fold_id=0, protocol="C",
                    train_idx=np.arange(0, 3000), test_idx=np.arange(3000, n))
        try:
            type(rung).fit = spy
            score_fold(rung, cols, y, fold, np.ones(n, dtype=bool), decouple_scale=True)
        finally:
            type(rung).fit = real_fit

        m = seen["mask"]
        self.assertIsNotNone(m, "decouple_scale did not reach the fit")
        tail = m[int(m.size * 0.8):]
        self.assertTrue(tail.any(), "the model mask reaching the fit is a positional block")


class TestTheDecoupledArmsAreWiredToTheProtocolsTheyName(unittest.TestCase):
    """`DECOUPLED_ARMS` is a dict of string pairs, so a typo in it is silent.

    Four mutations of that shape — `C_pos` pointing at protocol A, `A_rand` silently
    positional — passed a 429-test suite. The headline `A_rand - C_pos` would then be A minus
    A, i.e. approximately zero, and nothing would have complained.
    """

    def test_each_arm_names_the_protocol_and_rule_its_key_claims(self):
        for arm, (protocol, rule) in DECOUPLED_ARMS.items():
            expected_protocol, expected_rule = arm.split("_")
            with self.subTest(arm=arm):
                self.assertEqual(protocol, expected_protocol)
                self.assertEqual(rule, {"rand": "random", "pos": "positional"}[expected_rule])

    def test_the_four_arms_are_the_full_two_by_two(self):
        self.assertEqual(set(DECOUPLED_ARMS), {"A_rand", "A_pos", "C_rand", "C_pos"})

    def test_the_headline_arms_straddle_the_two_protocols(self):
        """`A_rand - C_pos` must compare A against C, or the headline is a self-difference."""
        self.assertEqual(DECOUPLED_ARMS["A_rand"][0], "A")
        self.assertEqual(DECOUPLED_ARMS["C_pos"][0], "C")
        self.assertNotEqual(DECOUPLED_ARMS["A_rand"][0], DECOUPLED_ARMS["C_pos"][0])

    def test_the_runner_actually_decouples_when_it_fills_those_arms(self):
        """`decouple_scale=False` in main would silently publish the CONFOUNDED arm.

        The four arms would still run, the table would still be emitted, every number would
        still look plausible — and it would be the `A′` construction whose attribution this
        control exists to withdraw. It lives in `main`, so it is asserted on the source; a
        mutation of exactly this shape passed the whole suite.
        """
        from edge.harness import run as run_mod

        lines = inspect.getsource(run_mod.main).splitlines()
        start = next(i for i, ln in enumerate(lines) if "DECOUPLED_ARMS" in ln)
        stanza = "\n".join(lines[start:start + 8])
        self.assertIn("decouple_scale=True", stanza)
        self.assertNotIn("decouple_scale=False", stanza)


class TestTheScaleRowsDoMoveWithTheRule(unittest.TestCase):
    """Property 2. A control where nothing moves measures nothing."""

    def test_the_two_rules_select_different_scale_rows(self):
        n = 1000
        rnd = scale_holdout_mask(n, "A", 0, "random")
        pos = scale_holdout_mask(n, "A", 0, "positional")
        self.assertFalse(np.array_equal(rnd, pos))


class TestDecoupledFitSeparatesTheTwoRowSets(unittest.TestCase):
    def test_the_raw_model_is_fitted_on_the_model_mask_only(self):
        rng = np.random.default_rng(0)
        n = 500
        X = rng.normal(size=(n, len(L3Ridge().features)))
        y = rng.normal(size=n)
        model_mask = model_fit_mask(n, fold_id=0, seed=0)

        a = L3Ridge()
        a.fit(X, y, scale_holdout=scale_holdout_mask(n, "A", 0, "random"),
              model_mask=model_mask)
        b = L3Ridge()
        b.fit(X, y, scale_holdout=scale_holdout_mask(n, "C", 0, "positional"),
              model_mask=model_mask)

        # Same model rows -> byte-identical raw score, under two different calibration rules.
        np.testing.assert_allclose(a._raw(X), b._raw(X), rtol=0, atol=0)
        # ...and different scales, because the rule moved.
        self.assertNotAlmostEqual(a.scale_, b.scale_, places=9)

    def test_the_scale_is_fitted_on_the_scale_rows_including_the_overlap(self):
        rng = np.random.default_rng(1)
        n = 500
        X = rng.normal(size=(n, len(L3Ridge().features)))
        y = rng.normal(size=n)
        model_mask = model_fit_mask(n, fold_id=0, seed=0)
        hold = scale_holdout_mask(n, "C", 0, "positional")

        r = L3Ridge()
        r.fit(X, y, scale_holdout=hold, model_mask=model_mask)
        self.assertAlmostEqual(r.scale_, _fit_scale(r._raw(X[hold]), y[hold]), places=12)

    def test_the_overlap_is_real_and_is_the_accepted_cost(self):
        """The scale rows are NOT disjoint from the model rows. Documented, not hidden."""
        n = 1000
        model_mask = model_fit_mask(n, fold_id=0, seed=0)
        hold = scale_holdout_mask(n, "C", 0, "positional")
        self.assertTrue((model_mask & hold).any())

    def test_the_default_path_is_untouched(self):
        """`model_mask=None` must reproduce the shipped calibration exactly."""
        rng = np.random.default_rng(2)
        n = 500
        X = rng.normal(size=(n, len(L3Ridge().features)))
        y = rng.normal(size=n)
        hold = scale_holdout_mask(n, "C", 0)

        a = L3Ridge()
        a.fit(X, y, scale_holdout=hold)
        b = L3Ridge()
        b.fit(X, y, scale_holdout=hold, model_mask=None)
        np.testing.assert_allclose(a.predict(X), b.predict(X), rtol=0, atol=0)


class TestScoreFoldWiresTheControlThrough(unittest.TestCase):
    """The mask must reach the fit. `scale_holdout_mask` being right in isolation is not enough."""

    @classmethod
    def setUpClass(cls):
        cls.cols, cls.y, cls.ts = _cell()

    def _fold(self, protocol):
        n = self.y.shape[0]
        return Fold(fold_id=0, protocol=protocol,
                    train_idx=np.arange(0, 3000), test_idx=np.arange(3000, n))

    def test_the_decoupled_arm_gives_the_same_model_under_both_rules(self):
        """The signature property, measured end to end: raw AUC cannot move with the rule.

        Run under BOTH protocols. An earlier version built `self._fold("A")` twice, so it
        could not see a mask that varied by protocol — the exact confound the control exists
        to remove, and a mutation of that shape passed the whole suite.
        """
        mask = np.ones(self.y.shape[0], dtype=bool)
        from edge.harness.models import build_ladder

        for protocol in ("A", "C"):
            rung = next(r for r in build_ladder(seed=0) if r.name.startswith("L3 "))
            with self.subTest(protocol=protocol):
                a = score_fold(rung, self.cols, self.y, self._fold(protocol), mask,
                               holdout_rule="random", decouple_scale=True)
                c = score_fold(rung, self.cols, self.y, self._fold(protocol), mask,
                               holdout_rule="positional", decouple_scale=True)
                # AUC on the RAW score is a pure function of the model. Same rows -> identical.
                self.assertAlmostEqual(a["auc_raw"], c["auc_raw"], places=12)
                # The rescaled prediction still differs, because the scale moved.
                self.assertNotAlmostEqual(a["scale"], c["scale"], places=9)

    def test_the_model_is_identical_across_PROTOCOLS_given_the_same_training_rows(self):
        """The mask must not vary by protocol. Same rows in, same model out, whatever the label.

        `Fold.protocol` is the only thing differing between these two calls, and the training
        indices are deliberately identical, so any difference in the raw score is protocol
        dependence in the mask.
        """
        mask = np.ones(self.y.shape[0], dtype=bool)
        from edge.harness.models import build_ladder

        outs = []
        for protocol in ("A", "C"):
            rung = next(r for r in build_ladder(seed=0) if r.name.startswith("L3 "))
            outs.append(score_fold(rung, self.cols, self.y, self._fold(protocol), mask,
                                   holdout_rule="random", decouple_scale=True)["auc_raw"])
        self.assertAlmostEqual(outs[0], outs[1], places=12)

    def test_without_the_control_the_model_DOES_move_with_the_rule(self):
        """Guards the test above: if these agreed, the control would be measuring nothing."""
        mask = np.ones(self.y.shape[0], dtype=bool)
        from edge.harness.models import build_ladder

        rung = next(r for r in build_ladder(seed=0) if r.name.startswith("L3 "))
        a = score_fold(rung, self.cols, self.y, self._fold("A"), mask,
                       holdout_rule="random")
        c = score_fold(rung, self.cols, self.y, self._fold("A"), mask,
                       holdout_rule="positional")
        self.assertNotAlmostEqual(a["auc_raw"], c["auc_raw"], places=9)

    def test_run_cell_accepts_the_arm(self):
        out = run_cell(self.cols, self.y, self.ts, 1_000_000_000, "A", 0,
                       holdout_rule="positional", decouple_scale=True)
        self.assertIn("L3 ridge, full set", out)


if __name__ == "__main__":
    unittest.main()
