"""Adversarial tests for the three-protocol splitter, WRITTEN BEFORE THE SPLITTER.

CLAUDE.md rule 2: "a splitter that lets a labelled sample straddle a fold boundary must
fail a test that exists before the splitter." This file is that test, plus the rest of the
invariants that separate protocol C from B and B from A.

The central asset here is `assert_no_label_leak`, an invariant checker written once and
pointed at every protocol. Two things make it credible rather than decorative:

  1. It is applied to a DELIBERATELY BROKEN splitter (`_leaky_folds`) inside this file, and
     asserted to fail. A leak detector that has never caught a leak is a comment.
  2. It is asserted to FAIL on protocol B and PASS on protocol C. B leaking is not a bug in
     B -- it is the entire measurement. If B ever stops leaking, the A->C / B->C
     decomposition is measuring nothing and the project's headline result is empty.

Vocabulary (docs/spec/EVAL_PROTOCOL.md sec. 1): a labelled sample at `t` with horizon `h`
occupies the closed interval [t, t+h]. It leaks into a test block [s, e) if that interval
intersects the block at all.
"""
import unittest

import numpy as np

from edge.harness.splits import Fold, make_folds


NS = 1_000_000_000


def _uniform_stamps(n=1000, step_ns=100_000_000, start=0):
    """n stamps on a regular grid; 100ms apart by default, like the calendar clock."""
    return start + np.arange(n, dtype=np.int64) * step_ns


def assert_no_label_leak(case, folds, ts_ns, horizon_ns):
    """No training sample's label window [t, t+h] may touch its fold's test block.

    This is the invariant CLAUDE.md rule 2 demands. Returns the number of violating
    (fold, sample) pairs so a caller can also assert a protocol DOES leak.
    """
    violations = 0
    for fold in folds:
        if fold.test_idx.size == 0 or fold.train_idx.size == 0:
            continue
        test_ts = ts_ns[fold.test_idx]
        block_start = test_ts.min()
        block_end = test_ts.max()  # inclusive: the last test sample's own stamp
        train_ts = ts_ns[fold.train_idx]
        label_end = train_ts + horizon_ns
        # [t, t+h] intersects [block_start, block_end] iff t <= block_end and t+h >= block_start
        overlap = (train_ts <= block_end) & (label_end >= block_start)
        violations += int(np.count_nonzero(overlap))
    return violations


def _leaky_folds(ts_ns, n_folds=5):
    """A deliberately WRONG walk-forward splitter: contiguous blocks, no purge, no embargo,
    train is everything outside the test block INCLUDING the samples immediately before it
    whose label windows run straight into it. This exists to be caught."""
    n = ts_ns.shape[0]
    bounds = np.linspace(0, n, n_folds + 1).astype(np.int64)
    out = []
    for k in range(n_folds):
        test_idx = np.arange(bounds[k], bounds[k + 1], dtype=np.int64)
        train_idx = np.setdiff1d(np.arange(n, dtype=np.int64), test_idx)
        out.append(Fold(fold_id=k, protocol="LEAKY", train_idx=train_idx, test_idx=test_idx))
    return out


class TestTheLeakDetectorItselfCatchesALeak(unittest.TestCase):
    """Guard on the guard. If this passes vacuously every other test in the file is worthless."""

    def test_the_broken_splitter_is_caught(self):
        ts = _uniform_stamps()
        horizon = 1 * NS  # 1s horizon over a 100ms grid: 10 stamps of overlap
        folds = _leaky_folds(ts)
        violations = assert_no_label_leak(self, folds, ts, horizon)
        self.assertGreater(
            violations, 0, "the deliberately leaky splitter must be caught by the detector"
        )

    def test_a_zero_horizon_sample_cannot_straddle_by_construction(self):
        # Sanity on the detector's arithmetic: with h = 0 a sample occupies a point, so the
        # only "overlap" is a training sample inside the test block, which no splitter emits.
        ts = _uniform_stamps()
        folds = make_folds(ts, horizon_ns=0, protocol="C", n_folds=5, seed=0)
        self.assertEqual(assert_no_label_leak(self, folds, ts, 0), 0)


class TestProtocolCIsClean(unittest.TestCase):
    """Protocol C -- purged walk-forward with embargo -- is the one that must not leak."""

    def setUp(self):
        self.ts = _uniform_stamps(n=2000)
        self.horizon = 1 * NS

    def test_no_training_label_window_touches_the_test_block(self):
        folds = make_folds(
            self.ts, horizon_ns=self.horizon, protocol="C", n_folds=5,
            embargo_ns=60 * NS, seed=0,
        )
        self.assertEqual(assert_no_label_leak(self, folds, self.ts, self.horizon), 0)

    def test_holds_across_horizons_and_embargoes(self):
        for horizon in (1 * NS, 2 * NS, 5 * NS, 10 * NS):
            for embargo in (0, 1 * NS, 60 * NS):
                folds = make_folds(
                    self.ts, horizon_ns=horizon, protocol="C", n_folds=5,
                    embargo_ns=embargo, seed=0,
                )
                with self.subTest(horizon=horizon, embargo=embargo):
                    self.assertEqual(
                        assert_no_label_leak(self, folds, self.ts, horizon), 0
                    )

    def test_embargo_widens_the_gap_before_the_test_block(self):
        # The gap between the last training stamp and the first test stamp must be at least
        # horizon + embargo: purge removes the label overlap, embargo removes the additional
        # serial-correlation window on top of it.
        horizon, embargo = 1 * NS, 60 * NS
        folds = make_folds(
            self.ts, horizon_ns=horizon, protocol="C", n_folds=5,
            embargo_ns=embargo, seed=0,
        )
        for fold in folds:
            if fold.train_idx.size == 0 or fold.test_idx.size == 0:
                continue
            gap = self.ts[fold.test_idx].min() - self.ts[fold.train_idx].max()
            with self.subTest(fold=fold.fold_id):
                self.assertGreaterEqual(gap, horizon + embargo)

    def test_a_larger_embargo_never_grows_the_training_set(self):
        horizon = 1 * NS
        sizes = []
        for embargo in (0, 10 * NS, 60 * NS, 300 * NS):
            folds = make_folds(
                self.ts, horizon_ns=horizon, protocol="C", n_folds=5,
                embargo_ns=embargo, seed=0,
            )
            sizes.append(sum(f.train_idx.size for f in folds))
        for earlier, later in zip(sizes, sizes[1:]):
            self.assertLessEqual(later, earlier)


class TestProtocolBLeaksOnPurpose(unittest.TestCase):
    """B is walk-forward with NO purge and NO embargo. It must leak, or B->C measures 0."""

    def test_b_straddles_the_boundary(self):
        ts = _uniform_stamps(n=2000)
        horizon = 1 * NS
        folds = make_folds(ts, horizon_ns=horizon, protocol="B", n_folds=5, seed=0)
        violations = assert_no_label_leak(self, folds, ts, horizon)
        self.assertGreater(
            violations, 0,
            "protocol B must straddle fold boundaries -- that overlap IS the B->C measurement",
        )

    def test_b_leaks_more_at_a_longer_horizon(self):
        ts = _uniform_stamps(n=2000)
        counts = [
            assert_no_label_leak(
                self, make_folds(ts, horizon_ns=h, protocol="B", n_folds=5, seed=0), ts, h
            )
            for h in (1 * NS, 2 * NS, 5 * NS, 10 * NS)
        ]
        for shorter, longer in zip(counts, counts[1:]):
            self.assertLess(shorter, longer)

    def test_b_is_still_strictly_forward_in_time(self):
        # B's defect is label overlap, NOT training on the future. Every training stamp must
        # still precede every test stamp, or B stops being walk-forward and the decomposition
        # conflates two different errors.
        ts = _uniform_stamps(n=2000)
        folds = make_folds(ts, horizon_ns=1 * NS, protocol="B", n_folds=5, seed=0)
        for fold in folds:
            if fold.train_idx.size == 0 or fold.test_idx.size == 0:
                continue
            with self.subTest(fold=fold.fold_id):
                self.assertLess(ts[fold.train_idx].max(), ts[fold.test_idx].min())


class TestProtocolAIsShuffled(unittest.TestCase):
    """A is shuffled k-fold -- what most of the literature does. It must train on the future."""

    def test_a_trains_on_the_future(self):
        ts = _uniform_stamps(n=2000)
        folds = make_folds(ts, horizon_ns=1 * NS, protocol="A", n_folds=5, seed=0)
        trained_on_future = 0
        for fold in folds:
            if fold.train_idx.size == 0 or fold.test_idx.size == 0:
                continue
            if ts[fold.train_idx].max() > ts[fold.test_idx].min():
                trained_on_future += 1
        self.assertEqual(
            trained_on_future, len(folds),
            "every shuffled fold should contain training samples after its first test sample",
        )

    def test_a_test_blocks_partition_the_eligible_universe_exactly_once(self):
        # A partitions the rows B and C can also test (segments 1..n_folds), NOT all rows:
        # the walk-forward protocols reserve segment 0 as initial training data, and scoring
        # A on rows the others never see would confound the optimism gap with a difference
        # in sample. Within that universe every row is still tested exactly once.
        ts = _uniform_stamps(n=2000)
        folds = make_folds(ts, horizon_ns=1 * NS, protocol="A", n_folds=5, seed=0)
        covered = np.concatenate([f.test_idx for f in folds])
        eligible = np.unique(
            np.concatenate([
                f.test_idx
                for f in make_folds(ts, horizon_ns=1 * NS, protocol="C", n_folds=5, seed=0)
            ])
        )
        self.assertEqual(covered.size, eligible.size)          # exactly once, no repeats
        np.testing.assert_array_equal(np.unique(covered), eligible)

    def test_a_is_deterministic_under_a_fixed_seed(self):
        ts = _uniform_stamps(n=2000)
        a = make_folds(ts, horizon_ns=1 * NS, protocol="A", n_folds=5, seed=0)
        b = make_folds(ts, horizon_ns=1 * NS, protocol="A", n_folds=5, seed=0)
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.train_idx, y.train_idx)
            np.testing.assert_array_equal(x.test_idx, y.test_idx)

    def test_a_changes_with_the_seed(self):
        ts = _uniform_stamps(n=2000)
        a = make_folds(ts, horizon_ns=1 * NS, protocol="A", n_folds=5, seed=0)
        b = make_folds(ts, horizon_ns=1 * NS, protocol="A", n_folds=5, seed=1)
        same = all(
            np.array_equal(x.test_idx, y.test_idx) for x, y in zip(a, b)
        )
        self.assertFalse(same)


class TestInvariantsCommonToEveryProtocol(unittest.TestCase):
    def setUp(self):
        self.ts = _uniform_stamps(n=2000)
        self.horizon = 1 * NS

    def test_train_and_test_never_share_a_sample(self):
        for protocol in ("A", "B", "C"):
            folds = make_folds(
                self.ts, horizon_ns=self.horizon, protocol=protocol, n_folds=5, seed=0
            )
            for fold in folds:
                with self.subTest(protocol=protocol, fold=fold.fold_id):
                    self.assertEqual(
                        np.intersect1d(fold.train_idx, fold.test_idx).size, 0
                    )

    def test_indices_are_int64_sorted_and_in_range(self):
        for protocol in ("A", "B", "C"):
            folds = make_folds(
                self.ts, horizon_ns=self.horizon, protocol=protocol, n_folds=5, seed=0
            )
            for fold in folds:
                for name in ("train_idx", "test_idx"):
                    idx = getattr(fold, name)
                    with self.subTest(protocol=protocol, fold=fold.fold_id, arr=name):
                        self.assertEqual(idx.dtype, np.int64)
                        self.assertTrue(np.all(np.diff(idx) > 0) or idx.size <= 1)
                        if idx.size:
                            self.assertGreaterEqual(idx.min(), 0)
                            self.assertLess(idx.max(), self.ts.shape[0])

    def test_walk_forward_test_blocks_are_ordered_and_disjoint(self):
        for protocol in ("B", "C"):
            folds = make_folds(
                self.ts, horizon_ns=self.horizon, protocol=protocol, n_folds=5, seed=0
            )
            ends = []
            for fold in folds:
                self.assertGreater(fold.test_idx.size, 0)
                ends.append((self.ts[fold.test_idx].min(), self.ts[fold.test_idx].max()))
            for (_, prev_end), (next_start, _) in zip(ends, ends[1:]):
                self.assertLess(prev_end, next_start)

    def test_unknown_protocol_raises(self):
        with self.assertRaises(ValueError):
            make_folds(self.ts, horizon_ns=self.horizon, protocol="Z", n_folds=5, seed=0)

    def test_unsorted_stamps_raise_rather_than_silently_mis_split(self):
        bad = self.ts.copy()
        bad[10], bad[11] = bad[11], bad[10]
        with self.assertRaises(ValueError):
            make_folds(bad, horizon_ns=self.horizon, protocol="C", n_folds=5, seed=0)

    def test_every_protocol_sees_the_same_sample_universe(self):
        # A, B and C must be scored on the SAME ROWS, or the optimism gap is partly a
        # difference in what was scored rather than how it was split. The walk-forward
        # protocols cannot test on segment 0 (it is their initial training data), and
        # segment 0 is the market open -- the most volatile part of the session. If A were
        # allowed to test there, it would be handed an easier sample and the difference
        # would be charged to leakage.
        tested = {}
        for protocol in ("A", "B", "C"):
            folds = make_folds(
                self.ts, horizon_ns=self.horizon, protocol=protocol, n_folds=5, seed=0
            )
            tested[protocol] = np.unique(np.concatenate([f.test_idx for f in folds]))
        np.testing.assert_array_equal(tested["A"], tested["B"])
        np.testing.assert_array_equal(tested["B"], tested["C"])

    def test_protocol_a_still_trains_on_the_initial_segment(self):
        # A is restricted in what it may TEST on, not in what it may train on: training on
        # all available data is what shuffled k-fold does, and crippling its training set
        # would understate the very optimism this project is trying to measure.
        folds = make_folds(
            self.ts, horizon_ns=self.horizon, protocol="A", n_folds=5, seed=0
        )
        first_test_stamp = min(self.ts[f.test_idx].min() for f in folds)
        trains_before = any(
            self.ts[f.train_idx].min() < first_test_stamp for f in folds
        )
        self.assertTrue(trains_before)


class TestIrregularStampsFromTheEventClock(unittest.TestCase):
    """The event clock is bursty -- 32% of INTC stamps sit within 0.5ms of the previous one
    (adversary phase-1 report). Fold boundaries cut on TIME must still hold."""

    def setUp(self):
        rng = np.random.default_rng(0)
        gaps = np.where(rng.random(4000) < 0.3, 400_000, rng.integers(1, 500_000_000, 4000))
        self.ts = np.cumsum(gaps.astype(np.int64))

    def test_protocol_c_is_clean_on_bursty_stamps(self):
        horizon = 1 * NS
        folds = make_folds(
            self.ts, horizon_ns=horizon, protocol="C", n_folds=5, embargo_ns=60 * NS, seed=0
        )
        self.assertEqual(assert_no_label_leak(self, folds, self.ts, horizon), 0)

    def test_a_burst_cannot_straddle_a_protocol_c_boundary(self):
        # The adversary's specific warning: if folds are cut on STAMP INDEX rather than wall
        # clock, one burst can land in two folds and both will hold the same episode.
        horizon = 1 * NS
        folds = make_folds(
            self.ts, horizon_ns=horizon, protocol="C", n_folds=5, embargo_ns=60 * NS, seed=0
        )
        for fold in folds:
            if fold.train_idx.size == 0 or fold.test_idx.size == 0:
                continue
            gap = self.ts[fold.test_idx].min() - self.ts[fold.train_idx].max()
            with self.subTest(fold=fold.fold_id):
                self.assertGreaterEqual(gap, horizon + 60 * NS)


if __name__ == "__main__":
    unittest.main()


class TestProtocolDControl(unittest.TestCase):
    """D: C's test blocks, trained on everything outside them, past AND future, no purge.

    D exists to separate two mechanisms the A-B / B-C decomposition conflated: "the model
    saw the future" (D vs C) from "the model saw rows adjacent to the test rows" (A vs D).
    """

    def _ts(self, n=1000, step=100_000_000):
        return (np.arange(n, dtype=np.int64) + 1) * step

    def test_d_test_blocks_are_identical_to_c(self):
        ts = self._ts()
        c = make_folds(ts, horizon_ns=1_000_000_000, protocol="C", n_folds=5, seed=0)
        d = make_folds(ts, horizon_ns=1_000_000_000, protocol="D", n_folds=5, seed=0)
        self.assertEqual(len(c), len(d))
        for fc, fd in zip(c, d):
            np.testing.assert_array_equal(fc.test_idx, fd.test_idx)

    def test_d_trains_on_the_future_where_c_does_not(self):
        ts = self._ts()
        d = make_folds(ts, horizon_ns=1_000_000_000, protocol="D", n_folds=5, seed=0)
        f = d[0]
        # There must be training rows AFTER the test block -- that is the whole point.
        self.assertTrue(np.any(f.train_idx > f.test_idx.max()))

    def test_d_train_and_test_are_still_disjoint(self):
        """D is leaky by design but must not be degenerate: no row is both train and test."""
        ts = self._ts()
        for f in make_folds(ts, horizon_ns=1_000_000_000, protocol="D", n_folds=5, seed=0):
            self.assertEqual(np.intersect1d(f.train_idx, f.test_idx).size, 0)

    def test_d_does_not_purge_so_it_trains_on_more_rows_than_c(self):
        ts = self._ts()
        c = make_folds(ts, horizon_ns=1_000_000_000, protocol="C", n_folds=5, seed=0)
        d = make_folds(ts, horizon_ns=1_000_000_000, protocol="D", n_folds=5, seed=0)
        for fc, fd in zip(c, d):
            self.assertGreater(fd.train_idx.size, fc.train_idx.size)
