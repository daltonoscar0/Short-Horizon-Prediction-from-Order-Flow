"""The three-protocol splitter (EVAL_PROTOCOL.md sec. 1). Hand-written; no sklearn.

Adversarial tests for this module were written BEFORE it existed (tests/test_splits.py,
RULES.md rule 2) and include a deliberately broken splitter that the leak detector is
asserted to catch.

TEST BLOCKS ARE CUT ON WALL-CLOCK TIME, NOT STAMP INDEX. The phase-1 red team measured that
32% of INTC event-clock stamps sit within 0.5 ms of the previous one and that those bursts
carry ~2.3x the correlation of isolated stamps; cutting on stamp index lets a single burst
land on both sides of a boundary, so the two folds hold the same episode and the result reads
as optimism that is really duplication. Cutting on time makes the purge and embargo windows
-- which are durations -- mean what they say. The cost is that per-fold sample counts vary,
which EVAL_PROTOCOL.md sec. 1 requires reporting anyway.

Timeline layout for the walk-forward protocols (B and C), n_folds = 5:

    |  seg 0  |  seg 1  |  seg 2  |  seg 3  |  seg 4  |  seg 5  |
    |<- train ->| test 0 |
    |<------ train ----->| test 1 |
    |<---------- train ----->| test 2 |            ... expanding

`n_folds + 1` equal-duration segments; segment 0 is initial training data only, and segments
1..n_folds are the test blocks. Training is EXPANDING (everything before the test block),
which is the variant every interviewer knows.

Purge and embargo (protocol C only) are additive at the left edge:

    keep a training sample t  iff  t + horizon + embargo  <  test_block_start

`horizon` removes the label overlap -- a sample at t owns the interval [t, t+h], so anything
whose interval reaches into the test block is contaminated. `embargo` removes an ADDITIONAL
serial-correlation window on top of that, per EVAL_PROTOCOL.md sec. 1's E. They are additive
rather than a max because they are answering two different questions (does my label overlap
the test window / is my feature still correlated with the test window), and the conservative
composition is the defensible one. The inequality is strict so a sample whose label ends
exactly at the boundary is still dropped.

EVAL_PROTOCOL.md also describes an embargo AFTER the test block, for the next fold. Under
expanding walk-forward every training sample precedes its test block, so a right-edge embargo
can never bind; it is implemented anyway (`_purge_embargo` masks both sides) so the function
stays correct if a later phase adds combinatorial purged CV, where training data does follow
test data.
"""
from collections import namedtuple

import numpy as np

from edge.harness import K_SHUFFLED, N_FOLDS, PROTOCOLS, default_embargo_ns

Fold = namedtuple("Fold", ["fold_id", "protocol", "train_idx", "test_idx"])


def _validate(ts_ns, horizon_ns, n_folds):
    ts = np.asarray(ts_ns, dtype=np.int64)
    if ts.ndim != 1:
        raise ValueError("ts_ns must be 1-D")
    if ts.size and np.any(np.diff(ts) < 0):
        raise ValueError("ts_ns must be non-decreasing; a splitter cannot fix unsorted time")
    if horizon_ns < 0:
        raise ValueError("horizon_ns must be >= 0")
    if n_folds < 2:
        raise ValueError("n_folds must be >= 2")
    return ts


def _time_segments(ts, n_folds):
    """`n_folds + 1` equal-DURATION segment edges spanning [ts.min(), ts.max()]."""
    lo = int(ts[0])
    hi = int(ts[-1])
    if hi <= lo:
        raise ValueError("ts_ns spans zero duration; cannot cut folds on time")
    # +1 on the last edge so the final stamp lands inside the last segment, not outside it.
    edges = np.linspace(lo, hi + 1, n_folds + 2)
    return edges.astype(np.int64)


def _purge_embargo(train_ts, block_start, block_end, horizon_ns, embargo_ns):
    """Boolean keep-mask over `train_ts`, dropping anything contaminated by the test block.

    Left edge:  t + horizon + embargo < block_start   (label overlap, then serial correlation)
    Right edge: t > block_end + embargo               (only binds if training follows test)
    """
    before = train_ts + horizon_ns + embargo_ns < block_start
    after = train_ts > block_end + embargo_ns
    return before | after


def make_folds(ts_ns, horizon_ns, protocol, n_folds=N_FOLDS, embargo_ns=None, seed=0):
    """Folds for one protocol over stamps `ts_ns` carrying a label of horizon `horizon_ns`.

    Returns `list[Fold]`. `train_idx`/`test_idx` are sorted int64 positions into `ts_ns`.
    Folds whose test block holds no stamps are dropped rather than returned empty, so a
    caller never divides by zero; the returned list may therefore be shorter than `n_folds`.
    """
    if protocol not in PROTOCOLS:
        raise ValueError(f"unknown protocol {protocol!r}; expected one of {PROTOCOLS}")
    ts = _validate(ts_ns, horizon_ns, n_folds)
    n = ts.shape[0]
    if n == 0:
        return []
    if embargo_ns is None:
        embargo_ns = default_embargo_ns(horizon_ns)
    embargo_ns = int(embargo_ns)

    edges = _time_segments(ts, n_folds)

    if protocol == "A":
        # A must be SCORED ON THE SAME ROWS as B and C, or part of the A->C gap is a
        # difference in which rows were tested rather than in how they were split. The
        # walk-forward protocols never test on segment 0 (it is their initial training
        # data), and segment 0 is the market open -- the most volatile, most predictable
        # part of the session. Letting A test there would hand it an easier sample and
        # charge the difference to leakage. So A's test folds partition the SAME universe
        # B and C test on; A still TRAINS on everything outside its test fold, including
        # segment 0, because training on all available data is what shuffled k-fold does.
        eligible = np.flatnonzero(ts >= int(edges[1])).astype(np.int64)
        return _shuffled_kfold(n, k=n_folds, seed=seed, eligible=eligible)

    all_idx = np.arange(n, dtype=np.int64)
    folds = []
    for k in range(n_folds):
        block_start = int(edges[k + 1])
        block_end = int(edges[k + 2])  # exclusive
        test_mask = (ts >= block_start) & (ts < block_end)
        test_idx = all_idx[test_mask]
        if test_idx.size == 0:
            continue  # a segment with no stamps: nothing to score, not an error

        # Both walk-forward protocols train only on the past. B differs from C ONLY in the
        # purge/embargo mask -- that is what makes B->C a clean measurement of label overlap
        # rather than a mixture of several changes at once.
        past_mask = ts < block_start
        if protocol == "D":
            # Protocol D: C's test block exactly, but training on EVERYTHING outside it --
            # past and future, no purge, no embargo. Added 2026-08-02 after the phase-2 red
            # team showed the A-B / B-C decomposition attributes the wrong mechanisms.
            #
            # D holds the test universe, the fold shapes and the fold aggregation fixed and
            # varies ONLY "may the model see the future". So:
            #   A - D  is the fold-geometry and adjacency channel
            #   D - C  is training on the future in DISJOINT blocks
            # Measured, D - C is worth ~0.01-0.05 in IC while A - D is worth 0.11-0.36:
            # nearly all of A's advantage comes from training rows temporally ADJACENT to
            # test rows, not from the future per se. That is label overlap, and it sits
            # inside A-B rather than in B-C.
            train_mask = ~test_mask
        elif protocol == "C":
            last_test_ts = int(ts[test_idx].max())
            keep = _purge_embargo(
                ts, block_start, last_test_ts, int(horizon_ns), embargo_ns
            )
            train_mask = past_mask & keep
        else:
            train_mask = past_mask

        folds.append(
            Fold(
                fold_id=len(folds),
                protocol=protocol,
                train_idx=all_idx[train_mask],
                test_idx=test_idx,
            )
        )
    return folds


def _shuffled_kfold(n, k, seed, eligible=None):
    """Protocol A: the pathology, implemented faithfully. Every eligible sample is tested
    exactly once and training sets contain samples from after the test block -- the point.

    `eligible` restricts which samples may be TESTED (see make_folds); training still draws
    on every row outside the current test fold.
    """
    rng = np.random.default_rng(seed)
    pool = np.arange(n, dtype=np.int64) if eligible is None else np.asarray(eligible, dtype=np.int64)
    order = pool[rng.permutation(pool.shape[0])]
    bounds = np.linspace(0, pool.shape[0], k + 1).astype(np.int64)
    all_idx = np.arange(n, dtype=np.int64)
    folds = []
    for i in range(k):
        test_idx = np.sort(order[bounds[i] : bounds[i + 1]]).astype(np.int64)
        if test_idx.size == 0:
            continue
        train_mask = np.ones(n, dtype=bool)
        train_mask[test_idx] = False
        folds.append(
            Fold(
                fold_id=len(folds),
                protocol="A",
                train_idx=all_idx[train_mask],
                test_idx=test_idx,
            )
        )
    return folds
