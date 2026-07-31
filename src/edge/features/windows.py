"""Trailing-window index helpers and the prefix-sum toolkit every family aggregates with.

For a stamp with time `t` and lookback `window_ns`, a family module aggregates event
contributions over the half-open interval `(t - window_ns, t]` (FEATURES.md sec. 1's
`OFI[t_{k-1}, t_k] = Sum e_n` notation). Given `end = stamps.index[i]` (the `<= t` last
event, from clocks.py), the contributing frame slice is `[start, end + 1)`, where
`start` comes from `trailing_start_index`. Clock-agnostic: takes plain ts_ns arrays, so
the same helpers serve calendar and event stamps identically.

`start <= end_exclusive` always holds, because `start` counts events with ts <= t - w
and `end_exclusive` counts events with ts <= t; a quiet window is therefore an empty
slice (sum 0, count 0), never a negative-length one.

The prefix-sum pair (`prefix_sums` + `window_sum`/`window_count`) is the only sanctioned
way to aggregate: 234,001 calendar stamps against 624,040 events cannot be done with a
Python loop, and a prefix difference is O(1) per stamp. NaN contributions are treated as
missing (excluded from both the sum and the count), so a family can decide for itself
whether a partially-defined window is NaN or usable.

`refit_*` implements the frozen refit schedule for any *fitted* statistic (PCA loadings,
Markov transition matrices, depth normalisers): one fit per REFIT_BLOCK_NS block of wall
clock, estimated on frame rows strictly before the block start, held constant inside the
block. See INTERFACE.md "Refit on a schedule".
"""
from collections import namedtuple

import numpy as np

from edge.features import REFIT_BLOCK_NS, SESSION_START_NS


def trailing_start_index(frame_ts_ns, stamp_ts_ns, window_ns):
    """First frame index with ts_ns > stamp - window_ns, i.e. the open left edge of (t-w, t]."""
    frame_ts_ns = np.asarray(frame_ts_ns)
    lower_bound = np.asarray(stamp_ts_ns) - window_ns
    # side="right" keeps the left boundary OPEN: an event exactly window_ns old is excluded.
    return np.searchsorted(frame_ts_ns, lower_bound, side="right").astype(np.int64)


def trailing_bounds(frame_ts_ns, stamps, window_ns):
    """(start, end_exclusive) frame slice for each stamp's (t - window_ns, t] interval."""
    start = trailing_start_index(frame_ts_ns, stamps.ts_ns, window_ns)
    end_exclusive = np.asarray(stamps.index, dtype=np.int64) + 1
    return start, end_exclusive


def between_stamps_start_index(stamps_index):
    """Start index of the (previous stamp, this stamp] slice; the first stamp anchors at row 0."""
    stamps_index = np.asarray(stamps_index, dtype=np.int64)
    start = np.empty_like(stamps_index)
    if start.shape[0] > 0:
        # Consecutive stamps share a frame, so the previous stamp's index + 1 is already
        # the first event strictly after it -- no second searchsorted needed.
        start[0] = 0
        start[1:] = stamps_index[:-1] + 1
    return start


def between_stamps_bounds(stamps):
    """(start, end_exclusive) frame slice for each stamp's (previous stamp, this stamp] interval."""
    end_exclusive = np.asarray(stamps.index, dtype=np.int64) + 1
    return between_stamps_start_index(stamps.index), end_exclusive


def prefix_sums(values):
    """(sum_prefix, count_prefix) along axis 0 with a leading zero row; NaN counts as missing."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    filled = np.where(finite, values, 0.0)
    zero_shape = (1,) + values.shape[1:]
    sum_prefix = np.concatenate([np.zeros(zero_shape, dtype=np.float64), np.cumsum(filled, axis=0)])
    count_prefix = np.concatenate(
        [np.zeros(zero_shape, dtype=np.int64), np.cumsum(finite.astype(np.int64), axis=0)]
    )
    return sum_prefix, count_prefix


def window_sum(sum_prefix, start, end_exclusive):
    """Sum of the finite contributions over [start, end_exclusive) from a prefix built above."""
    return sum_prefix[np.asarray(end_exclusive, dtype=np.int64)] - sum_prefix[
        np.asarray(start, dtype=np.int64)
    ]


def window_count(count_prefix, start, end_exclusive):
    """Count of finite contributions over [start, end_exclusive) from a prefix built above."""
    return count_prefix[np.asarray(end_exclusive, dtype=np.int64)] - count_prefix[
        np.asarray(start, dtype=np.int64)
    ]


RefitBlocks = namedtuple(
    "RefitBlocks", ["stamp_block_id", "block_id", "block_start_ns", "fit_end_index"]
)


def refit_block_id(stamp_ts_ns, session_start_ns=SESSION_START_NS, block_ns=REFIT_BLOCK_NS):
    """Index of the refit block containing each stamp; block 0 opens at the session start."""
    offset = np.asarray(stamp_ts_ns, dtype=np.int64) - session_start_ns
    return np.floor_divide(offset, block_ns).astype(np.int64)


def refit_block_start_ns(block_id, session_start_ns=SESSION_START_NS, block_ns=REFIT_BLOCK_NS):
    """Wall-clock ns at which a refit block opens."""
    return session_start_ns + np.asarray(block_id, dtype=np.int64) * block_ns


def refit_fit_end_index(frame_ts_ns, block_start_ns):
    """Frame rows strictly before a block start: the exclusive end of that block's fitting slice."""
    # side="left" makes the fitting slice STRICTLY causal: an event stamped exactly at the
    # block boundary belongs to the block being predicted, not to the sample that fits it.
    return np.searchsorted(
        np.asarray(frame_ts_ns), np.asarray(block_start_ns), side="left"
    ).astype(np.int64)


def refit_blocks(
    frame_ts_ns, stamp_ts_ns, session_start_ns=SESSION_START_NS, block_ns=REFIT_BLOCK_NS
):
    """Per-stamp block id plus each distinct block's start and its strictly-prior fitting slice end."""
    stamp_block_id = refit_block_id(stamp_ts_ns, session_start_ns, block_ns)
    block_id = np.unique(stamp_block_id)
    block_start = refit_block_start_ns(block_id, session_start_ns, block_ns)
    return RefitBlocks(
        stamp_block_id=stamp_block_id,
        block_id=block_id,
        block_start_ns=block_start,
        fit_end_index=refit_fit_end_index(frame_ts_ns, block_start),
    )
