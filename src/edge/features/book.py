"""Shared book spine: best quotes, L-level depth, and price references at every stamp.

Timestamp convention: **`<= t`**. `stamps.index[i]` is the position of the last event with
`ts_ns <= stamps.ts_ns[i]` (edge.features.clocks), and orderbook row `k` is the book state
*after* message `k` (docs/DATA_CONTRACT.md alignment invariant). So the state reported at a
stamp is the post-state of the last event at or before it, and a stamp landing strictly
between two events resolves to the EARLIER one -- never the later one. Nothing here reads a
row past `stamps.index[i]`; the violation detector is
tests/test_book.py::TestNoLookahead.

Every family module (ofi, queue, microprice, trade, state, labels) imports this instead of
re-extracting best quotes, so the weighted-mid crossing (FEATURES.md sec. 3) and the
sentinel/zero-size validity rule exist in exactly one place. Full contract in
docs/INTERFACE.md.

Units: prices and sizes stay in raw LOBSTER units (0.01 cents, shares) exactly as stored;
derived price references (mid, wmid, spread) are float64 raw units with NaN wherever
`valid` is False, so a sentinel price can never silently enter a feature. Report in ticks
via `ticks()`, never by dividing by 100 locally.
"""
from collections import namedtuple

import numpy as np

from edge.features import NUM_LEVELS, TICK_SIZE_RAW
from edge.io.lobster import ASK_SENTINEL_PRICE, BID_SENTINEL_PRICE

BookAtStamps = namedtuple(
    "BookAtStamps",
    [
        "bid_px",
        "bid_sz",
        "ask_px",
        "ask_sz",
        "bid_px_l",
        "bid_sz_l",
        "ask_px_l",
        "ask_sz_l",
        "mid",
        "wmid",
        "spread",
        "spread_ticks",
        "valid",
        "valid_l",
        "n_levels",
    ],
)

EventArrays = namedtuple("EventArrays", ["ts_ns", "event_type", "direction", "price", "size"])

_SIDES = ("bid", "ask")
_KINDS = ("px", "sz")


def ticks(x):
    """Raw LOBSTER price units -> ticks (1 tick = $0.01 = TICK_SIZE_RAW raw units)."""
    return np.asarray(x, dtype=np.float64) / TICK_SIZE_RAW


def frame_num_levels(frame):
    """Levels actually present in the frame schema, capped at NUM_LEVELS (10 real, 2 toy)."""
    names = set(frame.schema.names)
    lvl = 0
    while lvl < NUM_LEVELS and all(
        f"{side}_{kind}_{lvl + 1}" in names for side in _SIDES for kind in _KINDS
    ):
        lvl += 1
    return lvl


def event_frame_arrays(frame):
    """Message columns as int64 numpy arrays: ts_ns, event_type, direction, price, size.

    int64, not the frame's int8 for event_type/direction: under NEP 50 an int8 array times a
    Python int stays int8 and wraps silently (`direction * 100` -> overflow), which would
    corrupt any signed-volume aggregate. Copies, never views into the arrow buffers, so a
    family cannot mutate the shared frame through what it was handed.
    """
    return EventArrays(
        *(
            frame.column(name).to_numpy(zero_copy_only=False).astype(np.int64)
            for name in ("ts_ns", "event_type", "direction", "price", "size")
        )
    )


def _resolve_num_levels(frame, num_levels):
    available = frame_num_levels(frame)
    if num_levels is None:
        num_levels = available
    if num_levels < 1:
        raise ValueError("frame carries no complete book level (need bid_px_1..ask_sz_1)")
    if num_levels > available:
        raise ValueError(
            f"requested {num_levels} levels, frame carries {available} complete level(s)"
        )
    return num_levels


def _stack_levels(frame, side, kind, n_levels, take):
    """(n, n_levels) int64 matrix of `{side}_{kind}_1..n_levels`, gathered at `take` rows.

    Gathers column by column rather than materialising all 4*L full-length columns first:
    on the real frame that is a 200MB peak for information we immediately throw away.
    """
    cols = []
    for lvl in range(1, n_levels + 1):
        col = frame.column(f"{side}_{kind}_{lvl}").to_numpy(zero_copy_only=False)
        col = col.astype(np.int64, copy=False)
        cols.append(col if take is None else col[take])
    return np.stack(cols, axis=1)


def _build(frame, num_levels, take):
    n_levels = _resolve_num_levels(frame, num_levels)
    bid_px_l = _stack_levels(frame, "bid", "px", n_levels, take)
    bid_sz_l = _stack_levels(frame, "bid", "sz", n_levels, take)
    ask_px_l = _stack_levels(frame, "ask", "px", n_levels, take)
    ask_sz_l = _stack_levels(frame, "ask", "sz", n_levels, take)

    # A level is usable only if BOTH sides show a real price and positive size -- same rule
    # as io/quality.py::sentinel_row_count, extended to size 0 (LOBSTER pairs the sentinel
    # price with size 0, but a 0 size alone is equally unusable: it makes wmid 0/0).
    valid_l = (
        (bid_px_l != BID_SENTINEL_PRICE)
        & (ask_px_l != ASK_SENTINEL_PRICE)
        & (bid_sz_l > 0)
        & (ask_sz_l > 0)
    )
    valid = valid_l[:, 0]

    bid_px = bid_px_l[:, 0]
    bid_sz = bid_sz_l[:, 0]
    ask_px = ask_px_l[:, 0]
    ask_sz = ask_sz_l[:, 0]

    n = bid_px.shape[0]
    nan = np.full(n, np.nan, dtype=np.float64)

    mid = np.where(valid, (bid_px + ask_px) / 2.0, nan)

    # FEATURES.md sec. 3: wmid = (P^b*q^a + P^a*q^b) / (q^a + q^b). The weights are CROSSED
    # -- the bid price carries the ASK size -- so a heavy ask pulls wmid toward the bid.
    # Uncrossing them silently inverts every microprice deviation feature.
    numer = bid_px.astype(np.float64) * ask_sz + ask_px.astype(np.float64) * bid_sz
    denom = (bid_sz + ask_sz).astype(np.float64)
    wmid = np.full(n, np.nan, dtype=np.float64)
    np.divide(numer, denom, out=wmid, where=valid)  # valid => denom > 0, no 0/0

    spread = np.where(valid, (ask_px - bid_px).astype(np.float64), nan)

    return BookAtStamps(
        bid_px=bid_px,
        bid_sz=bid_sz,
        ask_px=ask_px,
        ask_sz=ask_sz,
        bid_px_l=bid_px_l,
        bid_sz_l=bid_sz_l,
        ask_px_l=ask_px_l,
        ask_sz_l=ask_sz_l,
        mid=mid,
        wmid=wmid,
        spread=spread,
        spread_ticks=ticks(spread),
        valid=valid,
        valid_l=valid_l,
        n_levels=n_levels,
    )


def book_at_stamps(frame, stamps, num_levels=None):
    """Book state at each stamp: the post-state of the last event with ts_ns <= stamp ts."""
    take = np.asarray(stamps.index, dtype=np.int64)
    return _build(frame, num_levels, take)


def book_at_events(frame, num_levels=None):
    """Same fields as `book_at_stamps`, one row per FRAME event -- what per-event families
    (OFI's e_n, effective spread of a trade) difference across consecutive events before
    aggregating into a stamp's trailing window."""
    return _build(frame, num_levels, None)
