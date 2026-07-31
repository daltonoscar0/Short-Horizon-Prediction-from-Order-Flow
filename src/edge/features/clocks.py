"""Calendar (100ms grid) and event (every N book-changing events) sampling clocks.

Both clocks emit a `Stamps(index, ts_ns)`: `index[i]` is the position in `frame` of the
*last* event with `frame.ts_ns <= ts_ns[i]` -- the frozen `<= t` convention (also see
INTERFACE.md). Every family module's `compute(frame, stamps)` consumes exactly this
pair, positionally aligned, regardless of which clock produced it; family code should
never need to know which clock it was handed.

Both accept an `EventFrame` or a raw pyarrow Table in the frozen schema.
"""
from collections import namedtuple

import numpy as np

from edge.features import SESSION_END_NS, SESSION_START_NS
from edge.features.frame import column

CALENDAR_GRID_NS = 100_000_000  # 100ms grid, FEATURES.md sec. 8.

# Book-changing event types (LOBSTER message col 2): 1 new limit order, 2 partial
# cancel, 3 full delete, 4 visible execution -- all four mutate the displayed book.
# Types 5 (hidden-order execution), 6 (cross/auction), 7 (halt indicator) are
# documented in docs/DATA_CONTRACT.md as book-unaffected (verified: the toy fixture's
# type-5 row leaves every orderbook column identical to the row before it), so they
# are never ticks of the event clock -- ticking on them would make the "event" clock
# advance on rows that carry zero order-flow information, defeating the point of an
# information-based alternative to the calendar clock.
BOOK_CHANGING_TYPES = (1, 2, 3, 4)

Stamps = namedtuple("Stamps", ["index", "ts_ns"])

# Frozen 2026-07-30: ONE N for both tickers, not a per-ticker calibration.
#
# The earlier per-ticker rule (N chosen so the event-stamp count matched the calendar
# grid: INTC 3, AMZN 1) is rejected for three reasons. (a) N=1 on AMZN is degenerate --
# the event clock becomes a relabelling of the event frame and every
# "(previous stamp, this stamp]" interval feature collapses to one event's
# contribution, which is not a window. (b) A per-ticker N confounds the clock axis with
# the ticker axis: in a table reporting INTC-vs-AMZN and calendar-vs-event side by side,
# "event time" must mean the same thing in both columns. (c) N=10 is a round a-priori
# number fitted to nothing, and it yields 62,048 stamps on INTC and 26,730 on AMZN --
# ample for Phase 2 folds.
#
# Measured wall-clock gap between consecutive event stamps at N=10 on 2012-06-21
# (this is the number that makes the two clocks' disagreement visible):
#   INTC  median 8.64 ms, IQR [0.19, 387.01] ms (width 386.82 ms), mean 377.1 ms
#   AMZN  median 352.22 ms, IQR [32.03, 1098.11] ms (width 1066.08 ms), mean 875.4 ms
EVENT_CLOCK_N = 10


def calendar_grid_stamp_count(
    session_start_ns=SESSION_START_NS, session_end_ns=SESSION_END_NS, grid_ns=CALENDAR_GRID_NS
):
    """Number of 100ms grid points in [session_start_ns, session_end_ns], both ends inclusive."""
    return (session_end_ns - session_start_ns) // grid_ns + 1


def calendar_stamps(
    frame, session_start_ns=SESSION_START_NS, session_end_ns=SESSION_END_NS, grid_ns=CALENDAR_GRID_NS
):
    """100ms grid stamps mapped to the `<= t` last-event index; pre-first-event grid points dropped."""
    ts = column(frame, "ts_ns")
    grid = np.arange(session_start_ns, session_end_ns + 1, grid_ns, dtype=np.int64)
    idx = np.searchsorted(ts, grid, side="right") - 1
    # A grid point before the first event has no "last event at or before" to report.
    # Dropping it is the frozen choice: numpy's searchsorted-1 leaves -1, which Python
    # silently reads as the LAST row of the frame -- a lookahead bug that would not
    # raise. See tests/test_clocks.py.
    valid = idx >= 0
    return Stamps(index=idx[valid].astype(np.int64), ts_ns=grid[valid])


def is_book_changing(event_type):
    """Boolean mask of book-mutating events (types 1-4); see BOOK_CHANGING_TYPES comment."""
    return np.isin(np.asarray(event_type), BOOK_CHANGING_TYPES)


def event_stamps(frame, n=EVENT_CLOCK_N):
    """Stamp the k*n-th book-changing event for k=1,2,... (1-indexed cumulative count)."""
    if n < 1:
        raise ValueError("event clock n must be >= 1")
    ts = column(frame, "ts_ns")
    changing_positions = np.flatnonzero(is_book_changing(column(frame, "event_type")))
    stamp_positions = changing_positions[n - 1 :: n]
    # The triggering event's OWN row satisfies ts <= t only by equality; INTERFACE.md
    # sec. 3 requires the LAST row with ts <= t, and LOBSTER puts many messages on the
    # same nanosecond (10.48% of INTC event stamps, up to 97 rows skipped). Using the
    # triggering row leaves a mid move that elapsed in ZERO wall-clock time -- and was
    # therefore already public at t under this project's own definition -- sitting inside
    # `label_y_mid_*`, because labels.py resolves its FORWARD endpoint with the correct
    # "last row <= t+h" rule while anchoring on this index. Same searchsorted as
    # calendar_stamps, so both clocks now mean the same thing by "the state at t".
    stamp_ts = ts[stamp_positions]
    idx = np.searchsorted(ts, stamp_ts, side="right") - 1
    # Collapsing to the last tied row makes every stamp sharing a nanosecond resolve to the
    # SAME index, i.e. an exact duplicate sample (588 on INTC, 0.95%; 6 on AMZN). Keep one
    # per distinct timestamp: INTERFACE.md sec. 3 requires ts_ns strictly increasing, and a
    # duplicated row is a real hazard downstream -- an identical sample on both sides of a
    # fold boundary is perfect leakage, and even inside one fold it silently double-weights
    # the busiest instants. There is no information in the copies: you cannot act at three
    # distinct moments within the same nanosecond.
    keep = np.ones(stamp_ts.shape[0], dtype=bool)
    keep[:-1] = stamp_ts[:-1] != stamp_ts[1:]  # keep the LAST stamp of each tied run
    return Stamps(index=idx[keep].astype(np.int64), ts_ns=stamp_ts[keep])


def stamps_for_clock(frame, clock, n=EVENT_CLOCK_N):
    """Dispatch on the frozen clock names "calendar" and "event"."""
    if clock == "calendar":
        return calendar_stamps(frame)
    if clock == "event":
        return event_stamps(frame, n=n)
    raise ValueError(f"unknown clock {clock!r}; expected 'calendar' or 'event'")
