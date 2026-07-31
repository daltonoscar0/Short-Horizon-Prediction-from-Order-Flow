"""Forward-looking labels: mid/weighted-mid returns and deadband sign, FEATURES.md sec. 7.

THE ONE MODULE ALLOWED TO LOOK FORWARD. Every other family reads only `ts_ns <= t`; a
label's entire job is the opposite, which is exactly why every forward-looking column in
the merged table lives under the single `label_` prefix -- one `grep label_` finds every
exception with no judgement call (INTERFACE.md sec. 2).

Causality convention, one rule for both ends of a label: `<= s` for a virtual stamp `s`.
The anchor at `t` reads the passed-in `book` (itself `<= t`, per book.py -- never
recomputed here). The forward endpoint at `t + h` (calendar horizons) or at stamp
position `i + N` (event horizons) is "the state at the LAST event/stamp with time
`<= t + h`" -- literally the same `<=` rule clocks.py and book.py use for `t`, just
applied to a future virtual timestamp instead of `t` itself. A virtual stamp landing
strictly between two events still resolves to the earlier one; nothing here ever reads
one row further forward than that last-qualifying event.

    y_h      = ticks(mid(last event <= t+h))  - ticks(mid(t))     h in LABEL_HORIZONS
    y_wmid_h analogous on book.wmid                     -- see substitution note below
    c_h      = classify(y_h, ALPHA_TICKS)      deadband sign, float64 in {-1, 0, +1}
    _e{N}    analogues at stamp position i+N instead of t+h      N in LABEL_HORIZONS_EVENTS

DEVIATION from FEATURES.md sec. 7's `y'_h = micro(t+h) - micro(t)`: this module may not
import microprice.py (INTERFACE.md sec. 5 rule 4, no cross-family imports), and the
Stoikov microprice is itself a FITTED quantity (INTERFACE.md sec. 8, block-refit
transition matrices) -- making a label depend on a model's refit schedule is a much
bigger commitment than the task description implies for a "lower-noise alternative". The
lower-noise alternative actually emitted is `book.wmid`, the exact (not fitted) weighted
mid, under `label_y_wmid_*` / `label_c_wmid_*`. This is a substitution, not a synonym,
and it is recorded here explicitly rather than silently renamed.

EVENT-TIME HORIZON `_e{N}`: N is a count of STAMP POSITIONS, not frame events
(INTERFACE.md sec. 6 / __init__.py's "in stamps of the event clock"), so compute() never
branches on which clock it was handed -- it just shifts `book.mid`/`book.wmid` by N
positions. Consequence: on the CALENDAR clock, consecutive stamps are exactly 100ms
apart wherever none were dropped, so `label_y_mid_e10` is arithmetically identical to
`label_y_mid_1s` there (both resolve to "last event <= t+1s"), modulo the different NaN
trigger near the very end of the session (one checks stamp-count, the other checks
wall-clock against SESSION_END_NS). On the EVENT clock the two are unrelated and `_e{N}`
is the only genuinely event-time label (median inter-stamp gap ~8.6ms on INTC, ~352ms on
AMZN at EVENT_CLOCK_N=10 -- see clocks.py). `_e{N}` is carried on both clocks anyway --
the price of one clock-agnostic module -- but it is only an interesting feature on the
event clock.

NaN, one cause per bullet, every one of them tested in tests/test_labels.py:
  - calendar horizon: `t+h` strictly past the LAST event in the frame -> NaN, never
    clipped to the last available price (a clipped label manufactures a zero return at
    the close and every model would learn it).
  - calendar horizon: `t+h` > SESSION_END_NS -> NaN, independent of the check above (a
    frame whose last printed event trails SESSION_END_NS must not backfill from it).
  - event horizon: the last N stamps (`i + N` runs off the end of `stamps`) -> NaN.
  - either endpoint's book is invalid (mid or wmid NaN there) -> NaN; this falls straight
    out of float NaN propagation through the subtraction, no separate branch needed,
    because both `book.mid`/`book.wmid` (INTERFACE.md sec. 4) and the forward-looking
    `frame.mid`/`frame.wmid` used below are already NaN wherever the touch is invalid.

DEADBAND: ALPHA_TICKS = 0.5 (half a tick) is frozen BY RULE -- FEATURES.md sec. 7's own
first suggestion, and the smallest move that is not a rounding artifact in a book quoted
in whole cents -- not fit against any data, training or otherwise. `classify()` is
exposed as a standalone pure function so a later report stage can sweep alpha for the
required sensitivity table without adding columns or re-deriving the rule. Boundary
convention: strictly greater than alpha is up, strictly less than -alpha is down, and
`|y| == alpha` exactly is FLAT (pinned by a test at exactly +-0.5).

BASE RATE WARNING: at 1s almost every `c_h` is 0 (flat) -- the deadband plus a
tick-quantised mid means most sub-second windows see no net tick move. Raw accuracy on
`c_*` is therefore meaningless on its own; Phase 2 must report balanced accuracy / MCC /
AUC against the observed base rate, never plain accuracy.

Forward endpoints are read from `frame.mid` / `frame.wmid` (EventFrame's own cached,
per-event properties, INTERFACE.md sec. 4), not a second `book_at_events(frame)` -- that
would materialise `4*L` int64 columns (~200MB on INTC at L=10) purely to throw away
everything except the touch price. `frame.ts_ns`/`frame.mid`/`frame.wmid` are read-only
and never mutated.

LOBSTER gives no aggressor-classification ambiguity to resolve here (that is trade.py's
concern, not this module's) -- labels only ever difference two touch-price references.
"""
import numpy as np

from edge.features import (
    LABEL_CLASSIFICATION_PREFIX,
    LABEL_HORIZONS,
    LABEL_HORIZONS_EVENTS,
    LABEL_REGRESSION_PREFIX,
    SESSION_END_NS,
    WINDOWS_NS,
)
from edge.features.book import book_at_stamps, ticks

# Frozen by rule (FEATURES.md sec. 7 / module docstring "DEADBAND"): half a tick, the
# smallest move that is not a rounding artifact in a whole-cent book. NOT tuned against
# any data.
ALPHA_TICKS = 0.5


def classify(y_ticks, alpha_ticks=ALPHA_TICKS):
    """Deadband sign of a tick return: +1.0 (y>alpha), -1.0 (y<-alpha), else 0.0; NaN in -> NaN out."""
    y = np.asarray(y_ticks, dtype=np.float64)
    out = np.full(y.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(y)
    out[finite] = 0.0
    out[finite & (y > alpha_ticks)] = 1.0
    out[finite & (y < -alpha_ticks)] = -1.0
    return out


def _forward_touch_index(frame_ts_ns, target_ns):
    """Position of the last frame row with ts_ns <= target_ns, plus an in-range mask.

    Same `<=` rule as clocks.calendar_stamps / book.book_at_stamps, applied to a virtual
    future timestamp instead of `t`. `in_range` is False -- never clipped -- wherever
    `target_ns` runs past the frame's own last printed event or past SESSION_END_NS.
    """
    n = frame_ts_ns.shape[0]
    if n == 0:
        empty_idx = np.zeros(target_ns.shape[0], dtype=np.int64)
        empty_mask = np.zeros(target_ns.shape[0], dtype=bool)
        return empty_idx, empty_mask
    idx = np.searchsorted(frame_ts_ns, target_ns, side="right") - 1
    idx = np.clip(idx, 0, n - 1)
    in_range = (target_ns <= frame_ts_ns[n - 1]) & (target_ns <= SESSION_END_NS)
    return idx, in_range


def compute(frame, stamps, book=None):
    """y_h / y_wmid_h / c_h / c_wmid_h at LABEL_HORIZONS, plus their _e{N} stamp-position analogues."""
    if book is None:
        book = book_at_stamps(frame, stamps)

    n = stamps.index.shape[0]
    mid_t = book.mid
    wmid_t = book.wmid

    frame_ts_ns = np.asarray(frame.ts_ns, dtype=np.int64)
    frame_mid = frame.mid
    frame_wmid = frame.wmid
    stamp_ts_ns = np.asarray(stamps.ts_ns, dtype=np.int64)

    out = {}

    # Loop over 4 calendar horizons -- a constant, not a loop over stamps (INTERFACE.md
    # sec. 5 rule 6).
    for h in LABEL_HORIZONS:
        target_ns = stamp_ts_ns + WINDOWS_NS[h]
        idx, in_range = _forward_touch_index(frame_ts_ns, target_ns)

        mid_target = np.where(in_range, frame_mid[idx], np.nan)
        wmid_target = np.where(in_range, frame_wmid[idx], np.nan)

        y_mid = ticks(mid_target - mid_t)
        y_wmid = ticks(wmid_target - wmid_t)

        out[f"{LABEL_REGRESSION_PREFIX}mid_{h}"] = y_mid
        out[f"{LABEL_REGRESSION_PREFIX}wmid_{h}"] = y_wmid
        out[f"{LABEL_CLASSIFICATION_PREFIX}mid_{h}"] = classify(y_mid)
        out[f"{LABEL_CLASSIFICATION_PREFIX}wmid_{h}"] = classify(y_wmid)

    # Loop over 4 event horizons -- likewise a constant.
    for step in LABEL_HORIZONS_EVENTS:
        mid_target = np.full(n, np.nan, dtype=np.float64)
        wmid_target = np.full(n, np.nan, dtype=np.float64)
        if n > step:
            mid_target[: n - step] = mid_t[step:]
            wmid_target[: n - step] = wmid_t[step:]

        y_mid = ticks(mid_target - mid_t)
        y_wmid = ticks(wmid_target - wmid_t)

        out[f"{LABEL_REGRESSION_PREFIX}mid_e{step}"] = y_mid
        out[f"{LABEL_REGRESSION_PREFIX}wmid_e{step}"] = y_wmid
        out[f"{LABEL_CLASSIFICATION_PREFIX}mid_e{step}"] = classify(y_mid)

    return out
