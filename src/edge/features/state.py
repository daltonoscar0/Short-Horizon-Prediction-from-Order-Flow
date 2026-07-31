"""State / control features: spread regime, depth, book slope, realized vol, event
intensity, cancel rate, average order size, and time-of-day (FEATURES.md sec. 5).

Causality: strictly `<= t` (INTERFACE.md sec. 2), no stricter convention than that. Every
windowed statistic aggregates frame rows with `ts_ns` in the half-open `(t - w, t]`
interval via `windows.trailing_bounds`, whose `end_exclusive = stamps.index[i] + 1` never
reaches past the last event `<= t`. Touch/level readouts come straight from `book`, which
is already `<= t` by construction (edge.features.book docstring).

`st_rv_{w}` is the one feature EVAL_PROTOCOL.md names by example as the easy way to build
a target-derived (leaking) feature: its per-event squared-return contribution is indexed
so that event n's contribution -- the (mid(n-1), mid(n)) transition -- is only visible
once event n has itself printed (contribution[0] is NaN: there is no event -1). A trailing
window ending at `stamps.index[i]` therefore can only ever include contributions up to and
including event `stamps.index[i]`, never mid(stamps.index[i] + 1). See
`TestRealizedVolRightEdge` for the test that pins this boundary.

NaN conventions used throughout (INTERFACE.md sec. 7):
  - touch/level readouts (`st_spread_ticks`, `st_depth_*_touch`, `st_slope_*`) are NaN
    wherever `book.valid` is False at that stamp, matching book.py's own convention for
    mid/wmid/spread -- the touch is one two-sided quantity, not two independent ones.
  - `st_depth_total` is a SUM over levels 1..L, masked per level by `book.valid_l`
    (a level contributes 0 if either side is absent there) -- a sum over zero valid levels
    is legitimately 0.0, not missing, so this column is never NaN.
  - `st_slope_bid` / `st_slope_ask` additionally use the PER-SIDE level mask
    (`frame.bid_valid` / `frame.ask_valid`, gathered at `stamps.index`) rather than the
    joint `book.valid_l`, because a bid-side regression should not be thrown a level just
    because the ask book happens to be thinner there that day -- book.py's own docstring
    flags this as a "pick the one your formula means" choice. NaN with fewer than 2 valid
    levels or zero x-variance across them (a straight vertical/horizontal book slice).
  - `st_rv_{w}` is NaN when the window holds zero valid per-event contributions (the first
    event in the frame, or a window narrower than one inter-event gap); 0.0 is a real "no
    price moved" observation once at least one contribution is available.
  - `st_cancel_rate_{w}` and `st_avg_order_size_{w}` are NaN on a zero denominator (no
    book-changing events / no type-1 events in the window respectively);
    `st_event_rate_{w}` never is -- its denominator is the fixed window length, not data.

`st_slope_bid` / `st_slope_ask` sign convention: distance from mid is a POSITIVE magnitude
on both sides (bid: `(mid - P^b)/tick`; ask: `(P^a - mid)/tick`), so the two slopes are
directly comparable -- a positive slope means depth builds away from the touch on that
side, on both sides, rather than one side's convention being a mirror image of the other's.

`st_tod_bucket` and `st_tod_frac` are pure functions of the stamp's own wall-clock time,
which is legitimately `<= t` information -- they are CONTROLS, not alpha. The assembly
stage separately builds the `_sadj` seasonality-adjusted twin of every raw column
(`SEASONAL_ADJ_SUFFIX`, FEATURES.md sec. 5's "produce every result twice"); this module
must not, and does not, emit any `_sadj` column itself.

No statistic here is FITTED (no PCA loadings, no transition matrix, no depth normaliser
estimated from past rows), so INTERFACE.md sec. 8's refit-block schedule does not apply
and there is no `refit_blocks` loop in this module.

LOBSTER gives direction explicitly for trade events; this module does not touch trade
direction at all (that is trade.py's job) -- noted here only because FEATURES.md sec. 4
asks every family touching aggressor flow to say so once.
"""
import numpy as np

from edge.features import (
    FEATURE_WINDOWS,
    NS_PER_SEC,
    SESSION_END_NS,
    SESSION_START_NS,
    TOD_BUCKET_NS,
    WINDOWS_NS,
)
from edge.features.book import book_at_stamps, ticks
from edge.features.clocks import BOOK_CHANGING_TYPES
from edge.features.windows import prefix_sums, trailing_bounds, window_count, window_sum

# LOBSTER event-type vocabulary (docs/DATA_AND_LINKS.md), frozen a priori -- not fit from
# the data. Cancel-rate numerator: type 2 (partial cancel) + type 3 (full delete).
_CANCEL_TYPES = (2, 3)
# Average-order-size numerator: type 1 (new limit order) only.
_NEW_LIMIT_TYPE = 1


def _mask_touch(raw_values, valid):
    """float64 touch quantity, NaN wherever the touch itself is invalid (~book.valid)."""
    out = np.full(raw_values.shape[0], np.nan, dtype=np.float64)
    out[valid] = raw_values[valid].astype(np.float64)
    return out


def _ols_slope(x, y, valid):
    """Vectorised closed-form OLS slope of y on x per row, using only `valid` columns:
    slope = Sum((x-xbar)(y-ybar)) / Sum((x-xbar)^2). No per-row Python loop -- reduced
    along axis=1 for every stamp at once. NaN when a row has fewer than 2 valid columns
    or zero x-variance across them (a divide-by-zero the spec asks to catch, not raise on).
    """
    valid = np.asarray(valid, dtype=bool)
    count = valid.sum(axis=1).astype(np.float64)
    safe_count = np.where(count > 0, count, 1.0)  # avoids 0/0 in the mean; result unused below

    # Mask BEFORE any subtraction so an invalid column's (possibly NaN, e.g. mid undefined)
    # raw value never enters the arithmetic -- np.where only picks branches, it does not
    # protect against NaN already baked into an unmasked intermediate.
    xf = np.where(valid, x, 0.0)
    yf = np.where(valid, y, 0.0)
    mean_x = xf.sum(axis=1) / safe_count
    mean_y = yf.sum(axis=1) / safe_count
    dx = np.where(valid, xf - mean_x[:, None], 0.0)
    dy = np.where(valid, yf - mean_y[:, None], 0.0)
    sum_xy = np.sum(dx * dy, axis=1)
    sum_xx = np.sum(dx * dx, axis=1)

    slope = np.full(x.shape[0], np.nan, dtype=np.float64)
    ok = (count >= 2) & (sum_xx > 0)
    np.divide(sum_xy, sum_xx, out=slope, where=ok)
    return slope


def _book_slope(mid, px_l, sz_l, level_valid, direction):
    """Book slope on one side: cumulative size vs. distance from mid, in ticks.

    `direction` is +1.0 for the bid side (`(mid - P^b)/tick`, positive since P^b <= mid)
    and -1.0 for the ask side (`-(mid - P^a)/tick = (P^a - mid)/tick`, positive since
    P^a >= mid) -- see module docstring for why both sides use a positive magnitude.
    """
    distance = direction * ticks(mid[:, None] - px_l.astype(np.float64))
    cum_size = np.cumsum(np.where(level_valid, sz_l.astype(np.float64), 0.0), axis=1)
    return _ols_slope(distance, cum_size, level_valid)


def compute(frame, stamps, book=None):
    """State/control features over FEATURE_WINDOWS plus time-of-day (FEATURES.md sec. 5).

    See module docstring for the `<= t` convention, the NaN rules and the slope sign
    convention.
    """
    if book is None:
        book = book_at_stamps(frame, stamps)

    n = stamps.index.shape[0]

    # -- spread regime ------------------------------------------------------------
    spread_ticks = book.spread_ticks.copy()  # defensive copy: `book` is shared across families
    spread_ticks_int = np.round(spread_ticks)  # still float64; prices are tick-aligned so this
    #                                            is a no-op except as a rounding safety net.

    # -- depth ----------------------------------------------------------------------
    depth_bid_touch = _mask_touch(book.bid_sz, book.valid)
    depth_ask_touch = _mask_touch(book.ask_sz, book.valid)
    depth_touch = depth_bid_touch + depth_ask_touch

    bid_sz_l = book.bid_sz_l.astype(np.float64)
    ask_sz_l = book.ask_sz_l.astype(np.float64)
    depth_total = np.sum(np.where(book.valid_l, bid_sz_l + ask_sz_l, 0.0), axis=1)

    # -- book slope -------------------------------------------------------------------
    # Per-side level validity (see module docstring for why this differs from the joint
    # book.valid_l), gathered at each stamp's `<= t` frame row, further intersected with
    # "mid is actually defined" -- a level with real displayed size but an invalid TOUCH
    # (e.g. the other side is a sentinel) has no `mid` to measure its distance from.
    finite_mid = np.isfinite(book.mid)
    bid_level_valid = frame.bid_valid[stamps.index][:, : book.n_levels] & finite_mid[:, None]
    ask_level_valid = frame.ask_valid[stamps.index][:, : book.n_levels] & finite_mid[:, None]
    slope_bid = _book_slope(book.mid, book.bid_px_l, book.bid_sz_l, bid_level_valid, 1.0)
    slope_ask = _book_slope(book.mid, book.ask_px_l, book.ask_sz_l, ask_level_valid, -1.0)

    # -- time of day (control, not alpha; see module docstring) -----------------------
    offset_ns = stamps.ts_ns.astype(np.int64) - SESSION_START_NS
    tod_bucket = np.floor_divide(offset_ns, TOD_BUCKET_NS).astype(np.float64)
    tod_frac = (stamps.ts_ns.astype(np.float64) - SESSION_START_NS) / float(
        SESSION_END_NS - SESSION_START_NS
    )

    out = {
        "st_spread_ticks": spread_ticks,
        "st_spread_ticks_int": spread_ticks_int,
        "st_depth_bid_touch": depth_bid_touch,
        "st_depth_ask_touch": depth_ask_touch,
        "st_depth_touch": depth_touch,
        "st_depth_total": depth_total,
        "st_slope_bid": slope_bid,
        "st_slope_ask": slope_ask,
        "st_tod_bucket": tod_bucket,
        "st_tod_frac": tod_frac,
    }

    # -- per-event contributions, built ONCE, aggregated per window like OFI ----------
    # (INTERFACE.md sec. 5 rule 6: a loop over the 3 FEATURE_WINDOWS is a loop over a
    # constant, not over stamps -- the O(n_stamps) work per window is a prefix diff.)
    mid_events = frame.mid
    if mid_events.shape[0] > 0:
        # Contribution[n] is the (mid(n-1) -> mid(n)) squared tick return, attached to
        # event n (the event at which that transition becomes public); contribution[0]
        # is NaN -- there is no event -1 to difference against.
        rv_contrib = np.concatenate(([np.nan], ticks(np.diff(mid_events)) ** 2))
    else:
        rv_contrib = np.zeros(0, dtype=np.float64)
    rv_sum_prefix, rv_count_prefix = prefix_sums(rv_contrib)

    event_type = frame.event_type
    cancel_indicator = np.isin(event_type, _CANCEL_TYPES).astype(np.float64)
    book_changing_indicator = np.isin(event_type, BOOK_CHANGING_TYPES).astype(np.float64)
    cancel_sum_prefix, _ = prefix_sums(cancel_indicator)
    book_changing_sum_prefix, _ = prefix_sums(book_changing_indicator)

    order_size = np.where(event_type == _NEW_LIMIT_TYPE, frame.size.astype(np.float64), np.nan)
    size_sum_prefix, size_count_prefix = prefix_sums(order_size)

    for w in FEATURE_WINDOWS:
        window_ns = WINDOWS_NS[w]
        start, end_exclusive = trailing_bounds(frame.ts_ns, stamps, window_ns)

        rv_sum = window_sum(rv_sum_prefix, start, end_exclusive)
        rv_count = window_count(rv_count_prefix, start, end_exclusive)
        rv = np.full(n, np.nan, dtype=np.float64)
        rv_ok = rv_count > 0
        rv[rv_ok] = np.sqrt(np.clip(rv_sum[rv_ok], 0.0, None))
        out[f"st_rv_{w}"] = rv

        n_in_window = (end_exclusive - start).astype(np.float64)
        out[f"st_event_rate_{w}"] = n_in_window / (window_ns / NS_PER_SEC)

        cancel_count = window_sum(cancel_sum_prefix, start, end_exclusive)
        book_changing_count = window_sum(book_changing_sum_prefix, start, end_exclusive)
        cancel_rate = np.full(n, np.nan, dtype=np.float64)
        np.divide(cancel_count, book_changing_count, out=cancel_rate, where=book_changing_count > 0)
        out[f"st_cancel_rate_{w}"] = cancel_rate

        size_sum = window_sum(size_sum_prefix, start, end_exclusive)
        size_count = window_count(size_count_prefix, start, end_exclusive)
        avg_order_size = np.full(n, np.nan, dtype=np.float64)
        np.divide(size_sum, size_count, out=avg_order_size, where=size_count > 0)
        out[f"st_avg_order_size_{w}"] = avg_order_size

    return out
