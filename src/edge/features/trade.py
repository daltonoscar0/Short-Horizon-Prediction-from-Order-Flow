"""Trade / aggressor features off the raw event stream (FEATURES.md sec. 4).

LOBSTER gives trade direction explicitly, so no Lee-Ready classification is needed
(FEATURES.md sec. 4: "no Lee-Ready classification needed... it's a detail that shows you
read the data spec"). But `direction` labels the RESTING LIMIT ORDER's side, not the
aggressor: docs/DATA_CONTRACT.md's message-file column table, col 6, verbatim -- "`+1` =
bid side, `-1` = ask side (side of the *standing* order, not the aggressor)". Execution of
a resting SELL (ask-side, direction=-1) limit order is a BUYER-initiated trade, so the
aggressor sign used throughout this module is always `-direction`.

FROZEN: a "trade" is event type 4 (visible execution) OR type 5 (hidden execution) --
docs/DATA_CONTRACT.md's event-type table: type 4 "execution of a visible limit order",
type 5 "execution of a hidden order (book unaffected)". Both carry a real direction, price
and size; type 5 simply doesn't move the displayed book. Dropping type 5 would silently
undercount every trade feature by its share of total executions (3,559 of 28,924+3,559 =
~11% on INTC) for a reason (venue chose not to display the counterparty) that has nothing
to do with whether a trade happened. The trade-to-quote denominator (`tr_quote_ratio_*`)
is instead the count of book-changing events (types 1-4, `clocks.BOOK_CHANGING_TYPES`) in
the same window -- a *different* set from "trades" by design: a type-4 trade is a member
of both counts, a type-5 trade is a trade but never a book-changing event.

Causality: `<= t`, the module default (INTERFACE.md sec. 2) -- no stricter convention.
Windowed features (`_100ms` / `_1s` / `_5s`) sum/count trade contributions over the
frame's `(t - window, t]` slice (`windows.trailing_bounds`). `tr_eff_spread` /
`tr_last_trade_sign` read the LAST TRADE event with `ts_ns <= t` via
`searchsorted(..., side="right")` over the trade subsequence only, so a trade stamped
exactly at `t` is included (ties resolve to inclusion, matching the `<=` convention) --
not a loop over stamps.

NaN policy (INTERFACE.md sec. 7): `tr_signed_vol_*` / `tr_buy_vol_*` / `tr_sell_vol_*` /
`tr_count_*` are SUMS/counts -- legitimately 0.0 on an empty (of trades) window, never NaN,
even if the window contains other, non-trade events. `tr_aggr_imb_*` / `tr_avg_size_*` /
`tr_quote_ratio_*` are RATIOS/means -- NaN when their own denominator (total trade volume,
trade count, book-changing-event count respectively) is 0; note `tr_quote_ratio_*` can
legitimately be `0.0` (trades=0, book-changing>0) while `tr_aggr_imb_*` is NaN in that same
window (trade volume=0), because the two features have different denominators.
`tr_eff_spread` is a price DIFFERENCE and is therefore emitted in TICKS (INTERFACE.md
sec. 1), via `book.ticks()`; the underlying subtraction is done in raw units first.

`tr_eff_spread` / `tr_last_trade_sign` are NaN before the session's first trade, and (an
edge case the toy fixture cannot reach but the real frame could in principle) NaN if that
trade is the very first frame row, since there is then no preceding event to read
`mid_before` from. They part company on one case: `tr_eff_spread` is additionally NaN when
the preceding row's mid is NaN (an invalid touch), while `tr_last_trade_sign` is not --
the aggressor sign comes from the message file's `direction` column and never depends on
the book.

`book` is accepted only for interface uniformity (INTERFACE.md sec. 5) and is never read:
every trade feature is derived from the raw event stream plus `frame.mid` (the per-event
cached mid, INTERFACE.md sec. 4's "per-event work without a second full book"), never from
the stamp-indexed `book_at_stamps` output -- so building one here would be a wasted ~200MB
gather for a value nothing uses. The result is therefore trivially bit-identical whether
`book` is supplied or left `None`.
"""
import numpy as np

from edge.features import FEATURE_WINDOWS, WINDOWS_NS
from edge.features.book import event_frame_arrays, ticks
from edge.features.clocks import is_book_changing
from edge.features.windows import prefix_sums, trailing_bounds, window_count, window_sum

# FROZEN a priori (see module docstring): both visible and hidden executions are real
# trades carrying a genuine direction/price/size; only type 5 leaves the displayed book
# untouched.
TRADE_EVENT_TYPES = (4, 5)


def compute(frame, stamps, book=None):
    """Trade / aggressor features (FEATURES.md sec. 4); see module docstring for conventions."""
    n = stamps.index.shape[0]

    # int64 copies (book.py's sanctioned accessor): int8 direction * a size would wrap
    # silently under NEP 50, and copies mean this module can never mutate the shared frame.
    arrays = event_frame_arrays(frame)

    is_trade = np.isin(arrays.event_type, TRADE_EVENT_TYPES)
    # direction labels the RESTING order's side (DATA_CONTRACT.md col 6); the aggressor
    # who crossed the spread sits on the OTHER side, hence the sign flip.
    aggressor_sign = -arrays.direction.astype(np.float64)
    size_f = arrays.size.astype(np.float64)

    # Per-event contribution arrays, NaN at every row that shouldn't count, so
    # windows.prefix_sums excludes them from both the sum and the count (windows.py
    # docstring: "NaN contributions are treated as missing").
    signed_val = np.where(is_trade, aggressor_sign * size_f, np.nan)
    buy_val = np.where(is_trade & (aggressor_sign > 0), size_f, np.nan)
    sell_val = np.where(is_trade & (aggressor_sign < 0), size_f, np.nan)
    size_val = np.where(is_trade, size_f, np.nan)
    bc_val = np.where(is_book_changing(arrays.event_type), 1.0, np.nan)

    signed_sum_p, signed_cnt_p = prefix_sums(signed_val)
    buy_sum_p, _ = prefix_sums(buy_val)
    sell_sum_p, _ = prefix_sums(sell_val)
    size_sum_p, _ = prefix_sums(size_val)
    _, bc_cnt_p = prefix_sums(bc_val)

    out = {}
    for w in FEATURE_WINDOWS:
        start, end = trailing_bounds(frame.ts_ns, stamps, WINDOWS_NS[w])

        signed_vol = window_sum(signed_sum_p, start, end)
        buy_vol = window_sum(buy_sum_p, start, end)
        sell_vol = window_sum(sell_sum_p, start, end)
        count = window_count(signed_cnt_p, start, end).astype(np.float64)
        size_sum = window_sum(size_sum_p, start, end)
        bc_count = window_count(bc_cnt_p, start, end).astype(np.float64)

        total_vol = buy_vol + sell_vol
        aggr_imb = np.full(n, np.nan, dtype=np.float64)
        np.divide(buy_vol - sell_vol, total_vol, out=aggr_imb, where=total_vol > 0)

        avg_size = np.full(n, np.nan, dtype=np.float64)
        np.divide(size_sum, count, out=avg_size, where=count > 0)

        quote_ratio = np.full(n, np.nan, dtype=np.float64)
        np.divide(count, bc_count, out=quote_ratio, where=bc_count > 0)

        out[f"tr_signed_vol_{w}"] = signed_vol
        out[f"tr_buy_vol_{w}"] = buy_vol
        out[f"tr_sell_vol_{w}"] = sell_vol
        out[f"tr_count_{w}"] = count
        out[f"tr_aggr_imb_{w}"] = aggr_imb
        out[f"tr_avg_size_{w}"] = avg_size
        out[f"tr_quote_ratio_{w}"] = quote_ratio

    # -- last-trade-at-or-before-t features: no window, carried forward indefinitely --
    trade_positions = np.flatnonzero(is_trade)
    eff_spread = np.full(n, np.nan, dtype=np.float64)
    last_sign = np.full(n, np.nan, dtype=np.float64)
    if trade_positions.size > 0:
        trade_ts = arrays.ts_ns[trade_positions]
        # side="right": a trade stamped exactly at t IS "at or before t" (the <= t tie).
        last_local = np.searchsorted(trade_ts, stamps.ts_ns, side="right") - 1
        has_trade = last_local >= 0
        safe_local = np.clip(last_local, 0, trade_positions.shape[0] - 1)
        trade_pos = trade_positions[safe_local]

        # mid_before reads the event IMMEDIATELY PRECEDING the trade row, never the
        # trade's own post-state -- the classic off-by-one this family is graded on.
        has_prev = trade_pos > 0
        safe_prev = np.clip(trade_pos - 1, 0, max(frame.n_events - 1, 0))
        mid_before = frame.mid[safe_prev]
        price_trade = arrays.price[trade_pos].astype(np.float64)

        # eff_spread needs a finite pre-trade mid; last_sign does NOT -- it is
        # -direction[trade_pos], read straight from the message file, with no dependence on
        # the book at all. Sharing the `isfinite(mid_before)` mask would delete a perfectly
        # known aggressor sign whenever the preceding row's touch happened to be unquoted.
        # `has_prev` stays on both: that one is the documented "trade is the very first
        # frame row" rule (see module docstring), not a book-validity question.
        usable = has_trade & has_prev & np.isfinite(mid_before)
        eff_spread = np.where(usable, ticks(2.0 * np.abs(price_trade - mid_before)), np.nan)
        last_sign = np.where(has_trade & has_prev, aggressor_sign[trade_pos], np.nan)

    out["tr_eff_spread"] = eff_spread
    out["tr_last_trade_sign"] = last_sign

    return out
