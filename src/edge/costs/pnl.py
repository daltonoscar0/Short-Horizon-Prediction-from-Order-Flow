"""Latency, fills, and transaction-cost-aware PnL (EVAL_PROTOCOL.md sec. 3).

Deliberately boring, per the protocol: one unit, no inventory accumulation, flat at the
horizon. Every optimistic bug in a backtest lives in the machinery this module refuses to
have.

THE DECISION SEQUENCE, and where each one is allowed to look:

    t          signal computed from information <= t          (the feature)
    t + delta  the earliest moment it can act                 (latency)
    t + delta + h   the position is closed                    (flat at horizon)

The signal is formed at `t` but the ENTRY PRICE is read at `t + delta`. That gap is the whole
point of the latency ladder: a signal whose edge has evaporated by the time it can be acted on
is latency arbitrage, not alpha. The phase-1 adversary measured that OFI's content decays
inside 100 ms, so the 100 ms rung is expected to be brutal here -- and if it is not, that is
itself suspicious.

AGGRESSIVE FILL, the only fill model used for the headline. Buying crosses to the ask and
selling crosses to the bid, so the round trip pays the full spread plus two take fees. Fill is
at the displayed price up to the displayed size; beyond it the book is walked, which
`walk_the_book` implements. This is pessimistic relative to a passive strategy and that is
deliberate -- a passive fill model needs queue-position assumptions, and every one of them is
a place to be accidentally generous.

PASSIVE FILL is provided but is NOT the headline: the honest simple rule is "fill only if the
level trades through your price", which ignores queue position and is therefore pessimistic in
a different direction. Reported as a sensitivity, never as the main number.
"""
import numpy as np

from edge.costs import (
    MAKE_REBATE_PER_SHARE,
    TAKE_FEE_PER_SHARE,
    TRADE_SIZE_SHARES,
)

TICK_SIZE_RAW = 100
PRICE_SCALE = 10_000  # raw LOBSTER units per dollar


def _to_dollars(raw):
    return np.asarray(raw, dtype=np.float64) / PRICE_SCALE


def apply_latency(ts_ns, delta_ns, target_ts_ns=None):
    """Index of the last row with ts <= t + delta, for each t. The action time.

    Uses the same `<= t` searchsorted convention as the feature clocks, so "the state you can
    act against" means exactly what "the state at t" means everywhere else in this project.
    Returns -1 where `t + delta` runs past the end of the session; callers must drop those
    rather than clamping, because clamping invents a fill at a price that never existed.
    """
    ts = np.asarray(ts_ns, dtype=np.int64)
    target = ts if target_ts_ns is None else np.asarray(target_ts_ns, dtype=np.int64)
    idx = np.searchsorted(ts, target + int(delta_ns), side="right") - 1
    beyond = target + int(delta_ns) > ts[-1] if ts.size else np.ones(target.shape, bool)
    idx = np.where(beyond, -1, idx)
    return idx.astype(np.int64)


def walk_the_book(prices_l, sizes_l, shares, side):
    """Volume-weighted fill price for `shares`, walking levels until filled.

    `side` is +1 to buy (walk the ask upward) or -1 to sell (walk the bid downward). Returns
    (vwap_raw, filled_shares). If the displayed book cannot fill the order, the shortfall is
    reported rather than filled at the last price -- pretending the book is deeper than it is
    is the single most common optimistic backtest bug.
    """
    px = np.asarray(prices_l, dtype=np.float64)
    sz = np.asarray(sizes_l, dtype=np.float64)
    remaining = float(shares)
    notional = 0.0
    filled = 0.0
    for level in range(px.shape[0]):
        if remaining <= 0:
            break
        available = sz[level]
        if not np.isfinite(available) or available <= 0 or not np.isfinite(px[level]):
            continue
        take = min(remaining, available)
        notional += take * px[level]
        filled += take
        remaining -= take
    if filled <= 0:
        return np.nan, 0.0
    return notional / filled, filled


def round_trip_pnl(
    entry_bid_raw, entry_ask_raw, exit_bid_raw, exit_ask_raw, direction,
    shares=TRADE_SIZE_SHARES, take_fee=TAKE_FEE_PER_SHARE, extra_cost_per_share=0.0,
):
    """PnL in DOLLARS for one aggressive round trip, flat at the horizon.

    direction +1: buy at the ask, sell back at the bid.  -1: sell at the bid, buy back at the
    ask. Either way the round trip pays the full spread plus two take fees, which is what
    makes short-horizon signals hard and is exactly the number that should not be softened.

    `extra_cost_per_share` is the knob the break-even table sweeps.
    """
    d = np.asarray(direction, dtype=np.float64)
    eb, ea = _to_dollars(entry_bid_raw), _to_dollars(entry_ask_raw)
    xb, xa = _to_dollars(exit_bid_raw), _to_dollars(exit_ask_raw)

    entry_px = np.where(d > 0, ea, eb)   # cross the spread on the way in
    exit_px = np.where(d > 0, xb, xa)    # and again on the way out
    gross = d * (exit_px - entry_px) * shares
    fees = 2.0 * (take_fee + extra_cost_per_share) * shares
    pnl = np.where(d == 0, 0.0, gross - fees)
    return pnl


def passive_round_trip_pnl(
    entry_mid_raw, exit_bid_raw, exit_ask_raw, direction, traded_through,
    shares=TRADE_SIZE_SHARES, make_rebate=MAKE_REBATE_PER_SHARE,
    take_fee=TAKE_FEE_PER_SHARE, extra_cost_per_share=0.0,
):
    """Sensitivity only. Post at the touch, fill ONLY where `traded_through` is True.

    Ignores queue position entirely, which is pessimistic on fill probability and optimistic
    on adverse selection -- both stated, neither modelled. Exit still crosses the spread, so
    the round trip earns one rebate and pays one take fee.
    """
    d = np.asarray(direction, dtype=np.float64)
    filled = np.asarray(traded_through, dtype=bool)
    entry_px = _to_dollars(entry_mid_raw)
    xb, xa = _to_dollars(exit_bid_raw), _to_dollars(exit_ask_raw)
    exit_px = np.where(d > 0, xb, xa)
    gross = d * (exit_px - entry_px) * shares
    net = gross + (make_rebate - take_fee - 2.0 * extra_cost_per_share) * shares
    return np.where(filled & (d != 0), net, 0.0)


def break_even_cost_per_share(pnl_before_extra_cost, n_trades, shares=TRADE_SIZE_SHARES):
    """The per-share cost at which total PnL crosses zero, INCREMENTAL to costs already paid.

    NOT the headline number, and the 2026-08-02 red team is the reason this docstring says so.
    Fed the NET series (spread crossed, take fees paid) this returns the cost on top of those,
    which for an always-on strategy is dominated by the spread and is therefore very nearly
    model-invariant: a constant always-long predictor scores -0.8072 ticks on INTC calendar 1s
    against a whole-ladder range of [-0.8025, -0.7702], i.e. inside it. A column that a
    constant cannot be distinguished from is measuring the trading rule, not the signal. Use
    `break_even_total_ticks` on the GROSS series for the question the project actually asks.

    Each round trip pays the extra cost twice (in and out), so total extra cost is
    `2 * c * shares * n_trades` and the break-even c solves

        sum(pnl) - 2 * c * shares * n_trades = 0

    Returns dollars per share. Negative means the strategy is already underwater before any
    additional cost, which is a result and is reported as one rather than clipped to zero.
    """
    total = float(np.nansum(pnl_before_extra_cost))
    if n_trades <= 0:
        return np.nan
    return total / (2.0 * shares * n_trades)


def break_even_cost_ticks(pnl_before_extra_cost, n_trades, shares=TRADE_SIZE_SHARES):
    """Break-even cost expressed in TICKS per share ($0.01), the unit the ladder speaks."""
    dollars = break_even_cost_per_share(pnl_before_extra_cost, n_trades, shares)
    return dollars / (TICK_SIZE_RAW / PRICE_SCALE)


def break_even_cost_bps(pnl_before_extra_cost, n_trades, avg_price_raw,
                        shares=TRADE_SIZE_SHARES):
    """Break-even cost in basis points of notional -- the cross-asset comparable unit."""
    dollars = break_even_cost_per_share(pnl_before_extra_cost, n_trades, shares)
    px = float(np.nanmean(_to_dollars(avg_price_raw)))
    if not np.isfinite(px) or px <= 0:
        return np.nan
    return 1e4 * dollars / px


def break_even_total_ticks(gross_pnl, n_trades, shares=TRADE_SIZE_SHARES):
    """TOTAL per-share, per-side cost the signal can bear before it stops paying. THE HEADLINE.

    Fed the GROSS (mid-to-mid, no spread, no fees) series, so the returned number is inclusive
    of every cost rather than incremental to an assumed one:

        sum(gross) - 2 * c * shares * n_trades = 0

    A reader who disagrees with the assumed $0.0030 take fee reads their own answer off this
    column directly -- which is what the break-even table was always supposed to allow, and
    what the incremental version could not, since the fee was already inside its input.

    Compare against the ACTUAL cost of trading: half-spread per side plus the take fee. On
    INTC (1.234-tick mean spread) that is ~0.62 + 0.30 = 0.92 ticks per side; this column
    reports 0.0106 for L1, so the signal pays about 1% of what it costs to trade it.
    """
    total = float(np.nansum(gross_pnl))
    if n_trades <= 0:
        return np.nan
    dollars = total / (2.0 * shares * n_trades)
    return dollars / (TICK_SIZE_RAW / PRICE_SCALE)


def turnover_and_holding(direction, horizon_ns):
    """Context for the cost number: trades taken and the (fixed) holding time in seconds."""
    d = np.asarray(direction, dtype=np.float64)
    n_trades = int(np.count_nonzero(d != 0))
    return {
        "n_trades": n_trades,
        "trade_rate": float(n_trades / d.size) if d.size else np.nan,
        "holding_seconds": float(horizon_ns) / 1e9,
    }
