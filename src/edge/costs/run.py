"""Phase-3 runner: cost-aware PnL and the break-even cost table (EVAL_PROTOCOL.md sec. 3-4).

    python3 -m edge.costs.run --features data/features --out reports/costs.md --seed 0

Scores the ladder under PROTOCOL C ONLY. That is deliberate: a PnL number under a leaky
protocol is not a weaker result, it is a misleading one, and the point of Phase 2 was to
establish which protocol to believe. The A/B numbers stay in the ladder table where the gap
is the subject; they do not get a PnL column.

For each rung, at each latency delta, the loop is:

    signal at t   ->   act at t + delta   ->   close at t + delta + h

Entry and exit prices are read from the BOOK at those two moments, so the round trip pays the
spread twice plus two take fees. The headline output is the break-even cost per trade: the
cost at which PnL crosses zero, in ticks and in bps.
"""
import argparse
import os

import numpy as np
import pyarrow.parquet as pq

from edge.costs import LATENCIES_NS, TRADE_SIZE_SHARES
from edge.costs.pnl import (
    PRICE_SCALE,
    TICK_SIZE_RAW,
    apply_latency,
    break_even_cost_bps,
    break_even_cost_ticks,
    break_even_total_ticks,
    round_trip_pnl,
    turnover_and_holding,
)
from edge.costs.stats import block_bootstrap_ci, deflated_sharpe, diebold_mariano
from edge.harness import N_FOLDS
from edge.harness.models import Rung, build_ladder
from edge.harness.run import HORIZON_NS, common_usable_mask, scale_holdout_mask
from edge.harness.splits import make_folds

TICKERS = ("INTC", "AMZN")
CLOCKS = ("calendar", "event")
DATE = "2012-06-21"

# Book columns needed to price a fill. The features table carries only derived columns, so
# the touch prices come from the event frame, joined on ts_ns.
FRAME_COLS = ("ts_ns", "bid_px_1", "ask_px_1")


def _load_features(features_dir, ticker, clock, columns):
    path = os.path.join(features_dir, f"{ticker}_{DATE}_{clock}.parquet")
    t = pq.read_table(path, columns=list(columns))
    return {n: t.column(n).to_numpy(zero_copy_only=False) for n in columns}


def _load_touch(frames_dir, ticker):
    path = os.path.join(frames_dir, f"{ticker}_{DATE}.parquet")
    t = pq.read_table(path, columns=list(FRAME_COLS))
    return {n: t.column(n).to_numpy(zero_copy_only=False) for n in FRAME_COLS}


def _non_overlapping(stamp_ts, horizon_ns, delta_ns, next_free=None):
    """Positions at least `delta + horizon` apart, so at most one is ever open.

    EVAL_PROTOCOL.md sec. 3 mandates "single unit, no inventory accumulation, flat at
    horizon". Taking a decision at every stamp would violate that outright: on the 100ms
    calendar grid with a 1s horizon, ten positions would be open simultaneously and the PnL
    would be a leveraged overlapping sum rather than the boring single-unit strategy the
    protocol asks for. Greedily keeping the earliest eligible stamp is the deterministic
    choice; it also makes the trade count an honest measure of turnover.

    `next_free` CARRIES ACROSS FOLDS. Resetting it per fold let a position opened at the end
    of one test block overlap one opened at the start of the next: the 2026-08-02 red team
    measured 4 such gaps on INTC calendar 1s (800/900/900/1000 ms against a required 1001 ms),
    one per fold boundary. Immaterial to the number, fatal to the docstring's invariant.
    Returns the mask and the updated watermark so the caller can thread it through folds.
    """
    keep = np.zeros(stamp_ts.shape[0], dtype=bool)
    span = int(delta_ns) + int(horizon_ns)
    if next_free is None:
        next_free = np.iinfo(np.int64).min
    for i, t in enumerate(stamp_ts):
        if t >= next_free:
            keep[i] = True
            next_free = t + span
    return keep, next_free


class _ConstantDirection(Rung):
    """Control: trade the same side every time, ignoring every feature.

    The 2026-08-02 red team's central finding is that the incremental break-even column cannot
    distinguish the ladder from this. It ships as a ROW so the reader sees that directly
    instead of having to recompute it.
    """

    features = ()

    def __init__(self, sign, name):
        super().__init__()
        self.sign = float(sign)
        self.name = name

    def _raw(self, X):
        return np.full(X.shape[0], self.sign, dtype=np.float64)

    def fit(self, X, y, scale_holdout=None, model_mask=None):
        # Both mask arguments are accepted and IGNORED, deliberately. A fitted rescale could
        # flip the sign, and a control that moves with the calibration is not a control.
        self.scale_ = 1.0
        return self


class _Oracle(Rung):
    """UPPER BOUND, not a rung: the realised label itself, i.e. perfect foresight.

    Uses the future by construction. It is here to bound what any predictor could earn under
    this fill model -- the red team measured that perfect foresight of the 1s mid sign still
    loses money once the spread is crossed, which is the single most informative number about
    whether this ladder could ever have paid.
    """

    name = "ORACLE perfect foresight (bound)"

    def __init__(self, label_col):
        super().__init__()
        self.features = (label_col,)

    def _raw(self, X):
        return np.sign(X[:, 0])

    def fit(self, X, y, scale_holdout=None, model_mask=None):
        self.scale_ = 1.0  # accepted and ignored: an upper bound must not be rescaled either
        return self


def build_controls(label_col):
    """The three control predictors, in report order. Not part of the L0-L4 ladder."""
    return [
        _ConstantDirection(+1.0, "CTRL always long"),
        _ConstantDirection(-1.0, "CTRL always short"),
        _Oracle(label_col),
    ]


def simulate(cols, touch, y, ts, horizon_ns, delta_ns, rung, folds, common_mask,
             deadband_ticks=0.0):
    """One rung, one latency. Returns per-trade PnL and the trade direction vector.

    Direction is the SIGN of the prediction, not a deadband on its magnitude. The first cut
    of this runner reused the labels' 0.5-tick deadband as a trading threshold, and L0/L1/L2
    then took ZERO trades across the whole session -- a scaled sign predictor never emits a
    magnitude above half a tick at a 1s horizon, where 96% of realised moves are zero. That
    left the break-even table empty for exactly the rungs it most needs to price. A deadband
    is a position-sizing choice, not part of the signal; `deadband_ticks` stays available as
    a sensitivity but defaults to 0.
    """
    feats = list(rung.features)
    X = (np.column_stack([cols[f] for f in feats]) if feats
         else np.zeros((y.shape[0], 0), dtype=np.float64))

    frame_ts = touch["ts_ns"].astype(np.int64)
    bid, ask = touch["bid_px_1"].astype(np.float64), touch["ask_px_1"].astype(np.float64)

    entry_rows = apply_latency(frame_ts, delta_ns, target_ts_ns=ts)
    exit_rows = apply_latency(frame_ts, delta_ns + int(horizon_ns), target_ts_ns=ts)

    pnl_all = np.full(y.shape[0], np.nan, dtype=np.float64)
    dir_all = np.zeros(y.shape[0], dtype=np.float64)
    gross_all = np.full(y.shape[0], np.nan, dtype=np.float64)
    fold_id = np.full(y.shape[0], -1, dtype=np.int64)
    next_free = None  # carried across folds: a position may not straddle a block boundary

    for f_i, fold in enumerate(folds):
        tr = fold.train_idx[common_mask[fold.train_idx]]
        te = fold.test_idx[common_mask[fold.test_idx]]
        if tr.size < 50 or te.size < 20:
            continue
        # The SAME causal calibration the ladder uses (2026-08-02). This path kept the
        # in-sample rescale until 2026-08-03, when the third red team found it here. It is
        # very nearly a no-op for the economics -- with a 0-tick trading deadband the traded
        # direction is sign(scale) * sign(raw), so only the SIGN of the scale can change a
        # trade, and an in-sample scale is negative only if a rung is anti-correlated with the
        # label on its own training rows. Fixed anyway: "very nearly" is not a property to
        # leave two calibrations apart over, and a negative fitted scale is not hypothetical --
        # the causal scale goes negative in 27 of 400 protocol-C folds in the ladder.
        hold = scale_holdout_mask(tr.size, fold.protocol, fold.fold_id)
        if hold.all() or not hold.any():
            continue
        rung.fit(X[tr], y[tr], scale_holdout=hold)
        pred = rung.predict(X[te])

        direction = np.zeros(te.shape[0], dtype=np.float64)
        direction[pred > deadband_ticks] = 1.0
        direction[pred < -deadband_ticks] = -1.0

        # Enforce "single unit, no inventory accumulation, flat at horizon": keep only
        # decision points spaced at least delta + h apart, so at most one position is ever
        # open. Without this, a 1s horizon on the 100ms grid runs ten overlapping positions
        # and the PnL is a leveraged sum, not the boring strategy the protocol specifies.
        spaced, next_free = _non_overlapping(ts[te], horizon_ns, delta_ns, next_free)
        direction = np.where(spaced, direction, 0.0)

        er, xr = entry_rows[te], exit_rows[te]
        valid = (er >= 0) & (xr >= 0) & (direction != 0)
        if not np.any(valid):
            continue
        idx = te[valid]
        pnl = round_trip_pnl(
            bid[er[valid]], ask[er[valid]], bid[xr[valid]], ask[xr[valid]],
            direction=direction[valid], shares=TRADE_SIZE_SHARES,
        )
        pnl_all[idx] = pnl
        dir_all[idx] = direction[valid]
        fold_id[idx] = f_i

        # Mid-to-mid PnL: no spread crossing, no fees. This isolates the SIGNAL's economic
        # value from the cost of acting on it. Without it the net table is uninformative
        # about the signal, because an always-on strategy loses almost exactly the spread
        # whatever it predicts, and every rung's break-even collapses to the same number.
        mid_entry = 0.5 * (bid[er[valid]] + ask[er[valid]])
        mid_exit = 0.5 * (bid[xr[valid]] + ask[xr[valid]])
        gross_all[idx] = (
            direction[valid] * (mid_exit - mid_entry) / PRICE_SCALE * TRADE_SIZE_SHARES
        )

    return pnl_all, dir_all, gross_all, fold_id


def _dm_adjacent(ticker, clock, h, gross_by_rung, ladder_names, block_len):
    """Diebold-Mariano between ADJACENT ladder rungs on per-trade gross PnL.

    EVAL_PROTOCOL.md sec. 2 asks for this and it was never wired up: `diebold_mariano` was
    imported into this module and never called. The loss is `-gross ticks` per trade, so a
    NEGATIVE statistic means the upper rung has the lower loss, i.e. earns more.

    Rungs trade at identical stamps (the decision set comes from `_non_overlapping` on the
    stamp vector alone, not from the prediction), so the two series are paired on the same
    trades wherever both traded -- which is what DM requires and what makes the test legible.
    """
    out = []
    for lower, upper in zip(ladder_names, ladder_names[1:]):
        if lower not in gross_by_rung or upper not in gross_by_rung:
            continue
        g_lo, idx_lo = gross_by_rung[lower]
        g_hi, idx_hi = gross_by_rung[upper]
        common, i_lo, i_hi = np.intersect1d(idx_lo, idx_hi, return_indices=True)
        if common.size < 8:
            continue
        # loss = -PnL, so "lower loss" reads as "more money".
        dm = diebold_mariano(-g_hi[i_hi], -g_lo[i_lo], lag=block_len, loss="linear")
        out.append({
            "ticker": ticker, "clock": clock, "h": h, "upper": upper, "lower": lower,
            "n": dm["n"], "mean_diff_ticks": -dm.get("mean_loss_diff", np.nan),
            "dm_stat": dm["dm_stat"], "p_value": dm["p_value"],
        })
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description="Cost-aware PnL and break-even cost table.")
    p.add_argument("--features", default="data/features")
    p.add_argument("--frames", default="data/frames")
    p.add_argument("--out", default="reports/costs.md")
    p.add_argument("--tickers", default=",".join(TICKERS))
    p.add_argument("--clocks", default="calendar")
    p.add_argument("--horizons", default="1s,5s")
    p.add_argument("--trials", type=int, default=1,
                   help="configurations tried, for the deflated Sharpe (CLAUDE.md rule 4)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    tickers = tuple(t for t in args.tickers.split(",") if t)
    clocks = tuple(c for c in args.clocks.split(",") if c)
    horizons = tuple(h for h in args.horizons.split(",") if h)

    needed = sorted({f for r in build_ladder(seed=args.seed) for f in r.features})
    rows = []
    dm_rows = []
    for ticker in tickers:
        touch = _load_touch(args.frames, ticker)
        for clock in clocks:
            labels = [f"label_y_mid_{h}" for h in horizons]
            cols = _load_features(args.features, ticker, clock, ["ts_ns"] + needed + labels)
            ts = cols["ts_ns"].astype(np.int64)
            for h in horizons:
                label_col = f"label_y_mid_{h}"
                y = cols[label_col]
                mask = common_usable_mask(cols, y, seed=args.seed)
                folds = make_folds(ts, horizon_ns=HORIZON_NS[h], protocol="C",
                                   n_folds=N_FOLDS, seed=args.seed)
                # Controls run on the SAME rows, folds, mask and fills as the ladder, so a
                # difference between a rung and a control cannot be a universe difference.
                predictors = list(build_ladder(seed=args.seed)) + build_controls(label_col)
                # Gross per-trade series, keyed by rung, for the DM tests between adjacent
                # rungs at the reference latency.
                gross_by_rung = {}
                for rung in predictors:
                    for delta in LATENCIES_NS:
                        pnl, direction, gross, fold_id = simulate(
                            cols, touch, y, ts, HORIZON_NS[h], delta, rung, folds, mask
                        )
                        traded = np.isfinite(pnl) & (direction != 0)
                        n_trades = int(np.count_nonzero(traded))
                        if n_trades == 0:
                            rows.append({
                                "ticker": ticker, "clock": clock, "h": h,
                                "rung": rung.name, "delta_ms": delta / 1e6,
                                "n_trades": 0, "total_pnl": 0.0, "be_ticks": np.nan,
                                "be_bps": np.nan, "dsr": np.nan, "ci_lo": np.nan,
                                "ci_hi": np.nan,
                            })
                            continue
                        vals = pnl[traded]
                        gross_vals = gross[traded]
                        gross_ticks = gross_vals / TRADE_SIZE_SHARES / (
                            TICK_SIZE_RAW / PRICE_SCALE
                        )
                        tk = turnover_and_holding(direction, HORIZON_NS[h])
                        block = max(10, int(HORIZON_NS[h] // 100_000_000))
                        # CI on the MEAN per trade -- which is what block_bootstrap_ci
                        # returns. The previous table printed this beside the TOTAL, off by
                        # a factor of n_trades in 120/120 rows (2026-08-02 red team).
                        ci = block_bootstrap_ci(vals, block_len=block, n_boot=500,
                                                seed=args.seed)
                        # The gross series is the one that discriminates between rungs, so
                        # it is the one that needs an interval.
                        ci_g = block_bootstrap_ci(gross_ticks, block_len=block, n_boot=500,
                                                  seed=args.seed)
                        # DSR on GROSS: net is a near-deterministic loss (Sharpe ~ -7.6), so
                        # its DSR is 0.000 at every trial count from 1 to 1e12 -- a tautology,
                        # not a finding. Deflation only bites where there is something to
                        # deflate.
                        d = deflated_sharpe(gross_vals, n_trials=args.trials)
                        d_net = deflated_sharpe(vals, n_trials=args.trials)
                        # bps denominator over TRADED rows only, not the whole session:
                        # segment 0 is never tested and does not belong in the average price.
                        entry_rows = apply_latency(
                            touch["ts_ns"].astype(np.int64), delta, target_ts_ns=ts
                        )[traded]
                        avg_px = float(np.nanmean(
                            touch["ask_px_1"].astype(np.float64)[entry_rows[entry_rows >= 0]]
                        ))
                        fids = fold_id[traded]
                        per_fold = [
                            float(np.nanmean(gross_ticks[fids == k]))
                            for k in range(len(folds)) if np.any(fids == k)
                        ]
                        rows.append({
                            "ticker": ticker, "clock": clock, "h": h, "rung": rung.name,
                            "delta_ms": delta / 1e6, "n_trades": n_trades,
                            "total_pnl": float(np.nansum(vals)),
                            "mean_pnl": float(np.nanmean(vals)),
                            "be_ticks": break_even_cost_ticks(vals, n_trades),
                            "be_bps": break_even_cost_bps(vals, n_trades, avg_px),
                            "be_total_ticks": break_even_total_ticks(gross_vals, n_trades),
                            "trade_rate": tk["trade_rate"],
                            "gross_total": float(np.nansum(gross_vals)),
                            "gross_per_trade": float(np.nanmean(gross_vals)),
                            "gross_ticks_per_trade": float(np.nanmean(gross_ticks)),
                            "gross_ci_lo": ci_g["lo"], "gross_ci_hi": ci_g["hi"],
                            "per_fold_gross": per_fold,
                            "worst_fold_gross": min(per_fold) if per_fold else np.nan,
                            "dsr": d["dsr"], "sharpe": d["sharpe"],
                            "dsr_net": d_net["dsr"], "sharpe_net": d_net["sharpe"],
                            "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                        })
                        if delta == LATENCIES_NS[0]:
                            gross_by_rung[rung.name] = (gross_ticks, np.flatnonzero(traded))
                        print(f"  {ticker}/{clock} h={h} {rung.name} d={delta/1e6:.0f}ms "
                              f"trades={n_trades} "
                              f"be_total={rows[-1]['be_total_ticks']:.4f} ticks")

                dm_rows.extend(
                    _dm_adjacent(ticker, clock, h, gross_by_rung,
                                 [r.name for r in build_ladder(seed=args.seed)],
                                 block_len=max(10, int(HORIZON_NS[h] // 100_000_000)))
                )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    _write(args, rows, dm_rows)
    print(f"wrote {args.out}")
    return 0


def _f(x, p=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def _write(args, rows, dm_rows):
    with open(args.out, "w") as fh:
        w = fh.write
        w("# Transaction costs, PnL and break-even cost (Phase 3)\n\n")
        # Echo the FULL effective argument set. The previous header printed only --seed and
        # --trials, and `--clocks` defaults to calendar, so the advertised command rebuilt
        # half the table (CLAUDE.md rule 7, 2026-08-02 red team).
        w(f"`python3 -m edge.costs.run --seed {args.seed} --trials {args.trials} "
          f"--tickers {args.tickers} --clocks {args.clocks} --horizons {args.horizons}`\n\n")
        w("**Protocol C only.** A PnL number computed under a leaky protocol is not a weaker ")
        w("result, it is a misleading one; Phase 2 established which protocol to believe.\n\n")
        w("## Stated assumptions\n\n")
        w("- **Fills are aggressive.** Buying crosses to the ask, selling to the bid, so a ")
        w("round trip pays the FULL spread plus two take fees. Pessimistic by design.\n")
        w("- **Fees**: $0.0030/share to take, $0.0020/share make rebate — the commonly-quoted ")
        w("2012 Nasdaq standard rates, entered as an ASSUMPTION and *not* verified against an ")
        w("archived price list. The break-even table is the answer to anyone who disagrees: ")
        w("it reports the cost at which PnL crosses zero, so a reader can substitute their own.\n")
        w("- **No market impact model.** Stated, not omitted. Ignoring impact biases PnL ")
        w("UPWARD; the break-even table partially absorbs this.\n")
        w(f"- **Position**: single unit of {TRADE_SIZE_SHARES} shares, no inventory, flat at ")
        w("the horizon. Direction is the SIGN of the prediction — **the deadband is 0**, not ")
        w("the labels' 0.5 ticks. A scaled sign predictor cannot emit a magnitude above half ")
        w("a tick at these horizons, so a 0.5-tick trading threshold took zero trades on ")
        w("every unfitted rung. That change was made after seeing an empty table, which makes ")
        w("it a choice informed by protocol-C output; it is logged in `DECISIONS.md` and the ")
        w("reasoning is a-priori, but it is disclosed here rather than presented as frozen.\n")
        w("- **Latency**: the signal formed at t acts at t+delta. A signal whose edge is gone ")
        w("by 100ms is latency arbitrage, not alpha — and the phase-1 adversary measured that ")
        w("OFI's content decays inside 100ms, so the 100ms rung is expected to be brutal.\n\n")
        if args.trials <= 1:
            w("> **The deflated Sharpe below is undeflated.** `--trials 1` was passed, which ")
            w("turns the DSR into an ordinary Sharpe probability wearing a rigorous name. ")
            w("The honest trial count comes from `RESULTS.md`'s trial log.\n\n")

        w("## What the headline column is, and what it is not\n\n")
        w("**`be total (ticks)` is the headline.** It is computed on the GROSS (mid-to-mid) ")
        w("series, so it is the total per-share, per-side cost the signal can bear before it ")
        w("stops paying — inclusive of every cost, which is what lets a reader who rejects ")
        w("the assumed fee schedule substitute their own number.\n\n")
        w("**`be incr (ticks)` is the old headline and it is nearly model-invariant.** It is ")
        w("computed on the NET series, so it reports the cost *on top of* a spread already ")
        w("crossed and a fee already paid, and for an always-on strategy that is dominated by ")
        w("the spread. The `CTRL always long` / `CTRL always short` rows are in the table for ")
        w("exactly this reason: the constant controls land INSIDE the ladder's range on that ")
        w("column, so it cannot answer PLAN.md's question. It is retained, demoted, because ")
        w("deleting a column the earlier draft led with would hide the correction.\n\n")
        w("**`ORACLE perfect foresight` is a bound, not a rung.** It trades the realised sign ")
        w("of the move, i.e. it sees the future. Read it as the ceiling this fill model ")
        w("permits: if the oracle is underwater, no predictor on this ladder could have paid.")
        w("\n\n")
        w("## Break-even cost table\n\n")
        w("Negative means underwater — reported, not clipped. `gross t/trade` is the column ")
        w("that discriminates between rungs; it carries a block-bootstrap CI and the worst of ")
        w("the five folds beside it, per CLAUDE.md rule 4.\n\n")
        w("| ticker | clock | h | rung | latency | trades | gross t/trade | gross CI lo | "
          "gross CI hi | worst fold | be total (ticks) | net PnL $ | mean PnL/trade $ | "
          "mean CI lo | mean CI hi | be incr (ticks) | be incr (bps) | DSR gross |\n")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            w(f"| {r['ticker']} | {r['clock']} | {r['h']} | {r['rung']} | "
              f"{r['delta_ms']:.0f}ms | {r['n_trades']} | "
              f"{_f(r.get('gross_ticks_per_trade'), 4)} | "
              f"{_f(r.get('gross_ci_lo'), 4)} | {_f(r.get('gross_ci_hi'), 4)} | "
              f"{_f(r.get('worst_fold_gross'), 4)} | "
              f"{_f(r.get('be_total_ticks'))} | {_f(r.get('total_pnl'), 2)} | "
              f"{_f(r.get('mean_pnl'), 4)} | "
              f"{_f(r.get('ci_lo'), 4)} | {_f(r.get('ci_hi'), 4)} | "
              f"{_f(r.get('be_ticks'))} | {_f(r.get('be_bps'), 2)} | "
              f"{_f(r.get('dsr'), 3)} |\n")
        w("\n")

        w("## Per-fold gross ticks/trade under protocol C\n\n")
        w("No pooled number travels alone (CLAUDE.md rule 4). Reference latency only.\n\n")
        w("| ticker | clock | h | rung | per-fold gross ticks/trade | worst |\n")
        w("|---|---|---|---|---|---|\n")
        ref_ms = LATENCIES_NS[0] / 1e6
        for r in rows:
            if abs(r["delta_ms"] - ref_ms) > 1e-9 or not r.get("per_fold_gross"):
                continue
            vals = ", ".join(f"{v:.4f}" for v in r["per_fold_gross"])
            w(f"| {r['ticker']} | {r['clock']} | {r['h']} | {r['rung']} | {vals} | "
              f"{_f(r.get('worst_fold_gross'), 4)} |\n")
        w("\n")

        w("## Diebold-Mariano between adjacent rungs (gross PnL per trade)\n\n")
        w("Loss is `-gross ticks`, so a NEGATIVE statistic means the upper rung earns more. ")
        w("Paired on the trades both rungs took; HAC lag = the bootstrap block length. ")
        w("Adjacent rungs are largely NESTED (L3's feature set contains L2's) and standard DM ")
        w("is undersized there, so read these as indicative, not decisive — ")
        w("`EVAL_PROTOCOL.md` sec. 2 asks for the Giacomini-White variant, which is not ")
        w("implemented.\n\n")
        w("| ticker | clock | h | upper | lower | n | mean diff (ticks) | DM stat | p |\n")
        w("|---|---|---|---|---|---|---|---|---|\n")
        for d in dm_rows:
            w(f"| {d['ticker']} | {d['clock']} | {d['h']} | {d['upper']} | {d['lower']} | "
              f"{d['n']} | {_f(d['mean_diff_ticks'], 4)} | {_f(d['dm_stat'], 3)} | "
              f"{_f(d['p_value'], 4)} |\n")
        w("\n")

        w("## Reading notes\n\n")
        w("- **The confidence intervals are on the MEAN per trade**, which is what a block ")
        w("bootstrap of a mean returns. An earlier draft printed them beside the TOTAL PnL, ")
        w("where they were off by a factor of `n_trades` in every row.\n")
        w("- Both intervals are circular block bootstraps with block length >= the label ")
        w("horizon; resampling individual trades would ignore the serial dependence that ")
        w("makes the naive interval far too narrow.\n")
        w("- **The DSR is on the GROSS series.** On the net series every rung has a Sharpe ")
        w("near -7.6 (it loses the spread almost deterministically), and the DSR is 0.000 at ")
        w("every trial count from 1 to 1e12 — a tautology rather than a finding. Deflation ")
        w("only means something where there is something to deflate.\n")
        w("- **Trade counts differ across rungs only where a prediction is exactly zero**, not ")
        w("because of a deadband (there isn't one). The decision set comes from the stamp ")
        w("vector, so every rung trades at identical times; `qi_touch == 0` on 10.5% of AMZN ")
        w("calendar stamps (bid size exactly equal to ask size) is the whole of the gap.\n")
        w("- **Net PnL is not comparable across latency rows on the event clock**, where the ")
        w("decision set itself shifts with latency. The per-trade columns are the comparable ")
        w("ones.\n")


if __name__ == "__main__":
    raise SystemExit(main())
