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
    apply_latency,
    break_even_cost_bps,
    break_even_cost_ticks,
    round_trip_pnl,
    turnover_and_holding,
)
from edge.costs.stats import block_bootstrap_ci, deflated_sharpe, diebold_mariano
from edge.harness import N_FOLDS
from edge.harness.models import build_ladder
from edge.harness.run import HORIZON_NS, common_usable_mask
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


def simulate(cols, touch, y, ts, horizon_ns, delta_ns, rung, folds, common_mask,
             deadband_ticks=0.5):
    """One rung, one latency. Returns per-trade PnL and the trade direction vector."""
    feats = list(rung.features)
    X = (np.column_stack([cols[f] for f in feats]) if feats
         else np.zeros((y.shape[0], 0), dtype=np.float64))

    frame_ts = touch["ts_ns"].astype(np.int64)
    bid, ask = touch["bid_px_1"].astype(np.float64), touch["ask_px_1"].astype(np.float64)

    entry_rows = apply_latency(frame_ts, delta_ns, target_ts_ns=ts)
    exit_rows = apply_latency(frame_ts, delta_ns + int(horizon_ns), target_ts_ns=ts)

    pnl_all = np.full(y.shape[0], np.nan, dtype=np.float64)
    dir_all = np.zeros(y.shape[0], dtype=np.float64)

    for fold in folds:
        tr = fold.train_idx[common_mask[fold.train_idx]]
        te = fold.test_idx[common_mask[fold.test_idx]]
        if tr.size < 50 or te.size < 20:
            continue
        rung.fit(X[tr], y[tr])
        pred = rung.predict(X[te])

        # Trade only on conviction above the same deadband the labels use, so the strategy
        # is not paying the spread on every 100ms tick to chase noise.
        direction = np.zeros(te.shape[0], dtype=np.float64)
        direction[pred > deadband_ticks] = 1.0
        direction[pred < -deadband_ticks] = -1.0

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

    return pnl_all, dir_all


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
    for ticker in tickers:
        touch = _load_touch(args.frames, ticker)
        for clock in clocks:
            labels = [f"label_y_mid_{h}" for h in horizons]
            cols = _load_features(args.features, ticker, clock, ["ts_ns"] + needed + labels)
            ts = cols["ts_ns"].astype(np.int64)
            for h in horizons:
                y = cols[f"label_y_mid_{h}"]
                mask = common_usable_mask(cols, y, seed=args.seed)
                folds = make_folds(ts, horizon_ns=HORIZON_NS[h], protocol="C",
                                   n_folds=N_FOLDS, seed=args.seed)
                for rung in build_ladder(seed=args.seed):
                    for delta in LATENCIES_NS:
                        pnl, direction = simulate(
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
                        tk = turnover_and_holding(direction, HORIZON_NS[h])
                        ci = block_bootstrap_ci(
                            vals, block_len=max(10, int(HORIZON_NS[h] // 100_000_000)),
                            n_boot=500, seed=args.seed,
                        )
                        d = deflated_sharpe(vals, n_trials=args.trials)
                        avg_px = np.nanmean(
                            touch["ask_px_1"].astype(np.float64)
                        )
                        rows.append({
                            "ticker": ticker, "clock": clock, "h": h, "rung": rung.name,
                            "delta_ms": delta / 1e6, "n_trades": n_trades,
                            "total_pnl": float(np.nansum(vals)),
                            "mean_pnl": float(np.nanmean(vals)),
                            "be_ticks": break_even_cost_ticks(vals, n_trades),
                            "be_bps": break_even_cost_bps(vals, n_trades, avg_px),
                            "trade_rate": tk["trade_rate"],
                            "dsr": d["dsr"], "sharpe": d["sharpe"],
                            "ci_lo": ci["lo"], "ci_hi": ci["hi"],
                        })
                        print(f"  {ticker}/{clock} h={h} {rung.name} d={delta/1e6:.0f}ms "
                              f"trades={n_trades} be={rows[-1]['be_ticks']:.4f} ticks")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    _write(args, rows)
    print(f"wrote {args.out}")
    return 0


def _f(x, p=4):
    return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"


def _write(args, rows):
    with open(args.out, "w") as fh:
        w = fh.write
        w("# Transaction costs, PnL and break-even cost (Phase 3)\n\n")
        w(f"`python3 -m edge.costs.run --seed {args.seed} --trials {args.trials}`\n\n")
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
        w("the horizon. Trades only when |prediction| exceeds the 0.5-tick deadband.\n")
        w("- **Latency**: the signal formed at t acts at t+delta. A signal whose edge is gone ")
        w("by 100ms is latency arbitrage, not alpha — and the phase-1 adversary measured that ")
        w("OFI's content decays inside 100ms, so the 100ms rung is expected to be brutal.\n\n")
        if args.trials <= 1:
            w("> **The deflated Sharpe below is undeflated.** `--trials 1` was passed, which ")
            w("turns the DSR into an ordinary Sharpe probability wearing a rigorous name. ")
            w("The honest trial count comes from `RESULTS.md`'s trial log.\n\n")

        w("## Break-even cost table (the headline)\n\n")
        w("The per-share cost at which each rung's PnL crosses zero. Negative means the rung ")
        w("is already underwater before any additional cost — reported, not clipped.\n\n")
        w("| ticker | clock | h | rung | latency | trades | total PnL $ | break-even (ticks) | "
          "break-even (bps) | PnL CI lo | PnL CI hi | DSR |\n")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for r in rows:
            w(f"| {r['ticker']} | {r['clock']} | {r['h']} | {r['rung']} | "
              f"{r['delta_ms']:.0f}ms | {r['n_trades']} | {_f(r.get('total_pnl'), 2)} | "
              f"{_f(r.get('be_ticks'))} | {_f(r.get('be_bps'), 2)} | "
              f"{_f(r.get('ci_lo'), 4)} | {_f(r.get('ci_hi'), 4)} | {_f(r.get('dsr'), 3)} |\n")
        w("\n")
        w("## Reading notes\n\n")
        w("- A break-even cost BELOW the assumed take fee of $0.0030/share (0.30 ticks) means ")
        w("the rung does not pay for its own fees, before any spread cost.\n")
        w("- The PnL confidence interval is a circular block bootstrap with block length >= ")
        w("the label horizon; resampling individual trades would ignore the serial dependence ")
        w("that makes the naive interval far too narrow.\n")
        w("- Trade counts differ across rungs because the deadband bites differently: a rung ")
        w("whose predictions rarely exceed half a tick simply does not trade, which is a ")
        w("result rather than a missing row.\n")


if __name__ == "__main__":
    raise SystemExit(main())
