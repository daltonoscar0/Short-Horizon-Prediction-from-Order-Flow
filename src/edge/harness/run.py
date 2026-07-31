"""Phase-2 runner: score the L0-L4 ladder under protocols A, B and C; emit the optimism table.

    python3 -m edge.harness.run --features data/features --out reports/ladder.md --seed 0

For each (ticker, clock, horizon, protocol, rung) this fits per fold, scores per fold, and
reports the DISTRIBUTION across folds -- mean, median and worst -- never a pooled number
alone (EVAL_PROTOCOL.md sec. 1). The headline artefact is the optimism table:

    A -> C   total optimism
    B -> C   the label-overlap component (B is already strictly forward in time)
    A -> B   the residual: training on the future outright

L1 is load-bearing. It has NO fitted parameters beyond a one-parameter rescale, so its A->C
gap is a floor on what the harness itself contributes: a large L1 gap would mean the
measurement is an artifact of how the protocols differ rather than of leakage, and the whole
decomposition would be suspect (DECISIONS.md 2026-07-31).

Rows with a NaN in any column a rung uses, or in the label, are dropped per fold and per rung
and the surviving counts are reported (INTERFACE.md sec. 7).
"""
import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq

from edge.harness import EMBARGO_SWEEP_NS, N_FOLDS, PROTOCOLS, default_embargo_ns
from edge.harness.metrics import (
    classification_metrics,
    fold_summary,
    r2_vs_test_mean,
    r2_vs_zero,
    spearman_ic,
)
from edge.harness.models import build_ladder
from edge.harness.splits import make_folds

TICKERS = ("INTC", "AMZN")
CLOCKS = ("calendar", "event")
DATE = "2012-06-21"
HORIZONS = ("1s", "2s", "5s", "10s")
HORIZON_NS = {"1s": 1_000_000_000, "2s": 2_000_000_000,
              "5s": 5_000_000_000, "10s": 10_000_000_000}
ALPHA_TICKS = 0.5  # imported convention from edge.features.labels; classification deadband


def _load(features_dir, ticker, clock, columns):
    path = os.path.join(features_dir, f"{ticker}_{DATE}_{clock}.parquet")
    table = pq.read_table(path, columns=list(columns))
    return {name: table.column(name).to_numpy(zero_copy_only=False) for name in columns}


def _classify(y_ticks, alpha=ALPHA_TICKS):
    out = np.full(y_ticks.shape, np.nan, dtype=np.float64)
    finite = np.isfinite(y_ticks)
    out[finite] = 0.0
    out[finite & (y_ticks > alpha)] = 1.0
    out[finite & (y_ticks < -alpha)] = -1.0
    return out


def score_fold(rung, cols, y, fold):
    """Fit on the fold's training rows, score its test rows. Returns a metrics dict or None."""
    feats = list(rung.features)
    if feats:
        X_all = np.column_stack([cols[f] for f in feats])
        usable = np.isfinite(X_all).all(axis=1) & np.isfinite(y)
    else:
        X_all = np.zeros((y.shape[0], 0), dtype=np.float64)
        usable = np.isfinite(y)

    tr = fold.train_idx[usable[fold.train_idx]]
    te = fold.test_idx[usable[fold.test_idx]]
    if tr.size < 50 or te.size < 20:
        return None  # too thin to mean anything; counted as a dropped fold, not a zero

    rung.fit(X_all[tr], y[tr])
    pred = rung.predict(X_all[te])
    y_te = y[te]

    cls = classification_metrics(_classify(y_te), _classify(pred))
    return {
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "r2": r2_vs_zero(y_te, pred),
        "r2_test_mean": r2_vs_test_mean(y_te, pred),
        "ic": spearman_ic(y_te, pred),
        "balanced_accuracy": cls["balanced_accuracy"],
        "mcc": cls["mcc"],
        "base_rate": cls["base_rate"],
    }


def run_cell(cols, y, ts, horizon_ns, protocol, seed, embargo_ns=None, n_folds=N_FOLDS):
    """Every rung under one protocol. Returns {rung_name: {metric: fold_summary}}."""
    folds = make_folds(
        ts, horizon_ns=horizon_ns, protocol=protocol,
        n_folds=n_folds, embargo_ns=embargo_ns, seed=seed,
    )
    out = {}
    for rung in build_ladder(seed=seed):
        per_fold = [score_fold(rung, cols, y, f) for f in folds]
        kept = [r for r in per_fold if r is not None]
        if not kept:
            continue
        out[rung.name] = {
            "n_folds_scored": len(kept),
            "n_folds_dropped": len(per_fold) - len(kept),
            "n_train": fold_summary([r["n_train"] for r in kept]),
            "n_test": fold_summary([r["n_test"] for r in kept]),
            "r2": fold_summary([r["r2"] for r in kept]),
            "r2_test_mean": fold_summary([r["r2_test_mean"] for r in kept]),
            "ic": fold_summary([r["ic"] for r in kept]),
            "balanced_accuracy": fold_summary([r["balanced_accuracy"] for r in kept]),
            "mcc": fold_summary([r["mcc"] for r in kept]),
            "base_rate": fold_summary([r["base_rate"] for r in kept]),
            "per_fold_ic": [r["ic"] for r in kept],
            "per_fold_r2": [r["r2"] for r in kept],
        }
    return out


def _fmt(x, places=5):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{places}f}"


def main(argv=None):
    p = argparse.ArgumentParser(description="Score the L0-L4 ladder under protocols A, B, C.")
    p.add_argument("--features", default="data/features")
    p.add_argument("--out", default="reports/ladder.md")
    p.add_argument("--horizons", default=",".join(HORIZONS))
    p.add_argument("--tickers", default=",".join(TICKERS))
    p.add_argument("--clocks", default=",".join(CLOCKS))
    p.add_argument("--embargo-sweep", action="store_true",
                   help="also emit the protocol-C embargo sensitivity table")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    horizons = tuple(h for h in args.horizons.split(",") if h)
    tickers = tuple(t for t in args.tickers.split(",") if t)
    clocks = tuple(c for c in args.clocks.split(",") if c)

    t0 = time.time()
    ladder_names = [r.name for r in build_ladder(seed=args.seed)]
    needed = sorted({f for r in build_ladder(seed=args.seed) for f in r.features})

    results = {}
    sweeps = {}
    for ticker in tickers:
        for clock in clocks:
            label_cols = [f"label_y_mid_{h}" for h in horizons]
            cols = _load(args.features, ticker, clock, ["ts_ns"] + needed + label_cols)
            ts = cols["ts_ns"].astype(np.int64)
            for h in horizons:
                y = cols[f"label_y_mid_{h}"]
                for protocol in PROTOCOLS:
                    key = (ticker, clock, h, protocol)
                    results[key] = run_cell(
                        cols, y, ts, HORIZON_NS[h], protocol, args.seed
                    )
                    print(f"  {ticker}/{clock} h={h} protocol {protocol}: "
                          f"{len(results[key])} rungs scored")
                if args.embargo_sweep:
                    for e in EMBARGO_SWEEP_NS:
                        sweeps[(ticker, clock, h, e)] = run_cell(
                            cols, y, ts, HORIZON_NS[h], "C", args.seed, embargo_ns=e
                        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        _write_report(fh, results, sweeps, ladder_names, tickers, clocks, horizons, args)
    dt = time.time() - t0
    print(f"wrote {args.out} in {dt:.1f}s")
    return 0


def _write_report(fh, results, sweeps, ladder_names, tickers, clocks, horizons, args):
    w = fh.write
    w("# Ladder under three evaluation protocols (Phase 2)\n\n")
    w(f"`python3 -m edge.harness.run --seed {args.seed}`\n\n")
    w("**A** shuffled k-fold (what most of the literature does) | ")
    w("**B** walk-forward, no purge, no embargo | ")
    w("**C** purged walk-forward with embargo (correct).\n\n")
    w("`A - C` is total optimism. `B - C` isolates the label-overlap component, since B is ")
    w("already strictly forward in time. `A - B` is the residual: training on the future ")
    w("outright. Every number is the distribution ACROSS FOLDS -- mean, median and worst -- ")
    w("never a pooled number alone (EVAL_PROTOCOL.md sec. 1).\n\n")
    w(f"Folds: {N_FOLDS} per protocol, identical count across protocols so the gap is not ")
    w("partly a training-set-size difference. Walk-forward test blocks are cut on WALL-CLOCK ")
    w("TIME, not stamp index. Embargo E = max(60s, 10h). Ridge lambda = 1.0, GBM 200 rounds, ")
    w("both frozen a priori.\n\n")
    w("R2 is computed against a ZERO forecast, never the test mean (EVAL_PROTOCOL.md sec. 2 ")
    w("names the test-mean version as the most common subtle leak in this literature). ")
    w("The `r2_test_mean` column is reported alongside purely to show the size of that gap.\n\n")

    w("## Optimism table (the headline)\n\n")
    w("Mean out-of-sample R2 vs zero, by protocol, with the two gaps.\n\n")
    w("| ticker | clock | h | rung | A | B | C | A-C (total) | B-C (overlap) | A-B (future) |\n")
    w("|---|---|---|---|---|---|---|---|---|---|\n")
    for ticker in tickers:
        for clock in clocks:
            for h in horizons:
                for name in ladder_names:
                    cells = {
                        pr: results.get((ticker, clock, h, pr), {}).get(name)
                        for pr in PROTOCOLS
                    }
                    if not any(cells.values()):
                        continue
                    vals = {}
                    for pr in PROTOCOLS:
                        c = cells[pr]
                        vals[pr] = c["r2"]["mean"] if c else np.nan
                    ac = vals["A"] - vals["C"]
                    bc = vals["B"] - vals["C"]
                    ab = vals["A"] - vals["B"]
                    w(f"| {ticker} | {clock} | {h} | {name} | {_fmt(vals['A'])} | "
                      f"{_fmt(vals['B'])} | {_fmt(vals['C'])} | {_fmt(ac)} | {_fmt(bc)} | "
                      f"{_fmt(ab)} |\n")
    w("\n")

    w("## Information coefficient, per protocol, with the worst fold\n\n")
    w("| ticker | clock | h | rung | protocol | IC mean | IC median | IC worst | folds | n_test median |\n")
    w("|---|---|---|---|---|---|---|---|---|---|\n")
    for ticker in tickers:
        for clock in clocks:
            for h in horizons:
                for name in ladder_names:
                    for pr in PROTOCOLS:
                        c = results.get((ticker, clock, h, pr), {}).get(name)
                        if not c:
                            continue
                        w(f"| {ticker} | {clock} | {h} | {name} | {pr} | "
                          f"{_fmt(c['ic']['mean'])} | {_fmt(c['ic']['median'])} | "
                          f"{_fmt(c['ic']['worst'])} | {c['n_folds_scored']} | "
                          f"{_fmt(c['n_test']['median'], 0)} |\n")
    w("\n")

    w("## Classification against the base rate\n\n")
    w("Balanced accuracy and MCC, printed next to the majority-class share. At 1s most ")
    w("labels are flat, so plain accuracy is meaningless (FEATURES.md sec. 7).\n\n")
    w("| ticker | clock | h | rung | protocol | bal. acc | MCC | base rate |\n")
    w("|---|---|---|---|---|---|---|---|\n")
    for ticker in tickers:
        for clock in clocks:
            for h in horizons:
                for name in ladder_names:
                    for pr in PROTOCOLS:
                        c = results.get((ticker, clock, h, pr), {}).get(name)
                        if not c:
                            continue
                        w(f"| {ticker} | {clock} | {h} | {name} | {pr} | "
                          f"{_fmt(c['balanced_accuracy']['mean'], 4)} | "
                          f"{_fmt(c['mcc']['mean'], 4)} | "
                          f"{_fmt(c['base_rate']['mean'], 4)} |\n")
    w("\n")

    w("## Per-fold detail under protocol C (the honest ladder)\n\n")
    w("| ticker | clock | h | rung | per-fold IC | per-fold R2 |\n")
    w("|---|---|---|---|---|---|\n")
    for ticker in tickers:
        for clock in clocks:
            for h in horizons:
                for name in ladder_names:
                    c = results.get((ticker, clock, h, "C"), {}).get(name)
                    if not c:
                        continue
                    ics = ", ".join(_fmt(v, 4) for v in c["per_fold_ic"])
                    r2s = ", ".join(_fmt(v, 5) for v in c["per_fold_r2"])
                    w(f"| {ticker} | {clock} | {h} | {name} | {ics} | {r2s} |\n")
    w("\n")

    if sweeps:
        w("## Embargo sensitivity (protocol C)\n\n")
        w("The frozen headline is E = max(60s, 10h); these rows exist to show the result is ")
        w("not an artifact of that choice.\n\n")
        w("| ticker | clock | h | embargo (s) | rung | R2 mean | IC mean | n_train median |\n")
        w("|---|---|---|---|---|---|---|---|\n")
        for (ticker, clock, h, e), cell in sorted(sweeps.items(), key=lambda kv: str(kv[0])):
            for name in ladder_names:
                c = cell.get(name)
                if not c:
                    continue
                w(f"| {ticker} | {clock} | {h} | {e / 1e9:.0f} | {name} | "
                  f"{_fmt(c['r2']['mean'])} | {_fmt(c['ic']['mean'])} | "
                  f"{_fmt(c['n_train']['median'], 0)} |\n")
        w("\n")

    w("## Reading notes\n\n")
    w("- **L1 is the control.** It has no fitted parameters beyond a one-parameter rescale, ")
    w("so its `A-C` gap is a floor on what the harness itself contributes. A large L1 gap ")
    w("would mean the decomposition is measuring the protocols rather than leakage.\n")
    w("- **The 1s label is not one-second alpha.** The phase-1 adversary measured that OFI's ")
    w("entire content decays inside 100 ms (rho 0.458 over the first 100 ms against 0.027 ")
    w("over the remaining 900 ms). Read these as short-horizon numbers.\n")
    w("- **Fold row counts vary by design.** Test blocks are equal in DURATION, not in ")
    w("sample count, and rungs using trade features lose rows where a window holds no ")
    w("trades. Both are reported rather than absorbed.\n")


if __name__ == "__main__":
    raise SystemExit(main())
