"""The frozen-predictor artifact floor, at several rung magnitudes.

    python3 -m edge.harness.floors --out reports/artifact_floor.md --seed 0

A separate entry point because a floor needs ONE fit per (cell, rung) while the ladder needs
five per protocol: there is no reason to pay for a full ladder re-run to get it.

WHAT THIS MEASURES. One model, fitted once on every usable row of the day, then scored
UNCHANGED under each protocol. Nothing is refitted between protocols, so any difference in the
reported metric is a pure fold-partition and aggregation artifact and none of it can be
leakage. It is the number a rung's A-C gap has to clear before it has measured anything.

WHY PER-MAGNITUDE. The floor is an interaction between the aggregation asymmetry -- protocol
A's fold mean approximates the sum(y^2)-WEIGHTED pooled metric, while C's is an unweighted
average over contiguous time segments of very unequal sum(y^2) -- and the predictor's own
variance. So it scales with the predictor. Measured on this data the L1-magnitude floor is
+0.0014 and the L4-magnitude floor is +0.0161: reporting the first beside an L4 gap
understates what that gap must clear by an order of magnitude. Both red-team passes on
2026-08-02 raised this; the second raised it against the first version of the fix.
"""
import argparse
import os

import numpy as np

from edge.harness import PROTOCOLS
from edge.harness.models import build_ladder
from edge.harness.run import (
    CLOCKS,
    HORIZONS,
    HORIZON_NS,
    TICKERS,
    _load,
    frozen_floor_multi,
    frozen_floor_windows,
)


def _f(x, places=5):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{places}f}"


def main(argv=None):
    p = argparse.ArgumentParser(description="Frozen-predictor artifact floor per rung.")
    p.add_argument("--features", default="data/features")
    p.add_argument("--out", default="reports/artifact_floor.md")
    p.add_argument("--tickers", default=",".join(TICKERS))
    p.add_argument("--clocks", default=",".join(CLOCKS))
    p.add_argument("--horizons", default=",".join(HORIZONS))
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    tickers = tuple(t for t in args.tickers.split(",") if t)
    clocks = tuple(c for c in args.clocks.split(",") if c)
    horizons = tuple(h for h in args.horizons.split(",") if h)

    needed = sorted({f for r in build_ladder(seed=args.seed) for f in r.features})
    rows = []
    for ticker in tickers:
        for clock in clocks:
            label_cols = [f"label_y_mid_{h}" for h in horizons]
            cols = _load(args.features, ticker, clock, ["ts_ns"] + needed + label_cols)
            ts = cols["ts_ns"].astype(np.int64)
            for h in horizons:
                y = cols[f"label_y_mid_{h}"]
                fl = frozen_floor_multi(cols, y, ts, HORIZON_NS[h], args.seed,
                                        protocols=PROTOCOLS, refit_scale=True)
                fz = frozen_floor_multi(cols, y, ts, HORIZON_NS[h], args.seed,
                                        protocols=PROTOCOLS, refit_scale=False)
                # The floor over every fit window, so the WORST one can be reported. Fitting
                # the frozen model on the whole day is the choice that minimises it.
                wins = {rung: frozen_floor_windows(
                            cols, y, ts, HORIZON_NS[h], args.seed, protocols=("A", "C"),
                            rung_prefix=rung + " ")
                        for rung in fl}
                for rung, per in fl.items():
                    rows.append((ticker, clock, h, rung, per, fz.get(rung, {}),
                                 wins.get(rung, {})))
                print(f"  {ticker}/{clock} h={h}: floors at "
                      f"{', '.join(sorted(fl))}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        w = fh.write
        w("# The frozen-predictor artifact floor (Phase 2)\n\n")
        w(f"`python3 -m edge.harness.floors --seed {args.seed} --tickers {args.tickers} "
          f"--clocks {args.clocks} --horizons {args.horizons}`\n\n")
        w("Each row is ONE model, fitted once on every usable row of the day, then scored ")
        w("under each protocol with **only the through-origin scale refitted per fold, under ")
        w("that protocol's own holdout rule** -- exactly the treatment the fitted rungs get. ")
        w("The raw model cannot have learned anything from any split, so every difference ")
        w("below is artifact and **none of it can be leakage**. A rung's `A - C` gap has to ")
        w("clear the floor measured AT ITS OWN MAGNITUDE.\n\n")
        w("> **Corrected 2026-08-03 after the third red-team pass.** The previous version of ")
        w("this table froze the SCALE as well as the model. That made it blind to the largest ")
        w("channel: under protocol A the calibration holdout is a random 20% of the training ")
        w("rows and is exchangeable with A's random test fold, while under B/C/D it is the ")
        w("last 20% in time and is not. A floor measured with that channel held fixed is not ")
        w("a bound on a rung exposed to it, and it was published as one -- the R2 floors were ")
        w("understated 2-6x and the AUC floor's worst cell was quoted as +0.0053 when the ")
        w("like-for-like figure is an order of magnitude larger. The frozen-scale column is ")
        w("retained beside the corrected one: it is the pure partition artifact, a real but ")
        w("different quantity, and the difference between the columns is the calibration ")
        w("channel.\n\n")
        w("The floor is the interaction between the aggregation asymmetry -- A's fold mean ")
        w("approximates the `sum(y^2)`-weighted pooled metric, C's is an unweighted average ")
        w("over contiguous segments of very unequal `sum(y^2)` -- and the predictor's own ")
        w("variance. It therefore SCALES WITH THE PREDICTOR, which is why one magnitude does ")
        w("not bound another and why the earlier practice of quoting the L1 floor beside an ")
        w("L4 gap was wrong.\n\n")
        w("| ticker | clock | h | frozen at | A R2 | B R2 | C R2 | D R2 | **A-C floor** | "
          "A-C (frozen scale) | A IC | C IC | A-C IC floor |\n")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for ticker, clock, h, rung, per, fz, wins in rows:
            def g(pr, m, per=per):
                return per[pr][m]["mean"] if pr in per else float("nan")
            fzg = (fz["A"]["r2"]["mean"] - fz["C"]["r2"]["mean"]
                   if "A" in fz and "C" in fz else float("nan"))
            w(f"| {ticker} | {clock} | {h} | {rung} | {_f(g('A', 'r2'))} | "
              f"{_f(g('B', 'r2'))} | {_f(g('C', 'r2'))} | {_f(g('D', 'r2'))} | "
              f"**{_f(g('A', 'r2') - g('C', 'r2'))}** | {_f(fzg)} | {_f(g('A', 'ic'))} | "
              f"{_f(g('C', 'ic'))} | {_f(g('A', 'ic') - g('C', 'ic'))} |\n")
        w("\n")

        w("## The same floor for AUC and the calibration slope\n\n")
        w("Added 2026-08-03 with those two metrics. A gap in a new metric needs a floor ")
        w("measured in THAT metric.\n\n")
        w("**The AUC floor is not small because rank statistics are immune to the ")
        w("aggregation asymmetry.** An earlier draft of `FINDINGS.md` claimed that, and it is ")
        w("wrong: protocol A's per-fold AUC is a whole-day POOLED AUC while C's is a ")
        w("WITHIN-BLOCK AUC, and any frozen score with a time-varying offset makes those two ")
        w("estimands diverge with nothing refitted at all -- the red team drove the floor to ")
        w("-0.137 that way. It is small HERE because these particular frozen predictors are ")
        w("temporally homogeneous. Read it as a measurement on this data, not as a property ")
        w("of the metric.\n\n")
        w("`neg C` counts protocol-C folds where the refitted scale came out NEGATIVE, which ")
        w("flips the ranking and sends AUC to 1-AUC; it is 0 under A everywhere. `A-C AUC raw` ")
        w("is the same floor on the raw score, immune to that sign by construction.\n\n")
        w("| ticker | clock | h | frozen at | A AUC | C AUC | **A-C AUC floor** | "
          "A-C AUC raw | neg C | A calib | C calib | A-C calib |\n")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for ticker, clock, h, rung, per, fz, wins in rows:
            def g(pr, m, per=per):
                return per[pr][m]["mean"] if pr in per else float("nan")
            neg = per["C"].get("n_negative_scale", 0) if "C" in per else 0
            w(f"| {ticker} | {clock} | {h} | {rung} | {_f(g('A', 'auc'), 4)} | "
              f"{_f(g('C', 'auc'), 4)} | **{_f(g('A', 'auc') - g('C', 'auc'), 4)}** | "
              f"{_f(g('A', 'auc_raw') - g('C', 'auc_raw'), 4)} | {neg} | "
              f"{_f(g('A', 'calibration_slope'), 4)} | "
              f"{_f(g('C', 'calibration_slope'), 4)} | "
              f"{_f(g('A', 'calibration_slope') - g('C', 'calibration_slope'), 4)} |\n")
        w("\n")
        w("The `A-C calib` column is a MEASUREMENT OF THE CALIBRATION DESIGN, not of the ")
        w("protocols. Under A the holdout is exchangeable with A's test fold, so A's slope is ")
        w("driven toward 1.0 for any predictor -- these frozen rows, which cannot leak, ")
        w("reproduce the fitted rungs' published A-vs-C slope split to three decimals. Do not ")
        w("read it as leakage.\n\n")

        w("## The floor over four fit windows -- and the WORST one is the one to quote\n\n")
        w("Added 2026-08-03 after the fourth red-team pass. Every floor above fits the frozen ")
        w("model on the WHOLE DAY, and that is the choice that MINIMISES the floor. At L4's ")
        w("magnitude it does so mechanically: a whole-day GBM scores AUC ~0.996, a ceiling ")
        w("where two protocols have almost no room to differ, while the rung it is meant to ")
        w("bound scores ~0.72 under C. Refitting the same frozen model on half the day puts ")
        w("its accuracy in the rungs' operating range and the floor grows several-fold. A ")
        w("margin quoted against the whole-day floor is therefore an overstatement, and the ")
        w("earlier '98x its floor' claim was exactly that.\n\n")
        w("`max` is the worst floor across fit windows for that cell. **Quote margins against ")
        w("`max`.**\n\n")
        w("| ticker | clock | h | frozen at | metric | whole | first 50% | last 50% | "
          "random 50% | **max** |\n")
        w("|---|---|---|---|---|---|---|---|---|---|\n")
        for ticker, clock, h, rung, per, fz, wins in rows:
            if not wins:
                continue
            for metric, places in (("r2", 5), ("auc_raw", 4)):
                vals = {}
                for win, res in wins.items():
                    if "A" in res and "C" in res:
                        vals[win] = (res["A"][metric]["mean"] - res["C"][metric]["mean"])
                    else:
                        vals[win] = float("nan")
                finite = [v for v in vals.values() if v == v]
                worst = max(finite) if finite else float("nan")
                w(f"| {ticker} | {clock} | {h} | {rung} | {metric} | "
                  f"{_f(vals.get('whole'), places)} | "
                  f"{_f(vals.get('first_half'), places)} | "
                  f"{_f(vals.get('last_half'), places)} | "
                  f"{_f(vals.get('random_half'), places)} | **{_f(worst, places)}** |\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
