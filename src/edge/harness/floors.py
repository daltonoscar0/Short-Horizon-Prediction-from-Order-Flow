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
                                        protocols=PROTOCOLS)
                for rung, per in fl.items():
                    rows.append((ticker, clock, h, rung, per))
                print(f"  {ticker}/{clock} h={h}: floors at "
                      f"{', '.join(sorted(fl))}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        w = fh.write
        w("# The frozen-predictor artifact floor (Phase 2)\n\n")
        w(f"`python3 -m edge.harness.floors --seed {args.seed} --tickers {args.tickers} "
          f"--clocks {args.clocks} --horizons {args.horizons}`\n\n")
        w("Each row is ONE model, fitted once on every usable row of the day, then scored ")
        w("UNCHANGED under each protocol. Nothing is refitted between protocols, so every ")
        w("difference below is a pure fold-partition and aggregation artifact and **none of ")
        w("it can be leakage**. A rung's `A - C` gap has to clear the floor measured AT ITS ")
        w("OWN MAGNITUDE.\n\n")
        w("The floor is the interaction between the aggregation asymmetry -- A's fold mean ")
        w("approximates the `sum(y^2)`-weighted pooled metric, C's is an unweighted average ")
        w("over contiguous segments of very unequal `sum(y^2)` -- and the predictor's own ")
        w("variance. It therefore SCALES WITH THE PREDICTOR, which is why one magnitude does ")
        w("not bound another and why the earlier practice of quoting the L1 floor beside an ")
        w("L4 gap was wrong.\n\n")
        w("| ticker | clock | h | frozen at | A R2 | B R2 | C R2 | D R2 | **A-C floor** | "
          "A IC | C IC | A-C IC floor |\n")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for ticker, clock, h, rung, per in rows:
            def g(pr, m, per=per):
                return per[pr][m]["mean"] if pr in per else float("nan")
            w(f"| {ticker} | {clock} | {h} | {rung} | {_f(g('A', 'r2'))} | "
              f"{_f(g('B', 'r2'))} | {_f(g('C', 'r2'))} | {_f(g('D', 'r2'))} | "
              f"**{_f(g('A', 'r2') - g('C', 'r2'))}** | {_f(g('A', 'ic'))} | "
              f"{_f(g('C', 'ic'))} | {_f(g('A', 'ic') - g('C', 'ic'))} |\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
