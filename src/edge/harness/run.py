"""Phase-2 runner: score the L0-L4 ladder under protocols A, B and C; emit the optimism table.

    python3 -m edge.harness.run --features data/features --out reports/ladder.md --seed 0

For each (ticker, clock, horizon, protocol, rung) this fits per fold, scores per fold, and
reports the DISTRIBUTION across folds -- mean, median and worst -- never a pooled number
alone (EVAL_PROTOCOL.md sec. 1). The headline artefact is the optimism table:

    A -> C   total optimism
    B -> C   the purge/embargo effect at a walk-forward boundary (NOT "label overlap" --
             it can only ever touch ~1% of an expanding training set)
    A -> B   the residual
    A -> D   fold geometry and TEMPORAL ADJACENCY of training rows, where label overlap
             actually lives
    D -> C   training on the future in disjoint blocks -- small, ~0.01-0.05 IC

L1 WAS the intended control, and it is not sufficient. Its "one-parameter rescale" is fitted,
and the phase-2 red team showed that rescale plus the fold-aggregation asymmetry accounts for
the ENTIRE L1/L1m/L2 gap. The real control is FROZEN: a predictor fitted once on the whole day
and scored unchanged under every protocol, so any difference is 100% partition artifact. That
is the `FROZEN` row in the report, and it is the floor the fitted rungs must clear.

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
# Imported, not copied: a hard-coded 0.5 here could silently diverge from the labels' own
# deadband, classifying predictions against a different threshold than the one the labels
# were built with (2026-08-02 red team).
from edge.features.labels import ALPHA_TICKS
from edge.harness.models import build_ladder
from edge.harness.splits import make_folds

TICKERS = ("INTC", "AMZN")
CLOCKS = ("calendar", "event")
DATE = "2012-06-21"
HORIZONS = ("1s", "2s", "5s", "10s")
HORIZON_NS = {"1s": 1_000_000_000, "2s": 2_000_000_000,
              "5s": 5_000_000_000, "10s": 10_000_000_000}


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


def common_usable_mask(cols, y, seed=0):
    """Rows where EVERY rung's features and the label are finite.

    The ladder's premise is that each rung must beat the one below it, which only means
    something if the rungs are scored on the same rows. Left to per-rung masks, L2 (one OFI
    column) would be scored on ~92% of stamps and L3/L4 on ~86%, and part of any difference
    between them would be a difference in sample. INTERFACE.md sec. 7 still puts the drop
    here in the harness rather than in the family modules; this just makes the drop shared.

    The dominant cost is the block-0 warmup on every fitted column (7.69% -- exactly the
    first refit block, the honest price of a causal fit), which is reported, not hidden.
    """
    needed = sorted({f for r in build_ladder(seed=seed) for f in r.features})
    mask = np.isfinite(y)
    for name in needed:
        mask &= np.isfinite(cols[name])
    return mask


def score_fold(rung, cols, y, fold, common_mask, causal_scale=True):
    """Fit on the fold's training rows, score its test rows. Returns a metrics dict or None."""
    feats = list(rung.features)
    if feats:
        X_all = np.column_stack([cols[f] for f in feats])
    else:
        X_all = np.zeros((y.shape[0], 0), dtype=np.float64)
    usable = common_mask

    tr = fold.train_idx[usable[fold.train_idx]]
    te = fold.test_idx[usable[fold.test_idx]]
    if tr.size < 50 or te.size < 20:
        return None  # too thin to mean anything; counted as a dropped fold, not a zero

    if causal_scale:
        # Strictly causal calibration: model on the first 80% of the training rows, the
        # through-origin scale on the held-out last 20%. Under A the training rows are not
        # time-ordered, so the holdout is a seeded random 20% -- A's own logic, applied to
        # A. See models.Rung.fit for why the in-sample alternative is an artifact.
        hold = np.zeros(tr.size, dtype=bool)
        if fold.protocol == "A":
            rng = np.random.default_rng(1000 + fold.fold_id)
            hold[rng.choice(tr.size, size=max(1, tr.size // 5), replace=False)] = True
        else:
            hold[int(tr.size * 0.8):] = True
        if hold.all() or not hold.any():
            return None
        rung.fit(X_all[tr], y[tr], scale_holdout=hold)
    else:
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


def run_cell(cols, y, ts, horizon_ns, protocol, seed, embargo_ns=None, n_folds=N_FOLDS,
             causal_scale=True):
    """Every rung under one protocol. Returns {rung_name: {metric: fold_summary}}."""
    folds = make_folds(
        ts, horizon_ns=horizon_ns, protocol=protocol,
        n_folds=n_folds, embargo_ns=embargo_ns, seed=seed,
    )
    common_mask = common_usable_mask(cols, y, seed=seed)
    out = {"_common_row_fraction": float(common_mask.mean())}
    for rung in build_ladder(seed=seed):
        per_fold = [score_fold(rung, cols, y, f, common_mask, causal_scale)
                    for f in folds]
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


def frozen_floor_multi(cols, y, ts, horizon_ns, seed, n_folds=N_FOLDS,
                       protocols=("A", "B", "C", "D"), rung_prefixes=("L1 ", "L3 ", "L4 ")):
    """Frozen floors at SEVERAL rung magnitudes, because one magnitude does not bound another.

    The 2026-08-02 second red-team pass showed this directly: the frozen floor measured with
    `sign(queue imbalance)` is +0.0014, but the same construction at L4's magnitude is
    +0.0161 -- an order of magnitude larger. A floor is an artifact of how the folds
    partition and aggregate INTERACTING WITH the predictor's variance, so it has to be
    measured at the magnitude of the rung being judged. Reporting only the L1 floor beside an
    L4 gap understates what that gap has to clear.
    """
    return {
        prefix.strip(): frozen_floor(cols, y, ts, horizon_ns, seed, n_folds=n_folds,
                                     protocols=protocols, rung_prefix=prefix)
        for prefix in rung_prefixes
    }


def frozen_floor(cols, y, ts, horizon_ns, seed, n_folds=N_FOLDS, protocols=("A", "B", "C"),
                 rung_prefix="L1 "):
    """The artifact floor: ONE predictor, fitted once, scored unchanged under every protocol.

    This is the control the L1 row was supposed to be and could not be. L1 has a fitted
    rescale, so its A-C gap mixes a partition artifact with a refit effect. Here the model is
    byte-identical across protocols, so ANY difference in the reported metric is 100% an
    artifact of how the folds partition and aggregate -- nothing was refitted, so nothing
    could have learned anything.

    The mechanism it exposes: A's five folds are random subsets of the same universe, so its
    fold mean approximates the sum(y^2)-WEIGHTED pooled metric. C's five folds are contiguous
    time segments with very unequal sum(y^2), so its fold mean is UNWEIGHTED. Mean-over-folds
    is therefore estimating two different quantities, and the difference is not leakage.

    Any rung whose A-C gap does not clear this floor has measured nothing.
    """
    common_mask = common_usable_mask(cols, y, seed=seed)
    ladder = build_ladder(seed=seed)
    rung = next(r for r in ladder if r.name.startswith(rung_prefix))
    feats = list(rung.features)
    X_all = (np.column_stack([cols[f] for f in feats]) if feats
             else np.zeros((y.shape[0], 0), dtype=np.float64))

    # Fit ONCE, on every usable row of the day. Deliberately leaky -- the point is that it is
    # equally leaky for every protocol, so it cannot explain a DIFFERENCE between them.
    fit_rows = np.flatnonzero(common_mask)
    rung.fit(X_all[fit_rows], y[fit_rows])
    frozen_pred = rung.predict(X_all)

    out = {}
    for protocol in protocols:
        folds = make_folds(ts, horizon_ns=horizon_ns, protocol=protocol,
                           n_folds=n_folds, seed=seed)
        r2s, ics = [], []
        for f in folds:
            te = f.test_idx[common_mask[f.test_idx]]
            if te.size < 20:
                continue
            r2s.append(r2_vs_zero(y[te], frozen_pred[te]))
            ics.append(spearman_ic(y[te], frozen_pred[te]))
        if r2s:
            out[protocol] = {"r2": fold_summary(r2s), "ic": fold_summary(ics)}
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
    p.add_argument("--in-sample-scale", action="store_true",
                   help="also emit the superseded in-sample-calibration columns (A and C), "
                        "so the size of the 2026-08-02 correction is visible")
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
    floors = {}
    insample = {}
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
                # The artifact floor: one frozen predictor, scored under every protocol.
                floors[(ticker, clock, h)] = frozen_floor(
                    cols, y, ts, HORIZON_NS[h], args.seed, protocols=PROTOCOLS
                )
                if args.in_sample_scale:
                    # The superseded calibration, kept as a diagnostic so the size of the
                    # correction is visible rather than asserted.
                    for protocol in ("A", "C"):
                        insample[(ticker, clock, h, protocol)] = run_cell(
                            cols, y, ts, HORIZON_NS[h], protocol, args.seed,
                            causal_scale=False,
                        )
                if args.embargo_sweep:
                    for e in EMBARGO_SWEEP_NS:
                        sweeps[(ticker, clock, h, e)] = run_cell(
                            cols, y, ts, HORIZON_NS[h], "C", args.seed, embargo_ns=e
                        )

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        _write_report(fh, results, sweeps, ladder_names, tickers, clocks, horizons, args,
                      floors, insample)
    dt = time.time() - t0
    print(f"wrote {args.out} in {dt:.1f}s")
    return 0


def _write_report(fh, results, sweeps, ladder_names, tickers, clocks, horizons, args,
                  floors=None, insample=None):
    floors = floors or {}
    insample = insample or {}
    w = fh.write
    w("# Ladder under three evaluation protocols (Phase 2)\n\n")
    # Echo the FULL effective argument set: the previous header printed only --seed, so the
    # command it advertised did not reproduce the sweep or the diagnostic columns.
    flags = ("".join([" --embargo-sweep" if args.embargo_sweep else "",
                      " --in-sample-scale" if args.in_sample_scale else ""]))
    w(f"`python3 -m edge.harness.run --seed {args.seed} --tickers {args.tickers} "
      f"--clocks {args.clocks} --horizons {args.horizons}{flags}`\n\n")
    w("**A** shuffled k-fold (what most of the literature does) | ")
    w("**B** walk-forward, no purge, no embargo | ")
    w("**C** purged walk-forward with embargo (correct) | ")
    w("**D** control: C's test blocks, trained on everything outside them, past AND future, ")
    w("no purge.\n\n")

    w("> **Corrected 2026-08-02 after the phase-2 red team**\n")
    w("> (`reports/adversary/phase2-ladder-2026-08-02.md`). Two changes to how these numbers ")
    w("are produced, and one to what they are called:\n")
    w("> 1. **Calibration is now causal.** The through-origin rescale is fitted on a held-out ")
    w("last 20% of each fold's training rows, not in-sample. The previous table fitted it on ")
    w("the same rows as the model, which for a high-capacity rung returns almost no ")
    w("shrinkage and makes the out-of-sample amplitude far too large. That, not leakage, is ")
    w("what produced the large negative protocol-C R2 in the earlier draft.\n")
    w("> 2. **The artifact floor is measured, not assumed.** A FROZEN predictor -- fitted ")
    w("once on the whole day, scored unchanged under every protocol -- still shows a nonzero ")
    w("A-C gap, because A's fold mean is the sum(y^2)-weighted average of the day while C's ")
    w("is unweighted. Any rung not clearing that floor has measured nothing.\n")
    w("> 3. **`B - C` is not the label-overlap component** and is no longer labelled as one. ")
    w("Under expanding walk-forward, purge+embargo can only remove the last h+E of a ")
    w("multi-hour training set (0.6-1.1% of rows). Label overlap lives in `A - D`.\n\n")

    w("`A - C` is total optimism. `A - D` is fold geometry plus the temporal ADJACENCY of ")
    w("training rows, which is where label overlap actually sits. `D - C` is training on the ")
    w("future in disjoint blocks. `B - C` is the purge/embargo effect at a walk-forward ")
    w("boundary. Every number is the distribution ACROSS FOLDS -- mean, median and worst -- ")
    w("never a pooled number alone (EVAL_PROTOCOL.md sec. 1).\n\n")
    w(f"Folds: {N_FOLDS} per protocol. Equal fold COUNT does not equalise training-set size ")
    w("-- A trains on roughly 1.8x C's median and 8.7x C's first fold -- but a size-matched ")
    w("control reproduces the unmatched result to within 0.01 R2, so size is not the driver. ")
    w("Walk-forward test blocks are cut on WALL-CLOCK TIME, not stamp index. Embargo ")
    w("E = max(60s, 10h). Ridge lambda = 1.0, GBM 200 rounds, both frozen a priori.\n\n")
    w("R2 is computed against a ZERO forecast, never the test mean (EVAL_PROTOCOL.md sec. 2 ")
    w("names the test-mean version as the most common subtle leak in this literature).\n\n")

    w("## Optimism table (the headline)\n\n")
    w("Out-of-sample R2 vs zero. `A-C med` is the gap recomputed from per-fold MEDIANS and ")
    w("`A-C ex0` drops fold 0 -- both because 45 of 80 rows in the earlier draft lost more ")
    w("than half their gap when fold 0 was dropped, and the mean is the number that gets ")
    w("quoted. `floor` is the frozen-predictor artifact gap for that cell.\n\n")
    w("| ticker | clock | h | rung | A | B | C | D | A-C | A-C med | A-C ex0 | A-D | D-C | "
      "B-C | floor |\n")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for ticker in tickers:
        for clock in clocks:
            for h in horizons:
                floor_cell = floors.get((ticker, clock, h), {})
                fl = np.nan
                if "A" in floor_cell and "C" in floor_cell:
                    fl = floor_cell["A"]["r2"]["mean"] - floor_cell["C"]["r2"]["mean"]
                for name in ladder_names:
                    cells = {
                        pr: results.get((ticker, clock, h, pr), {}).get(name)
                        for pr in PROTOCOLS
                    }
                    if not any(cells.values()):
                        continue
                    vals, meds, ex0 = {}, {}, {}
                    for pr in PROTOCOLS:
                        c = cells[pr]
                        vals[pr] = c["r2"]["mean"] if c else np.nan
                        meds[pr] = c["r2"]["median"] if c else np.nan
                        pf = c["per_fold_r2"] if c else []
                        ex0[pr] = float(np.mean(pf[1:])) if len(pf) > 1 else np.nan
                    ac = vals["A"] - vals["C"]
                    w(f"| {ticker} | {clock} | {h} | {name} | {_fmt(vals['A'])} | "
                      f"{_fmt(vals['B'])} | {_fmt(vals['C'])} | {_fmt(vals['D'])} | "
                      f"{_fmt(ac)} | {_fmt(meds['A'] - meds['C'])} | "
                      f"{_fmt(ex0['A'] - ex0['C'])} | {_fmt(vals['A'] - vals['D'])} | "
                      f"{_fmt(vals['D'] - vals['C'])} | {_fmt(vals['B'] - vals['C'])} | "
                      f"{_fmt(fl)} |\n")
    w("\n")

    w("## The artifact floor (frozen predictor, never refitted)\n\n")
    w("One `sign(queue imbalance)` model fitted once on every usable row of the day, then ")
    w("scored UNCHANGED under each protocol. Nothing is refitted between protocols, so every ")
    w("difference below is a pure fold-partition and aggregation artifact and none of it is ")
    w("leakage. This is the number the fitted rungs' gaps must be read against.\n\n")
    w("| ticker | clock | h | A R2 | B R2 | C R2 | D R2 | A-C (artifact) | A IC | C IC | "
      "A-C IC |\n")
    w("|---|---|---|---|---|---|---|---|---|---|---|\n")
    for (ticker, clock, h), f in sorted(floors.items()):
        if not f:
            continue
        g = lambda pr, m: f[pr][m]["mean"] if pr in f else np.nan  # noqa: E731
        w(f"| {ticker} | {clock} | {h} | {_fmt(g('A','r2'))} | {_fmt(g('B','r2'))} | "
          f"{_fmt(g('C','r2'))} | {_fmt(g('D','r2'))} | "
          f"{_fmt(g('A','r2') - g('C','r2'))} | {_fmt(g('A','ic'))} | {_fmt(g('C','ic'))} | "
          f"{_fmt(g('A','ic') - g('C','ic'))} |\n")
    w("\n")

    if insample:
        w("## Superseded: the in-sample calibration\n\n")
        w("What the earlier draft reported. The rescale is fitted on the same rows as the ")
        w("model, so a high-capacity rung's out-of-sample amplitude is far too large and its ")
        w("protocol-C R2 goes sharply negative. Kept so the size of the correction is ")
        w("visible rather than asserted.\n\n")
        w("| ticker | clock | h | rung | A (in-sample) | C (in-sample) | A-C (in-sample) | "
          "A-C (causal) |\n")
        w("|---|---|---|---|---|---|---|---|\n")
        for ticker in tickers:
            for clock in clocks:
                for h in horizons:
                    for name in ladder_names:
                        a = insample.get((ticker, clock, h, "A"), {}).get(name)
                        c = insample.get((ticker, clock, h, "C"), {}).get(name)
                        if not a or not c:
                            continue
                        ca = results.get((ticker, clock, h, "A"), {}).get(name)
                        cc = results.get((ticker, clock, h, "C"), {}).get(name)
                        causal = (ca["r2"]["mean"] - cc["r2"]["mean"]) if ca and cc else np.nan
                        w(f"| {ticker} | {clock} | {h} | {name} | {_fmt(a['r2']['mean'])} | "
                          f"{_fmt(c['r2']['mean'])} | "
                          f"{_fmt(a['r2']['mean'] - c['r2']['mean'])} | {_fmt(causal)} |\n")
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
        for (ticker, clock, h, e), cell in sorted(
                sweeps.items(), key=lambda kv: (kv[0][0], kv[0][1], kv[0][2], int(kv[0][3]))):
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
