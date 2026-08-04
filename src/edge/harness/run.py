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
    auc_binary,
    calibration_slope,
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
from edge.harness.models import _fit_scale, build_ladder
from edge.harness.splits import make_folds

# The decoupled-control arms. Every one fits the raw model on the SAME seeded random 80% of
# its fold's training rows, so the only thing that varies between them is which rows the
# through-origin scale is fitted on -- and, between A and C, the split itself.
DECOUPLED_ARMS = {
    "A_rand": ("A", "random"),        # shuffled split, random calibration slice
    "A_pos": ("A", "positional"),     # shuffled split, contiguous calibration slice
    "C_rand": ("C", "random"),        # walk-forward split, random calibration slice
    "C_pos": ("C", "positional"),     # walk-forward split, contiguous slice (C's own rule)
}

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


def scale_holdout_mask(n_train, protocol, fold_id, holdout_rule=None):
    """The calibration holdout over a fold's training rows.

    THE RULE IS PROTOCOL-DEPENDENT BY DEFAULT, and that is a design choice with consequences.
    Under A the training rows are not time-ordered, so a positional split would not be a
    held-out slice in any meaningful sense and the holdout is a seeded random 20% -- A's own
    logic applied to A. Under B/C/D it is the last 20% in time.

    The cost, found by the 2026-08-03 red team: under A the holdout is EXCHANGEABLE with A's
    random test fold, so the reported calibration slope is driven toward 1.0 mechanically, for
    any predictor, with or without skill. `holdout_rule` exists to hold the rule constant
    across protocols and measure how much of a gap is this channel rather than the split:
    pass "positional" to give A the walk-forward rule. Default None keeps the per-protocol
    behaviour the headline is computed with.
    """
    rule = holdout_rule or ("random" if protocol == "A" else "positional")
    hold = np.zeros(n_train, dtype=bool)
    if rule == "random":
        rng = np.random.default_rng(1000 + fold_id)
        hold[rng.choice(n_train, size=max(1, n_train // 5), replace=False)] = True
    elif rule == "positional":
        hold[int(n_train * 0.8):] = True
    else:
        raise ValueError(f"unknown holdout rule {rule!r}")
    return hold


def model_fit_mask(n_train, fold_id, seed=0):
    """The raw model's training rows for the DECOUPLED control: a seeded random 80%.

    Deliberately independent of the protocol and of the calibration rule, so that varying the
    rule cannot vary the model's training window. Scattered rather than positional for the same
    reason -- a positional 80% would hand one arm the day's tail and withhold it from the
    other, which is the exact confound this exists to remove.
    """
    rng = np.random.default_rng(2000 + 17 * seed + fold_id)
    mask = np.zeros(n_train, dtype=bool)
    mask[rng.choice(n_train, size=max(1, int(n_train * 0.8)), replace=False)] = True
    return mask


def _directional_auc(y_te, pred):
    """AUC on the rows the deadband calls up or down. Flats excluded; one-sided blocks NaN."""
    y_cls = _classify(y_te)
    d = np.isfinite(y_cls) & (y_cls != 0.0)
    if not np.any(d):
        return np.nan
    return auc_binary((y_cls[d] > 0.0).astype(np.float64), pred[d])


def score_fold(rung, cols, y, fold, common_mask, causal_scale=True, holdout_rule=None,
               decouple_scale=False, seed=0):
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
        # Strictly causal calibration: model on 80% of the training rows, the through-origin
        # scale on the held-out 20%. See models.Rung.fit for why the in-sample alternative is
        # an artifact, and scale_holdout_mask for what the per-protocol rule costs.
        hold = scale_holdout_mask(tr.size, fold.protocol, fold.fold_id, holdout_rule)
        if hold.all() or not hold.any():
            return None
        # The decoupled control holds the model's fit rows fixed across calibration rules,
        # accepting that the scale rows then overlap them. See models.Rung.fit.
        mm = model_fit_mask(tr.size, fold.fold_id, seed=seed) if decouple_scale else None
        rung.fit(X_all[tr], y[tr], scale_holdout=hold, model_mask=mm)
    else:
        rung.fit(X_all[tr], y[tr])
    pred = rung.predict(X_all[te])
    y_te = y[te]

    cls = classification_metrics(_classify(y_te), _classify(pred))

    # AUC on the DIRECTIONAL subset only: rows the deadband calls up or down, never the flats.
    # Scoring +1 against {0, -1} pooled would mostly measure the tie mass (96.3% of INTC
    # calendar 1s labels are exactly flat), which is the axis error the phase-1 red team
    # already caught once. Undefined -- correctly NaN -- when a test block is one-sided.
    auc = _directional_auc(y_te, pred)
    # The same AUC on the RAW score, before the through-origin rescale. The rescale is
    # rank-preserving only UP TO SIGN, and the 2026-08-03 red team found the scale fitted
    # negative in 27 of 400 protocol-C folds and 0 of 400 protocol-A folds -- a negative scale
    # flips the ranking and sends AUC to 1-AUC, which manufactured the whole apparent A-C gap
    # on the unfitted rungs. The raw column is immune to that channel by construction.
    auc_raw = _directional_auc(y_te, rung._raw(X_all[te]))

    return {
        "n_train": int(tr.size),
        "n_test": int(te.size),
        "r2": r2_vs_zero(y_te, pred),
        "r2_test_mean": r2_vs_test_mean(y_te, pred),
        "ic": spearman_ic(y_te, pred),
        # EVAL_PROTOCOL.md sec. 2: slope of realized on predicted. ~1 is calibrated, 0.2 means
        # confidently wrong at scale. Measured on the raw prediction, AFTER the through-origin
        # rescale, so it judges the amplitude the harness actually reports.
        "calibration_slope": calibration_slope(y_te, pred),
        "auc": auc,
        "auc_raw": auc_raw,
        "scale": float(getattr(rung, "scale_", np.nan)),
        "balanced_accuracy": cls["balanced_accuracy"],
        "mcc": cls["mcc"],
        "base_rate": cls["base_rate"],
    }


def run_cell(cols, y, ts, horizon_ns, protocol, seed, embargo_ns=None, n_folds=N_FOLDS,
             causal_scale=True, holdout_rule=None, decouple_scale=False):
    """Every rung under one protocol. Returns {rung_name: {metric: fold_summary}}."""
    folds = make_folds(
        ts, horizon_ns=horizon_ns, protocol=protocol,
        n_folds=n_folds, embargo_ns=embargo_ns, seed=seed,
    )
    common_mask = common_usable_mask(cols, y, seed=seed)
    out = {"_common_row_fraction": float(common_mask.mean())}
    for rung in build_ladder(seed=seed):
        per_fold = [score_fold(rung, cols, y, f, common_mask, causal_scale, holdout_rule,
                               decouple_scale, seed)
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
            "calibration_slope": fold_summary([r["calibration_slope"] for r in kept]),
            "auc": fold_summary([r["auc"] for r in kept]),
            "auc_raw": fold_summary([r["auc_raw"] for r in kept]),
            "n_negative_scale": sum(1 for r in kept if r["scale"] < 0.0),
            "balanced_accuracy": fold_summary([r["balanced_accuracy"] for r in kept]),
            "mcc": fold_summary([r["mcc"] for r in kept]),
            "base_rate": fold_summary([r["base_rate"] for r in kept]),
            "per_fold_ic": [r["ic"] for r in kept],
            "per_fold_r2": [r["r2"] for r in kept],
            # The headline metric needs its own fold distribution (RULES.md rule 4). It
            # shipped without one for two revisions, and it matters: on the clean arm L4 is
            # positive in 80 of 80 folds while L3 is positive in 55 of 80.
            "per_fold_auc_raw": [r["auc_raw"] for r in kept],
        }
    return out


def frozen_floor_multi(cols, y, ts, horizon_ns, seed, n_folds=N_FOLDS,
                       protocols=("A", "B", "C", "D"), rung_prefixes=("L1 ", "L3 ", "L4 "),
                       refit_scale=True):
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
                                     protocols=protocols, rung_prefix=prefix,
                                     refit_scale=refit_scale)
        for prefix in rung_prefixes
    }


FLOOR_FIT_WINDOWS = ("whole", "first_half", "last_half", "random_half")


def frozen_floor_windows(cols, y, ts, horizon_ns, seed, n_folds=N_FOLDS,
                         protocols=("A", "B", "C"), rung_prefix="L1 ",
                         windows=FLOOR_FIT_WINDOWS):
    """The floor over several fit windows, because WHERE the frozen model is fitted matters.

    The 2026-08-03 fourth red-team pass: fitting the frozen model on the whole day is the
    choice that MINIMISES the floor, and on AUC it does so for a mechanical reason. A
    whole-day GBM scores AUC 0.996 -- a ceiling, where two protocols have almost no room to
    differ -- while the rung it is meant to bound scores 0.72 under C. Refit the same frozen
    model on half the day, so its accuracy lands in the rungs' operating range, and the same
    floor is 5-13x larger at the median. The published "98x its floor" margin was that
    ceiling, not a property of the partition.

    The honest floor is therefore the WORST over fit windows, per cell, which is what
    `reports/artifact_floor.md` reports and what the surviving margins are quoted against.
    """
    return {w: frozen_floor(cols, y, ts, horizon_ns, seed, n_folds=n_folds,
                            protocols=protocols, rung_prefix=rung_prefix, fit_window=w)
            for w in windows}


def _fit_window_rows(fit_rows, window, seed):
    """Which of the usable rows the frozen model is fitted on."""
    if window == "whole":
        return fit_rows
    half = fit_rows.size // 2
    if window == "first_half":
        return fit_rows[:half]
    if window == "last_half":
        return fit_rows[half:]
    if window == "random_half":
        rng = np.random.default_rng(7000 + seed)
        return np.sort(rng.choice(fit_rows, size=half, replace=False))
    raise ValueError(f"unknown floor fit window {window!r}")


def frozen_floor(cols, y, ts, horizon_ns, seed, n_folds=N_FOLDS, protocols=("A", "B", "C"),
                 rung_prefix="L1 ", refit_scale=True, fit_window="whole"):
    """The artifact floor: ONE model, fitted once, scored under every protocol.

    This is the control the L1 row was supposed to be and could not be. L1 has a fitted
    rescale, so its A-C gap mixes a partition artifact with a refit effect. Here the RAW MODEL
    is byte-identical across protocols, so it cannot have learned anything from any split.

    `refit_scale` decides what the floor is a floor ON, and the 2026-08-03 red team killed the
    report for getting this wrong:

      * True (default, LIKE-FOR-LIKE) -- the through-origin scale is refitted per fold under
        that protocol's own holdout rule, exactly as the fitted rungs get it. This is the
        floor a rung's gap must actually clear, because the rungs are scored with a refitted
        scale and the scale is itself protocol-dependent. Measured here: +0.0043 (L1
        magnitude), +0.0112 (L3), +0.0192 (L4) against the frozen-scale figures below.
      * False (FROZEN SCALE) -- nothing at all is refitted. Isolates the pure fold-partition
        and aggregation artifact, which is a real and separate quantity, but it is NOT a
        bound on a refitted rung. The earlier report published it as one; the difference
        between the two columns IS the calibration channel, so both are emitted.

    The partition mechanism the frozen-scale column exposes: A's five folds are random subsets
    of the same universe, so its fold mean approximates the sum(y^2)-WEIGHTED pooled metric.
    C's five folds are contiguous time segments with very unequal sum(y^2), so its fold mean is
    UNWEIGHTED. Mean-over-folds is estimating two different quantities, and that is not
    leakage.
    """
    common_mask = common_usable_mask(cols, y, seed=seed)
    ladder = build_ladder(seed=seed)
    rung = next(r for r in ladder if r.name.startswith(rung_prefix))
    feats = list(rung.features)
    X_all = (np.column_stack([cols[f] for f in feats]) if feats
             else np.zeros((y.shape[0], 0), dtype=np.float64))

    # Fit ONCE, on `fit_window` of the usable rows. Deliberately leaky -- the point is that it
    # is equally leaky for every protocol, so it cannot explain a DIFFERENCE between them.
    # The window is a parameter because "whole day" minimises the floor and, at L4's
    # magnitude, does so by putting the frozen model at an AUC ceiling; see
    # frozen_floor_windows.
    fit_rows = _fit_window_rows(np.flatnonzero(common_mask), fit_window, seed)
    rung.fit(X_all[fit_rows], y[fit_rows])
    frozen_raw = rung._raw(X_all)
    frozen_pred = rung.predict(X_all)

    out = {}
    for protocol in protocols:
        folds = make_folds(ts, horizon_ns=horizon_ns, protocol=protocol,
                           n_folds=n_folds, seed=seed)
        r2s, ics, aucs, cals, raws, neg = [], [], [], [], [], 0
        # The per-fold scales are RETURNED, not just used. They are the only externally
        # checkable evidence that the floor refits the scale at all, that it refits it on the
        # holdout rather than in-sample, and that it uses each protocol's own rule -- three
        # mutations that a 400-test suite could not see on 2026-08-03 because the number never
        # left the function.
        scales = []
        for f in folds:
            te = f.test_idx[common_mask[f.test_idx]]
            if te.size < 20:
                continue
            if refit_scale:
                tr = f.train_idx[common_mask[f.train_idx]]
                if tr.size < 50:
                    continue
                hold = scale_holdout_mask(tr.size, f.protocol, f.fold_id)
                scale = _fit_scale(frozen_raw[tr][hold], y[tr][hold])
                pred = scale * frozen_raw
                neg += int(scale < 0.0)
                scales.append(scale)
            else:
                pred = frozen_pred
                scales.append(float(rung.scale_))
            r2s.append(r2_vs_zero(y[te], pred[te]))
            ics.append(spearman_ic(y[te], pred[te]))
            # AUC and the calibration slope get a floor too. An A-C gap in a NEW metric needs
            # its own floor measured the same way -- quoting one metric's floor beside
            # another's gap is the error the 2026-08-02 red team killed the ladder over.
            cals.append(calibration_slope(y[te], pred[te]))
            aucs.append(_directional_auc(y[te], pred[te]))
            raws.append(_directional_auc(y[te], frozen_raw[te]))
        if r2s:
            out[protocol] = {"r2": fold_summary(r2s), "ic": fold_summary(ics),
                             "auc": fold_summary(aucs), "auc_raw": fold_summary(raws),
                             "calibration_slope": fold_summary(cals),
                             "n_negative_scale": neg, "scales": scales,
                             "frozen_raw": frozen_raw}
    return out


def _floor_key(rung_name):
    """Which frozen magnitude a rung's gap is judged against. L3 and L4 get their own."""
    if rung_name.startswith("L3 "):
        return "L3"
    if rung_name.startswith("L4 "):
        return "L4"
    return "L1"


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
    p.add_argument("--holdout-control", action="store_true",
                   help="also re-run protocol A under C's positional calibration holdout, so "
                        "the share of the A-C gap that is the calibration RULE rather than "
                        "the split is visible (2026-08-03 red team). Implies the DECOUPLED "
                        "control below, which is the clean version of the same comparison")
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
    floors_frozen_scale = {}
    insample = {}
    holdout = {}
    decoupled = {}
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
                # The artifact floor. LIKE-FOR-LIKE: model frozen, scale refitted per fold
                # under each protocol's own rule, exactly as the fitted rungs get it. The
                # frozen-scale variant is kept beside it; the difference is the calibration
                # channel, which the 2026-08-03 red team showed the old floor could not see.
                # ...and measured at THREE rung magnitudes, then matched to each rung in the
                # headline table. The previous version measured only L1's magnitude and
                # printed it beside every rung's gap, which is the practice the 2026-08-02
                # pass called wrong in the prose while the generated column kept doing it.
                floors[(ticker, clock, h)] = frozen_floor_multi(
                    cols, y, ts, HORIZON_NS[h], args.seed, protocols=PROTOCOLS,
                    refit_scale=True,
                )
                floors_frozen_scale[(ticker, clock, h)] = frozen_floor_multi(
                    cols, y, ts, HORIZON_NS[h], args.seed, protocols=PROTOCOLS,
                    refit_scale=False,
                )
                if args.holdout_control:
                    # Protocol A under C's positional holdout. NOT a clean control: the rule
                    # also moves the model's training rows, so this arm is a temporally
                    # handicapped model as well as a differently-calibrated one.
                    holdout[(ticker, clock, h)] = run_cell(
                        cols, y, ts, HORIZON_NS[h], "A", args.seed,
                        holdout_rule="positional",
                    )
                    # The CLEAN control: the model's fit rows are the same seeded random 80%
                    # in every arm, and only the scale rows move. Both protocols get it, so
                    # the difference between these two columns is the calibration rule and
                    # nothing else.
                    for arm, (protocol, rule) in DECOUPLED_ARMS.items():
                        decoupled[(ticker, clock, h, arm)] = run_cell(
                            cols, y, ts, HORIZON_NS[h], protocol, args.seed,
                            holdout_rule=rule, decouple_scale=True,
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
                      floors, insample, holdout, floors_frozen_scale, decoupled)
    dt = time.time() - t0
    print(f"wrote {args.out} in {dt:.1f}s")
    return 0


def _write_report(fh, results, sweeps, ladder_names, tickers, clocks, horizons, args,
                  floors=None, insample=None, holdout=None, floors_frozen_scale=None,
                  decoupled=None):
    floors = floors or {}
    insample = insample or {}
    holdout = holdout or {}
    floors_frozen_scale = floors_frozen_scale or {}
    decoupled = decoupled or {}
    w = fh.write
    w("# Ladder under three evaluation protocols (Phase 2)\n\n")
    # Echo the FULL effective argument set: the previous header printed only --seed, so the
    # command it advertised did not reproduce the sweep or the diagnostic columns.
    flags = ("".join([" --embargo-sweep" if args.embargo_sweep else "",
                      " --in-sample-scale" if args.in_sample_scale else "",
                      " --holdout-control" if args.holdout_control else ""]))
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
    w("once on the whole day, scored under every protocol -- still shows a nonzero A-C gap, ")
    w("partly because A's fold mean is the sum(y^2)-weighted average of the day while C's is ")
    w("unweighted. Any rung not clearing that floor has measured nothing. (That was only half ")
    w("the mechanism; see the 2026-08-03 note below.)\n")
    w("> 3. **`B - C` is not the label-overlap component** and is no longer labelled as one. ")
    w("Under expanding walk-forward, purge+embargo can only remove the last h+E of a ")
    w("multi-hour training set (0.6-1.1% of rows). Label overlap lives in `A - D`.\n\n")

    w("> **Corrected again 2026-08-03 after the third red-team pass**\n")
    w("> (`reports/adversary/phase2-metrics-2026-08-03.md`). Adding the two metrics ")
    w("`EVAL_PROTOCOL.md` sec. 2 still wanted -- the calibration slope and AUC -- exposed two ")
    w("errors in the corrections above:\n")
    w("> 1. **The artifact floor froze the SCALE as well as the model**, so it could not see ")
    w("the largest artifact channel. Under A the calibration holdout is a random 20% of the ")
    w("training rows, exchangeable with A's random test fold; under B/C/D it is the last 20% ")
    w("in time and is not. Like-for-like floors, with the scale refitted per fold under each ")
    w("protocol's own rule, are 2-6x the published ones.\n")
    w("> 2. **About half of L4's `A - C` gap is that calibration rule, not the split.** Held ")
    w("constant, L4's median gap goes +0.3173 -> +0.1379 and L1's goes +0.0043 -> -0.0021. ")
    w("The capacity ordering survives; the magnitude was overstated.\n")
    w("> A calibration-slope reading published the same morning -- that A's near-1.0 slopes ")
    w("were a leakage signature -- was withdrawn: a frozen model that cannot leak reproduces ")
    w("it to three decimals.\n\n")

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
    w("quoted. `floor` is the frozen-model artifact gap **measured at that rung's own ")
    w("magnitude** (L0/L1/L1m/L2 against the L1-magnitude floor, L3 against L3's, L4 against ")
    w("L4's) with the scale refitted per fold, since the floor scales with the predictor and ")
    w("one magnitude does not bound another. Until 2026-08-03 this column printed the ")
    w("L1-magnitude floor beside every rung.\n\n")
    w("| ticker | clock | h | rung | A | B | C | D | A-C | A-C med | A-C ex0 | A-D | D-C | "
      "B-C | floor |\n")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for ticker in tickers:
        for clock in clocks:
            for h in horizons:
                floor_cell = floors.get((ticker, clock, h), {})
                for name in ladder_names:
                    per = floor_cell.get(_floor_key(name), {})
                    fl = np.nan
                    if "A" in per and "C" in per:
                        fl = per["A"]["r2"]["mean"] - per["C"]["r2"]["mean"]
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

    w("## The artifact floor (frozen model, scale refitted per fold)\n\n")
    w("One model fitted once on every usable row of the day, then scored under each protocol ")
    w("with **only the through-origin scale refitted per fold, under that protocol's own ")
    w("holdout rule** -- exactly the treatment the fitted rungs get. The raw model cannot have ")
    w("learned anything from any split, so every difference below is artifact; but it is the ")
    w("artifact of BOTH channels the rungs are exposed to, the fold partition and the ")
    w("calibration rule.\n\n")
    w("> **Corrected 2026-08-03 after the third red-team pass.** The previous version of this ")
    w("table froze the scale as well as the model, which made it blind to the channel that ")
    w("matters most: under protocol A the calibration holdout is a random 20% of the training ")
    w("rows and is therefore EXCHANGEABLE with A's random test fold, while under B/C/D it is ")
    w("the last 20% in time and is not. A floor measured with that channel held fixed is not ")
    w("a bound on a rung that is exposed to it, and it was published as one. The frozen-scale ")
    w("column is retained in the next section as the pure partition artifact, which is a real ")
    w("but different quantity.\n\n")
    w("`neg C` counts protocol-C folds where the refitted scale came out NEGATIVE, which "
      "flips the ranking and sends AUC to 1-AUC. It is 0 under A in every cell.\n\n")
    w("| ticker | clock | h | frozen at | A R2 | C R2 | **A-C floor** | A AUC | C AUC | "
      "A-C AUC floor | A-C (frozen scale) | neg C |\n")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for (ticker, clock, h), by_rung in sorted(floors.items()):
        frozen_by_rung = floors_frozen_scale.get((ticker, clock, h), {})
        for rung in sorted(by_rung):
            f = by_rung[rung]
            if not f:
                continue
            g = lambda pr, m, f=f: f[pr][m]["mean"] if pr in f else np.nan  # noqa: E731
            fz = frozen_by_rung.get(rung, {})
            fzg = (fz["A"]["r2"]["mean"] - fz["C"]["r2"]["mean"]
                   if "A" in fz and "C" in fz else np.nan)
            neg = f["C"].get("n_negative_scale", 0) if "C" in f else 0
            w(f"| {ticker} | {clock} | {h} | {rung} | {_fmt(g('A','r2'))} | "
              f"{_fmt(g('C','r2'))} | **{_fmt(g('A','r2') - g('C','r2'))}** | "
              f"{_fmt(g('A','auc'), 4)} | {_fmt(g('C','auc'), 4)} | "
              f"{_fmt(g('A','auc') - g('C','auc'), 4)} | {_fmt(fzg)} | {neg} |\n")
    w("\n")

    if holdout:
        w("## Control: how much of `A - C` is the calibration RULE, not the split\n\n")
        w("Protocol A re-run with **C's positional calibration holdout** (the last 20% of the ")
        w("training rows in time) instead of its own random 20%. Split, folds, seed and metrics ")
        w("are unchanged -- but **the model is NOT**, which is the defect: the rule and the ")
        w("model's training rows move together here.\n\n")
        w("> **SUPERSEDED as a decomposition -- see the CLEAN control below.** This arm was ")
        w("built to isolate the calibration rule and cannot: `Rung.fit` fits the raw model on ")
        w("the COMPLEMENT of the calibration holdout, so forcing A onto the positional rule ")
        w("also strips A's model of the day's tail. `A' - C` is a crippled model, not a ")
        w("differently-calibrated one, and the ~48%-of-the-gap attribution once drawn from it ")
        w("is withdrawn. It is retained because the correction is only legible beside it.\n\n")
        w("| ticker | clock | h | rung | A (own rule) | A' (C's rule) | C | A-C | **A'-C** | "
          "rule share | A-C AUC raw | **A'-C AUC raw** |\n")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for ticker in tickers:
            for clock in clocks:
                for h in horizons:
                    hc = holdout.get((ticker, clock, h), {})
                    for name in ladder_names:
                        a = results.get((ticker, clock, h, "A"), {}).get(name)
                        c = results.get((ticker, clock, h, "C"), {}).get(name)
                        ap = hc.get(name)
                        if not a or not c or not ap:
                            continue
                        va, vc, vp = (a["r2"]["mean"], c["r2"]["mean"], ap["r2"]["mean"])
                        gap, gap_p = va - vc, vp - vc
                        share = 1.0 - gap_p / gap if gap != 0 else np.nan
                        ua = a.get("auc_raw", {}).get("mean", np.nan)
                        uc = c.get("auc_raw", {}).get("mean", np.nan)
                        up = ap.get("auc_raw", {}).get("mean", np.nan)
                        w(f"| {ticker} | {clock} | {h} | {name} | {_fmt(va)} | {_fmt(vp)} | "
                          f"{_fmt(vc)} | {_fmt(gap)} | **{_fmt(gap_p)}** | "
                          f"{_fmt(share, 3)} | {_fmt(ua - uc, 4)} | "
                          f"**{_fmt(up - uc, 4)}** |\n")
        w("\n")

    if decoupled:
        w("## The CLEAN calibration control: model rows fixed, only the scale rows move\n\n")
        w("Every arm below fits the raw model on the **same seeded random 80%** of each fold's ")
        w("training rows -- the same rule under every protocol and every calibration slice -- ")
        w("and fits the through-origin scale on the slice named in the column. The scale rows ")
        w("therefore OVERLAP the model rows; that is the accepted cost, it is identical in ")
        w("every arm, and in exchange the calibration rule can be varied without also varying ")
        w("the model's training window.\n\n")
        w("Why this exists: `A' - C` above cannot separate the two. `Rung.fit` fits the model ")
        w("on the COMPLEMENT of the calibration holdout, so switching A to the positional rule ")
        w("also takes away the day's tail. The 2x2 gave two rule-matched answers that ")
        w("disagreed by a factor of ~4, which is why the earlier headline was a bracket. ")
        w("Here the row difference is the rule and nothing else.\n\n")
        w("`A_rand - C_pos` is the split-only comparison and **it is the number to quote**. ")
        w("`A_pos - C_pos` and `A_rand - C_rand` are the two rule-matched comparisons; their ")
        w("`rule spread` is a DIFFERENCE IN DIFFERENCES -- the interaction of rule and ")
        w("protocol, not 'what the rule is worth'. A rule worth +0.05 under both protocols ")
        w("would show a spread of zero.\n\n")
        w("> **THE R2 COLUMNS OF THIS TABLE ARE NOT USABLE, INCLUDING THE DIFFERENCES.** The ")
        w("overlap makes each arm's scale partly IN-SAMPLE -- the calibration killed on ")
        w("2026-08-02 -- and it bites asymmetrically, not identically: A's arm is unharmed ")
        w("(R2 0.18378 against the headline ladder's 0.18316 at INTC calendar 1s) while C's is ")
        w("destroyed (-0.25953 against +0.02134), because the in-sample scale bias grows with ")
        w("capacity per training row and C trains on a fraction of A's rows. Under permuted ")
        w("labels this construction manufactures an R2 gap of +0.062 +/- 0.058 out of pure ")
        w("noise, against +0.010 +/- 0.009 for the main ladder. An earlier draft claimed the ")
        w("overlap was identical in every arm and the differences therefore clean; that is ")
        w("false for R2 and it is withdrawn (2026-08-03 fifth red team). **The raw-AUC table ")
        w("below carries the same permutation null as the main ladder and is unaffected -- it ")
        w("is the only part of this control that may be quoted.**\n\n")
        w("| ticker | clock | h | rung | A_rand | A_pos | C_rand | C_pos | A_rand-C_pos | "
          "**A_pos-C_pos** | **A_rand-C_rand** | rule spread |\n")
        w("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
        for ticker in tickers:
            for clock in clocks:
                for h in horizons:
                    for name in ladder_names:
                        arms = {a: decoupled.get((ticker, clock, h, a), {}).get(name)
                                for a in DECOUPLED_ARMS}
                        if not all(arms.values()):
                            continue
                        v = {a: c["r2"]["mean"] for a, c in arms.items()}
                        unc = v["A_rand"] - v["C_pos"]
                        m_pos = v["A_pos"] - v["C_pos"]
                        m_rand = v["A_rand"] - v["C_rand"]
                        w(f"| {ticker} | {clock} | {h} | {name} | {_fmt(v['A_rand'])} | "
                          f"{_fmt(v['A_pos'])} | {_fmt(v['C_rand'])} | {_fmt(v['C_pos'])} | "
                          f"{_fmt(unc)} | **{_fmt(m_pos)}** | **{_fmt(m_rand)}** | "
                          f"{_fmt(abs(m_pos - m_rand))} |\n")
        w("\n")

        w("### The same control on raw-score AUC -- where the rule spread is ZERO BY "
          "CONSTRUCTION, and that is the finding\n\n")
        w("Raw AUC is a rank statistic on the raw model, so it never sees the scale at all. ")
        w("Once the model's fit rows no longer move with the calibration rule, the rule ")
        w("CANNOT move raw AUC, and the `rule spread` column below is identically 0.0000. ")
        w("That is an identity, not a measurement, and it is reported as one.\n\n")
        w("**What it settles.** The published `A' - C` shrank L4's raw-AUC gap from +0.1740 to ")
        w("+0.1371 and that shrink was attributed to the calibration rule. It cannot have ")
        w("been: with the model held fixed the rule is worth exactly nothing on this metric. ")
        w("The whole of that shrink was the model's LOST TRAINING WINDOW -- under the ")
        w("positional rule the model is fitted on the first 80% of the day and never sees its ")
        w("tail. The `A_rand - C_pos` column here is the split effect measured with a ")
        w("rule-independent model, and it is the number to quote.\n\n")
        w("`per-fold` and `worst` are the fold distribution of the split-only gap, which the ")
        w("headline metric shipped without for two revisions (RULES.md rule 4). Read them: ")
        w("two rungs with the same 16/16 cell count can have very different fold behaviour.\n\n")
        w("| ticker | clock | h | rung | **A_rand-C_pos** | A_pos-C_pos | A_rand-C_rand | "
          "rule spread | folds > 0 | worst fold |\n")
        w("|---|---|---|---|---|---|---|---|---|---|\n")
        for ticker in tickers:
            for clock in clocks:
                for h in horizons:
                    for name in ladder_names:
                        arms = {a: decoupled.get((ticker, clock, h, a), {}).get(name)
                                for a in DECOUPLED_ARMS}
                        if not all(arms.values()):
                            continue
                        v = {a: c.get("auc_raw", {}).get("mean", np.nan)
                             for a, c in arms.items()}
                        unc = v["A_rand"] - v["C_pos"]
                        m_pos = v["A_pos"] - v["C_pos"]
                        m_rand = v["A_rand"] - v["C_rand"]
                        pa = arms["A_rand"].get("per_fold_auc_raw", [])
                        pc = arms["C_pos"].get("per_fold_auc_raw", [])
                        per = [a - c for a, c in zip(pa, pc)
                               if np.isfinite(a) and np.isfinite(c)]
                        npos = sum(1 for x in per if x > 0)
                        worst = min(per) if per else np.nan
                        w(f"| {ticker} | {clock} | {h} | {name} | **{_fmt(unc, 4)}** | "
                          f"{_fmt(m_pos, 4)} | {_fmt(m_rand, 4)} | "
                          f"{_fmt(abs(m_pos - m_rand), 4)} | {npos}/{len(per)} | "
                          f"{_fmt(worst, 4)} |\n")
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

    w("## Information coefficient and calibration slope, per protocol, with the worst fold\n\n")
    w("`calib` is the slope of realized on predicted (EVAL_PROTOCOL.md sec. 2): 1.0 is ")
    w("calibrated, 0.2 means confidently wrong at scale, and a NEGATIVE slope means the ")
    w("prediction is pointed the wrong way at magnitude even where its rank correlation is ")
    w("positive.\n\n")
    w("> **DO NOT read the A-versus-C difference in this column as leakage.** An earlier draft ")
    w("did. Under protocol A the calibration holdout is a random 20% of the training rows, ")
    w("which is exchangeable with A's random test fold, so A's slope is driven toward 1.0 ")
    w("MECHANICALLY, for any predictor, with or without skill: a frozen whole-day model with ")
    w("only the scale refitted reproduces A = 1.03 / C = 1.39 at L1's magnitude to three ")
    w("decimals. Hold the calibration rule constant and the claim reverses for L4 -- median ")
    w("|slope-1| goes 0.018 to 1.034 under A while C's is 0.269. The column measures the ")
    w("calibration DESIGN, not the protocol (2026-08-03 red team).\n\n")
    w("Two further cautions. The slope is a heavy-tailed ratio whose denominator is the ")
    w("prediction's variance, so the MEAN across folds is not robust and the median column is ")
    w("the one to read: aggregated by median, protocol C's slopes come out near 1.0 and the ")
    w("'under-confident under C' reading disappears too. And it is reported here as a ")
    w("diagnostic that the spec asks for, not as evidence for any claim in this report.\n\n")
    w("| ticker | clock | h | rung | protocol | IC mean | IC median | IC worst | calib mean | "
      "calib median | calib worst | folds | n_test median |\n")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for ticker in tickers:
        for clock in clocks:
            for h in horizons:
                for name in ladder_names:
                    for pr in PROTOCOLS:
                        c = results.get((ticker, clock, h, pr), {}).get(name)
                        if not c:
                            continue
                        cal = c.get("calibration_slope", {})
                        w(f"| {ticker} | {clock} | {h} | {name} | {pr} | "
                          f"{_fmt(c['ic']['mean'])} | {_fmt(c['ic']['median'])} | "
                          f"{_fmt(c['ic']['worst'])} | {_fmt(cal.get('mean'), 4)} | "
                          f"{_fmt(cal.get('median'), 4)} | {_fmt(cal.get('worst'), 4)} | "
                          f"{c['n_folds_scored']} | {_fmt(c['n_test']['median'], 0)} |\n")
    w("\n")

    w("## Classification against the base rate\n\n")
    w("Balanced accuracy, MCC and AUC, printed next to the majority-class share ")
    w("(EVAL_PROTOCOL.md sec. 2). At 1s most labels are flat, so plain accuracy is ")
    w("meaningless (FEATURES.md sec. 7).\n\n")
    w("**Read balanced accuracy and MCC on the unfitted rungs as NULL, not as a result.** ")
    w("A rescaled sign predictor emits an amplitude far below the 0.5-tick deadband, so its ")
    w("predicted class is always 0, balanced accuracy is exactly 1/3 (chance on three ")
    w("classes) and MCC is NaN. AUC is the column that carries directional information for ")
    w("those rungs, because it ranks the raw prediction and never thresholds it.\n\n")
    w("AUC is computed on the DIRECTIONAL subset only -- rows the deadband calls up or down, ")
    w("flats excluded -- so 0.5 is chance regardless of how much tie mass the cell has.\n\n")
    w("**`AUC raw` is the one to quote.** The through-origin rescale is rank-preserving only ")
    w("UP TO SIGN, and the fitted scale comes out negative in a small number of protocol-C ")
    w("folds and never under A; a negative scale flips the ranking and sends AUC to 1-AUC. On ")
    w("the unfitted rungs that channel alone manufactured the entire apparent A-C gap -- ")
    w("L1m and L2's gaps are NEGATIVE on the raw score. `AUC raw` scores the rung's raw ")
    w("output and is immune to it by construction (2026-08-03 red team). `neg` counts folds ")
    w("with a negative fitted scale.\n\n")
    w("Every AUC column carries its WORST FOLD, because a median AUC of 0.71 built from folds "
      "of {0.79, 0.75, 0.63, 0.48, 0.71} is a story about the early session (RULES.md rule 4 "
      "-- the raw column shipped without one for a day).\n\n")
    w("| ticker | clock | h | rung | protocol | bal. acc | MCC | AUC | AUC worst | AUC raw | "
      "AUC raw worst | neg | base rate |\n")
    w("|---|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for ticker in tickers:
        for clock in clocks:
            for h in horizons:
                for name in ladder_names:
                    for pr in PROTOCOLS:
                        c = results.get((ticker, clock, h, pr), {}).get(name)
                        if not c:
                            continue
                        auc = c.get("auc", {})
                        raw = c.get("auc_raw", {})
                        w(f"| {ticker} | {clock} | {h} | {name} | {pr} | "
                          f"{_fmt(c['balanced_accuracy']['mean'], 4)} | "
                          f"{_fmt(c['mcc']['mean'], 4)} | "
                          f"{_fmt(auc.get('mean'), 4)} | {_fmt(auc.get('worst'), 4)} | "
                          f"{_fmt(raw.get('mean'), 4)} | {_fmt(raw.get('worst'), 4)} | "
                          f"{c.get('n_negative_scale', 0)} | "
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
    w("- **L1 is NOT the control -- the FROZEN row is.** L1 was the intended control and it ")
    w("does not work: its one-parameter rescale is still fitted per fold, so its `A-C` gap ")
    w("mixes a partition artifact with a refit effect, and the floor scales with prediction ")
    w("variance, so L1's small gap does not bound L4's. Like-for-like floors are +0.0043 (L1 ")
    w("magnitude), +0.0112 (L3) and +0.0192 (L4), medians across 16 cells. Read every gap ")
    w("against the `floor` column of the headline table, never against the L1 row ")
    w("(2026-08-02 red team, second pass).\n")
    w("- **The calibration rule is NOT a second channel.** An earlier note here said roughly ")
    w("half of L4's gap was the protocol-dependent calibration holdout and told the reader to ")
    w("quote `A' - C`. Both are withdrawn. With the model's fit rows held fixed (the DECOUPLED ")
    w("control), the rule moves raw AUC by exactly 0.000000 in all 96 rows -- it cannot move a ")
    w("rank statistic on a fixed model. `A' - C` was low because it stripped A's model of the ")
    w("day's tail. **Quote `A_rand - C_pos` from the decoupled raw-AUC table.**\n")
    w("- **On AUC, quote the raw column.** The rescale is rank-preserving only up to sign and ")
    w("the fitted scale goes negative in 27 of 400 protocol-C folds, which sends AUC to ")
    w("1-AUC. That channel alone produced the unfitted rungs' apparent gaps.\n")
    w("- **The 1s label is not one-second alpha.** The phase-1 adversary measured that OFI's ")
    w("entire content decays inside 100 ms (rho 0.458 over the first 100 ms against 0.027 ")
    w("over the remaining 900 ms). Read these as short-horizon numbers.\n")
    w("- **Fold row counts vary by design.** Test blocks are equal in DURATION, not in ")
    w("sample count, and rungs using trade features lose rows where a window holds no ")
    w("trades. Both are reported rather than absorbed.\n")


if __name__ == "__main__":
    raise SystemExit(main())
