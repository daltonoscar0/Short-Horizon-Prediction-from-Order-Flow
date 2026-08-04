# Edge — one page

## Problem

Short-horizon limit-order-book prediction has a well-known evaluation problem: labels at
horizon *h* overlap across neighbouring samples, and order flow is long-memory autocorrelated.
Shuffled k-fold on that data is not slightly optimistic — it trains on the future. Plenty of
published results use it. **How much of the reported performance is the protocol rather than
the signal?**

## Method

Build the classic microstructure baselines properly, then score the whole ladder under three
protocols with identical features, rows, folds and seed:

- **A** shuffled k-fold — what much of the literature does
- **B** walk-forward, no purge, no embargo
- **C** purged walk-forward with embargo — correct
- **D** *control*: C's test blocks, trained on everything outside them, past **and** future

`A−C` is total optimism. `A−D` is the temporal **adjacency** of training rows; `D−C` is seeing
the future in disjoint blocks. That split is what makes the gap interpretable.

Ladder: **L0** zero → **L1** `sign(queue imbalance)` (no parameters) → **L2** OLS on best-level
OFI (Cont–Kukanov–Stoikov) → **L3** ridge on 28 handcrafted features → **L4** gradient boosting.
Two tick regimes (INTC large-tick, AMZN small-tick), two clocks (100ms calendar, 10-event),
four horizons, five folds. LOBSTER level-10, 2012-06-21.

The cross-validation harness is **hand-written** — no `sklearn.model_selection` anywhere. So
are the metrics, the ridge solve, the Newey–West HAC, the block bootstrap and the deflated
Sharpe. The harness is the deliverable; the models are the detail.

## Result

**Optimism scales with model capacity, and it is adjacency — not the future.** Median R² vs
zero across 96 cells:

| rung | A − C | *floor* | A − D (adjacency) | D − C (future) | C (honest) | **split-only, raw AUC** |
|---|---|---|---|---|---|---|
| L1 `sign(queue imbalance)` | +0.0043 | *+0.0043* | +0.0126 | −0.0085 | +0.0091 | **+0.0023** (= its floor) |
| L3 ridge | +0.0587 | *+0.0112* | +0.0296 | +0.0231 | +0.0266 | **+0.0409** |
| L4 GBM | **+0.3173** | *+0.0192* | **+0.2964** | +0.0067 | +0.0273 | **+0.1913** |

1. **`A − C` has two channels and the control has to see both.** The floor is a frozen model
   scored under every protocol — but the harness also fits an output *scale* per fold, under a
   rule that differs by protocol (random 20% under A, last 20% under walk-forward), and A's is
   exchangeable with A's own test fold. Freeze the scale too and the control goes blind to the
   larger channel; refit it and the floors go up 2–6×. Then a *clean* control — raw model fitted
   on the same seeded random 80% in every arm, only the scale rows moving — shows the rule is
   worth **exactly zero** on a rank metric, because a rank metric never sees the scale. The
   split-only gap is **+0.1913** raw AUC for the GBM — **8.1×** its worst-case floor as a median
   of per-cell ratios, worst cell 3.4× — positive in 16 of 16 cells and in **80 of 80 folds**.
   The parameter-free rung's gap is **numerically identical to its own frozen-predictor floor in
   all 16 cells**, which is the strongest form of "it measured nothing" available.
2. **93% of the GBM's gap is temporal adjacency, not the future.** Protocol D hands the model
   the whole future in disjoint blocks and that buys it +0.007. Label overlap is the dominant
   mechanism — and it is invisible in `B − C`, which can only touch ~1% of an expanding
   training set.
3. **The honest ladder does not invert; it flattens.** L4 beats L1 in 13 of 16 cells, but its
   honest R² is **12.6× smaller** than shuffled k-fold advertises for the same model, and on
   *both* scale-free metrics — rank correlation and AUC — the ridge outranks the GBM. The
   shuffled protocol does not just inflate, it reorders the ladder, and which rung wins depends
   on whether you are paid for the amplitude or for the ordering.
4. **Nothing pays.** L1's signal is real (its gross-PnL CI excludes a constant control) but
   covers ~1% of its trading cost — 0.011 ticks earned against ~0.92 ticks of half-spread plus
   fee. **Perfect foresight of the 1s mid move still loses $458** under aggressive fills, which
   makes the cost model, not the predictor, the binding constraint.

## What I'd do with more data

**Days first, then names** — one axis at a time. The whole study is one session, so the
same-day-vs-next-day leak check cannot be run at all, and finding 2 is the one most likely to
change: overnight gaps change the geometry of label overlap entirely, and `B−C` could stop
being negligible once folds are cut on day boundaries. After that: the order-flow LM arm as L5/L6 against the
three controls the spec already fixes (untrained weights, lagged-raw, random-projection), which
is what the harness was built to make possible.

## What I'd want you to poke at

- Eleven red-team passes are in `reports/adversary/`, run read-only against the artifacts, and
  **three of them killed a headline this project had already written down**. The phase-3 pass
  showed the original break-even column was model-invariant — a constant always-long predictor
  scored *inside* the ladder's range on it. The phase-2 passes showed that an in-sample
  calibration, not leakage, produced every large negative protocol-C R² in the first draft;
  the corrected gap is roughly a third of what was first published and the "ladder inverts"
  claim was withdrawn entirely. The superseded columns are still in `reports/ladder.md`.
- **The floor was wrong twice, and both times a red team found it, not me.** The second phase-2
  pass killed the first version of the fix — the floor was measured only at L1's magnitude,
  which does not bound L4's. The third killed the second: it froze the calibration scale as well
  as the model, so it could not see the calibration channel at all. Its replacement was wrong
  too: it changed the calibration rule but, because the model is fitted on the complement of the
  calibration holdout, it also changed the model's training window — so its "half the gap is the
  rule" finding was itself confounded, and the clean control shows the rule is worth **zero**.
  A control that is not exposed to everything the measured thing is exposed to is not a control,
  and a control that moves something other than the one thing it names is not one either. That
  is the most portable lesson in this repo.
- Every configuration tried is in the trial log, including the two discarded harness
  configurations that invalidated their own runs. The deflated Sharpe uses n = 2,536, which
  deliberately over-counts, including superseded runs.
- `DECISIONS.md` records what was chosen without asking and why — including that the phase-3
  trading deadband was set *after* seeing protocol-C output, which is disclosed rather than
  presented as frozen.
