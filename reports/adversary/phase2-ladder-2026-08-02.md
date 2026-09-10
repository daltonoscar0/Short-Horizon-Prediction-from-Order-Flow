VERDICT: SURVIVED WITH CAVEATS

# Phase 2 red team, the three-protocol optimism table

Target: `reports/ladder.md`. Claim under review: the `A-C` column measures evaluation
optimism from leaky cross-validation, and `B-C` isolates the label-overlap component.

Everything below was produced by running code, not by reading prose. Scripts live in the
session scratchpad; each finding names the exact check. Nothing in `src/`, `tests/` or
`RESULTS.md` was modified.

Bottom line: **the sign of the result is real and survives every control I could build, a
shuffled k-fold protocol is massively optimistic for L3/L4, including on a scale-free metric.
The magnitudes in the published table are not what the table says they are.** Roughly half to
95% of most reported gaps is attributable to two things that are not leakage: an in-sample
through-origin scale fit, and an asymmetry in how the five folds are averaged. And the
column labels in the decomposition are attached to the wrong mechanisms.

---

## 1. Protocols differ in more than the split, PARTLY CONFIRMED

`run_cell` is one code path for A, B and C: same `common_usable_mask`, same
`build_ladder(seed)`, same `score_fold`, same metric functions. Only `make_folds` differs.
Seed, hyperparameters and metric code are genuinely identical. Verified by reading
`src/edge/harness/run.py:113-140` and by reproducing two cells of the committed table
bit-for-bit (`scratchpad/repro.py`; INTC/calendar/1s and INTC/event/1s match `ladder.md`
lines 19-24 and 43-48 to all five printed digits).

But two protocol differences beyond "the split" survive:

**(a) Training-set size.** `ladder.md` line 9 states: "Folds: 5 per protocol, identical count
across protocols so the gap is not partly a training-set-size difference." Equal fold count
does not equalise train size. Measured, INTC/calendar/1s: A `n_train` median **162,706**;
C **89,755**, with C's folds running 18,611 / 54,580 / 89,755 / 124,806 / 160,135. A trains
on 1.8x C's median and 8.7x C's first fold. The sentence is a non-sequitur.

I discharged it empirically rather than rhetorically. Protocol **Dmatch** (test blocks
identical to C, training drawn from the whole day but *randomly subsampled to exactly C's
per-fold train size*) reproduces protocol D to within 0.01 R2 in every cell
(`scratchpad/decomp.py`). Train size is not the driver. Good news for the claim.

**(b) Fold aggregation is a different estimand under A than under C.** This one is real and
it is not in the report. A's five folds are random subsets of the same universe, so each
fold's R2-vs-zero is approximately the SS-weighted pooled R2. C's five folds are contiguous
time segments of equal *duration* and very unequal `sum(y^2)` (measured shares for
INTC/calendar/1s: 0.188 / 0.260 / 0.158 / 0.108 / 0.285), so C's fold mean is an *unweighted*
average of segment R2s. Mean-over-folds therefore estimates two different quantities.
See item 6 for the size.

## 2. Row alignment, CLEAN

Re-derived from the raw CSV with no `edge` import (`scratchpad`, inline): for the first 60,000
INTC messages, every cancel/delete (`event in {2,3}`) whose price matched a level in orderbook
row `k-1` was checked. 27,419 matched events; the displayed size decrement appears in row `k`
in **27,273 (99.47%)** and in row `k+1` in **0**. LOBSTER convention "row k is the state after
message k" holds, as `EVAL_PROTOCOL.md` sec. 4 requires. The residual 0.53% are events where
the level was also touched by a same-timestamp neighbour.

## 3. Lookahead in features, CLEAN, but the ladder-level shift probe is weak

`grep` over `src/edge/features/` finds no `shift`, no centred window, no `.iloc[i+`. Every
window is a `searchsorted(..., side="right") - 1` plus a prefix-sum difference over `[t-w, t]`.
`normalize.rolling_zscore` uses the half-open `[t-window, t)` baseline and excludes `x[t]`
itself. Fitted feature statistics (integrated-OFI PCA loadings, microprice states) go through
`windows.refit_blocks`, fit on frame rows strictly before each 30-min block.

`EVAL_PROTOCOL.md` sec. 5's time-shift probe does **not** behave as specified at the ladder
level (`scratchpad/probes.py`, `probes2.py`). Shifting every feature forward by one event on
INTC/event/1s under protocol C: L1 R2 0.0719 -> 0.0543, L3 0.0956 -> 0.0711, L4 0.1239 ->
0.1338. Performance falls for three of five rungs. A wider sweep shows L4 rising monotonically
(+5: 0.234, +20: 0.414) before collapsing at +100, so the pipeline is not mis-signed, but a
one-event peek on the event clock is a sub-millisecond peek against a 1s label, and the
"performance should jump sharply" check is not doing work at this granularity. Phase 1 ran
this probe on raw correlations, not on the ladder, and the ladder version is not in `ladder.md`.

## 4. Scaler fit on the full sample, CLEAN for the standardiser, BROKEN for the scale

`grep -rn "\.fit(" src/` returns exactly two call sites, both inside a fold
(`harness/run.py:96`, `costs/run.py:177`). `L3Ridge._fit_raw` computes `mu_`/`sigma_` on the
fold's training rows only. No global scaler.

The problem is the *other* fitted parameter. `models.py` maps every rung's raw score into
label units with `scale = sum(s*y)/sum(s*s)`, **fitted on the same rows the model was fitted
on**. For a GBM that is an in-sample fit against near-perfectly-fit predictions, so it
returns ~no shrinkage; out of sample the amplitude is far too large and R2-vs-zero explodes
negative. Under A that never bites, because A's "out-of-sample" rows are the temporal
neighbours of its training rows.

I replaced it with a strictly causal alternative, model fit on the first 80% of each fold's
training rows, scale fit on the held-out last 20% (random 20% under A, which is A's own logic)
- and changed nothing else (`scratchpad/innercal.py`):

| cell | rung | shipped C | shipped A-C | causal-scale C | causal-scale A-C |
|---|---|---|---|---|---|
| AMZN cal 5s | L3 | -0.3423 | **0.3637** | +0.0037 | **0.0174** |
| AMZN evt 10s | L3 | -0.5502 | **0.6346** | +0.0060 | **0.0772** |
| INTC cal 10s | L3 | -0.0304 | **0.1112** | +0.0413 | **0.0396** |
| INTC cal 10s | L4 | -0.5523 | **1.0707** | +0.0151 | **0.4920** |
| INTC cal 1s | L4 | -0.2988 | **0.4898** | +0.0213 | **0.1620** |
| AMZN cal 5s | L4 | -0.2951 | **0.6601** | +0.0061 | **0.3484** |
| AMZN evt 10s | L4 | -0.0716 | **0.4099** | +0.0460 | **0.2819** |
| INTC evt 1s | L3 | +0.0956 | 0.1503 | +0.0712 | 0.1747 |

Every large negative protocol-C R2 in the report is an artifact of the in-sample scale, not a
finding. This is the answer to the open question about the L4 GBM: **the C-protocol
negative R2 is a calibration artifact.** The direction survives (C's IC is positive
throughout), the amplitude does not. Between 20% and 95% of the reported gap disappears with
a fix that touches neither the split nor the model.

## 5. R2 against the test mean, CLEAN

`metrics.r2_vs_zero` is `1 - sum((y-p)^2)/sum(y^2)`, read line by line. `r2_vs_test_mean`
exists but the headline table takes `c["r2"]["mean"]` (`run.py:235`). One prose defect:
`ladder.md` line 11 says "The `r2_test_mean` column is reported alongside", there is no such
column anywhere in the file.

Arithmetic sanity: R2-vs-zero is unbounded below and a pure amplitude error of factor `k`
costs `(k^2-1)*(1-rho^2)`-ish, so yes, the reported magnitudes (up to -2.97 on a single fold)
are arithmetically reachable with no leakage whatsoever. Item 4 shows they in fact are.

## 6. The L1 control, THE MEASUREMENT IS NOT ZERO, AND THE REPORT'S READING OF IT IS WRONG

This is the most important check in the project and it fails on its own terms.

`ladder.md` line 807: "L1 ... has no fitted parameters beyond a one-parameter rescale, so its
`A-C` gap is a floor on what the harness itself contributes." The parenthetical is doing all
the work: that one rescale is fitted, and it is the whole story.

I built the control the report should have built: a predictor that is **never refit**, frozen
sign(qi) with a globally fixed scale, a ridge fit once on the whole day, a GBM fit once on the
whole day, and scored the identical array under A, B and C (`scratchpad/fixedpred.py`,
`fixedic.py`). Any difference is 100% fold-partition artifact, since the model is byte
identical.

```
INTC calendar 1s   frozen sign(qi)   A=+0.01246 B=+0.01110 C=+0.01110  A-C = +0.00136
INTC calendar 1s   frozen GBM        A=+0.25089 B=+0.23474 C=+0.23474  A-C = +0.01615
INTC event 1s      frozen sign(qi)   A=+0.08528 B=+0.07704 C=+0.07704  A-C = +0.00824
INTC calendar 10s  frozen GBM        A=+0.54618 B=+0.51908 C=+0.51908  A-C = +0.02710
```

and on the scale-free metric the artifact reproduces the report's L1 row **exactly**:

```
frozen sign(qi) IC   INTC cal 1s  A=0.11812 C=0.11636  A-C = +0.00177
ladder.md L1    IC   INTC cal 1s  A=0.11812 C=0.11636  A-C = +0.00176   (lines 123,125)
frozen sign(qi) IC   INTC evt 1s  A=0.31371 C=0.29794  A-C = +0.01577
ladder.md L1    IC   INTC evt 1s  A=0.31371 C=0.29794  A-C = +0.01577   (lines 195,197)
```

The mechanism: A's fold mean is the `sum(y^2)`-weighted average of the time segments, C's is
the unweighted average. I verified this directly, the SS-weighted average of C's per-fold R2
equals A's fold mean to five decimals in every cell tested (`fixedpred.py`, column
"C SS-weighted").

Consequences:
- The L1, L1m and L2 rows of the optimism table contain **no leakage signal at all**. Their
  reported A-C (0.0005 to 0.02) is fully accounted for by the partition artifact plus the
  in-sample scale refit. Under an oracle scale the L1/L1m A-C gap goes to zero or negative in
  every cell tested (`scratchpad/oracle_scale.py`).
- `tests/test_splits.py::test_every_protocol_sees_the_same_sample_universe` asserts the *union*
  of test rows is identical across protocols. True, and insufficient: it says nothing about
  the aggregation, which is where the artifact lives. This is the test I had warned myself
  not to trust, and it does not cover the failure.
- The floor is small in absolute R2 (~0.002-0.03), so the L4 headline is not destroyed by it.
  The claim survives; the stated justification for it does not.

## 7. Base-rate artifact, NOT LOAD-BEARING

Base rate 0.9698 at INTC/calendar/1s. Balanced accuracy is 0.3333 (= chance on three classes)
for L0/L1/L1m/L2 under every protocol, because the rescaled continuous prediction never
exceeds the 0.5-tick deadband, so the classifier predicts "flat" always. The classification
table is therefore near-vacuous, which `DECISIONS.md` 2026-07-31 already concedes. No headline
claim rests on accuracy. The one non-trivial row (INTC/cal/1s L4: bal.acc 0.4178 / MCC 0.2919
under A vs 0.3427 / 0.0358 under C) is consistent with the R2 and IC story.

## 8. Deadband tuning, RULE, NOT HINDSIGHT (unverifiable from dates alone)

`ALPHA_TICKS = 0.5` is pre-registered in `docs/spec/FEATURES.md` sec. 7 and frozen in
`RESULTS.md` on 2026-07-31 with the sweep explicitly marked "reported for sensitivity, frozen
BY RULE". Every commit in the repo is dated 2026-07-30/31, so commit dates cannot discriminate
a priori from post hoc for anything in phase 1-2; the spec file predates the work and says
0.5, which is the strongest available evidence and I accept it. Defect: `run.py:45`
hard-codes `ALPHA_TICKS = 0.5` with the comment "imported convention from
edge.features.labels", it is a copy, not an import, so the two can silently diverge.

## 9. Trial count, UNDERSTATED, AND TWO FROZEN CHOICES ARE MISSING

`RESULTS.md` trial 5 logs "2 tickers x 2 clocks x 4 horizons x 3 protocols x 6 rungs = 288
fits". 288 is the configuration count; the fit count is 288 x 5 folds = 1,440. More
substantively:

- `RESULTS.md`'s frozen-decisions table has **embargo E blank** ("Phase 2") even though the
  shipped table uses `E = max(60s, 10h)`, and has **no row for k, protocol A's fold count**.
  `RULES.md` requires both.
- `EVAL_PROTOCOL.md` sec. 1 requires an E-sensitivity table. `run.py` implements
  `--embargo-sweep`; `ladder.md` has no such section, so it was never run.
- The `RESULTS.md` headline ladder table is still empty. No phase-2 number has been appended
  to the source-of-truth file; the whole result lives only in `reports/ladder.md`.

## 10. Pooled-number laundering, CONFIRMED FOR THE L3 ROWS

I recomputed all 80 (ticker, clock, horizon, rung) A and C cells per fold
(`scratchpad/perfold_all.py`, 800 fits, ~20 min).

- **45 of 80 rows lose more than half their reported A-C gap when fold 0 is dropped.**
- **30 of 80 lose more than half when the fold median replaces the fold mean.**
- Sum of gaps across the table: 14.11 by mean, 10.09 by median, 9.41 excluding fold 0.

The L3 rows are the worst. Three examples, mean-gap -> median-gap -> ex-fold-0 gap:

```
AMZN event 10s   L3 ridge   0.6346 -> 0.0160 -> 0.0300    (C fold 0 R2 = -2.9671)
AMZN calendar 10s L3 ridge  0.4183 -> 0.0232 -> 0.0190    (C fold 0 R2 = -1.9964)
AMZN event 1s    L3 ridge   0.4135 -> 0.0405 -> 0.0123    (C fold 0 R2 = -1.8550)
```

These are one-fold stories: fold 0 trains on ~1/6 of the day and its ridge is amplitude-
catastrophic (item 4). `ladder.md` does print the per-fold C detail at line 704, to its
credit, but the headline optimism table shows the mean only, and the mean is the number
that will be quoted.

The **L4 GBM rows are robust to this**: INTC/cal/10s 1.0707 -> 1.1165 (median) -> 1.0480
(ex-fold-0). The largest numbers in the table are not one-fold artifacts. They are
calibration artifacts (item 4).

---

## Additional experiments

**Labels shuffled within fold** (`scratchpad/probes.py`, INTC/event/1s). Every metric
collapses under all three protocols: IC in [-0.009, +0.011] for every rung, R2 in
[-0.010, +0.004] for L1/L1m/L2/L3. L4's R2 goes to -0.19 (the amplitude effect of item 4
again, on pure noise) with IC 0.000. PASS.

**Protocol A gap vs k** (`probes.py`, INTC/event/1s):

```
k= 2   L1 0.0012  L1m 0.0007  L2 0.0058  L3 0.0722  L4 0.4676
k= 3   L1 0.0067  L1m 0.0002  L2 0.0097  L3 0.2541  L4 0.4853
k= 5   L1 0.0133  L1m 0.0139  L2 0.0212  L3 0.1503  L4 0.4864
k=10   L1 0.0140  L1m 0.0409  L2 0.0381  L3 4.3915  L4 0.4601
```

L4's gap is flat in k, which is what the mechanism predicts (a test row has adjacent training
rows at any k), I do not read flatness as evidence of a broken splitter here. L3's is wildly
non-monotone, and I traced it: at k=10 C's fold 0 trains on 1,486 rows and scores
**R2 = -42.86**, which alone moves the mean gap from ~0.15 to 4.39. Again item 4, again fold 0.
Note also that A's eligible test universe is a function of k (`splits.py:116`, `edges[1]`
moves with `n_folds`), so a k-sweep is not a clean one-factor experiment as implemented.

**Constant predictor under all three protocols.** L0 gives exactly 0.00000 under A, B and C by
construction, which is uninformative. The informative version, a frozen *non-trivial*
predictor, is item 6, and its gaps do **not** vanish.

**Protocol D, the control the harness is missing.** Test blocks identical to C's; training =
everything outside the test block, past *and* future, no purge. D isolates "the model saw the
future" while holding the test universe, the fold shapes and the aggregation fixed. Combined
with the causal scale of item 4 (`scratchpad/final_decomp.py`):

```
INTC event 1s   L4   R2   A=+0.6009  D=+0.2062  C=+0.1912  | A-D=+0.3947  D-C=+0.0150
INTC event 1s   L4   IC   A=+0.6973  D=+0.4691  C=+0.4441  | A-D=+0.2282  D-C=+0.0249
INTC cal 1s     L4   IC   A=+0.2535  D=+0.1399  C=+0.1316  | A-D=+0.1136  D-C=+0.0083
AMZN cal 5s     L4   IC   A=+0.4940  D=+0.1325  C=+0.0867  | A-D=+0.3615  D-C=+0.0458
```

Handing the model the entire future in disjoint blocks buys it almost nothing. Essentially
all of A's advantage comes from training rows that are *temporally adjacent* to test rows.

That is a direct contradiction of the report's own column labels. `ladder.md` line 7 and
`harness/__init__.py:13-16` say `A-B` is "training on the future outright" and `B-C` is "the
label-overlap component". Measured: training on the future outright is worth ~0.01-0.05;
label overlap, adjacent rows whose `[t, t+h]` windows coincide, is worth 0.11-0.36 and sits
**inside the A-B column**, not the B-C column. `B-C` is near zero for an arithmetic reason:
under expanding walk-forward the purge+embargo removes only the last `h+E` of a multi-hour
training set, 578 of 90,333 rows (0.64%) at INTC/calendar/1s, 274 of 25,345 (1.1%) at
INTC/event/1s. A column that can only ever touch 1% of the training set is not a measurement
of label overlap in general; it is a measurement of label overlap *at a walk-forward
boundary*, where it is negligible by construction. The report's framing invites the reader to
conclude label overlap contributes ~nothing. The opposite is true.

(The embargo *is* applied where claimed, 578 dropped rows against a predicted 61s x ~10
stamps/s = 610, so this is a framing failure, not an implementation failure.)

## What actually survives

On the scale-free metric, with the causal calibration, with the aggregation artifact
subtracted, protocol A remains substantially optimistic for the high-capacity rungs:
IC A-C of +0.12 (INTC cal 1s), +0.25 (INTC evt 1s), +0.41 (AMZN cal 5s) for L4, against a
frozen-predictor artifact floor of +0.002 to +0.016. That is a real, large, correctly-signed
result, and it is the paper. It is roughly a third to a half of the number currently printed.

## Verdict rationale

Not KILLED: no arithmetic is wrong, the table reproduces from the code, alignment and
causality are clean, the label-shuffle control passes, and the qualitative claim holds on a
metric immune to every artifact I found. SURVIVED WITH CAVEATS because the published
magnitudes conflate at least three mechanisms, two of which are not leakage, and because the
decomposition's two columns are labelled with the wrong mechanisms.

Minimum changes before this number is quoted anywhere:
1. Fit the through-origin scale on a held-out slice of the training fold. Report both.
2. Report the frozen-predictor gap as the artifact floor, per cell, next to A-C.
3. Report the fold median and the ex-fold-0 gap next to the mean in the headline table.
4. Add protocol D, or stop describing A-B as "training on the future" and B-C as "the
   label-overlap component".
5. Fill in embargo E and k in the `RESULTS.md` frozen-decisions table; run the E sweep;
   append the phase-2 numbers to `RESULTS.md`.
