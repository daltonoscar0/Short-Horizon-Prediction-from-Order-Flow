VERDICT: KILLED

# Phase 2 red team (third pass) — the 2026-08-03 metrics

Reviewed: `reports/ladder.md`, `reports/artifact_floor.md`, `RESULTS.md` (2026-08-03 section,
leakage-diagnostic rows, trials 18-19), `FINDINGS.md` §1.1a/§1.3, `README.md`, `METHOD.md` §6,
`src/edge/harness/{run,floors,metrics,models,splits}.py`, `tests/test_score_fold.py`.
Read-only on `src/`, `tests/` and every `.md` except this file. Everything below was run with
`PYTHONPATH=src /opt/anaconda3/bin/python3`; scripts are in `scratch/`.

**The AUC headline survives every attack I made on it. The calibration-slope finding does not,
and the floor construction that is supposed to police both is measured with the responsible
mechanism held fixed.**

---

## The killing check

`score_fold` chooses the calibration holdout **by protocol**:

```python
if fold.protocol == "A":
    rng = np.random.default_rng(1000 + fold.fold_id)
    hold[rng.choice(tr.size, size=max(1, tr.size // 5), replace=False)] = True
else:
    hold[int(tr.size * 0.8):] = True
```

Under A the through-origin scale is fitted on a random 20% of the training rows, which is
*exchangeable with A's random test fold*. Under B/C/D it is fitted on the last 20% in time,
which is *not* exchangeable with a later test block. The reported calibration slope is
`beta_test(y on raw) / scale`, so under A it is forced toward 1.0 for **any** predictor whose
`beta` is stable between two random subsets of the same day — skill or no skill, leakage or no
leakage.

Two controls, both of which the project already had the machinery for:

**Control 1 — freeze the model, refit only the scale.** One fit on the whole day, byte-identical
under every protocol, so there is zero differential exposure. Only the per-fold scale is refitted
with each protocol's own holdout rule (`scratch/calib_probe.py`, `scratch/scalefloor.py`):

| rung, frozen model | slope under A | slope under C | published A | published C |
|---|---|---|---|---|
| L1 magnitude | **1.0319** | **1.3905** | 1.0319 | 1.3904 |
| L3 magnitude | **0.9784** | 1.1219 | 0.9775 | 1.6839 |
| L4 magnitude | **1.0043** | 0.9998 | 1.0026 | 1.2700 |

A predictor that cannot leak reproduces "protocol A is calibrated to ≈1.0" to three decimals.
For L1 — which has no fitted model at all — it reproduces *both* published columns exactly,
because for L1 the control **is** L1.

**Control 2 — hold the calibration rule constant.** Re-run protocol A with C's last-20% holdout.
The split, the folds, the seed and the model are unchanged; only the calibration rule moves
(`scratch/holdoutrule.py`, all 16 cells):

| rung | median \|slope−1\| under A, as shipped | under A, with C's rule | under C |
|---|---|---|---|
| L3 | 0.031 | 0.151 | 1.583 |
| L4 | **0.018** | **1.034** | **0.269** |

For L4 the claim **reverses**: with one calibration rule, protocol A is four times further from
calibrated than protocol C. `RESULTS.md`'s leakage-diagnostic row ("**CONFIRMS the gap,
independently of R².** … A model is calibrated on rows it has effectively seen"), the body text
("a leakage signature in its own right and an independent one") and `FINDINGS.md` §1.3's fourth
bullet are all reading a protocol-dependent calibration rule as evidence about the model. The
previous pass flagged this asymmetry as newly introduced by its own fix (its item 1); it has been
promoted to a finding instead of controlled for.

**And the floor that was supposed to catch this cannot.** `frozen_floor` calls
`rung.fit(X[fit_rows], y[fit_rows])` with no `scale_holdout` and scores one `frozen_pred` under
every protocol, so the scale is frozen too. The published `A-C calib floor` column is therefore
±0.01 where the real artifact is 0.36 at L1's magnitude — a control that is vacuous by
construction, presented as a control. This is the same class of error the 2026-08-02 pass killed
the report over.

**Two further consequences of the same hole.**

1. *The R² floor is understated 2–6×.* Frozen model, per-fold scale refit — the like-for-like
   floor — is L1 **+0.0043**, L3 **+0.0112**, L4 **+0.0192** (median of 16 cells) against the
   published +0.0020 / +0.0020 / +0.0153. The L1 row reproduces the published L1 A−C gap
   *exactly, cell by cell* (+0.00302, +0.00602, +0.00816, +0.00536, +0.01815, …), confirming
   that L1 carries no leakage signal at all — by the mechanism the report does not name.
2. *The AUC floor's "worst single cell +0.0053" is not a bound.* With the same construction the
   worst cell is **+0.0645** (AMZN event 5s at L3's magnitude), and in 4 of 16 cells L3's
   published AUC gap is *smaller* than the like-for-like floor at its own magnitude
   (AMZN event 5s 0.0413 vs 0.0645; event 10s 0.0495 vs 0.0593; calendar 2s 0.0280 vs 0.0461;
   calendar 5s 0.0326 vs 0.0351). The channel is the through-origin scale going **negative** —
   27 of 400 protocol-C folds, 0 of 400 protocol-A folds — which flips the ranking and sends
   AUC to 1−AUC. Recomputing every AUC on the **raw** score, immune to the scale sign, the
   published progression 0.003 → 0.004 → 0.008 → 0.042 → 0.174 becomes
   +0.0022 → **−0.0018** → **−0.0002** → +0.0409 → +0.1740: two of its five entries are
   entirely a calibration-sign artifact.

None of this overturns the headline. L3's and L4's AUC gaps are 97% and 100% intact on the raw
score, and with the calibration rule held constant L4's median R² gap is still +0.1379 against a
like-for-like floor of +0.0192. What is dead is the calibration-slope finding, the published
floors as bounds, and the L1m/L2 entries of the AUC ladder.

---

## What I tried and what happened (all ten)

1. **Protocols differ in more than the split — CONFIRMED, and it is material.** `run_cell`
   computes the feature matrix, `common_usable_mask`, the rung list and the metric code once per
   cell and loops protocols, so features/seed/hyperparameters/metrics are protocol-independent
   (read, not taken on trust). But `score_fold` branches on `fold.protocol` for the calibration
   holdout. Holding that rule constant removes **57%** of L4's median R² A−C gap
   (+0.3174 → +0.1379) and **21%** of its AUC gap (+0.1740 → +0.1371). Defensible as part of
   what "shuffled k-fold" means; not disclosed anywhere a gap is quoted, and not covered by any
   published floor.
2. **Row alignment — CLEAN, verified independently.** From the raw LOBSTER CSVs with no
   `edge.*` import (`scratch/align.py`): on INTC's first 20,000 messages, 9,805 type-1 adds show
   the exact `+size` delta under "orderbook row *k* is the state **after** message *k*"
   (9,805/9,805); the off-by-one convention holds on 1,248/9,805.
3. **Lookahead — none on the harness path.** No `L3_FEATURES` column is a `_sadj` rolling
   z-score, so `normalize.py` (the only `min_periods`/rolling code) is not on the path.
   Forward-shift probe through the identical code: features from *t*+1 event take INTC cal 1s
   L3 under C from 0.0215 to 0.0882 and L4 from 0.0213 to 0.0722. The probe fires; the shipped
   columns do not.
4. **Scaler fit on the full sample — no.** `L3Ridge._fit_raw` fits `mu_`/`sigma_` inside the
   fold on `X[~hold]` only; the only other `.fit(` sites are `harness/run.py` and
   `costs/run.py`. The *other* fitted scalar, `Rung.scale_`, is now causal — but its holdout is
   protocol-dependent, which is the killing check.
5. **R² against the test mean — no.** `r2_vs_zero` is `1 - SSE/sum(y*y)`; `r2_vs_test_mean` is
   never a headline. L0 scores exactly 0.00000 in all 96 headline rows, which is the arithmetic
   proof.
6. **The L1 control — recomputed, and the report's own conclusion is right for the wrong
   reason.** L1's published A−C R² gap is reproduced to five decimals in every cell by a frozen
   model with only the scale refitted, i.e. it is 100% calibration + aggregation artifact. On
   AUC, L1's raw-score gap is +0.0022 against a published floor of +0.0023 — its entire gap is
   the floor, as claimed. L1m's and L2's AUC gaps are *negative* on the raw score.
7. **Base-rate artifact — disclosed and correct.** Balanced accuracy is exactly 0.3333 with MCC
   NaN for L0/L1/L1m/L2 under every protocol because the rescaled sign never clears the 0.5-tick
   deadband; base rate runs 0.75–0.96. `METHOD.md` §6, `ladder.md` and `FINDINGS.md` §1.3 all
   say to read those rows as null. AUC is genuinely the only directional column for them, and
   L0's AUC is exactly 0.5000 under all four protocols (constant predictor, gaps vanish).
8. **Deadband tuning — rule, not hindsight.** α = 0.5 ticks frozen 2026-07-31 in `DECISIONS.md`
   line 281 and in `labels.py`, sourced to `FEATURES.md` §7, a full day before any phase-2
   number. The AUC binary (directional vs pooled) was chosen 2026-08-03 on a stated a-priori
   argument; I checked whether it was chosen for the outcome — the pooled definition would give
   L4 **+0.1804** (larger) and L3 **+0.0199** (2.1× smaller). Mixed, so not obviously
   outcome-driven, but L3's headline AUC gap doubles under the chosen definition and that is
   worth stating.
9. **Trial count — +0 is defensible, with one unlogged degree of freedom.** 2,056 is unchanged;
   trials 18–19 re-score existing fits, search no new configuration, and a deflated Sharpe
   deflates over configurations. I could not reconcile every historical line to the total but
   found no configuration run and unlogged. The unlogged freedom is the *metric axis*: two new
   metrics and a choice of AUC binary are three new chances to restate a claim, and only the
   AUC-binary choice is logged.
10. **Pooled-number laundering — one real instance, in the calibration table.** The AUC headline
    is not laundered: L4's per-cell gaps run +0.049 to +0.314, 16/16 positive, and dropping
    fold 0 moves the median from +0.1740 to +0.1654 (L3: +0.0420 → +0.0261, still positive).
    The calibration table is: it aggregates the **mean across folds** of a heavy-tailed ratio.
    Using the per-cell **median** across folds — the aggregation the same page tells you to
    prefer ("unstable per cell … read the medians") — protocol C's slopes become L1 **1.0206**,
    L3 **0.9712**, L4 1.1185, and "under C none of them is calibrated" evaporates. So does
    "under the correct protocol every rung is UNDER-confident": L2's median-aggregated C slope
    is 0.7991.

## Additional experiments

- **Reproduction of the artifacts.** I recomputed the AUC and calibration tables from the
  parquet files with my own driver and reproduce `reports/ladder.md` exactly: L4 per-cell A−C
  AUC gaps {0.0493, 0.0673, 0.0740, 0.1008, 0.1091, 0.1337, 0.1499, 0.1671, 0.1810, 0.2001,
  0.2165, 0.2426, 0.2610, 0.3021, 0.3054, 0.3138}; medians 0.6545/0.6573, 0.7811/0.7319,
  0.9059/0.7122; cell counts 13/16, 9/16, 10/16, 16/16, 16/16; worst folds 0.4294/0.4552/0.4725;
  L4>L3 in 6/16, L3>L1 in 16/16. Every number in `RESULTS.md`'s 2026-08-03 tables, `FINDINGS.md`
  §1.1a and `README.md`'s headline table is arithmetically correct.
- **The re-run changed nothing else — VERIFIED.** Parsed `git show HEAD:reports/ladder.md`
  against the working copy: the headline table (97 rows), the frozen-floor table, the superseded
  in-sample table, the per-fold-C table and the 481-row embargo sweep are **byte-identical**;
  all 384 IC rows and all 384 classification rows match on every shared column, with only the
  three new columns inserted. `ladder.md`'s "L1 is the control" reading note is deleted and
  replaced, and the "identical training-set size" sentence is replaced by "A trains on roughly
  1.8× C's median". All three instructions from the 2026-08-02 KILLED report were carried out.
- **Median vs difference-of-medians vs mean, AUC.** median-of-per-cell-gaps
  0.0032/0.0035/0.0078/0.0419/0.1740; difference-of-medians
  −0.0027/−0.0011/+0.0176/+0.0492/+0.1937; mean-of-per-cell-gaps
  0.0088/0.0081/0.0144/0.0468/0.1796. The capacity-scaling shape survives all three, and
  `RESULTS.md`/`FINDINGS.md` disclose which one they quote and why. Clean.
- **L3's calibration range [−4.74, 43.20] — VERIFIED** from the cell-level mean column
  (min AMZN calendar 10s, max AMZN event 10s). L4's [0.88, 1.80] verified. On the median column
  L3's range is a far tamer [0.64, 1.32], which is the same aggregation problem as item 10.
- **Labels shuffled.** AUC → 0.4924/0.5032/0.5005 under A and 0.4950/0.4968/0.5057 under C for
  L1/L3/L4; R² → ±0.0001; gaps ≤0.0065. Chance under both protocols. PASS.
- **Features shifted +1 event.** L3's protocol-C R² 0.0215 → 0.0882, L4's 0.0213 → 0.0722, AUC
  0.891 → 0.905 and 0.857 → 0.891. Behaves as a leak probe should.
- **Protocol A at k ∈ {2,3,5,10,20}.** L4 R² gap +0.1205, +0.1443, +0.1618, +0.1704, +0.1935 —
  smooth and monotone. L4 AUC gap +0.1243, +0.1304, +0.1337, +0.1348, +0.1356 — smooth, monotone,
  saturating. L1's AUC gap is non-monotone and sign-changing across k (−0.0022 … −0.0040), which
  is what a null should look like. The splitter does what it claims.
- **Constant predictor.** L0 scores R² exactly 0.00000 and AUC exactly 0.5000 under all four
  protocols in all 96 cells. Gaps vanish.
- **Can the AUC floor be made large?** Yes, by the mechanism the report says has no analogue.
  A's per-fold AUC is a *whole-day pooled* AUC; C's is a *within-block* AUC. Any frozen score
  with a time-varying offset makes those two estimands diverge with no refitting: frozen
  `sign(qi)` plus an offset aligned with the block up-rate gives A−C floors of −0.032, −0.074,
  −0.137 as the offset grows, against +0.0026 for the published construction
  (`scratch/drift_floor.py`). The claim "the artifact floor of a rank statistic does not scale
  with predictor **variance**" is trivially true; the claim carried in `METHOD.md` §6,
  `README.md` and `FINDINGS.md` §1.1a — that the AUC null is flat *because* the weighting
  asymmetry has no analogue in a rank statistic — is wrong. It is flat here because these three
  frozen predictors happen to be temporally homogeneous, and because L4's frozen AUC sits at
  0.996, where a ceiling leaves almost no room to differ.
- **The AUC definition and fold dropping — no asymmetry found.** Across all 800 (cell × rung ×
  protocol × fold) evaluations, **zero** AUC values are NaN, every cell reports 5 folds under
  every protocol, and no protocol loses a fold. The directional subset is 1.6%–86% of test rows
  with an up-share of 0.31–0.54, so the subsets differ per fold but never degenerate. The
  concern is real in principle and empirically absent here.
- **`tests/test_score_fold.py` — the tests can fail.** I mutated a copy of `src/`:
  pooling flats into the negative class → 3 failures including the headline assertion;
  returning 0.5 instead of NaN on a one-sided block → 2 failures; computing the slope on
  `rung._raw` instead of `rung.predict` → 9 of 10 fail. The last one fails via `AttributeError`
  on the stub rather than on a value, because `StubRung` applies its `scale` inside its own
  `predict`: the test verifies that the slope is taken from `predict()`'s output, which is
  weaker than the docstring's claim that it verifies the through-origin rescale specifically.
  It still catches the regression. Full suite: 392 passed, 585 subtests.

## Most likely remaining explanation that is not "the effect is real"

For the surviving AUC headline: **regime coverage, not the future.** Even with the calibration
rule held constant, L4's A−C AUC gap of +0.137 is measured on one trading day whose protocol-C
folds are contiguous *regime* blocks. Protocol A's training set spans the whole session; C's
fold-*k* training set is strictly the morning. Protocol D does not separate this either, because
D also sees every regime. The per-fold protocol-C AUCs I extracted are monotone in training-set
size in most cells (AMZN calendar 1s L4: 0.530, 0.583, 0.623, 0.636, 0.651), which is exactly
what "C has not yet seen enough of the day" looks like and exactly what "A saw the future" looks
like. On one day the two are not separable, and "optimism scales with capacity" and "small
training sets hurt high-capacity models most" predict the same table.

## What I could not check and why

- **Cross-day generality.** One date, two names. `B − C` is near zero *because* the fold geometry
  is intraday; with day-boundary folds the purge removes a different share. Untestable here.
- **Whether the phase-3 cost numbers carry the superseded calibration.** `src/edge/costs/run.py`
  line 177 calls `rung.fit(X[tr], y[tr])` with no `scale_holdout`, i.e. the in-sample rescale
  that was killed on 2026-08-02, and `reports/costs.md` has not been regenerated since. Trading
  direction is the *sign* of the prediction, so a positive scale is harmless — but I measured
  27 protocol-C folds where the fitted scale is **negative**, and I did not trace whether any of
  them land in the cost grid. Out of scope for this pass; it needs one.
- **GBM hyperparameters.** Frozen a priori, never swept, so I cannot say how much of L4's
  behaviour under C is those specific settings.
- **The exact provenance of the trial total 2,056.** I verified that no configuration was run
  and left unlogged, but not every historical line against the sum.

## What would have to change for this to survive

1. Withdraw the calibration-slope leakage reading from `RESULTS.md` (the 2026-08-03 body text
   and the leakage-diagnostics row), `FINDINGS.md` §1.3 and anywhere else it appears. The
   defensible statement is: *under protocol A the calibration holdout is exchangeable with the
   test fold, so A's slope is mechanically ≈1; the slope therefore measures the calibration
   design, not leakage.* Report it beside the frozen-model control.
2. Re-measure every floor with the scale refitted per fold under the protocol's own rule — the
   same machinery the fitted rungs get. That is +0.0043 / +0.0112 / +0.0192 for R², and it
   changes the AUC floor's worst cell from +0.0053 to +0.0645. Re-state the AUC table against
   the larger floor and drop the L1m and L2 entries from the capacity progression, or quote them
   on the raw score.
3. Disclose the calibration-rule decomposition beside the headline: with one calibration rule,
   L4's median A−C R² gap is +0.1379, not +0.3174. The claim is still large and still scales
   with capacity; it is roughly half the advertised size once the calibration channel is
   separated from the split.
4. Fix `METHOD.md` §6, `README.md` and `FINDINGS.md` §1.1a: a rank statistic's floor is immune
   to the predictor's *scale*, not to cross-block heterogeneity, and it is not immune to the
   *sign* of a fitted rescale.

The AUC restatement of the capacity claim is a good result and it holds. It is being reported
next to a second "independent" result that is an artifact of the harness, and behind floors that
are measured with the artifact switched off.
