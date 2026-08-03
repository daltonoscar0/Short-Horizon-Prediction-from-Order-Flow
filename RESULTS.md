# RESULTS — Edge

## Headline ladder

Report per fold. Never pool without the distribution.

**Protocol C (purged walk-forward with embargo) — the honest ladder.** Median across all 96
(ticker × clock × horizon) cells, 2026-08-02, `--seed 0`. Full per-cell detail in
`reports/ladder.md`; the artifact floor each gap must clear is in `reports/artifact_floor.md`.

| date | scope | rung | OOS R² vs zero (median) | worst cell | IC (median) | IC worst fold | A−C optimism | artifact floor | break-even (ticks) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-02 | 96 cells | L0 zero | 0.0000 | 0.0000 | n/a | n/a | 0.0000 | — | n/a (no trades) |
| 2026-08-02 | 96 cells | L1 sign(queue imbalance) | +0.0091 | −0.0019 | +0.1465 | −0.1110 | +0.0043 | +0.0043 | 0.0106 |
| 2026-08-02 | 96 cells | L1m sign(microprice − mid) | +0.0042 | −0.0040 | — | — | +0.0058 | +0.0043 | 0.0068 |
| 2026-08-02 | 96 cells | L2 OLS on OFI | −0.0018 | −0.0971 | +0.1236 | −0.2666 | +0.0204 | +0.0043 | 0.0037 |
| 2026-08-02 | 96 cells | L3 ridge, full set | **+0.0266** | −0.0450 | **+0.2262** | −0.0718 | +0.0587 | +0.0112 | 0.0100 |
| 2026-08-02 | 96 cells | L4 GBM, full set | **+0.0273** | −0.0003 | +0.1751 | −0.0654 | **+0.3173** | +0.0192 | 0.0088 |
| — | cut | L5 full set + LM states | — | — | — | — | — | — | — |
| — | cut | L6 LM states only | — | — | — | — | — | — | — |
| — | cut | C1/C2/C3 LM controls | — | — | — | — | — | — | — |

Break-even is the INTC calendar 1s, 1ms-latency cell (`reports/costs.md`), on the gross series.
L5/L6 and their controls are cut: no order-flow-LM checkpoint exists (`DECISIONS.md`
2026-07-31, `FUTURE.md` §2).

**Reading it.** The A−C column scales with capacity — from +0.004 on a rung with no fitted
parameters to +0.317 on the GBM, against artifact floors of +0.004 and +0.019 respectively.
**But `A − C` bundles two mechanisms and the next section takes them apart**: about half of
L4's gap and all of L1's is the harness's own protocol-dependent calibration rule, not the
split. Held constant, the figures are −0.002 (L1) and +0.138 (L4), and the capacity ordering
survives. Decomposed against protocol D, 93% of the GBM's remaining gap is the temporal
*adjacency* of training rows rather than the future itself. Under the correct protocol the
fitted rungs do beat the unfitted baseline (L4 in 13 of 16 cells), but L4's honest R² is 12.6×
smaller than protocol A advertises for the same model, and on both scale-free metrics — IC and
AUC — the ridge outranks the GBM. After costs nothing pays: the best break-even is 0.011 ticks
against ~0.92 ticks of actual cost, and perfect foresight of the 1s mid move still loses money.

**Superseded.** An earlier version of this table (never committed) reported L4 at A−C +0.5453
and C −0.2164, concluding that the ladder inverts. That was an in-sample calibration artifact,
found by the 2026-08-02 phase-2 red team; see `DECISIONS.md` and the superseded columns
retained in `reports/ladder.md`.

## The two channels inside `A − C`, and the headline on a scale-free metric (2026-08-03)

`EVAL_PROTOCOL.md` sec. 2's last two unreported metrics — the calibration slope and AUC — are
now emitted (`reports/ladder.md`). Wiring them in cost this project its published floors: the
red team used them to show that the frozen-predictor floor was measured with the largest
artifact channel held fixed, and that the harness's own calibration design contributes about
half of the flagship gap. The corrected picture is narrower and better supported.

### The calibration rule is a channel, and it has to be separated from the split

`score_fold` picks the through-origin scale's holdout **by protocol**: a random 20% of the
training rows under A, the last 20% in time under B/C/D. Under A that holdout is *exchangeable
with A's random test fold*; under the walk-forward protocols it is not. That asymmetry is a real
part of what "shuffled k-fold" means — a practitioner who shuffles really does calibrate on
exchangeable rows — but it is a **different mechanism from seeing the future**, and until
2026-08-03 nothing separated the two.

Protocol A re-run under C's positional holdout (`--holdout-control`; same split, folds, seed and
model, only the rule moves), median across 16 cells:

| rung | `A − C` as published | **`A′ − C`, rule held constant** | like-for-like floor | bracket over rules |
|---|---|---|---|---|
| L1 `sign(queue imbalance)` | +0.0043 | **−0.0021** | *+0.0043* | never clears its floor |
| L3 ridge, full set | +0.0587 | **+0.0542** | *+0.0112* | [+0.054, +0.183] |
| L4 GBM, full set | +0.3173 | **+0.1379** | *+0.0192* | [+0.138, +0.522] |

L4's headline gap is roughly **half** what the A−C column advertises once the calibration rule
is held constant. The claim survives in the narrower form and is arguably cleaner: with the rule
fixed, the parameter-free rung shows **no optimism at all** (−0.0021, below its own floor),
while the ridge and the GBM clear theirs. The capacity ordering is intact; the magnitude was
overstated.

**`A′` is not a clean control, and the bracket column is why.** `Rung.fit` fits the model on the
complement of the calibration holdout, so changing the rule also changes which rows the model
trains on. Standardising on the *other* rule instead — give C protocol A's random holdout — puts
L4's gap at **+0.5223**, above the published figure rather than below it. Each protocol's default
rule flatters it, `A − C` sits in the middle of the bracket, and a third rule (holdout = the
first 20% in time) returns essentially the whole uncontrolled gap on AUC (+0.1727 of +0.1740),
which says the AUC shrink under `A′` is a lost training window rather than the rule. The clean
control would hold the model's fit rows fixed and move only the scale rows; it is not
implemented. `FINDINGS.md` §1.1a carries the full 2×2.

### The artifact floor was measured with that channel switched off

The floor is supposed to be the number a gap must clear. The version published earlier on
2026-08-03 froze the *scale* as well as the model, so it could not see the calibration channel at
all — a control that was vacuous by construction, presented as a control. This is the same class
of error the 2026-08-02 pass killed the report over, one level down.

Refitting the scale per fold under each protocol's own rule, exactly as the fitted rungs get it:

| frozen at | like-for-like R² floor | frozen-scale R² floor (what was published) | AUC floor | worst AUC cell |
|---|---|---|---|---|
| L1 magnitude | **+0.0043** | +0.0020 | +0.0031 | +0.0324 |
| L3 magnitude | **+0.0112** | +0.0020 | +0.0025 | **+0.0645** |
| L4 magnitude | **+0.0192** | +0.0153 | +0.0014 | +0.0045 |

The R² floors are 2–6× the published ones. The frozen-scale column is retained because the pure
partition artifact is a real and separate quantity — the difference between the two columns *is*
the calibration channel, measured on a predictor that cannot leak.

### AUC, quoted on the raw score

| rung | AUC under A | AUC under C | A − C (raw score) | **A′ − C** (rule held constant) | cells positive | *floor* | *rescaled gap, withdrawn* |
|---|---|---|---|---|---|---|---|
| L0 zero | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0 / 16 | — | *0.0000* |
| L1 `sign(queue imbalance)` | 0.6545 | 0.6573 | +0.0022 | +0.0023 | 13 / 16 | *+0.0031* | *+0.0032* |
| L1m `sign(microprice − mid)` | 0.5880 | 0.5892 | **−0.0018** | −0.0018 | 6 / 16 | *+0.0031* | *+0.0035* |
| L2 OLS on OFI | 0.6546 | 0.6567 | **−0.0002** | −0.0002 | 7 / 16 | *+0.0031* | *+0.0078* |
| L3 ridge, full set | 0.7811 | **0.7319** | +0.0409 | **+0.0283** | **16 / 16** | *+0.0025* | *+0.0419* |
| L4 GBM, full set | 0.9059 | 0.7122 | **+0.1740** | **+0.1371** | **16 / 16** | *+0.0014* | *+0.1740* |

The two controls are independent and compose: the raw score removes the rescale's sign, and
`A′ − C` removes the calibration rule. Neither touches the unfitted rungs — their AUC gap is
whatever the fold partition gives, and it is zero to three decimals.

**Quote the margin against the WORST floor, not the published one.** Every floor above fits the
frozen model on the whole day, and that is the choice that minimises it — at L4's magnitude for
a mechanical reason: a whole-day GBM scores AUC ≈0.996, a **ceiling** where two protocols have
almost no room to differ, while the rung it is meant to bound scores ≈0.72 under C. Refit the
frozen model on half the day, so its accuracy lands in the rungs' operating range, and the same
floor is several-fold larger (worst cell +0.0383 at L4's magnitude). `reports/artifact_floor.md`
now reports all four fit windows and their maximum.

Against the worst-window floor, at each rung's own magnitude, raw against raw:

| rung | `A′ − C` raw AUC | worst-window floor | margin | folds positive |
|---|---|---|---|---|
| L3 ridge | +0.0283 | up to +0.0066 | median 18.5×, **min cell 5.6×** | 55 / 80 |
| L4 GBM | +0.1371 | up to +0.0383 | median **6.9×**, **min cell 1.5×** | **79 / 80** |

So L4 survives at roughly **7×**, not the 98× an earlier version of this table claimed against
the ceiling-measured floor. **That is the narrowest version of the headline this data supports,
and it is the one to quote.** L3's is real but thin — its gap is a fold-0 story (positive in 55
of 80 folds, halving to +0.0131 when fold 0 is dropped) where L4's is not (79 of 80, +0.1273
without fold 0).

**Why the raw score.** The through-origin rescale is rank-preserving only *up to sign*, and the
fitted scale comes out negative in **27 of 400 protocol-C folds and 0 of 400 protocol-A folds**.
A negative scale flips the ranking and sends AUC to 1−AUC. On the unfitted rungs that channel
alone manufactured the entire apparent gap: L1m and L2 are **negative** on the raw score where
the rescaled column reported +0.0035 and +0.0078. The fitted rungs are **near enough** untouched
by it at the median (L4 identical to four decimals, L3 −0.0010), which is why the capacity result
survives and the "smooth progression across all six rungs" reading does not — though three cells
do move materially (AMZN calendar 10s L3 +0.0584 → +0.0405, and two AMZN L4 cells), so this is a
median statement and not a general one. It is also exactly right for L1m and L2 and only
*mostly* right for L1, whose gap goes +0.0032 → +0.0022 rather than to zero.

**What the AUC table supports, stated narrowly.** L3 and L4 show a positive A−C gap in 16 of 16
cells on a metric with no amplitude in it, and the fitted rungs clear their floors while the
unfitted ones show no gap. That is capacity-scaling optimism. The margins are quoted against the
worst-window floor above — ~7× for L4, thinner for L3 — and **not** against the whole-day floor,
which is measured at a ceiling. What the table does **not** support: a smooth five-point
progression, or L3 cell by cell, where its gap is a fold-0 story.

**And the floor is not flat for the reason first given.** An earlier draft of this section said
the AUC floor cannot scale because a rank statistic has no analogue of the Σy² weighting
asymmetry. That is wrong and it was the stated reason the restatement was worth anything.
Protocol A's per-fold AUC is a whole-day **pooled** AUC while C's is a **within-block** AUC; any
frozen score with a time-varying offset makes those two estimands diverge with nothing refitted,
and the red team drove the floor to −0.137 that way. The floor is small *here* because these
particular frozen predictors are temporally homogeneous — a measurement on this data, not a
property of the metric.

### Calibration slope: a diagnostic, and NOT evidence of leakage

| rung | slope under A | slope under C | C range across cells |
|---|---|---|---|
| L1 `sign(queue imbalance)` | 1.0319 | 1.3904 | [−0.21, 3.24] |
| L2 OLS on OFI | 0.9174 | 4.8604 | [−0.58, 19.37] |
| L3 ridge, full set | 0.9775 | 1.6839 | [−4.74, 43.20] |
| L4 GBM, full set | 1.0026 | 1.2700 | [0.88, 1.80] |

**A claim published earlier today is withdrawn.** It read: "under protocol A every fitted rung is
calibrated to ≈1.0 and under C none of them is — this is a leakage signature in its own right,
because a model is calibrated on rows it has effectively already seen." It is an artifact of the
calibration design described above. Under A the holdout is exchangeable with A's test fold, so
the slope is driven toward 1.0 **mechanically, for any predictor, with or without skill**. Two
controls:

- A **frozen** whole-day model with only the scale refitted — it cannot leak — reproduces
  A = 1.0319 / C = 1.3905 at L1's magnitude against the published 1.0319 / 1.3904.
- **Holding the calibration rule constant reverses it for L4**: median |slope − 1| goes 0.018 →
  1.034 under A, against C's 0.269.

Two further cautions. The slope is a heavy-tailed ratio whose denominator is the prediction's
variance, so the mean across folds is not robust; aggregated by median, protocol C's slopes come
out near 1.0 and the companion claim that "under C every rung is under-confident" disappears as
well (L2's median-aggregated C slope is 0.80). The slope is reported because the spec asks for
it, and **nothing in this study rests on it**.

## Leakage diagnostics

| test | expected | observed | date |
|---|---|---|---|
| label shuffle | metrics → chance | **PASS** — every feature collapses to \|ρ\| < 0.01 (INTC/event, `ofi_best_100ms` 0.3743 → 0.0022) | 2026-07-31 |
| time-shift probe (+1 event) | large jump | **PASS** — shipped 0.3743, fwd+1 0.4076, fwd+5 0.4992, fwd+10 0.5420; bwd-1 0.3458, bwd-10 0.1639. The shipped column sits strictly on the causal side, and a leaky pipeline could not improve when handed more future | 2026-07-31 |
| future-scramble (rows ≥ 300,000) | non-label columns unmoved | **PASS** — 0 of 229 non-label columns moved; all 28 `label_` columns moved, row counts matching horizons exactly (e1:1, e2:2, e5:5, e10:10, 1s:17, 2s:18, 5s:28, 10s:101). Also discharges INTERFACE.md sec. 8's reuse-a-single-fit failure mode | 2026-07-31 |
| independent re-derivation | pipeline = definitions | **PASS** — stamps, `ofi_best_100ms`, `qi_l1`, `label_y_mid_1s` rebuilt from raw with no `edge.features` import: max\|diff\| = 0.0, NaN mismatch 0 | 2026-07-31 |
| purge off vs on | on < off; record the gap | **RUN, and the gap is near zero — for an arithmetic reason.** Median `B − C` R² across 96 cells: L1 +0.0006, L3 +0.0019, L4 +0.0025. Purge is real (it removes exactly 610/620/650/1100 rows at h = 1/2/5/10s) but under expanding walk-forward it can only touch the last `h + E` of a multi-hour training set — 0.6–1.1% of training rows. **This does NOT mean label overlap is unimportant**: measured against protocol D, adjacency is 93% of L4's total gap. The purge column measures label overlap only at a walk-forward boundary, where it is negligible by construction | 2026-08-02 |
| same-day vs next-day | drop expected | **CANNOT BE RUN** — one ticker-day per name in this universe. Recorded as unavailable, not as passed; it is the check most likely to find something the intraday checks cannot (`FUTURE.md` §1) | 2026-08-02 |
| frozen-predictor floor | gap → 0 if the harness is neutral | **FAILS, and that is the finding.** A predictor fitted once and scored unchanged under every protocol still shows A−C of +0.0020 (L1 magnitude) to +0.0153 (L4 magnitude), because A's fold mean approximates the Σy²-weighted pooled metric while C's is unweighted. Every reported gap is now quoted against its own-magnitude floor | 2026-08-02 |
| calibration slope under A vs C | a leaky protocol should look better calibrated | **FAILS AS A LEAKAGE TEST — the difference is the harness's own calibration design.** Under A the scale's holdout is a random 20% of training rows, exchangeable with A's random test fold, so the slope is driven to ≈1.0 for *any* predictor. A frozen model that cannot leak reproduces A 1.0319 / C 1.3905 to three decimals, and holding the rule constant reverses the ranking for L4. Published as a leakage signature on 2026-08-03 and **withdrawn the same day** | 2026-08-03 |
| calibration rule held constant | if the gap is the split, changing the holdout rule should not move it | **MOVES IT A LOT, IN BOTH DIRECTIONS.** Protocol A under C's positional holdout puts L4's median A−C R² at **+0.1379** (from +0.3173) and L1's at **−0.0021**; standardising on A's random rule instead puts L4 at **+0.5223**. The published gap sits mid-bracket [+0.138, +0.522], not at the top. The capacity ordering survives every rule tested and L4's raw-AUC gap stays in [+0.137, +0.190] | 2026-08-03 |
| AUC gap vs an AUC floor | the gap should clear a floor measured in the same metric | **PASS for L3/L4, FAILS for the unfitted rungs.** On the raw score the gaps are +0.0022 (L1), −0.0018 (L1m), −0.0002 (L2), +0.0409 (L3), +0.1740 (L4). The fitted rungs clear in 16 of 16 cells; the unfitted rungs show no gap at all, and their positive figures in the first draft were the rescale's **sign** flipping AUC to 1−AUC in 27 protocol-C folds. Margins are quoted against the worst-fit-window floor (~7× for L4), not the whole-day one | 2026-08-03 |
| floor measured at several fit windows | the floor should not depend on where the frozen model was fitted | **IT DOES, AND "WHOLE DAY" IS THE FLATTERING CHOICE.** A whole-day frozen GBM scores AUC ≈0.996 — a ceiling, where two protocols cannot differ much — against ≈0.72 for the rung it bounds. Refit it on half the day and the AUC floor at L4's magnitude grows to +0.0383 in the worst cell, from +0.0045. The margin claimed in the first draft of §1.1a (98×) was that ceiling; the real figure is ~7×. All four windows and their max now ship in `reports/artifact_floor.md` | 2026-08-03 |
| frozen-scale vs refit-scale floor | a control must be exposed to every channel the thing it controls is exposed to | **THE PUBLISHED FLOOR WAS NOT.** Freezing the scale along with the model hid the calibration channel entirely; refitting it per fold under each protocol's own rule raises the R² floor from +0.0020/+0.0020/+0.0153 to **+0.0043/+0.0112/+0.0192** at L1/L3/L4 magnitudes. Both columns are now printed, and their difference is the channel | 2026-08-03 |
| in-sample vs causal calibration | should not change the sign of a result | **DID change it.** Fitting the through-origin rescale in-sample drove protocol-C R² sharply negative for high-capacity rungs (L4 median −0.2164 → +0.0273) and inflated the measured optimism (A−C +0.5453 → +0.3173). Superseded columns retained in `reports/ladder.md` | 2026-08-02 |

### Phase 1 substantive finding — the flagged correlations are real, and short

The correlation report flags 446 feature/label pairs above \|ρ\| = 0.15 at the 1s horizon
(282 INTC, 164 AMZN). Adversary verdict `SURVIVED WITH CAVEATS`: genuine microstructure,
**correctly measured, reported on the wrong axis**.

- **Not leakage** — six independent executable checks above.
- **Pooled ρ was measuring tie mass.** 96.3% of INTC calendar 1s labels are exactly flat vs
  87.6% on the event clock, and 77.7% of calendar stamps have OFI exactly 0. Conditioning on
  a non-flat label and an active window collapses the clock gap for queue imbalance:
  ρ = 0.616 calendar vs 0.603 event, at 0.80–0.83 balanced directional accuracy — the
  textbook large-tick queue-imbalance number, not a discovery.
- **The OFI residual is burst oversampling.** 32% of INTC event stamps sit within 0.5 ms of
  the previous one and carry ρ = 0.437, against 0.19 for isolated stamps.
- **The signal decays inside 100 ms.** ρ(`ofi_best_100ms`, mid(t+1s) − mid(t+100ms)) = 0.027
  against 0.458 for the first 100 ms. A `label_y_mid_1s` number is **not** one-second alpha.

## Trial log (mandatory — feeds the deflated Sharpe)

Every configuration tried, including abandoned ones. If it isn't here, the trial count is
wrong and the deflated Sharpe is a lie.

| # | date | what changed | outcome | kept? |
|---|---|---|---|---|
| 1 | 2026-07-31 | Phase-1 feature build, 4 (ticker × clock) tables, 257 columns each | bit-reproducible; CKS self-check PASS on all 4 combos | yes |
| 2 | 2026-07-31 | Univariate correlation triage, 7,264 (feature × label) cells | triage only, explicitly not a result; 480 pairs self-flagged | yes (as triage) |
| 3 | 2026-07-31 | Deadband α sweep, 5 points {0, 0.25, 0.5, 1, 2} | reported for sensitivity; α frozen at 0.5 BY RULE, not by this sweep | frozen a priori |
| 4 | 2026-07-31 | OFI self-check regressions, 52 (13 regressors × 4 ticker/clock) | PASS | yes |
| 5 | 2026-07-31 | Ladder grid: 2 tickers × 2 clocks × 4 horizons × 3 protocols × 6 rungs = **288 fits** | the optimism table | pending adversary |
| 6 | 2026-07-31 | L3/L4 feature set: dropped `tr_aggr_imb_1s` before scoring any fold | recovered 87% of rows; chosen on NaN structure, not on a result | yes |
| 7 | 2026-07-31 | Protocol A test universe restricted to match B/C (first attempt tested the market open A-only) | discarded, re-run | no — superseded |
| 8 | 2026-07-31 | Per-rung row masks replaced by a common mask (first attempt scored L3/L4 on 4,576 rows vs L1/L2's 39,000) | discarded, re-run | no — superseded |

| 9 | 2026-08-02 | Embargo sensitivity sweep, E ∈ {0, 1s, 10s, 60s, 300s} × 16 cells × 6 rungs = **480 protocol-C fits** | R² decays monotonically in E and flattens between 60s and 300s (INTC/event/1s L1: 0.07228 → 0.07189 → 0.07186); the frozen E is not a lucky point on the curve | yes (sensitivity) |
| 10 | 2026-08-02 | Phase-3 cost grid: 2 tickers × 2 clocks × 2 horizons × 9 predictors × 3 latencies = **216 configurations** (6 rungs + 3 controls) | the break-even and DM tables | yes |
| 11 | 2026-08-02 | Break-even column recomputed on the gross series after the red team showed the net-based column was model-invariant | superseded the phase-3 headline; incremental column retained and demoted | yes — corrected |
| 12 | 2026-08-02 | Deflated Sharpe moved from the net series to the gross series | net DSR is 0.000 at every trial count from 1 to 1e12 (per-trade Sharpe ≈ −7.6); a tautology, not a result | yes — corrected |

| 13 | 2026-08-02 | **Phase-2 rework after the red team.** Ladder re-scored under a CAUSAL through-origin calibration (held-out last 20% of each fold's training rows) across A/B/C plus the new control protocol D: 4 × 16 × 6 = **384 fits** | supersedes trial 5. Every large negative protocol-C R² was an artifact of the old in-sample scale; INTC cal 1s L4 C goes −0.2988 → +0.0213 and its A−C gap 0.4898 → 0.1618 | yes — this is the headline |
| 14 | 2026-08-02 | Frozen-predictor artifact floor: 1 fit per cell, scored unchanged under 4 protocols = **16 fits** | the control L1 was supposed to be. A−C of +0.0014 to +0.0271 with nothing refitted; reproduces the old L1 IC gap to 5 decimals | yes |
| 15 | 2026-08-02 | Superseded in-sample calibration retained as a diagnostic, protocols A and C: **192 fits** | kept so the size of the correction is visible rather than asserted | yes (as diagnostic) |
| 16 | 2026-08-02 | Embargo sweep re-run under the causal calibration: **480 fits** | supersedes trial 9 | yes |
| 17 | 2026-08-02 | Calibration holdout fraction fixed at 80/20, **not swept** | a new free parameter, chosen after the problem was known. Disclosed in `FINDINGS.md` §5, not frozen a priori | disclosed |

| 18 | 2026-08-03 | Ladder re-run at the identical configuration and seed to emit the calibration slope and directional AUC: **384 fits, +0 trials** | not a new configuration. Same folds, same rungs, same hyperparameters, two extra diagnostics computed from the same predictions; all 384 IC rows and the entire headline R² table reproduce 2026-08-02 exactly, which is the check that the additions changed nothing else. Counting it as 384 new trials would inflate the denominator with fits that searched nothing | yes — +0, logged rather than silent |
| 19 | 2026-08-03 | Frozen-predictor floor extended to AUC and the calibration slope: **48 fits re-scored, +0 trials** | the numbers it produced were wrong — the scale was frozen with the model, so the floor could not see the calibration channel. Superseded by trial 21 | no — superseded |
| 20 | 2026-08-03 | **Calibration-rule control**: protocol A re-run under C's positional holdout, 2×2×4×6 = **96 fits** | a genuinely new configuration and it counts. L4's median A−C R² gap goes +0.3173 → +0.1379 and L1's to −0.0021. The reported control is one of at least three rules and it is the one that shrinks the gap most — the bracket is [+0.138, +0.522], disclosed in the section rather than presented as a point estimate | yes — **+96** |
| 21 | 2026-08-03 | Frozen floors re-measured with the scale refitted per fold under each protocol's own rule, at three magnitudes, both variants: **96 fits, +0 trials** | supersedes trials 14 and 19. R² floors go +0.0020/+0.0020/+0.0153 → +0.0043/+0.0112/+0.0192; the frozen-scale variant is retained beside it, and the difference is the calibration channel | yes — +0 |
| 22 | 2026-08-03 | `edge.costs.run` switched from the in-sample rescale to the causal one and the cost grid re-run: **216 configurations re-scored, +0 trials** | the same grid, not a new one. Phase 3 had kept the calibration killed for the ladder on 2026-08-02; found by the third red team outside its own scope. **The fitted rungs moved**: L4's INTC calendar 1s break-even 0.0080 → 0.0088, L3's 0.0098 → 0.0100. The causal holdout changes what the raw model is fitted on (80% of training rows, not 100%), not merely the scale — so the first estimate that this was direction-invariant was wrong. **So did 18 of 72 unfitted-rung rows**, the largest being L1m INTC event 5s 0.0256 → 0.0149: the causal scale can come out negative, and with a 0-tick deadband that inverts an unfitted rung's traded direction outright (AMZN event 5s L1 fold 0 goes −0.0173 → +0.0173 gross ticks/trade). A claim that "the unfitted rungs are unchanged, which is the check that nothing else moved" stood here for one revision and was false | yes — +0, numbers corrected |

**Honest trial count for the deflated Sharpe: 2,152** = 2,056 through 2026-08-02 + **96** for the
calibration-rule control (trial 20), the only 2026-08-03 entry that searched a new configuration.
Trials 18, 19, 21 and 22 re-scored existing fits under additional metrics or a corrected
calibration and add nothing; the reasoning is in their rows rather than left as an unexplained
gap between the run log and the count. The 2,056 was = 984 previous + 1,072 from the
2026-08-02 rework (384 causal ladder + 16 frozen floor + 192 in-sample diagnostic + 480
re-run embargo sweep). Trials 5 and 9 are superseded but stay counted: they were fitted, they
produced numbers that were reported at the time, and removing them from the count would flatter
the deflation. Historical note — the previous figure was **984** = 288 ladder + 480 embargo
sweep + 216 cost configurations. The embargo sweep is arguably excludable (pre-registered sensitivity, frozen by
rule, never selected anything); it is counted anyway, because undercounting is the failure mode
that makes a deflated Sharpe dishonest and overcounting only makes the threshold harder to
clear. Historical note — the earlier figure was **288** for the ladder grid proper, plus trials 1–4
above as pre-registered diagnostics rather than model selection. Trials 7 and 8 are *discarded
harness configurations*, logged because CLAUDE.md rule 4 requires the failures too — both were
found by reading per-fold row counts, both invalidated their runs, and neither contributed a
number to any table. Any deflated Sharpe computed at `--trials 1` is undeflated and the report
says so on its face.

## Decisions frozen (and when)

| decision | value | frozen on | evidence |
|---|---|---|---|
| deadband α | 0.5 ticks (half a tick) | 2026-07-31 | Rule, not tuning: `FEATURES.md` sec. 7's own first suggestion and the smallest move that is not a rounding artifact in a whole-cent book. `edge.features.labels.ALPHA_TICKS`. Sensitivity swept over {0, 0.25, 0.5, 1, 2} in `reports/feature_correlations.md` and reported, never used to select. |
| microprice recursion depth | `MICRO_ITERS = 6` | 2026-07-31 | Stoikov's own repo value. `I - B` is singular by construction so the limit is not directly solvable; the depth must be an a-priori constant. `edge.features.microprice`. |
| microprice state grid | 5 imbalance × 3 spread buckets | 2026-07-31 | Frozen a priori per `INTERFACE.md` sec. 8, not chosen by looking at which grid correlated better. |
| refit schedule | 30 min, fit on strictly-prior rows | 2026-07-31 | `REFIT_BLOCK_NS`; matches `TOD_BUCKET_NS` so the seasonality control and refit grid align. Block 0 is NaN. |
| embargo E | `max(60s, 10·h)` | 2026-07-31 | `EVAL_PROTOCOL.md` sec. 1's own default, taken as a rule rather than chosen. `edge.harness.default_embargo_ns`. Sensitivity swept over E ∈ {0, 1s, 10s, 60s, 300s} in `reports/ladder.md` and reported; the frozen value was never selected by comparing sweep rows. |
| k for protocol A | 5, equal to `N_FOLDS` | 2026-07-31 | `K_SHUFFLED`. Deliberately equal to the walk-forward fold count so an A−C gap cannot be partly a difference in the number of training rows. |
| trading deadband (phase 3) | 0 ticks | 2026-07-31 | **Not frozen a priori — disclosed.** The first cost run reused the labels' 0.5-tick deadband and every unfitted rung took zero trades, leaving the break-even table empty for L1, the baseline the project exists to test. The reasoning for 0 is a-priori (a scaled sign predictor cannot emit a magnitude above half a tick at these horizons) but the change was made after seeing protocol-C output, which is CLAUDE.md rule 5 territory. Recorded as informed-by-output rather than presented as clean. |
| latency ladder Δ | {1ms, 10ms, 100ms} | 2026-07-31 | `EVAL_PROTOCOL.md` sec. 3 names these three exactly. `edge.costs.LATENCIES_NS`. |
| GBM hyperparameters | 200 rounds, lr 0.05, 31 leaves, min 200/leaf, L2 1.0, no early stopping | 2026-07-31 | Frozen a priori. Early stopping on a slice of the test fold is itself a leak; carving a validation slice out of training is one more free parameter to log. Identical count under all three protocols. |
