# RESULTS — Edge

## Headline ladder

Report per fold. Never pool without the distribution.

**Protocol C (purged walk-forward with embargo) — the honest ladder.** Median across all 96
(ticker × clock × horizon) cells, 2026-08-02, `--seed 0`. Full per-cell detail in
`reports/ladder.md`; the artifact floor each gap must clear is in `reports/artifact_floor.md`.

| date | scope | rung | OOS R² vs zero (median) | worst cell | IC (median) | IC worst fold | A−C optimism | artifact floor | break-even (ticks) |
|---|---|---|---|---|---|---|---|---|---|
| 2026-08-02 | 96 cells | L0 zero | 0.0000 | 0.0000 | n/a | n/a | 0.0000 | — | n/a (no trades) |
| 2026-08-02 | 96 cells | L1 sign(queue imbalance) | +0.0091 | −0.0019 | +0.1465 | −0.1110 | +0.0043 | +0.0020 | 0.0106 |
| 2026-08-02 | 96 cells | L1m sign(microprice − mid) | +0.0042 | −0.0040 | — | — | +0.0058 | +0.0020 | 0.0068 |
| 2026-08-02 | 96 cells | L2 OLS on OFI | −0.0018 | −0.0971 | +0.1236 | −0.2666 | +0.0204 | +0.0020 | 0.0035 |
| 2026-08-02 | 96 cells | L3 ridge, full set | **+0.0266** | −0.0450 | **+0.2262** | −0.0718 | +0.0587 | +0.0020 | 0.0098 |
| 2026-08-02 | 96 cells | L4 GBM, full set | **+0.0273** | −0.0003 | +0.1751 | −0.0654 | **+0.3173** | +0.0153 | 0.0080 |
| — | cut | L5 full set + LM states | — | — | — | — | — | — | — |
| — | cut | L6 LM states only | — | — | — | — | — | — | — |
| — | cut | C1/C2/C3 LM controls | — | — | — | — | — | — | — |

Break-even is the INTC calendar 1s, 1ms-latency cell (`reports/costs.md`), on the gross series.
L5/L6 and their controls are cut: no order-flow-LM checkpoint exists (`DECISIONS.md`
2026-07-31, `FUTURE.md` §2).

**Reading it.** The A−C column is the contribution and it scales with capacity — from +0.004 on
a rung with no fitted parameters to +0.317 on the GBM, against artifact floors of +0.002 and
+0.015 respectively. Decomposed against protocol D, 93% of the GBM's gap is the temporal
*adjacency* of training rows rather than the future itself. Under the correct protocol the
fitted rungs do beat the unfitted baseline (L4 in 13 of 16 cells), but L4's honest R² is 12.6×
smaller than protocol A advertises for the same model, and on IC the ridge outranks the GBM.
After costs nothing pays: the best break-even is 0.011 ticks against ~0.92 ticks of actual
cost, and perfect foresight of the 1s mid move still loses money.

**Superseded.** An earlier version of this table (never committed) reported L4 at A−C +0.5453
and C −0.2164, concluding that the ladder inverts. That was an in-sample calibration artifact,
found by the 2026-08-02 phase-2 red team; see `DECISIONS.md` and the superseded columns
retained in `reports/ladder.md`.

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

**Honest trial count for the deflated Sharpe: 2,056** = 984 previous + 1,072 from the
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
