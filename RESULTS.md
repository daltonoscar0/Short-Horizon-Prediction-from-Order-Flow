# RESULTS — Edge

## Headline ladder

Report per fold. Never pool without the distribution.

| date | ticker | clock | horizon | rung | OOS R² (vs zero) | IC (median/worst fold) | bal. acc | break-even cost (ticks) | notes |
|---|---|---|---|---|---|---|---|---|---|
| | | | | L0 constant | | | | | |
| | | | | L1 sign(imbalance) | | | | | |
| | | | | L2 OLS on OFI | | | | | |
| | | | | L3 ridge/logistic, full set | | | | | |
| | | | | L4 GBM, full set | | | | | |
| | | | | L5 full set + LM states | | | | | |
| | | | | L6 LM states only | | | | | |
| | | | | C1 untrained-weights control | | | | | |
| | | | | C2 lagged-raw control | | | | | |
| | | | | C3 random-projection control | | | | | |

## Leakage diagnostics

| test | expected | observed | date |
|---|---|---|---|
| label shuffle | metrics → chance | **PASS** — every feature collapses to \|ρ\| < 0.01 (INTC/event, `ofi_best_100ms` 0.3743 → 0.0022) | 2026-07-31 |
| time-shift probe (+1 event) | large jump | **PASS** — shipped 0.3743, fwd+1 0.4076, fwd+5 0.4992, fwd+10 0.5420; bwd-1 0.3458, bwd-10 0.1639. The shipped column sits strictly on the causal side, and a leaky pipeline could not improve when handed more future | 2026-07-31 |
| future-scramble (rows ≥ 300,000) | non-label columns unmoved | **PASS** — 0 of 229 non-label columns moved; all 28 `label_` columns moved, row counts matching horizons exactly (e1:1, e2:2, e5:5, e10:10, 1s:17, 2s:18, 5s:28, 10s:101). Also discharges INTERFACE.md sec. 8's reuse-a-single-fit failure mode | 2026-07-31 |
| independent re-derivation | pipeline = definitions | **PASS** — stamps, `ofi_best_100ms`, `qi_l1`, `label_y_mid_1s` rebuilt from raw with no `edge.features` import: max\|diff\| = 0.0, NaN mismatch 0 | 2026-07-31 |
| purge off vs on | on < off; record the gap | Phase 2 | |
| same-day vs next-day | drop expected | Phase 2 (single ticker-day in this universe) | |

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

**Honest trial count for the deflated Sharpe: 288** for the ladder grid proper, plus trials 1–4
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
| embargo E | | | Phase 2 |
