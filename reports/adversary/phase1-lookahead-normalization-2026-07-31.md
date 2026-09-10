VERDICT: SURVIVED WITH CAVEATS

# Phase 1 red team, lookahead and normalization

All numbers below were produced by scripts I wrote, run against
`data/frames/*.parquet` and `data/features/*.parquet` as they stand on 2026-07-31.
Interpreter `/opt/anaconda3/bin/python3`, `PYTHONPATH=src`. Scratch scripts live in
`scratch/redteam/`.
Nothing outside `reports/adversary/` was modified. `make features` was not run.

---

## THE DECISIVE QUESTION: genuine effect or artifact?

**Answer: genuine microstructure, correctly measured, but reported on the wrong axis.**
The flagged correlations are not lookahead. They are (a) queue-imbalance / order-flow
continuation, a textbook effect, (b) inflated in the *pooled* Spearman by the event clock's
tie structure and burst oversampling, and (c) mislabelled as a 1s effect when it is a
<=100ms effect.

Six independent lines of evidence, each executable.

### 1. The columns are what the definitions say (independent re-derivation)

`rt/indep.py` rebuilds the INTC event clock, the CKS best-level `e_n`, the 100ms trailing
window, `qi_l1` and `label_y_mid_1s` from `data/frames/INTC_2012-06-21.parquet` with no
import of `edge.features.*`:

```
stamp ts identical: True
stamp index identical: True
  ofi_best_100ms: max|diff|=0.000e+00  nan mismatch=0
  qi_l1:          max|diff|=0.000e+00  nan mismatch=0
  label_y_mid_1s: max|diff|=0.000e+00  nan mismatch=0
```

So the number is not a pipeline bug. It is what these definitions produce on this data.

### 2. Scramble the entire future: no non-label column moves

`rt/perturb2.py` takes the real INTC frame, adds +5 ticks to every price and rescales every
size **from row 300,000 onward**, rebuilds the whole 257-column table, and compares the
29,903 stamps strictly before `ts[300000]`:

```
INTC/event K=300000 cut_ts=44581686887489 stamps strictly before cut: 29903
  NON-LABEL columns differing: 0
  label_ columns that DID change: 28   (e1:1 row, e2:2, e5:5, e10:10, 1s:17, 2s:18, 5s:28, 10s:101)
  label_ columns that did NOT change: 0
```

This simultaneously discharges four separate items: the `<= t` compliance of all 229
non-label columns (`grep label_` really does find every forward-looking column); the
inverted labels.py requirement (every label changes, with row counts that match the
horizons exactly); and INTERFACE.md sec. 8's first failure mode, a statistic fitted once on
`fit_end_index[-1]` and reused would have moved `ofi_integrated_*` and `mp_*` at *every*
stamp under this perturbation, and none moved. Truncation at row 300,000 (`rt/trunc.py`)
gives the same result: 0 non-label columns differ, labels differ only in the 17-101 rows at
the cut. Also run on the calendar clock: identical outcome.

### 3. The leaky counterfactual scores *higher* than the shipped column

`rt/probes.py` shifts each feature FORWARD by k stamps (i.e. hands it the future) and
BACKWARD by k (staler). A pipeline that had already leaked would not improve when given more
future. INTC/event, vs `label_y_mid_1s`:

| feature | shuffled y | bwd-10 | bwd-1 | **shipped** | fwd+1 | fwd+5 | fwd+10 |
|---|---|---|---|---|---|---|---|
| `ofi_best_100ms` | 0.0022 | 0.1639 | 0.3458 | **0.3743** | 0.4076 | 0.4992 | 0.5420 |
| `ofi_l1_100ms` | 0.0035 | 0.1933 | 0.3819 | **0.4100** | 0.4406 | 0.5281 | 0.5803 |

AMZN/event `ofi_best_100ms`: 0.2705 shipped -> 0.4780 at fwd+1. `tr_aggr_imb_100ms` on
AMZN/calendar: 0.3221 shipped -> 0.6202 at fwd+1. The shipped column always sits strictly on
the causal side of the boundary. Label shuffle collapses every feature to |rho| < 0.01. A
constant predictor gives Spearman NaN (no variance), as it must.

### 4. The contemporaneous relation is far stronger than the forward one

`rt/decomp.py`, INTC/event: `rho(ofi_best_100ms, mid(t) - mid(t-1s)) = 0.7102` against
`rho(..., mid(t+1s) - mid(t)) = 0.3743`. A boundary leak pulls the forward number *toward*
the contemporaneous one. It is half of it. Same test on INTC/calendar: 0.0401 backward,
0.0342 forward.

### 5. The event-clock advantage is a mix effect, not a signal effect

`rt/final.py`. Condition on the label actually being non-flat (`label_c_mid_1s != 0`) and on
the window being active (`ofi != 0`):

| INTC | frac non-flat | `qi_touch` rho pooled | rho \| non-flat & active | balanced dir-acc |
|---|---|---|---|---|
| calendar | 0.037 | 0.1611 | **0.6158** | 0.801 |
| event | 0.124 | 0.3961 | **0.6033** | 0.829 |

For queue imbalance the clock gap vanishes completely once the tie mass is removed: 96.3% of
INTC calendar 1s labels are exactly flat vs 87.6% on the event clock, and 77.7% of calendar
stamps have `ofi` exactly 0. The pooled Spearman was measuring the tie fraction. 78-83%
balanced directional accuracy for queue imbalance on the next mid move of a large-tick name
is the textbook number, not a discovery and not a leak.

For OFI a residual gap survives conditioning (calendar 0.263 vs event 0.474). `rt/clockcond.py`
localises it to burst oversampling, stratifying event stamps by the gap to the previous
event stamp:

```
gap [   0.0,   0.5)ms  n=20043  rho(ofi)= 0.4371  rho(qi)= 0.5305
gap [   2.0,  10.0)ms  n= 5897  rho(ofi)= 0.1846  rho(qi)= 0.3591
gap [ 200.0,1000.0)ms  n=11974  rho(ofi)= 0.1918  rho(qi)= 0.2551
gap > 1000ms           n= 7929  rho(ofi)= 0.1994  rho(qi)= 0.2106
```

32% of INTC event stamps sit within 0.5 ms of the previous one. Those carry rho 0.44; the
isolated ones carry 0.19. Matching on activity instead (calendar stamps whose trailing 100ms
holds >=10 events) gives calendar rho(ofi)=0.1457 against event 0.3650 at the same cut -
the remaining factor is that an event-clock stamp's 100ms window holds a median of 27 events
(mean 97) against a calendar median of 0 (mean 2.7). The correlations report's own stated
boring explanation ("a selection effect on WHEN you look, not a leak in WHAT you look at")
is correct and now has a number behind it.

### 6. It is not a 1s effect

`rt/decomp.py` splits the label into `mid(t+d) - mid(t)` and `mid(t+1s) - mid(t+d)`.
INTC/event, `ofi_best_100ms`:

| d | rho(ofi, fast part) | rho(ofi, slow part) |
|---|---|---|
| 1 ms | 0.2549 | 0.2948 |
| 10 ms | 0.4150 | 0.1331 |
| 100 ms | **0.4584** | **0.0273** |
| 500 ms | 0.4004 | 0.0132 |

Everything the feature knows is spent within 100 ms. Pushing the feature window itself back
by 1/10/100/1000 ms drops rho to 0.279 / 0.129 / 0.068 / -0.010. This is a real but
extremely short-lived alpha being scored against a 1s label. See finding **F4**.

---

## Findings, ranked

### F1 (medium), the event clock violates INTERFACE.md sec. 3's `<= t` index contract, untested

`clocks.event_stamps` (src/edge/features/clocks.py:87-88) sets
`index[i] = stamp_positions[i]`, the triggering event's own row. INTERFACE.md sec. 3 says
`index[i]` is "position in `frame` of the **last** event with `frame.ts_ns <= ts_ns[i]`".
When several messages share a nanosecond these differ. `rt/ties.py`:

```
INTC/event: stamps=62048  stamps where stamps.index != last row with ts<=t: 6503 (10.48%)  max rows skipped=97
AMZN/event: stamps=26730  ...                                                804 ( 3.01%)  max rows skipped=22
```

`tests/test_clocks.py::test_index_is_last_qualifying_event_not_merely_a_qualifying_one`
(line 55) asserts this contract, but it is written against `calendar_stamps` only; there is
no event-clock equivalent, and the 8-event toy fixture contains no timestamp tie, so the
case is unreachable in the suite.

Direction of harm: for *features* it is conservative (they read fewer rows than they are
entitled to). For *labels* it is not. `labels.compute` anchors on `book.mid[i]` (row
`stamps.index[i]`) but resolves the forward endpoint with the correct "last row <= t+h" rule.
So on the event clock `label_y_mid_1s` silently contains a `mid(t) - mid_partial(t)`
component that elapsed in **zero wall-clock time** and was already public at `t` under this
project's own definition of "the state at t":

```
INTC: mid(last row<=t) - mid(stamp row), ticks: nonzero on 821 stamps, mean=-0.00044, max|.|=1.00
      rho(ofi_as_shipped, that same-nanosecond mid move) = 0.1050
AS SHIPPED (index=triggering event row)   rho(ofi)=0.3743  rho(qi)=0.3961
CONTRACT   (index=last row ts<=t)         rho(ofi)=0.3738  rho(qi)=0.3754
```

Cost: +0.021 on `qi_touch`/`qi_l1` (5.5% relative), +0.0005 on OFI. Small, and it would
appear identically under protocols A, B and C, so it does not bias the optimism
decomposition. But it is a real definitional leak into the label on the event clock and it
is unasserted. Fix is one line (`np.searchsorted(ts, ts[stamp_positions], side="right") - 1`)
plus the missing test.

### F2 (medium), the microprice recursion does not converge; `mp_micro` is a truncation of a divergent-in-the-mean series

`microprice._micro_adjustment` runs `G^(i+1) = G^1 + B G^i` six times with a `TOL` early
exit, and the docstring justifies the fixed depth by "B^i G^1 -> 0 when the mid is a
martingale under the fitted chain". On this data it is not. `rt/mpconv.py`, per-iteration
`||g^(i+1) - g^i||_inf` in raw units:

```
INTC block  1: |g1|max=  42.54  |g6|max=  63.45  step deltas=['14.16','9.42','9.23','9.02','9.00','8.78']
INTC block 13: |g1|max=  43.61  |g6|max=  65.07  step deltas=[ '8.80','3.98','3.88','3.87','3.87','3.85']
AMZN block  1: |g1|max=1650.00  |g6|max=1582.29  step deltas=['39.69','12.52','6.19','3.81','2.92','2.58']
```

The increments plateau instead of decaying. `rt/brows.py` diagnoses it: the arithmetic is
correct (`B` row sums are exactly 1.0, `Q`+`R2` row sums are 1.0), but `|eig(B)|` = {1.0,
0.9906, 0.272} and the stationary drift `pi . g1 = -1.80` raw units = **-0.018 ticks per
iteration**. So `G^i` has a persistent linear term plus a transient that decays as `0.99^i`
- it needs hundreds of iterations, not six, and the limit does not exist. `MICRO_ITERS = 6`
is therefore not "the cap, not a target": it *is* the feature's definition, and `G^7` would
be a materially different column. Frozen a priori from Stoikov's repo, so this is not
tuning, but the docstring's convergence claim is false and should say so.

Consequences on AMZN (small-tick, median spread 12 ticks): `mp_micro` sits **outside
[bid, ask] on 0.32%** of finite stamps, and `|mp_dev_mid|` reaches 15.82 ticks (p50 0.39,
p99 0.70). INTC is clean (0.00% outside, `|mp_dev_mid|` max 0.459 ticks). Not a leak, the
block-refit causality is intact (F-clean list below), but `mp_micro`/`mp_dev_*` on AMZN
carry a fat, meaningless tail that Phase 2 should either winsorise or exclude, with the
decision logged.

### F3 (medium, reporting), the phase-1 leak triage was read off a truncated console print

`correlations.main` prints `suspicious[:20]`. `reports/feature_correlations.md` contains
**488** flagged items, not 19:

```
LEAK SMELL  INTC/calendar   7      SADJ>>RAW  INTC/calendar   8
LEAK SMELL  INTC/event    275      SADJ>>RAW  INTC/event     23
LEAK SMELL  AMZN/calendar  16      SADJ>>RAW  AMZN/calendar   2
LEAK SMELL  AMZN/event    148      SADJ>>RAW  AMZN/event      9
```

"AMZN: zero leak smells at any threshold" is false, AMZN has 164 pairs over |rho|=0.15 at
the 1s horizon, and the **largest single flagged correlation in the entire phase-1 report is
AMZN, not INTC**: `AMZN/event tr_aggr_imb_100ms vs label_y_mid_1s, rho = +0.5307`. It passes
the same causality probes (`rt/aggr.py`: shipped 0.5307, fwd+1 0.6293, bwd-1 0.4538, 81.5%
balanced directional accuracy on non-flat labels), so it is not a leak either, but a triage
that concluded "AMZN is clean" was reading a 20-line window on a 488-line list.

### F4 (low), the 1s label is measuring a <=100ms effect

Quantified in section 6 above: `rho(ofi_best_100ms, mid(t+1s) - mid(t+100ms)) = 0.027` on
INTC/event against 0.458 for the first 100ms. Phase 2 must not present a `label_y_mid_1s`
result as one-second alpha, and the cost model in Phase 3 has to price a signal whose entire
content decays inside 100 ms.

### F5 (low), the `tr_last_trade_sign` widening is a 0-row no-op on this sample

The 2026-07-31 change is correct in principle. Measured effect (`rt/labelnan.py`): on all
four ticker/clock tables, `tr_last_trade_sign` and `tr_eff_spread` have **identical**
non-NaN sets (`sign-finite where eff NaN: 0` everywhere), because the sample has zero invalid
touches. So the claim "widened the column's non-NaN set only where the aggressor sign is
genuinely known" widens it by exactly zero rows here, and the difference is exercised only by
synthetic fixtures. `sign values = [-1.0, 1.0]`, never 0, never NaN-where-known. Correct,
just untested on real data.

### F6 (low), `tr_aggr_imb_*` correlations are conditioned on a trade having occurred

`tr_aggr_imb_100ms` is NaN when the window holds no trades. Finite on **1.8%** of AMZN
calendar stamps, 28.6% of AMZN event stamps, 53.7% of INTC event stamps. Its rho is
therefore not comparable with `qi_touch`'s (finite on ~100%). INTERFACE.md sec. 7 puts the
NaN drop in the harness, per fold, this column will make fold row counts differ by more than
an order of magnitude across rungs, which Phase 2 has to report rather than absorb.

### F7 (low), the trial log is empty against a ~7,264-cell multiple-comparison surface

`RESULTS.md`'s trial log has one blank row. Phase 1 ran 4 (ticker x clock) feature builds,
a 7,264-row univariate correlation grid (`reports/feature_correlations.csv`, 7,265 lines),
a 5-point alpha sweep and 52 OFI self-check regressions. No model has been fitted, so there
is arguably nothing to log yet, but the "LEAK SMELL" thresholds are applied to the max over
that surface, and the surface size is nowhere recorded.

---

## Clean, verified, not assumed

**Row alignment.** Re-verified independently from the raw CSVs (`rt/` inline script), not
taken on trust: at message row 102 a type-1 bid add at 275200 with the previous best bid at
275100 shows `orderbook[102].bid_px_1 == 275200` and `orderbook[101].bid_px_1 != 275200`; a
type-3 delete of the entire best bid at row 1760 vacates the level at row 1760. Row k is the
state **after** message k. The `<= t` convention is safe on that basis.

**Boundary conventions.** `rt/bnd.py`: an event exactly `w` old is excluded, an event exactly
at `t` is included, an event at `t + eps` is excluded; `refit_fit_end_index` with a frame row
exactly at the block start returns 1 not 2, `side="left"`, strictly causal.
`trailing_bounds`'s `end_exclusive = stamps.index[i] + 1` never reaches past the stamp row.
`_event_contributions` and `_estimate_block_adjustment` both attribute a `[1:] vs [:-1]`
difference to the **later** row (`contrib[1:] = ...`), which is the correct direction.

**No lookahead in features.** Grep found no `shift` / `rolling` / centred window /
`min_periods` / `.iloc[i+` anywhere under `src/edge/features/`. The only `x.mean(axis=0)` is
`ofi._estimate_first_pc` operating on a strictly-prior fit slice. The perturbation and
truncation tests above are the executable proof.

**Normalization, `rolling_zscore`.** `rt/norm.py` checks it against a naive O(n^2)
`[t-window, t)` reference over 18 configurations (NaN holes, zero variance, heavy ties,
a 1e5 offset, windows 5/50/400): all agree to 1e-9. Mutating `x[300:]` leaves `z[0:300]`
bit-identical while `z[300]` changes, the baseline excludes `x[t]` and cannot see past it.
`_causal_offset` is a constant shift taken from the first finite value; it cancels exactly in
`(x - mean)/std` and leaks nothing.

**Normalization, `expanding_bucket_zscore` (the 113 `_sadj` twins).** Same naive-reference
test against "strictly earlier stamps in the same bucket": agrees to 1e-9 on every trial.
`window=pos.size` with `lo = max(0, idx - window)` makes the baseline genuinely expanding,
never full-sample. `build.build_table` line 101 exempts `label_`-prefixed columns from the
twin; the built tables contain no `label_*_sadj` column. And the whole-future-scramble test
covers all 113 `_sadj` columns at once: none moved. The 42 `SADJ>>RAW` flags are all
noise-level (largest is `AMZN/event mp_micro_sadj` at |IC| = 0.0556 against a raw 0.0007) -
a ratio artifact of dividing by a near-zero raw correlation, not manufactured signal.

**No full-sample scaler.** `grep -rn "\.fit(\|StandardScaler\|sklearn\|scipy\|pandas" src/`
returns nothing under `src/edge/features/` (only prose in `src/edge/analysis/*` saying they
are not used). Treated as necessary-not-sufficient per instruction: the arithmetic was
checked directly above, and the perturbation test proves no statistic anywhere in the table
is estimated on data later than the row it is applied to.

**Fitted statistics (INTERFACE.md sec. 8's three failure modes).** `rt/fitted.py`,
INTC/calendar: `fit_end_index[0] == 0`; block 0's 17,999 stamps are all-NaN for `mp_micro`,
`ofi_integrated_1s` and `ofi_l1_1s`, while the unfitted `ofi_raw_l1_1s` is not, the NaN is
the refit rule, not a blanket. The 13 microprice adjustment vectors are pairwise distinct
(no reuse of one fit). The 13 PCA loading vectors all have `sign(loading[0]) = +1` with
consecutive-block inner products >= 0.996 (INTC) / 0.921 (AMZN), **no sign flips**. The PCA
is fitted on per-**event** rows (`raw_win_events[:fe, :]`), not on the stamp array. Fitting
once on `fit_end_index[-1]` is ruled out by the perturbation test (it would have moved every
early stamp; nothing moved).

**Microprice units, no double conversion.** `mp_dev_mid == mp_micro - ticks(book.mid)` to
2.2e-13 on INTC and 1.7e-12 on AMZN. `mp_micro` (2661-2749 ticks on INTC) is on the same
scale as `ticks(mid)` (2661-2759). The raw->ticks move did not double-convert the deviations.

**Label NaN rules.** `rt/labelnan.py`, all 4 tables x {1s, 10s}: the observed NaN count
equals `count(t + h > min(last event ts, SESSION_END_NS))` exactly, mismatch 0 everywhere;
the final stamp's label is NaN, never clipped to the last available price; `_e10`'s last 10
rows are NaN and exactly 10 rows are NaN in total.

**Deadband tuning.** Frozen by rule, not hindsight. `DECISIONS.md` line 281 (2026-07-31) and
the `RESULTS.md` frozen-decisions row carry the same date, and `RESULTS.md`'s headline ladder
and leakage-diagnostics tables are **entirely blank**, there is no fold result in existence
that alpha could have been chosen against. The 5-point sweep in
`reports/feature_correlations.md` is reported as sensitivity and is not used to select.

**Base-rate artifact.** Correctly pre-empted, not discovered by me. `labels.py`'s own BASE
RATE WARNING and the report's base-rate table are both present and correct: INTC/calendar
`label_c_mid_1s` is 96.3% flat, so an always-flat predictor scores 0.963 accuracy and 0.333
three-class balanced accuracy. My independent recomputation matches. Every accuracy figure I
quote above is balanced, computed on the non-flat subset, and stated as such.

---

## The four standard checks I could not run, and why

- **The L1 control's A-C gap.** Not measurable at Phase 1. There is no harness, no splitter,
  and no protocol A/B/C implementation in `src/`; `RESULTS.md`'s ladder is blank. This is the
  single most important check in the project and it is entirely ahead of us.
- **Protocol A with k=2 and k=10.** Same reason, no splitter exists yet.
- **Constant predictor under all three protocols.** Same reason. I ran the Phase-1 analogue:
  a constant column yields Spearman NaN (zero variance), which is the correct degenerate
  answer, not a spurious number.
- **Pooled-number laundering / worst fold.** No folds exist. But note that every number in
  `reports/feature_correlations.md` is a whole-session pooled statistic, and section 5 above
  shows two of them (INTC/event `qi_touch` 0.3961, `ofi_best_100ms` 0.3743) decompose into
  strata differing by more than 2x. The report says "THIS IS NOT A RESULT" in bold; that
  disclaimer is doing real work and must survive into Phase 2's framing.
