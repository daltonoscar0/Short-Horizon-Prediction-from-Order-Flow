# Feature x label correlation triage (Phase 1)

`python3 -m edge.analysis.correlations --seed 0`

**THIS IS NOT A RESULT.** Every number in this report is a univariate, in-sample, whole-session Pearson/Spearman correlation between one feature column and one forward label, computed for feature triage only. There is no purged walk-forward split, no purge, no embargo, and no held-out fold behind any of it (EVAL_PROTOCOL.md sec. 1-2). Nothing here may be quoted as evidence of predictive value; Phase 2's harness is where numbers count. The Spearman rho column is nonetheless the information coefficient as EVAL_PROTOCOL.md sec. 2 defines it: Spearman rho between prediction and realized move.

Grid: 2 tickers x 2 clocks x 8 labels x 227 non-label feature columns = 7264 rows in `feature_correlations.csv`. Labels: `label_y_mid_h` / `label_y_wmid_h` for h in ('1s', '2s', '5s', '10s') (FEATURES.md sec. 7's `y_h` primary and its wmid twin).

**Methodology, hand-written on numpy (no scipy/pandas/sklearn):** Pearson r (mean-centred dot product); Spearman IC via a hand-written average-rank transform (tied values share the mean of the ranks they span), ranked once per column on that column's own finite values and then masked pairwise per (feature, label) pair -- see module docstring's "Spearman rank convention" for why this is not bit-identical to re-ranking the exact pairwise intersection, and why that gap is negligible here; `n` is pairwise-complete (`isfinite(x) & isfinite(y)`), reported per row, since a column with a NaN block (e.g. a refit-block-0 warmup) legitimately has fewer usable rows than one without; `t_hac` reuses `edge.analysis.ofi_selfcheck._ols_with_hac` (hand-written Newey-West/Bartlett HAC), because overlapping labels on a fine grid make the naive t wildly overstated. HAC lag is `ofi_selfcheck`'s own a-priori rule, generalised from its fixed 1s case to all four horizons: calendar clock lag = 10 x horizon-seconds (10 stamps/sec on the 100ms grid, exact); event clock lag = median event-clock stamps per calendar second (computed per ticker) x horizon-seconds.

Event-clock HAC lag inputs:

- INTC/event median event-clock stamps per calendar second: 1 (mean 2.63, p90 4, p99 38, max 151) -> HAC lags of 1/2/5/10 at 1s/2s/5s/10s.
- AMZN/event median event-clock stamps per calendar second: 1 (mean 1.14, p90 3, p99 9, max 36) -> HAC lags of 1/2/5/10 at 1s/2s/5s/10s.

## Stamps and NaN accounting

Total stamps per (ticker, clock), and how many are usable (finite label) at each horizon x label-type. A horizon's usable count is strictly less than the stamp count at the session's tail, where `t+h` runs past the last event or past `SESSION_END_NS` (labels.py's NaN rule, never clipped/back-filled).

| ticker | clock | n stamps | usable mid_1s | usable wmid_1s | usable mid_2s | usable wmid_2s | usable mid_5s | usable wmid_5s | usable mid_10s | usable wmid_10s |
|---|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | 234000 | 233989 (100.0%) | 233989 (100.0%) | 233979 (100.0%) | 233979 (100.0%) | 233949 (100.0%) | 233949 (100.0%) | 233899 (100.0%) | 233899 (100.0%) |
| INTC | event | 61460 | 61437 (100.0%) | 61437 (100.0%) | 61422 (99.9%) | 61422 (99.9%) | 61328 (99.8%) | 61328 (99.8%) | 61173 (99.5%) | 61173 (99.5%) |
| AMZN | calendar | 234000 | 233989 (100.0%) | 233989 (100.0%) | 233979 (100.0%) | 233979 (100.0%) | 233949 (100.0%) | 233949 (100.0%) | 233899 (100.0%) | 233899 (100.0%) |
| AMZN | event | 26724 | 26719 (100.0%) | 26719 (100.0%) | 26716 (100.0%) | 26716 (100.0%) | 26707 (99.9%) | 26707 (99.9%) | 26688 (99.9%) | 26688 (99.9%) |

## Alpha sensitivity of the classification labels

**alpha = 0.5 ticks (half a tick) is FROZEN BY RULE** (FEATURES.md sec. 7's own first suggestion, `edge.features.labels.ALPHA_TICKS`) -- not fit to this or any other data. This table is reported to show sensitivity, not to pick a value: a bigger alpha manufactures a bigger "flat" class and can make a trivial always-flat predictor look accurate, which is exactly why FEATURES.md sec. 7 warns against tuning it against test folds. Swept on `label_y_mid_h` (the primary `y_h`, not the wmid twin) via `edge.features.labels.classify` -- imported, not reimplemented.

| ticker | clock | horizon | alpha (ticks) | n | frac up | frac flat | frac down |
|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | 0.0 | 233989 | 2.0% | 95.7% | 2.3% |
| INTC | calendar | 1s | 0.25 | 233989 | 2.0% | 95.7% | 2.3% |
| INTC | calendar | 1s | 0.5 **FROZEN** | 233989 | 1.7% | 96.3% | 2.0% |
| INTC | calendar | 1s | 1.0 | 233989 | 0.1% | 99.7% | 0.1% |
| INTC | calendar | 1s | 2.0 | 233989 | 0.0% | 100.0% | 0.0% |
| INTC | calendar | 2s | 0.0 | 233979 | 3.7% | 92.1% | 4.2% |
| INTC | calendar | 2s | 0.25 | 233979 | 3.7% | 92.1% | 4.2% |
| INTC | calendar | 2s | 0.5 **FROZEN** | 233979 | 3.2% | 93.0% | 3.8% |
| INTC | calendar | 2s | 1.0 | 233979 | 0.3% | 99.4% | 0.4% |
| INTC | calendar | 2s | 2.0 | 233979 | 0.0% | 99.9% | 0.0% |
| INTC | calendar | 5s | 0.0 | 233949 | 7.8% | 83.4% | 8.8% |
| INTC | calendar | 5s | 0.25 | 233949 | 7.8% | 83.4% | 8.8% |
| INTC | calendar | 5s | 0.5 **FROZEN** | 233949 | 7.1% | 84.7% | 8.2% |
| INTC | calendar | 5s | 1.0 | 233949 | 0.8% | 98.0% | 1.2% |
| INTC | calendar | 5s | 2.0 | 233949 | 0.1% | 99.6% | 0.2% |
| INTC | calendar | 10s | 0.0 | 233899 | 12.6% | 72.9% | 14.5% |
| INTC | calendar | 10s | 0.25 | 233899 | 12.6% | 72.9% | 14.5% |
| INTC | calendar | 10s | 0.5 **FROZEN** | 233899 | 11.8% | 74.3% | 13.9% |
| INTC | calendar | 10s | 1.0 | 233899 | 1.9% | 95.4% | 2.7% |
| INTC | calendar | 10s | 2.0 | 233899 | 0.3% | 99.0% | 0.7% |
| INTC | event | 1s | 0.0 | 61437 | 13.4% | 69.3% | 17.3% |
| INTC | event | 1s | 0.25 | 61437 | 13.4% | 69.3% | 17.3% |
| INTC | event | 1s | 0.5 **FROZEN** | 61437 | 4.5% | 88.8% | 6.7% |
| INTC | event | 1s | 1.0 | 61437 | 0.8% | 98.1% | 1.1% |
| INTC | event | 1s | 2.0 | 61437 | 0.1% | 99.7% | 0.2% |
| INTC | event | 2s | 0.0 | 61422 | 14.5% | 64.8% | 20.7% |
| INTC | event | 2s | 0.25 | 61422 | 14.5% | 64.8% | 20.7% |
| INTC | event | 2s | 0.5 **FROZEN** | 61422 | 6.0% | 84.7% | 9.3% |
| INTC | event | 2s | 1.0 | 61422 | 1.2% | 96.8% | 2.1% |
| INTC | event | 2s | 2.0 | 61422 | 0.3% | 99.3% | 0.4% |
| INTC | event | 5s | 0.0 | 61328 | 19.1% | 56.0% | 24.9% |
| INTC | event | 5s | 0.25 | 61328 | 19.1% | 56.0% | 24.9% |
| INTC | event | 5s | 0.5 **FROZEN** | 61328 | 10.5% | 75.0% | 14.4% |
| INTC | event | 5s | 1.0 | 61328 | 2.9% | 92.3% | 4.8% |
| INTC | event | 5s | 2.0 | 61328 | 0.8% | 97.8% | 1.4% |
| INTC | event | 10s | 0.0 | 61173 | 22.1% | 48.4% | 29.5% |
| INTC | event | 10s | 0.25 | 61173 | 22.1% | 48.4% | 29.5% |
| INTC | event | 10s | 0.5 **FROZEN** | 61173 | 13.7% | 66.7% | 19.6% |
| INTC | event | 10s | 1.0 | 61173 | 3.8% | 89.0% | 7.2% |
| INTC | event | 10s | 2.0 | 61173 | 1.0% | 96.5% | 2.5% |
| AMZN | calendar | 1s | 0.0 | 233989 | 14.4% | 69.9% | 15.7% |
| AMZN | calendar | 1s | 0.25 | 233989 | 14.4% | 69.9% | 15.7% |
| AMZN | calendar | 1s | 0.5 **FROZEN** | 233989 | 10.3% | 78.4% | 11.3% |
| AMZN | calendar | 1s | 1.0 | 233989 | 7.7% | 83.8% | 8.5% |
| AMZN | calendar | 1s | 2.0 | 233989 | 5.1% | 89.6% | 5.3% |
| AMZN | calendar | 2s | 0.0 | 233979 | 21.8% | 54.2% | 24.0% |
| AMZN | calendar | 2s | 0.25 | 233979 | 21.8% | 54.2% | 24.0% |
| AMZN | calendar | 2s | 0.5 **FROZEN** | 233979 | 16.4% | 65.3% | 18.3% |
| AMZN | calendar | 2s | 1.0 | 233979 | 12.9% | 73.0% | 14.1% |
| AMZN | calendar | 2s | 2.0 | 233979 | 8.8% | 81.9% | 9.3% |
| AMZN | calendar | 5s | 0.0 | 233949 | 33.1% | 30.2% | 36.6% |
| AMZN | calendar | 5s | 0.25 | 233949 | 33.1% | 30.2% | 36.6% |
| AMZN | calendar | 5s | 0.5 **FROZEN** | 233949 | 26.8% | 43.2% | 30.0% |
| AMZN | calendar | 5s | 1.0 | 233949 | 22.3% | 53.0% | 24.7% |
| AMZN | calendar | 5s | 2.0 | 233949 | 16.7% | 65.3% | 18.0% |
| AMZN | calendar | 10s | 0.0 | 233899 | 40.0% | 15.0% | 45.0% |
| AMZN | calendar | 10s | 0.25 | 233899 | 40.0% | 15.0% | 45.0% |
| AMZN | calendar | 10s | 0.5 **FROZEN** | 233899 | 34.6% | 26.4% | 39.0% |
| AMZN | calendar | 10s | 1.0 | 233899 | 30.4% | 35.4% | 34.2% |
| AMZN | calendar | 10s | 2.0 | 233899 | 24.2% | 48.5% | 27.3% |
| AMZN | event | 1s | 0.0 | 26719 | 28.0% | 42.7% | 29.4% |
| AMZN | event | 1s | 0.25 | 26719 | 28.0% | 42.7% | 29.4% |
| AMZN | event | 1s | 0.5 **FROZEN** | 26719 | 22.0% | 55.0% | 23.0% |
| AMZN | event | 1s | 1.0 | 26719 | 17.7% | 63.9% | 18.4% |
| AMZN | event | 1s | 2.0 | 26719 | 12.4% | 75.1% | 12.5% |
| AMZN | event | 2s | 0.0 | 26716 | 33.1% | 31.8% | 35.1% |
| AMZN | event | 2s | 0.25 | 26716 | 33.1% | 31.8% | 35.1% |
| AMZN | event | 2s | 0.5 **FROZEN** | 26716 | 27.0% | 44.6% | 28.5% |
| AMZN | event | 2s | 1.0 | 26716 | 22.3% | 54.4% | 23.3% |
| AMZN | event | 2s | 2.0 | 26716 | 16.1% | 67.3% | 16.6% |
| AMZN | event | 5s | 0.0 | 26707 | 40.1% | 16.7% | 43.2% |
| AMZN | event | 5s | 0.25 | 26707 | 40.1% | 16.7% | 43.2% |
| AMZN | event | 5s | 0.5 **FROZEN** | 26707 | 34.7% | 27.9% | 37.3% |
| AMZN | event | 5s | 1.0 | 26707 | 30.2% | 37.5% | 32.3% |
| AMZN | event | 5s | 2.0 | 26707 | 23.7% | 51.3% | 24.9% |
| AMZN | event | 10s | 0.0 | 26688 | 42.9% | 8.7% | 48.3% |
| AMZN | event | 10s | 0.25 | 26688 | 42.9% | 8.7% | 48.3% |
| AMZN | event | 10s | 0.5 **FROZEN** | 26688 | 38.9% | 17.7% | 43.3% |
| AMZN | event | 10s | 1.0 | 26688 | 35.3% | 25.4% | 39.3% |
| AMZN | event | 10s | 2.0 | 26688 | 29.3% | 38.0% | 32.7% |

## Base rate of the frozen (alpha=0.5) classification labels

Any Phase-2 classification accuracy has to be read against these base rates, not against 33% or 50% -- at short horizons almost every label is flat (labels.py's own "BASE RATE WARNING").

| ticker | clock | label | n total | n finite | frac up | frac flat | frac down |
|---|---|---|---|---|---|---|---|
| INTC | calendar | label_c_mid_1s | 234000 | 233989 | 1.7% | 96.3% | 2.0% |
| INTC | calendar | label_c_mid_2s | 234000 | 233979 | 3.2% | 93.0% | 3.8% |
| INTC | calendar | label_c_mid_5s | 234000 | 233949 | 7.1% | 84.7% | 8.2% |
| INTC | calendar | label_c_mid_10s | 234000 | 233899 | 11.8% | 74.3% | 13.9% |
| INTC | calendar | label_c_wmid_1s | 234000 | 233989 | 1.8% | 96.1% | 2.1% |
| INTC | calendar | label_c_wmid_2s | 234000 | 233979 | 3.3% | 92.8% | 3.9% |
| INTC | calendar | label_c_wmid_5s | 234000 | 233949 | 7.2% | 84.4% | 8.4% |
| INTC | calendar | label_c_wmid_10s | 234000 | 233899 | 12.0% | 73.8% | 14.1% |
| INTC | calendar | label_c_mid_e1 | 234000 | 233999 | 0.2% | 99.6% | 0.2% |
| INTC | calendar | label_c_mid_e2 | 234000 | 233998 | 0.4% | 99.2% | 0.4% |
| INTC | calendar | label_c_mid_e5 | 234000 | 233995 | 0.9% | 98.0% | 1.0% |
| INTC | calendar | label_c_mid_e10 | 234000 | 233990 | 1.7% | 96.3% | 2.0% |
| INTC | event | label_c_mid_1s | 61460 | 61437 | 4.5% | 88.8% | 6.7% |
| INTC | event | label_c_mid_2s | 61460 | 61422 | 6.0% | 84.7% | 9.3% |
| INTC | event | label_c_mid_5s | 61460 | 61328 | 10.5% | 75.0% | 14.4% |
| INTC | event | label_c_mid_10s | 61460 | 61173 | 13.7% | 66.7% | 19.6% |
| INTC | event | label_c_wmid_1s | 61460 | 61437 | 5.5% | 83.7% | 10.8% |
| INTC | event | label_c_wmid_2s | 61460 | 61422 | 7.1% | 78.6% | 14.3% |
| INTC | event | label_c_wmid_5s | 61460 | 61328 | 11.8% | 68.7% | 19.5% |
| INTC | event | label_c_wmid_10s | 61460 | 61173 | 15.8% | 59.8% | 24.4% |
| INTC | event | label_c_mid_e1 | 61460 | 61459 | 0.1% | 99.8% | 0.1% |
| INTC | event | label_c_mid_e2 | 61460 | 61458 | 0.3% | 99.3% | 0.4% |
| INTC | event | label_c_mid_e5 | 61460 | 61455 | 1.0% | 97.5% | 1.5% |
| INTC | event | label_c_mid_e10 | 61460 | 61450 | 2.8% | 93.0% | 4.1% |
| AMZN | calendar | label_c_mid_1s | 234000 | 233989 | 10.3% | 78.4% | 11.3% |
| AMZN | calendar | label_c_mid_2s | 234000 | 233979 | 16.4% | 65.3% | 18.3% |
| AMZN | calendar | label_c_mid_5s | 234000 | 233949 | 26.8% | 43.2% | 30.0% |
| AMZN | calendar | label_c_mid_10s | 234000 | 233899 | 34.6% | 26.4% | 39.0% |
| AMZN | calendar | label_c_wmid_1s | 234000 | 233989 | 16.6% | 66.1% | 17.3% |
| AMZN | calendar | label_c_wmid_2s | 234000 | 233979 | 24.5% | 50.0% | 25.6% |
| AMZN | calendar | label_c_wmid_5s | 234000 | 233949 | 35.2% | 27.4% | 37.4% |
| AMZN | calendar | label_c_wmid_10s | 234000 | 233899 | 41.2% | 14.7% | 44.1% |
| AMZN | calendar | label_c_mid_e1 | 234000 | 233999 | 1.5% | 96.9% | 1.6% |
| AMZN | calendar | label_c_mid_e2 | 234000 | 233998 | 2.8% | 94.3% | 3.0% |
| AMZN | calendar | label_c_mid_e5 | 234000 | 233995 | 6.1% | 87.3% | 6.6% |
| AMZN | calendar | label_c_mid_e10 | 234000 | 233990 | 10.3% | 78.4% | 11.3% |
| AMZN | event | label_c_mid_1s | 26724 | 26719 | 22.0% | 55.0% | 23.0% |
| AMZN | event | label_c_mid_2s | 26724 | 26716 | 27.0% | 44.6% | 28.5% |
| AMZN | event | label_c_mid_5s | 26724 | 26707 | 34.7% | 27.9% | 37.3% |
| AMZN | event | label_c_mid_10s | 26724 | 26688 | 38.9% | 17.7% | 43.3% |
| AMZN | event | label_c_wmid_1s | 26724 | 26719 | 30.8% | 38.6% | 30.6% |
| AMZN | event | label_c_wmid_2s | 26724 | 26716 | 35.6% | 28.4% | 36.0% |
| AMZN | event | label_c_wmid_5s | 26724 | 26707 | 41.9% | 15.7% | 42.5% |
| AMZN | event | label_c_wmid_10s | 26724 | 26688 | 44.0% | 9.1% | 46.9% |
| AMZN | event | label_c_mid_e1 | 26724 | 26723 | 16.1% | 67.0% | 16.9% |
| AMZN | event | label_c_mid_e2 | 26724 | 26722 | 24.4% | 49.5% | 26.1% |
| AMZN | event | label_c_mid_e5 | 26724 | 26719 | 35.6% | 26.2% | 38.2% |
| AMZN | event | label_c_mid_e10 | 26724 | 26714 | 40.8% | 14.3% | 44.9% |

## Seasonality adjustment: what it did to the ranking

Whole grid (3584 raw/sadj pairs with finite IC on both sides, mostly noise-level correlations): seasonality adjustment makes |IC| larger in 1260 (35.2%) and smaller in 2324 (64.8%); mean |IC| shift (sadj - raw) = -0.00520, median -0.00276 -- an absolute shift this small is not informative on its own when most raw |IC| values here are themselves under 0.05.

Higher-signal subset (|raw IC| > 0.1, n=1176, the features a reviewer would actually look at): seasonality adjustment shrinks |IC| on average, by -5.4% relative (median -4.2%); shrinks 854 (72.6%) and grows 322 (27.4%) of them. Read as: a modest, mostly-shrinking correction -- some of the raw signal in the strongest features is intraday level/seasonality, not all of it.

## Suspicious

**Read the LEAK SMELL list below against the clock, not just the number.** At the 1s horizon, raw-feature mean |IC| is 0.031 on the calendar clock vs 0.145 on the event clock (median 0.015 vs 0.124); 2.6% of calendar-clock features clear the 0.15 threshold vs 45.8% of event-clock features. The likely (boring) explanation: event-clock stamps are book-changing events by construction (clocks.py), so they sample exactly the moments order flow is active, not uniformly in time -- a selection effect on WHEN you look, not a leak in WHAT you look at. This does not un-flag the individual items (the threshold is on the number, and the number is what Phase 2 would have to survive out-of-sample on this clock), but it is the reason most of them cluster on `clock=event`.

483 flagged item(s) below. Flagging does not mean the number is wrong -- it means it clears a threshold chosen because a real univariate relationship at this horizon, in 2012 large/small-tick equities, should not be this large or this shaped, so it is reported rather than presented as a result.

- LEAK SMELL: INTC/calendar qi_touch vs label_y_mid_1s: Spearman rho=0.1611 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/calendar qi_touch_sadj vs label_y_mid_1s: Spearman rho=0.1587 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/calendar qi_l1 vs label_y_mid_1s: Spearman rho=0.1611 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/calendar qi_l1_sadj vs label_y_mid_1s: Spearman rho=0.1587 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/calendar mp_wmid_dev_mid vs label_y_mid_1s: Spearman rho=0.1616 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/calendar mp_wmid_dev_mid_sadj vs label_y_mid_1s: Spearman rho=0.1566 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/calendar tr_aggr_imb_100ms vs label_y_mid_1s: Spearman rho=0.1722 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_100ms vs label_y_mid_1s: Spearman rho=0.3661 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_100ms_sadj vs label_y_mid_1s: Spearman rho=0.3254 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_100ms vs label_y_mid_1s: Spearman rho=0.3246 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2535 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_100ms vs label_y_mid_1s: Spearman rho=0.4012 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_100ms_sadj vs label_y_mid_1s: Spearman rho=0.3605 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_100ms vs label_y_mid_1s: Spearman rho=0.3145 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2765 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_100ms vs label_y_mid_1s: Spearman rho=0.2888 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2699 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_100ms vs label_y_mid_1s: Spearman rho=0.2529 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2649 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_100ms vs label_y_mid_1s: Spearman rho=0.2644 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2599 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_100ms vs label_y_mid_1s: Spearman rho=0.2757 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2610 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_100ms vs label_y_mid_1s: Spearman rho=0.2962 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2660 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_100ms vs label_y_mid_1s: Spearman rho=0.2724 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2697 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_100ms vs label_y_mid_1s: Spearman rho=0.2620 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2603 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_100ms vs label_y_mid_1s: Spearman rho=0.2818 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2603 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_100ms vs label_y_mid_1s: Spearman rho=0.3727 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2977 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_1s vs label_y_mid_1s: Spearman rho=0.3177 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_1s_sadj vs label_y_mid_1s: Spearman rho=0.2856 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_1s vs label_y_mid_1s: Spearman rho=0.2758 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_1s_sadj vs label_y_mid_1s: Spearman rho=0.2169 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l1_1s vs label_y_mid_1s: Spearman rho=0.3177 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l1_1s_sadj vs label_y_mid_1s: Spearman rho=0.2856 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l2_1s vs label_y_mid_1s: Spearman rho=0.2239 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l2_1s_sadj vs label_y_mid_1s: Spearman rho=0.1921 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l3_1s vs label_y_mid_1s: Spearman rho=0.2235 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l3_1s_sadj vs label_y_mid_1s: Spearman rho=0.1981 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l4_1s vs label_y_mid_1s: Spearman rho=0.1992 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l4_1s_sadj vs label_y_mid_1s: Spearman rho=0.1986 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l5_1s vs label_y_mid_1s: Spearman rho=0.1883 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l5_1s_sadj vs label_y_mid_1s: Spearman rho=0.1927 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l6_1s vs label_y_mid_1s: Spearman rho=0.1911 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l6_1s_sadj vs label_y_mid_1s: Spearman rho=0.1891 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l7_1s vs label_y_mid_1s: Spearman rho=0.2138 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l7_1s_sadj vs label_y_mid_1s: Spearman rho=0.2034 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l8_1s vs label_y_mid_1s: Spearman rho=0.1941 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l8_1s_sadj vs label_y_mid_1s: Spearman rho=0.1988 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l9_1s vs label_y_mid_1s: Spearman rho=0.1923 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l9_1s_sadj vs label_y_mid_1s: Spearman rho=0.2012 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l10_1s vs label_y_mid_1s: Spearman rho=0.1983 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l10_1s_sadj vs label_y_mid_1s: Spearman rho=0.1959 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_1s vs label_y_mid_1s: Spearman rho=0.3525 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_1s_sadj vs label_y_mid_1s: Spearman rho=0.3229 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_1s vs label_y_mid_1s: Spearman rho=0.2580 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_1s_sadj vs label_y_mid_1s: Spearman rho=0.2300 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_1s vs label_y_mid_1s: Spearman rho=0.2514 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_1s_sadj vs label_y_mid_1s: Spearman rho=0.2291 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_1s vs label_y_mid_1s: Spearman rho=0.2175 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_1s_sadj vs label_y_mid_1s: Spearman rho=0.2251 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_1s vs label_y_mid_1s: Spearman rho=0.2101 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_1s_sadj vs label_y_mid_1s: Spearman rho=0.2206 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_1s vs label_y_mid_1s: Spearman rho=0.2205 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_1s_sadj vs label_y_mid_1s: Spearman rho=0.2220 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_1s vs label_y_mid_1s: Spearman rho=0.2431 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_1s_sadj vs label_y_mid_1s: Spearman rho=0.2299 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_1s vs label_y_mid_1s: Spearman rho=0.2180 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_1s_sadj vs label_y_mid_1s: Spearman rho=0.2239 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_1s vs label_y_mid_1s: Spearman rho=0.2161 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_1s_sadj vs label_y_mid_1s: Spearman rho=0.2253 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_1s vs label_y_mid_1s: Spearman rho=0.2297 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_1s_sadj vs label_y_mid_1s: Spearman rho=0.2203 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_1s vs label_y_mid_1s: Spearman rho=0.3152 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_1s_sadj vs label_y_mid_1s: Spearman rho=0.2544 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_5s vs label_y_mid_1s: Spearman rho=0.2653 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_5s_sadj vs label_y_mid_1s: Spearman rho=0.2434 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_5s vs label_y_mid_1s: Spearman rho=0.2095 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_5s_sadj vs label_y_mid_1s: Spearman rho=0.1701 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_5s vs label_y_mid_1s: Spearman rho=0.2892 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_5s_sadj vs label_y_mid_1s: Spearman rho=0.2676 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_5s vs label_y_mid_1s: Spearman rho=0.2003 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_5s_sadj vs label_y_mid_1s: Spearman rho=0.1773 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_5s vs label_y_mid_1s: Spearman rho=0.1930 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_5s_sadj vs label_y_mid_1s: Spearman rho=0.1747 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_5s vs label_y_mid_1s: Spearman rho=0.1768 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_5s_sadj vs label_y_mid_1s: Spearman rho=0.1776 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_5s vs label_y_mid_1s: Spearman rho=0.1747 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_5s_sadj vs label_y_mid_1s: Spearman rho=0.1748 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_5s vs label_y_mid_1s: Spearman rho=0.1739 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_5s_sadj vs label_y_mid_1s: Spearman rho=0.1723 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_5s vs label_y_mid_1s: Spearman rho=0.1871 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_5s_sadj vs label_y_mid_1s: Spearman rho=0.1792 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_5s vs label_y_mid_1s: Spearman rho=0.1716 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_5s_sadj vs label_y_mid_1s: Spearman rho=0.1783 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_5s vs label_y_mid_1s: Spearman rho=0.1680 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_5s_sadj vs label_y_mid_1s: Spearman rho=0.1737 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_5s vs label_y_mid_1s: Spearman rho=0.1785 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_5s_sadj vs label_y_mid_1s: Spearman rho=0.1723 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_5s vs label_y_mid_1s: Spearman rho=0.2412 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_5s_sadj vs label_y_mid_1s: Spearman rho=0.1995 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_touch vs label_y_mid_1s: Spearman rho=0.3727 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_touch_sadj vs label_y_mid_1s: Spearman rho=0.3759 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_sign vs label_y_mid_1s: Spearman rho=0.3113 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_sign_sadj vs label_y_mid_1s: Spearman rho=0.2931 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_l1 vs label_y_mid_1s: Spearman rho=0.3727 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_l1_sadj vs label_y_mid_1s: Spearman rho=0.3759 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_uniform vs label_y_mid_1s: Spearman rho=0.1765 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_uniform_sadj vs label_y_mid_1s: Spearman rho=0.2184 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_inv vs label_y_mid_1s: Spearman rho=0.3251 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_inv_sadj vs label_y_mid_1s: Spearman rho=0.3372 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_exp vs label_y_mid_1s: Spearman rho=0.3080 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_exp_sadj vs label_y_mid_1s: Spearman rho=0.3219 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event mp_dev_mid vs label_y_mid_1s: Spearman rho=0.2925 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event mp_dev_mid_sadj vs label_y_mid_1s: Spearman rho=0.3076 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event mp_dev_wmid vs label_y_mid_1s: Spearman rho=-0.3842 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event mp_dev_wmid_sadj vs label_y_mid_1s: Spearman rho=-0.3955 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event mp_wmid_dev_mid vs label_y_mid_1s: Spearman rho=0.4098 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event mp_wmid_dev_mid_sadj vs label_y_mid_1s: Spearman rho=0.4102 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_100ms vs label_y_mid_1s: Spearman rho=0.4072 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_100ms_sadj vs label_y_mid_1s: Spearman rho=0.3550 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_buy_vol_100ms vs label_y_mid_1s: Spearman rho=0.2584 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_buy_vol_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2339 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_100ms vs label_y_mid_1s: Spearman rho=-0.3431 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_100ms_sadj vs label_y_mid_1s: Spearman rho=-0.2649 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_100ms vs label_y_mid_1s: Spearman rho=0.4246 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_100ms_sadj vs label_y_mid_1s: Spearman rho=0.4138 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_1s vs label_y_mid_1s: Spearman rho=0.3679 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_1s_sadj vs label_y_mid_1s: Spearman rho=0.3304 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_buy_vol_1s vs label_y_mid_1s: Spearman rho=0.2150 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_buy_vol_1s_sadj vs label_y_mid_1s: Spearman rho=0.1844 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_1s vs label_y_mid_1s: Spearman rho=-0.3259 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_1s_sadj vs label_y_mid_1s: Spearman rho=-0.2830 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_1s vs label_y_mid_1s: Spearman rho=0.3704 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_1s_sadj vs label_y_mid_1s: Spearman rho=0.3640 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_5s vs label_y_mid_1s: Spearman rho=0.3017 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_5s_sadj vs label_y_mid_1s: Spearman rho=0.2781 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_buy_vol_5s vs label_y_mid_1s: Spearman rho=0.1683 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_5s vs label_y_mid_1s: Spearman rho=-0.2807 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_5s_sadj vs label_y_mid_1s: Spearman rho=-0.2643 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_5s vs label_y_mid_1s: Spearman rho=0.3038 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_5s_sadj vs label_y_mid_1s: Spearman rho=0.2904 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_last_trade_sign vs label_y_mid_1s: Spearman rho=0.2657 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_last_trade_sign_sadj vs label_y_mid_1s: Spearman rho=0.2592 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event st_depth_bid_touch vs label_y_mid_1s: Spearman rho=0.2302 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event st_depth_bid_touch_sadj vs label_y_mid_1s: Spearman rho=0.2506 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event st_depth_ask_touch vs label_y_mid_1s: Spearman rho=-0.2063 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event st_depth_ask_touch_sadj vs label_y_mid_1s: Spearman rho=-0.2813 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_100ms vs label_y_wmid_1s: Spearman rho=0.2931 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2410 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_100ms vs label_y_wmid_1s: Spearman rho=0.2662 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2033 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_100ms vs label_y_wmid_1s: Spearman rho=0.3491 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.3066 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_100ms vs label_y_wmid_1s: Spearman rho=0.2909 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2584 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_100ms vs label_y_wmid_1s: Spearman rho=0.2488 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2438 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_100ms vs label_y_wmid_1s: Spearman rho=0.2216 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2422 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_100ms vs label_y_wmid_1s: Spearman rho=0.2382 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2366 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_100ms vs label_y_wmid_1s: Spearman rho=0.2467 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2386 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_100ms vs label_y_wmid_1s: Spearman rho=0.2639 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2424 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_100ms vs label_y_wmid_1s: Spearman rho=0.2442 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2454 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_100ms vs label_y_wmid_1s: Spearman rho=0.2321 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2380 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_100ms vs label_y_wmid_1s: Spearman rho=0.2527 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2384 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_100ms vs label_y_wmid_1s: Spearman rho=0.3371 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2657 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_1s vs label_y_wmid_1s: Spearman rho=0.2456 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2097 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_1s vs label_y_wmid_1s: Spearman rho=0.2300 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1839 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l1_1s vs label_y_wmid_1s: Spearman rho=0.2456 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l1_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2097 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l2_1s vs label_y_wmid_1s: Spearman rho=0.2072 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l2_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1800 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l3_1s vs label_y_wmid_1s: Spearman rho=0.1877 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l3_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1731 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l4_1s vs label_y_wmid_1s: Spearman rho=0.1679 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l4_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1713 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l5_1s vs label_y_wmid_1s: Spearman rho=0.1625 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l5_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1670 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l6_1s vs label_y_wmid_1s: Spearman rho=0.1627 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l6_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1672 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l7_1s vs label_y_wmid_1s: Spearman rho=0.1786 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l7_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1747 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l8_1s vs label_y_wmid_1s: Spearman rho=0.1653 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l8_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1710 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l9_1s vs label_y_wmid_1s: Spearman rho=0.1602 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l9_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1735 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l10_1s vs label_y_wmid_1s: Spearman rho=0.1691 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_raw_l10_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1681 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_1s vs label_y_wmid_1s: Spearman rho=0.2902 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2661 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_1s vs label_y_wmid_1s: Spearman rho=0.2515 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2334 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_1s vs label_y_wmid_1s: Spearman rho=0.2280 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2206 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_1s vs label_y_wmid_1s: Spearman rho=0.2030 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2167 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_1s vs label_y_wmid_1s: Spearman rho=0.2022 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2129 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_1s vs label_y_wmid_1s: Spearman rho=0.2073 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2147 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_1s vs label_y_wmid_1s: Spearman rho=0.2240 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2198 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_1s vs label_y_wmid_1s: Spearman rho=0.2080 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2163 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_1s vs label_y_wmid_1s: Spearman rho=0.2054 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2173 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_1s vs label_y_wmid_1s: Spearman rho=0.2179 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2123 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_1s vs label_y_wmid_1s: Spearman rho=0.2816 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2368 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_5s vs label_y_wmid_1s: Spearman rho=0.2065 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_best_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1808 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_5s vs label_y_wmid_1s: Spearman rho=0.1834 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_deep_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1580 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_5s vs label_y_wmid_1s: Spearman rho=0.2353 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l1_5s_sadj vs label_y_wmid_1s: Spearman rho=0.2223 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_5s vs label_y_wmid_1s: Spearman rho=0.2045 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l2_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1937 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_5s vs label_y_wmid_1s: Spearman rho=0.1871 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l3_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1785 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_5s vs label_y_wmid_1s: Spearman rho=0.1752 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l4_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1776 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_5s vs label_y_wmid_1s: Spearman rho=0.1759 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l5_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1788 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_5s vs label_y_wmid_1s: Spearman rho=0.1757 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l6_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1764 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_5s vs label_y_wmid_1s: Spearman rho=0.1800 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l7_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1810 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_5s vs label_y_wmid_1s: Spearman rho=0.1729 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l8_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1824 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_5s vs label_y_wmid_1s: Spearman rho=0.1685 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l9_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1792 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_5s vs label_y_wmid_1s: Spearman rho=0.1789 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_l10_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1791 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_5s vs label_y_wmid_1s: Spearman rho=0.2166 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event ofi_integrated_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1938 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_uniform_sadj vs label_y_wmid_1s: Spearman rho=0.1509 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_inv vs label_y_wmid_1s: Spearman rho=0.1575 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_inv_sadj vs label_y_wmid_1s: Spearman rho=0.1693 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_exp vs label_y_wmid_1s: Spearman rho=0.1555 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event qi_depth_exp_sadj vs label_y_wmid_1s: Spearman rho=0.1686 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_100ms vs label_y_wmid_1s: Spearman rho=0.2852 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2446 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_100ms vs label_y_wmid_1s: Spearman rho=-0.2805 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_100ms_sadj vs label_y_wmid_1s: Spearman rho=-0.2182 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_100ms vs label_y_wmid_1s: Spearman rho=0.3135 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.3268 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_1s vs label_y_wmid_1s: Spearman rho=0.2501 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2230 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_1s vs label_y_wmid_1s: Spearman rho=-0.2607 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_1s_sadj vs label_y_wmid_1s: Spearman rho=-0.2331 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_count_1s vs label_y_wmid_1s: Spearman rho=-0.1528 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_1s vs label_y_wmid_1s: Spearman rho=0.2588 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_1s_sadj vs label_y_wmid_1s: Spearman rho=0.2658 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_5s vs label_y_wmid_1s: Spearman rho=0.2137 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_signed_vol_5s_sadj vs label_y_wmid_1s: Spearman rho=0.1925 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_5s vs label_y_wmid_1s: Spearman rho=-0.2291 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_sell_vol_5s_sadj vs label_y_wmid_1s: Spearman rho=-0.2125 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_5s vs label_y_wmid_1s: Spearman rho=0.2150 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_aggr_imb_5s_sadj vs label_y_wmid_1s: Spearman rho=0.2031 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_last_trade_sign vs label_y_wmid_1s: Spearman rho=0.1513 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: INTC/event tr_last_trade_sign_sadj vs label_y_wmid_1s: Spearman rho=0.1648 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar tr_aggr_imb_100ms vs label_y_mid_1s: Spearman rho=0.3152 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar tr_aggr_imb_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2751 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar tr_aggr_imb_1s vs label_y_mid_1s: Spearman rho=0.2263 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar tr_aggr_imb_1s_sadj vs label_y_mid_1s: Spearman rho=0.2054 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar qi_touch vs label_y_wmid_1s: Spearman rho=-0.1705 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar qi_touch_sadj vs label_y_wmid_1s: Spearman rho=-0.1742 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar qi_sign vs label_y_wmid_1s: Spearman rho=-0.1628 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar qi_sign_sadj vs label_y_wmid_1s: Spearman rho=-0.1594 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar qi_l1 vs label_y_wmid_1s: Spearman rho=-0.1705 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar qi_l1_sadj vs label_y_wmid_1s: Spearman rho=-0.1742 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar mp_dev_mid_sadj vs label_y_wmid_1s: Spearman rho=-0.1575 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar mp_dev_wmid vs label_y_wmid_1s: Spearman rho=0.1800 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar mp_dev_wmid_sadj vs label_y_wmid_1s: Spearman rho=0.1795 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar mp_wmid_dev_mid vs label_y_wmid_1s: Spearman rho=-0.1853 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar mp_wmid_dev_mid_sadj vs label_y_wmid_1s: Spearman rho=-0.1830 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/calendar tr_aggr_imb_100ms vs label_y_wmid_1s: Spearman rho=0.1541 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_best_100ms vs label_y_mid_1s: Spearman rho=0.2712 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_best_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2278 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_deep_100ms vs label_y_mid_1s: Spearman rho=0.2703 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_deep_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2552 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l1_100ms vs label_y_mid_1s: Spearman rho=0.2761 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l1_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2338 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l2_100ms vs label_y_mid_1s: Spearman rho=0.2477 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l2_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2107 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l3_100ms vs label_y_mid_1s: Spearman rho=0.2576 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l3_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2299 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l4_100ms vs label_y_mid_1s: Spearman rho=0.2627 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l4_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2422 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l5_100ms vs label_y_mid_1s: Spearman rho=0.2637 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l5_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2447 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l6_100ms vs label_y_mid_1s: Spearman rho=0.2555 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l6_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2428 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l7_100ms vs label_y_mid_1s: Spearman rho=0.2579 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l7_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2465 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l8_100ms vs label_y_mid_1s: Spearman rho=0.2514 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l8_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2402 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l9_100ms vs label_y_mid_1s: Spearman rho=0.2466 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l9_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2379 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l10_100ms vs label_y_mid_1s: Spearman rho=0.2447 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l10_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2342 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_integrated_100ms vs label_y_mid_1s: Spearman rho=0.2787 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_integrated_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2667 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_best_1s vs label_y_mid_1s: Spearman rho=0.2640 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_best_1s_sadj vs label_y_mid_1s: Spearman rho=0.2315 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_deep_1s vs label_y_mid_1s: Spearman rho=0.2814 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_deep_1s_sadj vs label_y_mid_1s: Spearman rho=0.2711 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l1_1s vs label_y_mid_1s: Spearman rho=0.2640 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l1_1s_sadj vs label_y_mid_1s: Spearman rho=0.2315 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l2_1s vs label_y_mid_1s: Spearman rho=0.2401 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l2_1s_sadj vs label_y_mid_1s: Spearman rho=0.2169 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l3_1s vs label_y_mid_1s: Spearman rho=0.2539 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l3_1s_sadj vs label_y_mid_1s: Spearman rho=0.2343 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l4_1s vs label_y_mid_1s: Spearman rho=0.2658 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l4_1s_sadj vs label_y_mid_1s: Spearman rho=0.2496 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l5_1s vs label_y_mid_1s: Spearman rho=0.2637 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l5_1s_sadj vs label_y_mid_1s: Spearman rho=0.2510 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l6_1s vs label_y_mid_1s: Spearman rho=0.2649 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l6_1s_sadj vs label_y_mid_1s: Spearman rho=0.2539 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l7_1s vs label_y_mid_1s: Spearman rho=0.2664 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l7_1s_sadj vs label_y_mid_1s: Spearman rho=0.2588 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l8_1s vs label_y_mid_1s: Spearman rho=0.2610 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l8_1s_sadj vs label_y_mid_1s: Spearman rho=0.2542 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l9_1s vs label_y_mid_1s: Spearman rho=0.2559 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l9_1s_sadj vs label_y_mid_1s: Spearman rho=0.2506 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l10_1s vs label_y_mid_1s: Spearman rho=0.2549 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_raw_l10_1s_sadj vs label_y_mid_1s: Spearman rho=0.2481 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l1_1s vs label_y_mid_1s: Spearman rho=0.2678 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l1_1s_sadj vs label_y_mid_1s: Spearman rho=0.2374 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l2_1s vs label_y_mid_1s: Spearman rho=0.2470 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l2_1s_sadj vs label_y_mid_1s: Spearman rho=0.2239 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l3_1s vs label_y_mid_1s: Spearman rho=0.2604 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l3_1s_sadj vs label_y_mid_1s: Spearman rho=0.2407 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l4_1s vs label_y_mid_1s: Spearman rho=0.2745 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l4_1s_sadj vs label_y_mid_1s: Spearman rho=0.2587 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l5_1s vs label_y_mid_1s: Spearman rho=0.2730 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l5_1s_sadj vs label_y_mid_1s: Spearman rho=0.2622 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l6_1s vs label_y_mid_1s: Spearman rho=0.2742 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l6_1s_sadj vs label_y_mid_1s: Spearman rho=0.2653 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l7_1s vs label_y_mid_1s: Spearman rho=0.2753 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l7_1s_sadj vs label_y_mid_1s: Spearman rho=0.2696 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l8_1s vs label_y_mid_1s: Spearman rho=0.2705 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l8_1s_sadj vs label_y_mid_1s: Spearman rho=0.2650 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l9_1s vs label_y_mid_1s: Spearman rho=0.2661 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l9_1s_sadj vs label_y_mid_1s: Spearman rho=0.2622 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l10_1s vs label_y_mid_1s: Spearman rho=0.2642 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l10_1s_sadj vs label_y_mid_1s: Spearman rho=0.2583 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_integrated_1s vs label_y_mid_1s: Spearman rho=0.2899 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_integrated_1s_sadj vs label_y_mid_1s: Spearman rho=0.2818 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_best_5s vs label_y_mid_1s: Spearman rho=0.2539 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_best_5s_sadj vs label_y_mid_1s: Spearman rho=0.2366 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_deep_5s vs label_y_mid_1s: Spearman rho=0.2535 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_deep_5s_sadj vs label_y_mid_1s: Spearman rho=0.2495 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l1_5s vs label_y_mid_1s: Spearman rho=0.2577 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l1_5s_sadj vs label_y_mid_1s: Spearman rho=0.2416 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l2_5s vs label_y_mid_1s: Spearman rho=0.2287 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l2_5s_sadj vs label_y_mid_1s: Spearman rho=0.2186 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l3_5s vs label_y_mid_1s: Spearman rho=0.2371 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l3_5s_sadj vs label_y_mid_1s: Spearman rho=0.2272 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l4_5s vs label_y_mid_1s: Spearman rho=0.2443 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l4_5s_sadj vs label_y_mid_1s: Spearman rho=0.2391 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l5_5s vs label_y_mid_1s: Spearman rho=0.2419 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l5_5s_sadj vs label_y_mid_1s: Spearman rho=0.2386 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l6_5s vs label_y_mid_1s: Spearman rho=0.2483 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l6_5s_sadj vs label_y_mid_1s: Spearman rho=0.2465 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l7_5s vs label_y_mid_1s: Spearman rho=0.2456 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l7_5s_sadj vs label_y_mid_1s: Spearman rho=0.2457 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l8_5s vs label_y_mid_1s: Spearman rho=0.2489 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l8_5s_sadj vs label_y_mid_1s: Spearman rho=0.2470 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l9_5s vs label_y_mid_1s: Spearman rho=0.2542 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l9_5s_sadj vs label_y_mid_1s: Spearman rho=0.2529 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l10_5s vs label_y_mid_1s: Spearman rho=0.2477 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_l10_5s_sadj vs label_y_mid_1s: Spearman rho=0.2479 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_integrated_5s vs label_y_mid_1s: Spearman rho=0.2595 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event ofi_integrated_5s_sadj vs label_y_mid_1s: Spearman rho=0.2577 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event qi_touch_sadj vs label_y_mid_1s: Spearman rho=0.1504 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event qi_l1_sadj vs label_y_mid_1s: Spearman rho=0.1504 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_signed_vol_100ms vs label_y_mid_1s: Spearman rho=0.3426 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_signed_vol_100ms_sadj vs label_y_mid_1s: Spearman rho=0.2519 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_buy_vol_100ms vs label_y_mid_1s: Spearman rho=0.2530 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_sell_vol_100ms vs label_y_mid_1s: Spearman rho=-0.2479 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_sell_vol_100ms_sadj vs label_y_mid_1s: Spearman rho=-0.1678 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_100ms vs label_y_mid_1s: Spearman rho=0.5222 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_100ms_sadj vs label_y_mid_1s: Spearman rho=0.4614 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_signed_vol_1s vs label_y_mid_1s: Spearman rho=0.3320 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_signed_vol_1s_sadj vs label_y_mid_1s: Spearman rho=0.2756 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_buy_vol_1s vs label_y_mid_1s: Spearman rho=0.2447 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_buy_vol_1s_sadj vs label_y_mid_1s: Spearman rho=0.1642 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_sell_vol_1s vs label_y_mid_1s: Spearman rho=-0.2390 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_sell_vol_1s_sadj vs label_y_mid_1s: Spearman rho=-0.1934 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_1s vs label_y_mid_1s: Spearman rho=0.4328 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_1s_sadj vs label_y_mid_1s: Spearman rho=0.3970 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_signed_vol_5s vs label_y_mid_1s: Spearman rho=0.2717 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_signed_vol_5s_sadj vs label_y_mid_1s: Spearman rho=0.2450 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_buy_vol_5s vs label_y_mid_1s: Spearman rho=0.2086 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_buy_vol_5s_sadj vs label_y_mid_1s: Spearman rho=0.1763 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_sell_vol_5s vs label_y_mid_1s: Spearman rho=-0.1923 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_sell_vol_5s_sadj vs label_y_mid_1s: Spearman rho=-0.1786 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_5s vs label_y_mid_1s: Spearman rho=0.3150 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_5s_sadj vs label_y_mid_1s: Spearman rho=0.2949 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_last_trade_sign vs label_y_mid_1s: Spearman rho=0.2400 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_last_trade_sign_sadj vs label_y_mid_1s: Spearman rho=0.2183 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event qi_touch vs label_y_wmid_1s: Spearman rho=-0.2541 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event qi_touch_sadj vs label_y_wmid_1s: Spearman rho=-0.2581 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event qi_sign vs label_y_wmid_1s: Spearman rho=-0.2311 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event qi_sign_sadj vs label_y_wmid_1s: Spearman rho=-0.2182 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event qi_l1 vs label_y_wmid_1s: Spearman rho=-0.2541 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event qi_l1_sadj vs label_y_wmid_1s: Spearman rho=-0.2581 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event mp_dev_mid vs label_y_wmid_1s: Spearman rho=-0.2151 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event mp_dev_mid_sadj vs label_y_wmid_1s: Spearman rho=-0.2267 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event mp_dev_wmid vs label_y_wmid_1s: Spearman rho=0.2741 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event mp_dev_wmid_sadj vs label_y_wmid_1s: Spearman rho=0.2752 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event mp_wmid_dev_mid vs label_y_wmid_1s: Spearman rho=-0.2801 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event mp_wmid_dev_mid_sadj vs label_y_wmid_1s: Spearman rho=-0.2779 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_signed_vol_100ms vs label_y_wmid_1s: Spearman rho=0.1512 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_100ms vs label_y_wmid_1s: Spearman rho=0.2376 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_100ms_sadj vs label_y_wmid_1s: Spearman rho=0.2126 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_signed_vol_1s vs label_y_wmid_1s: Spearman rho=0.1596 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_1s vs label_y_wmid_1s: Spearman rho=0.2085 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event tr_aggr_imb_1s_sadj vs label_y_wmid_1s: Spearman rho=0.1908 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event st_depth_bid_touch vs label_y_wmid_1s: Spearman rho=-0.1726 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event st_depth_ask_touch vs label_y_wmid_1s: Spearman rho=0.1608 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- LEAK SMELL: AMZN/event st_depth_ask_touch_sadj vs label_y_wmid_1s: Spearman rho=0.1564 (|rho|>0.15 at a 1s horizon on 2012 equity data is not a discovery)
- SADJ>>RAW: INTC/calendar qi_l4 vs label_y_mid_10s: raw rho=-0.0107, sadj rho=-0.0331 (3.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/calendar mp_micro vs label_y_mid_10s: raw rho=-0.0158, sadj rho=0.0337 (2.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/calendar st_spread_ticks vs label_y_mid_10s: raw rho=-0.0055, sadj rho=0.0496 (9.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/calendar st_spread_ticks_int vs label_y_mid_10s: raw rho=-0.0055, sadj rho=0.0496 (9.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/calendar mp_dev_mid vs label_y_wmid_10s: raw rho=0.0157, sadj rho=0.0321 (2.0x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/calendar tr_signed_vol_100ms vs label_y_wmid_10s: raw rho=0.0137, sadj rho=0.0304 (2.2x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/calendar st_spread_ticks vs label_y_wmid_10s: raw rho=-0.0081, sadj rho=0.0475 (5.9x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/calendar st_spread_ticks_int vs label_y_wmid_10s: raw rho=-0.0081, sadj rho=0.0475 (5.9x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event qi_l9 vs label_y_mid_1s: raw rho=0.0196, sadj rho=0.0475 (2.4x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_slope_bid vs label_y_mid_1s: raw rho=0.0253, sadj rho=0.0520 (2.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_slope_ask vs label_y_mid_1s: raw rho=0.0000, sadj rho=-0.0492 (5665.3x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event qi_l8 vs label_y_wmid_1s: raw rho=0.0136, sadj rho=0.0352 (2.6x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event qi_l9 vs label_y_wmid_1s: raw rho=0.0210, sadj rho=0.0549 (2.6x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event qi_l10 vs label_y_wmid_1s: raw rho=0.0208, sadj rho=0.0488 (2.3x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event mp_micro vs label_y_wmid_1s: raw rho=-0.0280, sadj rho=0.0633 (2.3x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_depth_touch vs label_y_wmid_1s: raw rho=0.0101, sadj rho=-0.0490 (4.9x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_slope_ask vs label_y_wmid_1s: raw rho=0.0099, sadj rho=-0.0644 (6.5x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event tr_avg_size_5s vs label_y_mid_2s: raw rho=-0.0114, sadj rho=-0.0390 (3.4x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event qi_l8 vs label_y_wmid_2s: raw rho=0.0140, sadj rho=0.0306 (2.2x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event qi_l9 vs label_y_wmid_2s: raw rho=0.0106, sadj rho=0.0397 (3.8x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event qi_l10 vs label_y_wmid_2s: raw rho=0.0087, sadj rho=0.0383 (4.4x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event mp_micro vs label_y_wmid_2s: raw rho=-0.0183, sadj rho=0.0377 (2.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_depth_touch vs label_y_wmid_2s: raw rho=0.0076, sadj rho=-0.0538 (7.0x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_slope_ask vs label_y_wmid_2s: raw rho=0.0227, sadj rho=-0.0479 (2.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event tr_avg_size_100ms vs label_y_mid_5s: raw rho=-0.0138, sadj rho=-0.0332 (2.4x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event tr_avg_size_1s vs label_y_mid_5s: raw rho=-0.0161, sadj rho=-0.0454 (2.8x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event mp_micro vs label_y_wmid_5s: raw rho=-0.0159, sadj rho=0.0388 (2.4x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event tr_avg_size_100ms vs label_y_wmid_5s: raw rho=-0.0149, sadj rho=-0.0416 (2.8x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event tr_avg_size_1s vs label_y_wmid_5s: raw rho=-0.0282, sadj rho=-0.0629 (2.2x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_depth_ask_touch vs label_y_wmid_5s: raw rho=-0.0559, sadj rho=-0.1314 (2.4x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_depth_touch vs label_y_wmid_5s: raw rho=0.0161, sadj rho=-0.0482 (3.0x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event tr_avg_size_100ms vs label_y_wmid_10s: raw rho=-0.0129, sadj rho=-0.0317 (2.5x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_depth_ask_touch vs label_y_wmid_10s: raw rho=-0.0315, sadj rho=-0.1020 (3.2x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: INTC/event st_avg_order_size_1s vs label_y_wmid_10s: raw rho=-0.0149, sadj rho=-0.0310 (2.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/calendar st_avg_order_size_5s vs label_y_mid_5s: raw rho=0.0147, sadj rho=0.0353 (2.4x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/calendar st_avg_order_size_5s vs label_y_mid_10s: raw rho=0.0163, sadj rho=0.0339 (2.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/event mp_micro vs label_y_mid_1s: raw rho=-0.0016, sadj rho=0.0545 (33.5x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/event mp_micro vs label_y_mid_2s: raw rho=-0.0091, sadj rho=0.0469 (5.1x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/event mp_micro vs label_y_mid_5s: raw rho=-0.0085, sadj rho=0.0381 (4.5x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/event st_avg_order_size_5s vs label_y_mid_5s: raw rho=0.0061, sadj rho=0.0419 (6.9x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/event st_avg_order_size_5s vs label_y_wmid_5s: raw rho=0.0069, sadj rho=0.0394 (5.7x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/event mp_micro vs label_y_mid_10s: raw rho=-0.0130, sadj rho=0.0322 (2.5x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/event st_avg_order_size_5s vs label_y_mid_10s: raw rho=0.0099, sadj rho=0.0478 (4.8x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/event st_slope_ask vs label_y_wmid_10s: raw rho=-0.0082, sadj rho=-0.0310 (3.8x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)
- SADJ>>RAW: AMZN/event st_avg_order_size_5s vs label_y_wmid_10s: raw rho=0.0138, sadj rho=0.0517 (3.8x, a seasonality adjustment should not manufacture signal out of a near-zero raw correlation)

Thresholds: |Spearman rho| > 0.15 at the 1s horizon; Spearman rho identical (spread < 1e-06) across all of 1s/2s/5s/10s for the same feature/label-type; a `_sadj` column's |IC| at least 2.0x its raw parent's AND at least 0.03 larger in absolute terms.

## Top-20 features by |IC| per (ticker, clock, label), raw and sadj side by side

Ranked by |Spearman IC| of the RAW (non-`_sadj`) feature; the seasonality-adjusted twin's IC and t_hac are shown alongside for the same rank so the effect of the adjustment is legible in the row, not buried in a second table.

### INTC / calendar / `label_y_mid_1s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 5008 | 0.1624 | 0.1722 | 10.19 | 0.1491 | 1.91 | -0.0231 |
| 2 | mp_wmid_dev_mid | 233989 | 0.1635 | 0.1616 | 23.58 | 0.1566 | 1.45 | -0.0050 |
| 3 | qi_touch | 233989 | 0.1650 | 0.1611 | 24.15 | 0.1587 | 1.45 | -0.0024 |
| 4 | qi_l1 | 233989 | 0.1650 | 0.1611 | 24.15 | 0.1587 | 1.45 | -0.0024 |
| 5 | mp_dev_mid | 215990 | 0.1425 | 0.1347 | 20.49 | 0.1493 | 21.91 | 0.0146 |
| 6 | qi_depth_inv | 233989 | 0.1227 | 0.1233 | 14.95 | 0.1292 | 7.04 | 0.0058 |
| 7 | qi_depth_exp | 233989 | 0.1174 | 0.1186 | 13.80 | 0.1238 | 8.16 | 0.0052 |
| 8 | qi_sign | 233989 | 0.1105 | 0.1178 | 21.39 | 0.1072 | 20.81 | -0.0107 |
| 9 | mp_dev_wmid | 215990 | -0.1160 | -0.1165 | -18.12 | -0.1352 | -1.36 | -0.0186 |
| 10 | tr_aggr_imb_1s | 35762 | 0.1037 | 0.1154 | 9.10 | 0.1090 | 1.53 | -0.0064 |
| 11 | st_depth_bid_touch | 233989 | 0.0677 | 0.1021 | 11.60 | 0.1150 | 6.97 | 0.0129 |
| 12 | st_depth_ask_touch | 233989 | -0.0631 | -0.0858 | -9.55 | -0.1150 | -7.20 | -0.0292 |
| 13 | ofi_l1_5s | 215990 | 0.0551 | 0.0835 | 6.18 | 0.0729 | 7.14 | -0.0106 |
| 14 | ofi_best_5s | 233989 | 0.0516 | 0.0818 | 7.23 | 0.0744 | 7.69 | -0.0074 |
| 15 | tr_aggr_imb_5s | 110115 | 0.0758 | 0.0817 | 10.61 | 0.0829 | 1.97 | 0.0012 |
| 16 | ofi_l1_1s | 215990 | 0.0429 | 0.0804 | 5.58 | 0.0744 | 2.04 | -0.0060 |
| 17 | ofi_best_1s | 233989 | 0.0375 | 0.0765 | 6.12 | 0.0730 | 2.17 | -0.0035 |
| 18 | ofi_raw_l1_1s | 233989 | 0.0375 | 0.0765 | 6.12 | 0.0730 | 2.17 | -0.0035 |
| 19 | ofi_integrated_5s | 215990 | 0.0175 | 0.0752 | 2.01 | 0.0394 | 3.38 | -0.0358 |
| 20 | tr_signed_vol_5s | 233989 | 0.0493 | 0.0745 | 6.03 | 0.0627 | 2.19 | -0.0118 |

### INTC / calendar / `label_y_wmid_1s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_avg_size_100ms | 5008 | -0.0362 | -0.0585 | -2.26 | -0.0503 | -3.23 | 0.0082 |
| 2 | tr_avg_size_1s | 35762 | -0.0164 | -0.0417 | -1.54 | -0.0430 | -2.67 | -0.0014 |
| 3 | st_rv_1s | 215175 | -0.0548 | -0.0322 | -4.26 | -0.0234 | -5.45 | 0.0088 |
| 4 | tr_aggr_imb_1s | 35762 | 0.0532 | -0.0286 | 4.65 | -0.0176 | 1.55 | 0.0110 |
| 5 | tr_buy_vol_1s | 233989 | -0.0181 | -0.0258 | -2.20 | -0.0245 | -1.66 | 0.0013 |
| 6 | tr_avg_size_5s | 110115 | -0.0171 | -0.0249 | -2.54 | -0.0291 | -3.41 | -0.0042 |
| 7 | ofi_l5_5s | 215990 | -0.0033 | 0.0245 | -0.38 | 0.0233 | 0.47 | -0.0012 |
| 8 | tr_count_1s | 233989 | -0.0536 | -0.0241 | -5.80 | -0.0289 | -3.74 | -0.0048 |
| 9 | tr_aggr_imb_100ms | 5008 | 0.0991 | 0.0237 | 6.12 | 0.0171 | 1.92 | -0.0066 |
| 10 | tr_quote_ratio_1s | 215012 | -0.0136 | -0.0232 | -2.36 | -0.0232 | -2.12 | -0.0001 |
| 11 | ofi_l6_5s | 215990 | -0.0054 | 0.0208 | -0.62 | 0.0161 | 0.24 | -0.0047 |
| 12 | ofi_raw_l6_1s | 233989 | -0.0204 | -0.0205 | -2.67 | -0.0126 | -1.54 | 0.0079 |
| 13 | ofi_l1_5s | 215990 | 0.0287 | -0.0202 | 3.21 | -0.0186 | 3.95 | 0.0016 |
| 14 | st_slope_ask | 233989 | 0.0107 | -0.0195 | 1.07 | -0.0223 | -0.62 | -0.0028 |
| 15 | st_spread_ticks | 233989 | -0.0285 | -0.0186 | -3.09 | 0.0007 | -3.26 | 0.0194 |
| 16 | st_spread_ticks_int | 233989 | -0.0285 | -0.0186 | -3.09 | 0.0007 | -3.26 | 0.0194 |
| 17 | tr_quote_ratio_5s | 233960 | -0.0367 | -0.0184 | -5.66 | -0.0246 | -6.00 | -0.0062 |
| 18 | ofi_l4_5s | 215990 | -0.0028 | 0.0178 | -0.31 | 0.0160 | 0.57 | -0.0018 |
| 19 | tr_count_5s | 233989 | -0.0409 | -0.0177 | -4.09 | -0.0258 | -5.37 | -0.0081 |
| 20 | ofi_l8_5s | 215990 | -0.0078 | 0.0177 | -0.96 | 0.0157 | -0.15 | -0.0020 |

### INTC / calendar / `label_y_mid_2s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 5003 | 0.1614 | 0.2045 | 9.56 | 0.1785 | 2.56 | -0.0261 |
| 2 | mp_wmid_dev_mid | 233979 | 0.1981 | 0.2025 | 22.25 | 0.1966 | 1.45 | -0.0059 |
| 3 | qi_touch | 233979 | 0.1998 | 0.2022 | 22.79 | 0.1992 | 1.45 | -0.0029 |
| 4 | qi_l1 | 233979 | 0.1998 | 0.2022 | 22.79 | 0.1992 | 1.45 | -0.0029 |
| 5 | mp_dev_mid | 215980 | 0.1762 | 0.1705 | 19.39 | 0.1895 | 20.94 | 0.0190 |
| 6 | qi_depth_inv | 233979 | 0.1513 | 0.1579 | 13.78 | 0.1637 | 6.35 | 0.0058 |
| 7 | tr_aggr_imb_1s | 35752 | 0.1288 | 0.1556 | 8.97 | 0.1413 | 1.58 | -0.0142 |
| 8 | qi_depth_exp | 233979 | 0.1448 | 0.1521 | 12.71 | 0.1570 | 7.28 | 0.0049 |
| 9 | qi_sign | 233979 | 0.1373 | 0.1482 | 19.77 | 0.1336 | 19.41 | -0.0146 |
| 10 | mp_dev_wmid | 215980 | -0.1403 | -0.1426 | -16.51 | -0.1655 | -1.36 | -0.0229 |
| 11 | st_depth_bid_touch | 233979 | 0.0836 | 0.1265 | 9.94 | 0.1421 | 5.80 | 0.0156 |
| 12 | st_depth_ask_touch | 233979 | -0.0770 | -0.1105 | -8.32 | -0.1473 | -6.39 | -0.0368 |
| 13 | tr_aggr_imb_5s | 110105 | 0.0948 | 0.1055 | 9.62 | 0.1063 | 2.52 | 0.0008 |
| 14 | ofi_l1_5s | 215980 | 0.0665 | 0.1047 | 6.34 | 0.0928 | 6.69 | -0.0120 |
| 15 | ofi_best_5s | 233979 | 0.0637 | 0.1014 | 7.08 | 0.0927 | 6.75 | -0.0087 |
| 16 | ofi_l1_1s | 215980 | 0.0536 | 0.0968 | 5.90 | 0.0908 | 2.09 | -0.0059 |
| 17 | ofi_integrated_5s | 215980 | 0.0211 | 0.0946 | 2.01 | 0.0500 | 2.88 | -0.0445 |
| 18 | ofi_best_1s | 233979 | 0.0498 | 0.0937 | 6.55 | 0.0910 | 2.22 | -0.0027 |
| 19 | ofi_raw_l1_1s | 233979 | 0.0498 | 0.0937 | 6.55 | 0.0910 | 2.22 | -0.0027 |
| 20 | tr_signed_vol_5s | 233979 | 0.0596 | 0.0910 | 5.64 | 0.0766 | 2.17 | -0.0144 |

### INTC / calendar / `label_y_wmid_2s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_avg_size_1s | 35752 | -0.0222 | -0.0530 | -1.78 | -0.0576 | -3.13 | -0.0046 |
| 2 | tr_avg_size_100ms | 5003 | -0.0365 | -0.0521 | -2.33 | -0.0476 | -3.10 | 0.0045 |
| 3 | st_rv_1s | 215165 | -0.0670 | -0.0429 | -3.66 | -0.0297 | -5.52 | 0.0131 |
| 4 | tr_avg_size_5s | 110105 | -0.0250 | -0.0356 | -2.65 | -0.0389 | -3.23 | -0.0033 |
| 5 | tr_count_1s | 233979 | -0.0593 | -0.0347 | -5.18 | -0.0362 | -3.96 | -0.0015 |
| 6 | tr_quote_ratio_1s | 215002 | -0.0193 | -0.0334 | -2.87 | -0.0314 | -3.13 | 0.0020 |
| 7 | st_spread_ticks | 233979 | -0.0325 | -0.0296 | -2.96 | 0.0092 | -2.99 | 0.0387 |
| 8 | st_spread_ticks_int | 233979 | -0.0325 | -0.0296 | -2.96 | 0.0092 | -2.99 | 0.0387 |
| 9 | tr_aggr_imb_100ms | 5003 | 0.0995 | 0.0293 | 6.05 | 0.0286 | 2.42 | -0.0008 |
| 10 | tr_buy_vol_1s | 233979 | -0.0203 | -0.0291 | -2.09 | -0.0290 | -2.00 | 0.0001 |
| 11 | tr_count_5s | 233979 | -0.0441 | -0.0283 | -3.53 | -0.0364 | -5.30 | -0.0080 |
| 12 | ofi_l1_5s | 215980 | 0.0336 | -0.0275 | 3.28 | -0.0262 | 3.61 | 0.0012 |
| 13 | tr_quote_ratio_5s | 233950 | -0.0473 | -0.0269 | -5.19 | -0.0343 | -5.77 | -0.0075 |
| 14 | st_slope_ask | 233979 | 0.0160 | -0.0260 | 1.14 | -0.0219 | -0.42 | 0.0041 |
| 15 | qi_l2 | 233979 | 0.0572 | 0.0243 | 6.17 | 0.0208 | 8.04 | -0.0034 |
| 16 | tr_sell_vol_1s | 233979 | -0.0558 | -0.0242 | -5.62 | -0.0231 | -3.94 | 0.0011 |
| 17 | tr_sell_vol_5s | 233979 | -0.0514 | -0.0242 | -5.56 | -0.0308 | -5.24 | -0.0066 |
| 18 | st_event_rate_1s | 233979 | -0.0477 | -0.0234 | -4.59 | -0.0314 | -4.70 | -0.0080 |
| 19 | st_rv_5s | 233950 | -0.0425 | -0.0229 | -2.54 | -0.0279 | -4.25 | -0.0050 |
| 20 | st_rv_100ms | 76904 | -0.0375 | -0.0220 | -2.22 | -0.0089 | -3.73 | 0.0131 |

### INTC / calendar / `label_y_mid_5s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | qi_touch | 233949 | 0.2343 | 0.2474 | 19.28 | 0.2462 | 1.45 | -0.0012 |
| 2 | qi_l1 | 233949 | 0.2343 | 0.2474 | 19.28 | 0.2462 | 1.45 | -0.0012 |
| 3 | mp_wmid_dev_mid | 233949 | 0.2344 | 0.2471 | 19.19 | 0.2432 | 1.44 | -0.0039 |
| 4 | tr_aggr_imb_100ms | 4981 | 0.1732 | 0.2272 | 8.04 | 0.2051 | 2.34 | -0.0221 |
| 5 | mp_dev_mid | 215950 | 0.2055 | 0.2076 | 16.12 | 0.2358 | 17.99 | 0.0283 |
| 6 | qi_depth_inv | 233949 | 0.1863 | 0.2025 | 12.66 | 0.2089 | 5.53 | 0.0064 |
| 7 | qi_depth_exp | 233949 | 0.1812 | 0.1966 | 12.07 | 0.2017 | 6.25 | 0.0051 |
| 8 | qi_sign | 233949 | 0.1680 | 0.1861 | 16.05 | 0.1656 | 15.50 | -0.0204 |
| 9 | tr_aggr_imb_1s | 35722 | 0.1502 | 0.1843 | 8.64 | 0.1781 | 1.57 | -0.0062 |
| 10 | mp_dev_wmid | 215950 | -0.1722 | -0.1804 | -13.99 | -0.2109 | -1.36 | -0.0305 |
| 11 | st_depth_bid_touch | 233949 | 0.1053 | 0.1591 | 7.72 | 0.1808 | 5.56 | 0.0217 |
| 12 | st_depth_ask_touch | 233949 | -0.0837 | -0.1345 | -5.82 | -0.1749 | -5.36 | -0.0404 |
| 13 | tr_aggr_imb_5s | 110075 | 0.1149 | 0.1300 | 7.92 | 0.1229 | 2.21 | -0.0071 |
| 14 | ofi_l1_5s | 215950 | 0.0577 | 0.1210 | 4.51 | 0.1046 | 4.70 | -0.0164 |
| 15 | ofi_best_5s | 233949 | 0.0573 | 0.1159 | 5.20 | 0.1032 | 4.43 | -0.0128 |
| 16 | ofi_integrated_5s | 215950 | 0.0099 | 0.1094 | 0.78 | 0.0489 | 1.45 | -0.0605 |
| 17 | tr_signed_vol_5s | 233949 | 0.0636 | 0.1002 | 5.32 | 0.0810 | 2.17 | -0.0192 |
| 18 | ofi_best_1s | 233949 | 0.0529 | 0.0993 | 6.73 | 0.0951 | 2.59 | -0.0042 |
| 19 | ofi_raw_l1_1s | 233949 | 0.0529 | 0.0993 | 6.73 | 0.0951 | 2.59 | -0.0042 |
| 20 | ofi_l1_1s | 215950 | 0.0555 | 0.0976 | 6.20 | 0.0901 | 2.56 | -0.0075 |

### INTC / calendar / `label_y_wmid_5s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 4981 | 0.1042 | 0.0621 | 4.93 | 0.0652 | 2.25 | 0.0031 |
| 2 | tr_avg_size_1s | 35722 | -0.0154 | -0.0490 | -0.92 | -0.0558 | -2.22 | -0.0068 |
| 3 | tr_avg_size_5s | 110075 | -0.0476 | -0.0449 | -3.43 | -0.0406 | -3.68 | 0.0043 |
| 4 | tr_avg_size_100ms | 4981 | -0.0248 | -0.0382 | -1.32 | -0.0432 | -2.32 | -0.0050 |
| 5 | st_rv_1s | 215135 | -0.0524 | -0.0382 | -3.02 | -0.0244 | -4.28 | 0.0138 |
| 6 | qi_l2 | 233949 | 0.0789 | 0.0379 | 6.08 | 0.0335 | 7.35 | -0.0045 |
| 7 | tr_sell_vol_5s | 233949 | -0.0508 | -0.0350 | -4.45 | -0.0324 | -4.72 | 0.0026 |
| 8 | tr_count_1s | 233949 | -0.0449 | -0.0344 | -4.19 | -0.0281 | -3.77 | 0.0063 |
| 9 | tr_sell_vol_1s | 233949 | -0.0490 | -0.0332 | -5.25 | -0.0205 | -4.22 | 0.0126 |
| 10 | tr_quote_ratio_1s | 214972 | -0.0175 | -0.0328 | -1.75 | -0.0330 | -2.19 | -0.0003 |
| 11 | tr_last_trade_sign | 233947 | 0.0585 | 0.0327 | 5.56 | 0.0349 | 6.17 | 0.0023 |
| 12 | st_spread_ticks | 233949 | -0.0236 | -0.0320 | -1.77 | 0.0255 | -1.95 | 0.0576 |
| 13 | st_spread_ticks_int | 233949 | -0.0236 | -0.0320 | -1.77 | 0.0255 | -1.95 | 0.0576 |
| 14 | tr_eff_spread | 233947 | -0.0495 | -0.0319 | -4.89 | -0.0257 | -4.67 | 0.0063 |
| 15 | tr_count_5s | 233949 | -0.0406 | -0.0315 | -2.83 | -0.0348 | -4.37 | -0.0033 |
| 16 | st_slope_ask | 233949 | 0.0249 | -0.0315 | 1.17 | -0.0214 | -0.05 | 0.0101 |
| 17 | tr_quote_ratio_5s | 233920 | -0.0484 | -0.0311 | -3.64 | -0.0416 | -4.39 | -0.0105 |
| 18 | tr_quote_ratio_100ms | 76603 | -0.0126 | -0.0293 | -1.30 | -0.0082 | -2.08 | 0.0211 |
| 19 | st_event_rate_5s | 233949 | -0.0353 | -0.0240 | -1.95 | -0.0220 | -2.33 | 0.0019 |
| 20 | st_rv_100ms | 76876 | -0.0345 | -0.0239 | -2.63 | -0.0047 | -3.94 | 0.0193 |

### INTC / calendar / `label_y_mid_10s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | qi_touch | 233899 | 0.2399 | 0.2622 | 15.66 | 0.2623 | 1.44 | 0.0002 |
| 2 | qi_l1 | 233899 | 0.2399 | 0.2622 | 15.66 | 0.2623 | 1.44 | 0.0002 |
| 3 | mp_wmid_dev_mid | 233899 | 0.2382 | 0.2620 | 15.52 | 0.2598 | 1.44 | -0.0022 |
| 4 | tr_aggr_imb_100ms | 4951 | 0.1696 | 0.2237 | 7.02 | 0.2102 | 2.25 | -0.0135 |
| 5 | qi_depth_inv | 233899 | 0.1993 | 0.2221 | 10.12 | 0.2272 | 5.32 | 0.0050 |
| 6 | mp_dev_mid | 215900 | 0.2090 | 0.2167 | 12.45 | 0.2530 | 14.58 | 0.0363 |
| 7 | qi_depth_exp | 233899 | 0.1951 | 0.2150 | 9.73 | 0.2178 | 5.95 | 0.0028 |
| 8 | mp_dev_wmid | 215900 | -0.1888 | -0.2048 | -11.61 | -0.2369 | -1.36 | -0.0321 |
| 9 | qi_sign | 233899 | 0.1751 | 0.2002 | 12.77 | 0.1756 | 12.12 | -0.0246 |
| 10 | tr_aggr_imb_1s | 35672 | 0.1549 | 0.1914 | 8.11 | 0.1853 | 1.72 | -0.0061 |
| 11 | st_depth_bid_touch | 233899 | 0.1162 | 0.1816 | 5.99 | 0.1969 | 5.42 | 0.0153 |
| 12 | tr_aggr_imb_5s | 110025 | 0.1212 | 0.1404 | 6.72 | 0.1325 | 2.00 | -0.0080 |
| 13 | st_depth_ask_touch | 233899 | -0.0842 | -0.1390 | -4.17 | -0.1809 | -4.94 | -0.0419 |
| 14 | ofi_l1_5s | 215900 | 0.0588 | 0.1351 | 4.45 | 0.1225 | 4.67 | -0.0127 |
| 15 | ofi_best_5s | 233899 | 0.0560 | 0.1291 | 4.96 | 0.1201 | 4.80 | -0.0091 |
| 16 | ofi_integrated_5s | 215900 | 0.0110 | 0.1217 | 0.82 | 0.0615 | 1.41 | -0.0603 |
| 17 | tr_last_trade_sign | 233897 | 0.1014 | 0.1146 | 7.47 | 0.1098 | 8.13 | -0.0048 |
| 18 | qi_depth_uniform | 233899 | 0.0941 | 0.1068 | 3.64 | 0.1258 | 3.20 | 0.0190 |
| 19 | ofi_deep_5s | 233899 | 0.0115 | 0.1067 | 0.96 | 0.0510 | 1.27 | -0.0557 |
| 20 | tr_signed_vol_5s | 233899 | 0.0590 | 0.1016 | 4.66 | 0.0875 | 2.37 | -0.0141 |

### INTC / calendar / `label_y_wmid_10s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 4951 | 0.0928 | 0.0752 | 3.89 | 0.0796 | 2.24 | 0.0044 |
| 2 | tr_avg_size_5s | 110025 | -0.0654 | -0.0610 | -3.68 | -0.0528 | -3.61 | 0.0081 |
| 3 | tr_avg_size_1s | 35672 | -0.0364 | -0.0584 | -1.92 | -0.0578 | -2.57 | 0.0006 |
| 4 | tr_last_trade_sign | 233897 | 0.0740 | 0.0569 | 5.38 | 0.0602 | 6.20 | 0.0033 |
| 5 | tr_avg_size_100ms | 4951 | -0.0361 | -0.0539 | -1.68 | -0.0528 | -2.88 | 0.0011 |
| 6 | tr_aggr_imb_1s | 35672 | 0.0904 | 0.0522 | 4.66 | 0.0598 | 1.79 | 0.0076 |
| 7 | tr_aggr_imb_5s | 110025 | 0.0761 | 0.0494 | 4.14 | 0.0455 | 1.87 | -0.0038 |
| 8 | tr_sell_vol_5s | 233899 | -0.0459 | -0.0469 | -3.56 | -0.0359 | -3.95 | 0.0111 |
| 9 | tr_eff_spread | 233897 | -0.0564 | -0.0461 | -4.17 | -0.0304 | -3.93 | 0.0157 |
| 10 | qi_l2 | 233899 | 0.0810 | 0.0459 | 4.77 | 0.0397 | 5.72 | -0.0062 |
| 11 | tr_sell_vol_1s | 233899 | -0.0408 | -0.0389 | -5.35 | -0.0172 | -3.56 | 0.0217 |
| 12 | qi_depth_exp | 233899 | 0.1190 | 0.0377 | 5.97 | 0.0396 | 4.96 | 0.0019 |
| 13 | qi_depth_inv | 233899 | 0.1176 | 0.0375 | 6.03 | 0.0419 | 4.56 | 0.0045 |
| 14 | tr_quote_ratio_5s | 233870 | -0.0429 | -0.0372 | -3.02 | -0.0487 | -3.59 | -0.0115 |
| 15 | tr_count_5s | 233899 | -0.0341 | -0.0366 | -2.09 | -0.0384 | -3.49 | -0.0019 |
| 16 | ofi_l8_5s | 215900 | -0.0157 | -0.0352 | -1.15 | -0.0125 | -0.22 | 0.0227 |
| 17 | st_event_rate_5s | 233899 | -0.0329 | -0.0349 | -1.55 | -0.0302 | -2.18 | 0.0047 |
| 18 | st_slope_ask | 233899 | 0.0312 | -0.0336 | 1.04 | -0.0211 | -0.06 | 0.0125 |
| 19 | st_rv_1s | 215085 | -0.0487 | -0.0326 | -3.06 | -0.0147 | -4.05 | 0.0179 |
| 20 | tr_count_1s | 233899 | -0.0384 | -0.0306 | -4.62 | -0.0206 | -4.25 | 0.0100 |

### INTC / event / `label_y_mid_1s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 32710 | 0.3829 | 0.4246 | 55.70 | 0.4138 | 1.74 | -0.0109 |
| 2 | mp_wmid_dev_mid | 61437 | 0.3802 | 0.4098 | 58.44 | 0.4102 | 58.38 | 0.0004 |
| 3 | tr_signed_vol_100ms | 61437 | 0.2802 | 0.4072 | 38.18 | 0.3550 | 9.85 | -0.0522 |
| 4 | ofi_l1_100ms | 52994 | 0.2508 | 0.4012 | 36.61 | 0.3605 | 4.46 | -0.0407 |
| 5 | mp_dev_wmid | 52994 | -0.3552 | -0.3842 | -57.88 | -0.3955 | -51.17 | -0.0113 |
| 6 | ofi_integrated_100ms | 52994 | 0.1757 | 0.3727 | 29.30 | 0.2977 | 6.06 | -0.0750 |
| 7 | qi_touch | 61437 | 0.3614 | 0.3727 | 59.82 | 0.3759 | 55.34 | 0.0032 |
| 8 | qi_l1 | 61437 | 0.3614 | 0.3727 | 59.82 | 0.3759 | 55.34 | 0.0032 |
| 9 | tr_aggr_imb_1s | 40539 | 0.3240 | 0.3704 | 51.98 | 0.3640 | 1.99 | -0.0064 |
| 10 | tr_signed_vol_1s | 61437 | 0.2630 | 0.3679 | 39.29 | 0.3304 | 14.14 | -0.0374 |
| 11 | ofi_best_100ms | 61437 | 0.2129 | 0.3661 | 35.92 | 0.3254 | 5.46 | -0.0407 |
| 12 | ofi_l1_1s | 52994 | 0.2310 | 0.3525 | 33.12 | 0.3229 | 31.29 | -0.0296 |
| 13 | tr_sell_vol_100ms | 61437 | -0.2369 | -0.3431 | -34.80 | -0.2649 | -12.59 | 0.0783 |
| 14 | tr_sell_vol_1s | 61437 | -0.2257 | -0.3259 | -35.30 | -0.2830 | -14.20 | 0.0429 |
| 15 | qi_depth_inv | 61437 | 0.2987 | 0.3251 | 38.48 | 0.3372 | 7.10 | 0.0121 |
| 16 | ofi_deep_100ms | 61437 | 0.1322 | 0.3246 | 24.12 | 0.2535 | 4.86 | -0.0710 |
| 17 | ofi_best_1s | 61437 | 0.1866 | 0.3177 | 31.10 | 0.2856 | 25.60 | -0.0321 |
| 18 | ofi_raw_l1_1s | 61437 | 0.1866 | 0.3177 | 31.10 | 0.2856 | 25.60 | -0.0321 |
| 19 | ofi_integrated_1s | 52994 | 0.1513 | 0.3152 | 23.88 | 0.2544 | 15.39 | -0.0608 |
| 20 | ofi_l2_100ms | 52994 | 0.1431 | 0.3145 | 24.51 | 0.2765 | 4.97 | -0.0380 |

### INTC / event / `label_y_wmid_1s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | ofi_l1_100ms | 52994 | 0.1868 | 0.3491 | 27.32 | 0.3066 | 4.47 | -0.0426 |
| 2 | ofi_integrated_100ms | 52994 | 0.1611 | 0.3371 | 25.51 | 0.2657 | 6.26 | -0.0713 |
| 3 | tr_aggr_imb_100ms | 32710 | 0.2350 | 0.3135 | 32.57 | 0.3268 | 1.86 | 0.0132 |
| 4 | ofi_best_100ms | 61437 | 0.1412 | 0.2931 | 25.40 | 0.2410 | 5.21 | -0.0521 |
| 5 | ofi_l2_100ms | 52994 | 0.1621 | 0.2909 | 25.63 | 0.2584 | 5.13 | -0.0325 |
| 6 | ofi_l1_1s | 52994 | 0.1739 | 0.2902 | 24.45 | 0.2661 | 24.26 | -0.0242 |
| 7 | tr_signed_vol_100ms | 61437 | 0.1974 | 0.2852 | 30.04 | 0.2446 | 10.39 | -0.0406 |
| 8 | ofi_integrated_1s | 52994 | 0.1438 | 0.2816 | 21.43 | 0.2368 | 15.88 | -0.0448 |
| 9 | tr_sell_vol_100ms | 61437 | -0.1810 | -0.2805 | -29.07 | -0.2182 | -12.27 | 0.0623 |
| 10 | ofi_deep_100ms | 61437 | 0.0976 | 0.2662 | 16.44 | 0.2033 | 5.03 | -0.0629 |
| 11 | ofi_l7_100ms | 52994 | 0.1460 | 0.2639 | 22.67 | 0.2424 | 1.42 | -0.0215 |
| 12 | tr_sell_vol_1s | 61437 | -0.1704 | -0.2607 | -28.35 | -0.2331 | -13.70 | 0.0276 |
| 13 | tr_aggr_imb_1s | 40539 | 0.1948 | 0.2588 | 30.32 | 0.2658 | 2.11 | 0.0070 |
| 14 | ofi_l10_100ms | 52994 | 0.1469 | 0.2527 | 23.88 | 0.2384 | 6.22 | -0.0143 |
| 15 | ofi_l2_1s | 52994 | 0.1487 | 0.2515 | 21.51 | 0.2334 | 3.56 | -0.0181 |
| 16 | tr_signed_vol_1s | 61437 | 0.1807 | 0.2501 | 29.41 | 0.2230 | 15.47 | -0.0271 |
| 17 | ofi_l3_100ms | 52994 | 0.1496 | 0.2488 | 23.63 | 0.2438 | 7.78 | -0.0050 |
| 18 | ofi_l6_100ms | 52994 | 0.1443 | 0.2467 | 23.09 | 0.2386 | 2.39 | -0.0081 |
| 19 | ofi_best_1s | 61437 | 0.1235 | 0.2456 | 22.05 | 0.2097 | 13.49 | -0.0359 |
| 20 | ofi_raw_l1_1s | 61437 | 0.1235 | 0.2456 | 22.05 | 0.2097 | 13.49 | -0.0359 |

### INTC / event / `label_y_mid_2s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 32698 | 0.3299 | 0.3956 | 39.17 | 0.3971 | 1.78 | 0.0015 |
| 2 | mp_wmid_dev_mid | 61422 | 0.3265 | 0.3741 | 39.00 | 0.3806 | 45.40 | 0.0066 |
| 3 | tr_signed_vol_100ms | 61422 | 0.2472 | 0.3689 | 30.45 | 0.3220 | 9.97 | -0.0468 |
| 4 | ofi_l1_100ms | 52979 | 0.2198 | 0.3573 | 26.28 | 0.3156 | 4.44 | -0.0418 |
| 5 | qi_touch | 61422 | 0.3226 | 0.3504 | 43.93 | 0.3603 | 44.04 | 0.0099 |
| 6 | qi_l1 | 61422 | 0.3226 | 0.3504 | 43.93 | 0.3603 | 44.04 | 0.0099 |
| 7 | tr_aggr_imb_1s | 40524 | 0.2901 | 0.3495 | 38.23 | 0.3550 | 2.08 | 0.0056 |
| 8 | mp_dev_wmid | 52979 | -0.3074 | -0.3488 | -42.16 | -0.3674 | -41.31 | -0.0186 |
| 9 | tr_signed_vol_1s | 61422 | 0.2467 | 0.3438 | 31.87 | 0.3047 | 13.51 | -0.0391 |
| 10 | tr_sell_vol_100ms | 61422 | -0.2227 | -0.3312 | -27.35 | -0.2600 | -13.13 | 0.0711 |
| 11 | ofi_integrated_100ms | 52979 | 0.1435 | 0.3288 | 19.39 | 0.2546 | 5.91 | -0.0742 |
| 12 | ofi_best_100ms | 61422 | 0.1767 | 0.3263 | 26.67 | 0.2880 | 5.38 | -0.0384 |
| 13 | ofi_l1_1s | 52979 | 0.2302 | 0.3257 | 26.25 | 0.2927 | 24.94 | -0.0330 |
| 14 | tr_sell_vol_1s | 61422 | -0.2276 | -0.3186 | -29.31 | -0.2822 | -13.70 | 0.0364 |
| 15 | mp_dev_mid | 52979 | 0.2997 | 0.3050 | 41.75 | 0.3160 | 45.54 | 0.0110 |
| 16 | ofi_best_1s | 61422 | 0.1805 | 0.2994 | 26.86 | 0.2676 | 21.78 | -0.0318 |
| 17 | ofi_raw_l1_1s | 61422 | 0.1805 | 0.2994 | 26.86 | 0.2676 | 21.78 | -0.0318 |
| 18 | qi_sign | 61422 | 0.2523 | 0.2974 | 40.97 | 0.2930 | 41.65 | -0.0044 |
| 19 | qi_depth_inv | 61422 | 0.2590 | 0.2938 | 25.94 | 0.3111 | 7.03 | 0.0173 |
| 20 | tr_aggr_imb_5s | 50598 | 0.2335 | 0.2906 | 36.11 | 0.2834 | 1.97 | -0.0072 |

### INTC / event / `label_y_wmid_2s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | ofi_l1_100ms | 52979 | 0.1551 | 0.2944 | 18.41 | 0.2582 | 4.40 | -0.0363 |
| 2 | tr_aggr_imb_100ms | 32698 | 0.2021 | 0.2885 | 23.30 | 0.3049 | 1.84 | 0.0164 |
| 3 | ofi_integrated_100ms | 52979 | 0.1262 | 0.2857 | 16.29 | 0.2250 | 6.04 | -0.0606 |
| 4 | tr_sell_vol_100ms | 61422 | -0.1747 | -0.2656 | -22.89 | -0.2094 | -13.19 | 0.0562 |
| 5 | tr_signed_vol_100ms | 61422 | 0.1746 | 0.2593 | 23.56 | 0.2252 | 10.54 | -0.0340 |
| 6 | tr_sell_vol_1s | 61422 | -0.1828 | -0.2544 | -24.65 | -0.2275 | -13.15 | 0.0269 |
| 7 | ofi_l1_1s | 52979 | 0.1692 | 0.2527 | 18.60 | 0.2311 | 18.73 | -0.0217 |
| 8 | ofi_best_100ms | 61422 | 0.1129 | 0.2489 | 18.11 | 0.2067 | 5.05 | -0.0423 |
| 9 | ofi_integrated_1s | 52979 | 0.1325 | 0.2447 | 15.27 | 0.2075 | 13.26 | -0.0372 |
| 10 | tr_aggr_imb_1s | 40524 | 0.1807 | 0.2440 | 23.37 | 0.2557 | 2.13 | 0.0117 |
| 11 | ofi_l2_100ms | 52979 | 0.1343 | 0.2437 | 17.13 | 0.2166 | 5.00 | -0.0271 |
| 12 | tr_signed_vol_1s | 61422 | 0.1787 | 0.2375 | 24.93 | 0.2107 | 14.07 | -0.0268 |
| 13 | tr_sell_vol_5s | 61422 | -0.1397 | -0.2278 | -19.17 | -0.2091 | -15.96 | 0.0186 |
| 14 | ofi_deep_100ms | 61422 | 0.0751 | 0.2248 | 11.01 | 0.1742 | 4.69 | -0.0506 |
| 15 | ofi_best_1s | 61422 | 0.1185 | 0.2172 | 18.45 | 0.1867 | 12.91 | -0.0305 |
| 16 | ofi_raw_l1_1s | 61422 | 0.1185 | 0.2172 | 18.45 | 0.1867 | 12.91 | -0.0305 |
| 17 | ofi_l7_100ms | 52979 | 0.1112 | 0.2170 | 14.30 | 0.2024 | 1.41 | -0.0147 |
| 18 | ofi_l2_1s | 52979 | 0.1436 | 0.2159 | 15.89 | 0.2021 | 3.20 | -0.0138 |
| 19 | ofi_l10_100ms | 52979 | 0.1058 | 0.2084 | 15.30 | 0.2009 | 6.27 | -0.0075 |
| 20 | ofi_l3_100ms | 52979 | 0.1184 | 0.2038 | 15.00 | 0.2034 | 7.42 | -0.0004 |

### INTC / event / `label_y_mid_5s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 32609 | 0.2919 | 0.3733 | 25.09 | 0.3701 | 1.76 | -0.0032 |
| 2 | mp_wmid_dev_mid | 61328 | 0.3171 | 0.3597 | 30.13 | 0.3649 | 35.26 | 0.0052 |
| 3 | qi_touch | 61328 | 0.3108 | 0.3481 | 34.15 | 0.3552 | 35.39 | 0.0071 |
| 4 | qi_l1 | 61328 | 0.3108 | 0.3481 | 34.15 | 0.3552 | 35.39 | 0.0071 |
| 5 | tr_signed_vol_100ms | 61328 | 0.2214 | 0.3235 | 21.19 | 0.2846 | 9.53 | -0.0389 |
| 6 | tr_aggr_imb_1s | 40430 | 0.2403 | 0.3179 | 22.72 | 0.3257 | 2.05 | 0.0079 |
| 7 | mp_dev_wmid | 52885 | -0.2906 | -0.3163 | -29.23 | -0.3327 | -29.89 | -0.0163 |
| 8 | ofi_l1_100ms | 52885 | 0.2018 | 0.3130 | 17.01 | 0.2742 | 4.35 | -0.0388 |
| 9 | qi_depth_inv | 61328 | 0.2914 | 0.3072 | 23.19 | 0.3134 | 6.89 | 0.0062 |
| 10 | tr_signed_vol_1s | 61328 | 0.2205 | 0.3010 | 21.30 | 0.2695 | 12.13 | -0.0314 |
| 11 | qi_depth_exp | 61328 | 0.2876 | 0.2985 | 22.09 | 0.3053 | 15.97 | 0.0067 |
| 12 | mp_dev_mid | 52885 | 0.2800 | 0.2923 | 31.93 | 0.3110 | 35.41 | 0.0186 |
| 13 | ofi_best_100ms | 61328 | 0.1578 | 0.2911 | 16.68 | 0.2606 | 5.15 | -0.0304 |
| 14 | ofi_integrated_100ms | 52885 | 0.1383 | 0.2891 | 12.81 | 0.2187 | 5.56 | -0.0704 |
| 15 | ofi_l1_1s | 52885 | 0.2157 | 0.2860 | 17.71 | 0.2557 | 17.63 | -0.0303 |
| 16 | qi_sign | 61328 | 0.2363 | 0.2824 | 28.94 | 0.2685 | 29.43 | -0.0139 |
| 17 | tr_sell_vol_100ms | 61328 | -0.1852 | -0.2783 | -17.92 | -0.2192 | -12.79 | 0.0591 |
| 18 | tr_aggr_imb_5s | 50504 | 0.2036 | 0.2719 | 22.32 | 0.2623 | 1.97 | -0.0095 |
| 19 | tr_sell_vol_1s | 61328 | -0.1808 | -0.2645 | -16.69 | -0.2349 | -12.80 | 0.0296 |
| 20 | ofi_deep_100ms | 61328 | 0.0975 | 0.2570 | 10.32 | 0.1980 | 4.38 | -0.0590 |

### INTC / event / `label_y_wmid_5s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 32609 | 0.1752 | 0.2442 | 14.59 | 0.2617 | 1.80 | 0.0174 |
| 2 | ofi_l1_100ms | 52885 | 0.1334 | 0.2263 | 11.01 | 0.1976 | 4.19 | -0.0287 |
| 3 | ofi_integrated_100ms | 52885 | 0.1055 | 0.2200 | 9.38 | 0.1694 | 5.56 | -0.0506 |
| 4 | tr_sell_vol_100ms | 61328 | -0.1451 | -0.2190 | -14.77 | -0.1754 | -12.13 | 0.0436 |
| 5 | tr_signed_vol_100ms | 61328 | 0.1575 | 0.2108 | 15.69 | 0.1925 | 9.28 | -0.0183 |
| 6 | tr_sell_vol_1s | 61328 | -0.1443 | -0.2080 | -13.95 | -0.1907 | -12.02 | 0.0173 |
| 7 | tr_aggr_imb_1s | 40430 | 0.1428 | 0.1977 | 13.15 | 0.2189 | 2.09 | 0.0212 |
| 8 | ofi_best_100ms | 61328 | 0.0998 | 0.1961 | 10.47 | 0.1691 | 4.47 | -0.0270 |
| 9 | tr_sell_vol_5s | 61328 | -0.0963 | -0.1951 | -9.35 | -0.1795 | -12.52 | 0.0156 |
| 10 | tr_signed_vol_1s | 61328 | 0.1586 | 0.1925 | 16.00 | 0.1796 | 11.31 | -0.0128 |
| 11 | ofi_l1_1s | 52885 | 0.1483 | 0.1917 | 11.75 | 0.1757 | 12.12 | -0.0161 |
| 12 | ofi_integrated_1s | 52885 | 0.1098 | 0.1838 | 8.93 | 0.1549 | 9.55 | -0.0288 |
| 13 | ofi_l2_100ms | 52885 | 0.1213 | 0.1833 | 10.61 | 0.1624 | 4.68 | -0.0209 |
| 14 | ofi_deep_100ms | 61328 | 0.0643 | 0.1794 | 6.52 | 0.1402 | 4.15 | -0.0392 |
| 15 | tr_aggr_imb_5s | 50504 | 0.1264 | 0.1718 | 13.61 | 0.1742 | 2.02 | 0.0023 |
| 16 | ofi_best_1s | 61328 | 0.1010 | 0.1629 | 10.62 | 0.1439 | 7.22 | -0.0190 |
| 17 | ofi_raw_l1_1s | 61328 | 0.1010 | 0.1629 | 10.62 | 0.1439 | 7.22 | -0.0190 |
| 18 | ofi_l7_100ms | 52885 | 0.0893 | 0.1577 | 7.74 | 0.1524 | 1.46 | -0.0053 |
| 19 | ofi_l2_1s | 52885 | 0.1277 | 0.1571 | 10.19 | 0.1508 | 2.81 | -0.0062 |
| 20 | ofi_deep_1s | 61328 | 0.0489 | 0.1539 | 4.47 | 0.1267 | 5.91 | -0.0272 |

### INTC / event / `label_y_mid_10s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | qi_touch | 61173 | 0.2758 | 0.3084 | 24.14 | 0.3114 | 25.26 | 0.0031 |
| 2 | qi_l1 | 61173 | 0.2758 | 0.3084 | 24.14 | 0.3114 | 25.26 | 0.0031 |
| 3 | mp_wmid_dev_mid | 61173 | 0.2713 | 0.3075 | 20.60 | 0.3086 | 24.33 | 0.0011 |
| 4 | tr_aggr_imb_100ms | 32480 | 0.2240 | 0.2849 | 14.16 | 0.2975 | 1.61 | 0.0125 |
| 5 | qi_depth_inv | 61173 | 0.2573 | 0.2739 | 15.95 | 0.2692 | 7.27 | -0.0048 |
| 6 | qi_depth_exp | 61173 | 0.2604 | 0.2695 | 15.72 | 0.2637 | 14.56 | -0.0057 |
| 7 | mp_dev_mid | 52730 | 0.2481 | 0.2621 | 22.23 | 0.2928 | 25.73 | 0.0307 |
| 8 | mp_dev_wmid | 52730 | -0.2302 | -0.2570 | -18.15 | -0.2706 | -19.15 | -0.0136 |
| 9 | tr_aggr_imb_1s | 40275 | 0.1838 | 0.2509 | 12.70 | 0.2681 | 1.90 | 0.0172 |
| 10 | qi_sign | 61173 | 0.2023 | 0.2449 | 18.88 | 0.2336 | 19.30 | -0.0114 |
| 11 | tr_signed_vol_100ms | 61173 | 0.1643 | 0.2438 | 14.48 | 0.2362 | 8.67 | -0.0076 |
| 12 | ofi_l1_100ms | 52730 | 0.1401 | 0.2272 | 10.10 | 0.2085 | 4.16 | -0.0187 |
| 13 | tr_sell_vol_1s | 61173 | -0.1365 | -0.2270 | -11.16 | -0.2020 | -10.68 | 0.0250 |
| 14 | tr_signed_vol_1s | 61173 | 0.1510 | 0.2251 | 12.94 | 0.2123 | 9.77 | -0.0128 |
| 15 | tr_sell_vol_100ms | 61173 | -0.1478 | -0.2242 | -13.17 | -0.1838 | -10.52 | 0.0404 |
| 16 | tr_aggr_imb_5s | 50349 | 0.1596 | 0.2197 | 12.93 | 0.2307 | 1.81 | 0.0111 |
| 17 | ofi_best_100ms | 61173 | 0.1113 | 0.2158 | 10.65 | 0.2046 | 4.83 | -0.0111 |
| 18 | tr_sell_vol_5s | 61173 | -0.1039 | -0.2145 | -8.07 | -0.2040 | -10.94 | 0.0105 |
| 19 | ofi_integrated_100ms | 52730 | 0.0783 | 0.2063 | 5.86 | 0.1526 | 4.83 | -0.0536 |
| 20 | ofi_l1_1s | 52730 | 0.1368 | 0.2023 | 9.25 | 0.1845 | 9.84 | -0.0178 |

### INTC / event / `label_y_wmid_10s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_sell_vol_1s | 61173 | -0.1016 | -0.1663 | -8.77 | -0.1549 | -9.34 | 0.0114 |
| 2 | tr_sell_vol_100ms | 61173 | -0.1077 | -0.1615 | -9.94 | -0.1372 | -9.16 | 0.0243 |
| 3 | tr_aggr_imb_100ms | 32480 | 0.1034 | 0.1564 | 6.40 | 0.1887 | 1.52 | 0.0323 |
| 4 | tr_sell_vol_5s | 61173 | -0.0678 | -0.1536 | -5.56 | -0.1428 | -8.95 | 0.0108 |
| 5 | ofi_l1_100ms | 52730 | 0.0757 | 0.1472 | 5.47 | 0.1389 | 3.74 | -0.0083 |
| 6 | ofi_integrated_100ms | 52730 | 0.0443 | 0.1409 | 3.23 | 0.1090 | 4.23 | -0.0319 |
| 7 | tr_signed_vol_100ms | 61173 | 0.1039 | 0.1371 | 9.40 | 0.1511 | 7.52 | 0.0140 |
| 8 | tr_aggr_imb_1s | 40275 | 0.0832 | 0.1345 | 5.67 | 0.1625 | 1.80 | 0.0280 |
| 9 | tr_signed_vol_1s | 61173 | 0.0940 | 0.1247 | 8.23 | 0.1317 | 7.78 | 0.0070 |
| 10 | ofi_best_100ms | 61173 | 0.0563 | 0.1243 | 5.42 | 0.1179 | 3.68 | -0.0064 |
| 11 | tr_count_1s | 61173 | -0.0587 | -0.1237 | -4.35 | -0.1319 | -6.90 | -0.0081 |
| 12 | tr_aggr_imb_5s | 50349 | 0.0761 | 0.1201 | 6.10 | 0.1338 | 1.72 | 0.0137 |
| 13 | ofi_l1_1s | 52730 | 0.0740 | 0.1199 | 5.06 | 0.1161 | 6.00 | -0.0038 |
| 14 | tr_count_5s | 61173 | -0.0501 | -0.1164 | -2.90 | -0.1188 | -6.96 | -0.0024 |
| 15 | st_rv_1s | 61173 | -0.0928 | -0.1161 | -4.55 | -0.0950 | -6.40 | 0.0211 |
| 16 | tr_count_100ms | 61173 | -0.0644 | -0.1150 | -5.93 | -0.1013 | -5.08 | 0.0136 |
| 17 | tr_quote_ratio_100ms | 61173 | -0.0679 | -0.1140 | -7.61 | -0.1214 | -7.52 | -0.0074 |
| 18 | ofi_integrated_1s | 52730 | 0.0391 | 0.1138 | 2.64 | 0.0946 | 4.56 | -0.0192 |
| 19 | ofi_deep_100ms | 61173 | 0.0263 | 0.1105 | 2.42 | 0.0885 | 3.34 | -0.0220 |
| 20 | tr_quote_ratio_1s | 61173 | -0.0648 | -0.1097 | -6.57 | -0.1142 | -6.64 | -0.0045 |

### AMZN / calendar / `label_y_mid_1s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 4220 | 0.2748 | 0.3152 | 12.49 | 0.2751 | 12.20 | -0.0400 |
| 2 | tr_aggr_imb_1s | 31534 | 0.2007 | 0.2263 | 16.29 | 0.2054 | 15.66 | -0.0209 |
| 3 | tr_aggr_imb_5s | 102435 | 0.1224 | 0.1349 | 15.94 | 0.1217 | 5.00 | -0.0132 |
| 4 | tr_signed_vol_1s | 233989 | 0.1022 | 0.1020 | 7.27 | 0.0520 | 6.34 | -0.0499 |
| 5 | tr_signed_vol_5s | 233989 | 0.0774 | 0.0984 | 8.22 | 0.0806 | 6.90 | -0.0178 |
| 6 | ofi_integrated_5s | 215990 | 0.0692 | 0.0924 | 9.03 | 0.0884 | 10.54 | -0.0040 |
| 7 | ofi_l6_5s | 215990 | 0.0527 | 0.0876 | 5.85 | 0.0833 | 9.37 | -0.0043 |
| 8 | ofi_l9_5s | 215990 | 0.0572 | 0.0874 | 5.68 | 0.0841 | 8.55 | -0.0033 |
| 9 | ofi_deep_5s | 233989 | 0.0638 | 0.0874 | 9.10 | 0.0833 | 10.67 | -0.0042 |
| 10 | ofi_l1_5s | 215990 | 0.0382 | 0.0863 | 4.34 | 0.0786 | 1.30 | -0.0077 |
| 11 | ofi_l7_5s | 215990 | 0.0559 | 0.0861 | 6.51 | 0.0826 | 9.43 | -0.0035 |
| 12 | ofi_l8_5s | 215990 | 0.0594 | 0.0859 | 6.74 | 0.0819 | 9.32 | -0.0040 |
| 13 | ofi_l3_5s | 215990 | 0.0572 | 0.0853 | 6.00 | 0.0812 | 8.62 | -0.0042 |
| 14 | ofi_l4_5s | 215990 | 0.0558 | 0.0847 | 6.42 | 0.0801 | 8.94 | -0.0046 |
| 15 | ofi_best_5s | 233989 | 0.0369 | 0.0835 | 4.56 | 0.0764 | 1.32 | -0.0071 |
| 16 | ofi_l5_5s | 215990 | 0.0514 | 0.0832 | 5.66 | 0.0783 | 8.94 | -0.0048 |
| 17 | ofi_l10_5s | 215990 | 0.0712 | 0.0823 | 10.27 | 0.0794 | 10.39 | -0.0029 |
| 18 | tr_last_trade_sign | 233988 | 0.0664 | 0.0797 | 12.71 | 0.0706 | 12.24 | -0.0090 |
| 19 | qi_touch | 233989 | 0.0437 | 0.0749 | 8.50 | 0.0734 | 2.47 | -0.0015 |
| 20 | qi_l1 | 233989 | 0.0437 | 0.0749 | 8.50 | 0.0734 | 2.47 | -0.0015 |

### AMZN / calendar / `label_y_wmid_1s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | mp_wmid_dev_mid | 233989 | -0.2691 | -0.1853 | -29.51 | -0.1830 | -12.01 | 0.0023 |
| 2 | mp_dev_wmid | 215963 | 0.2500 | 0.1800 | 40.08 | 0.1795 | 13.99 | -0.0004 |
| 3 | qi_touch | 233989 | -0.2151 | -0.1705 | -39.18 | -0.1742 | -2.50 | -0.0037 |
| 4 | qi_l1 | 233989 | -0.2151 | -0.1705 | -39.18 | -0.1742 | -2.50 | -0.0037 |
| 5 | qi_sign | 233989 | -0.1953 | -0.1628 | -37.82 | -0.1594 | -36.15 | 0.0034 |
| 6 | tr_aggr_imb_100ms | 4220 | 0.1695 | 0.1541 | 8.49 | 0.1368 | 8.52 | -0.0172 |
| 7 | mp_dev_mid | 215963 | -0.1701 | -0.1483 | -16.78 | -0.1575 | -9.06 | -0.0092 |
| 8 | st_depth_bid_touch | 233989 | -0.0427 | -0.1215 | -4.50 | -0.1088 | -1.01 | 0.0126 |
| 9 | tr_aggr_imb_1s | 31534 | 0.1250 | 0.1197 | 10.33 | 0.1048 | 10.24 | -0.0149 |
| 10 | st_depth_ask_touch | 233989 | 0.1097 | 0.1177 | 13.77 | 0.1113 | 19.37 | -0.0064 |
| 11 | qi_depth_inv | 233989 | -0.0967 | -0.0843 | -18.18 | -0.0908 | -2.56 | -0.0066 |
| 12 | qi_depth_exp | 233989 | -0.0761 | -0.0645 | -14.70 | -0.0699 | -4.44 | -0.0054 |
| 13 | ofi_l2_5s | 215990 | 0.0491 | 0.0632 | 5.87 | 0.0594 | 8.61 | -0.0038 |
| 14 | tr_aggr_imb_5s | 102435 | 0.0700 | 0.0616 | 9.37 | 0.0563 | 4.62 | -0.0053 |
| 15 | tr_signed_vol_1s | 233989 | 0.0660 | 0.0524 | 5.98 | 0.0243 | 6.23 | -0.0281 |
| 16 | qi_l2 | 233989 | 0.0440 | 0.0499 | 8.79 | 0.0535 | 7.25 | 0.0036 |
| 17 | tr_signed_vol_5s | 233989 | 0.0437 | 0.0457 | 5.33 | 0.0343 | 5.75 | -0.0114 |
| 18 | tr_sell_vol_1s | 233989 | -0.0324 | -0.0437 | -4.07 | -0.0162 | -1.26 | 0.0275 |
| 19 | ofi_integrated_5s | 215990 | 0.0321 | 0.0407 | 4.58 | 0.0383 | 5.47 | -0.0024 |
| 20 | ofi_l3_5s | 215990 | 0.0357 | 0.0397 | 5.38 | 0.0372 | 6.19 | -0.0025 |

### AMZN / calendar / `label_y_mid_2s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 4216 | 0.3028 | 0.3489 | 13.55 | 0.3028 | 13.54 | -0.0461 |
| 2 | tr_aggr_imb_1s | 31524 | 0.2377 | 0.2716 | 16.35 | 0.2376 | 15.58 | -0.0340 |
| 3 | tr_aggr_imb_5s | 102425 | 0.1457 | 0.1653 | 14.01 | 0.1439 | 4.89 | -0.0214 |
| 4 | tr_signed_vol_5s | 233979 | 0.0829 | 0.1186 | 7.15 | 0.0987 | 6.69 | -0.0199 |
| 5 | tr_signed_vol_1s | 233979 | 0.1002 | 0.1169 | 8.52 | 0.0593 | 7.16 | -0.0576 |
| 6 | ofi_integrated_5s | 215980 | 0.0807 | 0.1082 | 8.32 | 0.1035 | 9.70 | -0.0047 |
| 7 | ofi_l9_5s | 215980 | 0.0659 | 0.1045 | 4.44 | 0.1005 | 7.36 | -0.0040 |
| 8 | ofi_deep_5s | 233979 | 0.0740 | 0.1039 | 8.10 | 0.0988 | 9.68 | -0.0051 |
| 9 | ofi_l1_5s | 215980 | 0.0434 | 0.1034 | 3.01 | 0.0935 | 1.28 | -0.0099 |
| 10 | ofi_l6_5s | 215980 | 0.0610 | 0.1031 | 5.12 | 0.0983 | 8.66 | -0.0049 |
| 11 | ofi_best_5s | 233979 | 0.0417 | 0.1016 | 3.08 | 0.0924 | 1.31 | -0.0092 |
| 12 | ofi_l8_5s | 215980 | 0.0694 | 0.1008 | 5.83 | 0.0970 | 8.66 | -0.0038 |
| 13 | ofi_l3_5s | 215980 | 0.0691 | 0.0998 | 5.36 | 0.0946 | 8.55 | -0.0053 |
| 14 | ofi_l4_5s | 215980 | 0.0679 | 0.0996 | 5.97 | 0.0946 | 8.62 | -0.0050 |
| 15 | ofi_l7_5s | 215980 | 0.0631 | 0.0983 | 5.59 | 0.0943 | 8.44 | -0.0040 |
| 16 | tr_last_trade_sign | 233978 | 0.0750 | 0.0962 | 10.45 | 0.0839 | 10.18 | -0.0123 |
| 17 | ofi_l10_5s | 215980 | 0.0803 | 0.0936 | 9.17 | 0.0901 | 9.33 | -0.0035 |
| 18 | ofi_l5_5s | 215980 | 0.0562 | 0.0935 | 5.06 | 0.0877 | 7.95 | -0.0058 |
| 19 | qi_touch | 233979 | 0.0497 | 0.0919 | 7.19 | 0.0898 | 2.41 | -0.0021 |
| 20 | qi_l1 | 233979 | 0.0497 | 0.0919 | 7.19 | 0.0898 | 2.41 | -0.0021 |

### AMZN / calendar / `label_y_wmid_2s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | mp_wmid_dev_mid | 233979 | -0.3229 | -0.2485 | -29.77 | -0.2450 | -11.17 | 0.0036 |
| 2 | mp_dev_wmid | 215953 | 0.2988 | 0.2413 | 37.16 | 0.2408 | 12.81 | -0.0005 |
| 3 | qi_touch | 233979 | -0.2596 | -0.2266 | -35.65 | -0.2315 | -2.48 | -0.0048 |
| 4 | qi_l1 | 233979 | -0.2596 | -0.2266 | -35.65 | -0.2315 | -2.48 | -0.0048 |
| 5 | qi_sign | 233979 | -0.2345 | -0.2146 | -34.30 | -0.2089 | -32.88 | 0.0057 |
| 6 | tr_aggr_imb_100ms | 4216 | 0.2209 | 0.2081 | 10.52 | 0.1781 | 10.72 | -0.0300 |
| 7 | mp_dev_mid | 215953 | -0.1981 | -0.1951 | -15.45 | -0.2073 | -8.54 | -0.0122 |
| 8 | st_depth_bid_touch | 233979 | -0.0594 | -0.1551 | -4.42 | -0.1362 | -1.06 | 0.0189 |
| 9 | st_depth_ask_touch | 233979 | 0.1302 | 0.1537 | 11.40 | 0.1469 | 17.09 | -0.0068 |
| 10 | tr_aggr_imb_1s | 31524 | 0.1610 | 0.1536 | 11.27 | 0.1302 | 10.90 | -0.0234 |
| 11 | qi_depth_inv | 233979 | -0.1175 | -0.1096 | -16.22 | -0.1185 | -2.55 | -0.0089 |
| 12 | qi_depth_exp | 233979 | -0.0930 | -0.0843 | -13.13 | -0.0915 | -4.38 | -0.0072 |
| 13 | tr_aggr_imb_5s | 102425 | 0.0861 | 0.0759 | 8.51 | 0.0654 | 4.99 | -0.0105 |
| 14 | ofi_l2_5s | 215980 | 0.0640 | 0.0748 | 5.72 | 0.0702 | 8.22 | -0.0046 |
| 15 | tr_signed_vol_1s | 233979 | 0.0679 | 0.0629 | 6.85 | 0.0253 | 7.12 | -0.0376 |
| 16 | qi_l2 | 233979 | 0.0497 | 0.0575 | 7.49 | 0.0626 | 7.17 | 0.0051 |
| 17 | tr_signed_vol_5s | 233979 | 0.0488 | 0.0552 | 4.73 | 0.0399 | 5.44 | -0.0153 |
| 18 | tr_sell_vol_1s | 233979 | -0.0362 | -0.0491 | -4.38 | -0.0214 | -1.30 | 0.0277 |
| 19 | tr_last_trade_sign | 233978 | 0.0439 | 0.0477 | 6.37 | 0.0420 | 6.54 | -0.0057 |
| 20 | ofi_integrated_5s | 215980 | 0.0426 | 0.0468 | 4.85 | 0.0435 | 5.37 | -0.0033 |

### AMZN / calendar / `label_y_mid_5s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 4205 | 0.3079 | 0.3450 | 11.33 | 0.2901 | 11.17 | -0.0548 |
| 2 | tr_aggr_imb_1s | 31494 | 0.2479 | 0.2777 | 12.92 | 0.2409 | 12.52 | -0.0368 |
| 3 | tr_aggr_imb_5s | 102395 | 0.1561 | 0.1738 | 10.14 | 0.1416 | 4.62 | -0.0322 |
| 4 | tr_signed_vol_5s | 233949 | 0.0859 | 0.1210 | 5.44 | 0.1050 | 5.65 | -0.0160 |
| 5 | tr_signed_vol_1s | 233949 | 0.0830 | 0.1146 | 8.52 | 0.0638 | 7.42 | -0.0508 |
| 6 | ofi_integrated_5s | 215950 | 0.0948 | 0.1123 | 6.78 | 0.1088 | 8.10 | -0.0035 |
| 7 | ofi_l1_5s | 215950 | 0.0450 | 0.1122 | 2.04 | 0.1020 | 1.11 | -0.0103 |
| 8 | ofi_best_5s | 233949 | 0.0401 | 0.1087 | 1.95 | 0.0984 | 1.14 | -0.0103 |
| 9 | ofi_l9_5s | 215950 | 0.0821 | 0.1074 | 3.94 | 0.1042 | 5.99 | -0.0032 |
| 10 | ofi_deep_5s | 233949 | 0.0850 | 0.1071 | 6.45 | 0.1023 | 7.67 | -0.0048 |
| 11 | ofi_l6_5s | 215950 | 0.0725 | 0.1070 | 4.17 | 0.1052 | 6.85 | -0.0018 |
| 12 | tr_last_trade_sign | 233948 | 0.0829 | 0.1069 | 7.48 | 0.0935 | 7.20 | -0.0134 |
| 13 | ofi_l7_5s | 215950 | 0.0748 | 0.1028 | 4.25 | 0.0998 | 6.62 | -0.0029 |
| 14 | ofi_l3_5s | 215950 | 0.0805 | 0.1018 | 4.45 | 0.0984 | 7.23 | -0.0034 |
| 15 | ofi_l8_5s | 215950 | 0.0829 | 0.1018 | 4.71 | 0.0989 | 7.06 | -0.0029 |
| 16 | ofi_l4_5s | 215950 | 0.0755 | 0.1014 | 4.82 | 0.0976 | 6.92 | -0.0038 |
| 17 | ofi_l5_5s | 215950 | 0.0678 | 0.0980 | 4.56 | 0.0940 | 6.60 | -0.0040 |
| 18 | tr_buy_vol_5s | 233949 | 0.0545 | 0.0960 | 2.77 | 0.0776 | 3.01 | -0.0184 |
| 19 | ofi_l10_5s | 215950 | 0.0929 | 0.0940 | 7.69 | 0.0916 | 8.20 | -0.0024 |
| 20 | ofi_l2_5s | 215950 | 0.0794 | 0.0935 | 4.45 | 0.0858 | 6.63 | -0.0077 |

### AMZN / calendar / `label_y_wmid_5s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | mp_wmid_dev_mid | 233949 | -0.3627 | -0.3234 | -28.12 | -0.3193 | -10.43 | 0.0041 |
| 2 | mp_dev_wmid | 215923 | 0.3347 | 0.3155 | 30.64 | 0.3148 | 11.79 | -0.0007 |
| 3 | qi_touch | 233949 | -0.2979 | -0.2945 | -28.82 | -0.3005 | -2.47 | -0.0061 |
| 4 | qi_l1 | 233949 | -0.2979 | -0.2945 | -28.82 | -0.3005 | -2.47 | -0.0061 |
| 5 | qi_sign | 233949 | -0.2679 | -0.2731 | -27.46 | -0.2631 | -26.29 | 0.0100 |
| 6 | mp_dev_mid | 215923 | -0.2319 | -0.2541 | -14.59 | -0.2702 | -8.16 | -0.0161 |
| 7 | tr_aggr_imb_100ms | 4205 | 0.2331 | 0.2320 | 9.97 | 0.1859 | 9.85 | -0.0461 |
| 8 | st_depth_bid_touch | 233949 | -0.0749 | -0.1977 | -3.48 | -0.1676 | -1.64 | 0.0301 |
| 9 | st_depth_ask_touch | 233949 | 0.1474 | 0.1899 | 8.65 | 0.1862 | 13.90 | -0.0036 |
| 10 | tr_aggr_imb_1s | 31494 | 0.1848 | 0.1838 | 10.09 | 0.1524 | 9.76 | -0.0314 |
| 11 | qi_depth_inv | 233949 | -0.1435 | -0.1440 | -12.97 | -0.1542 | -2.50 | -0.0102 |
| 12 | qi_depth_exp | 233949 | -0.1173 | -0.1142 | -10.81 | -0.1223 | -4.21 | -0.0081 |
| 13 | tr_aggr_imb_5s | 102395 | 0.0966 | 0.0893 | 6.74 | 0.0715 | 4.39 | -0.0178 |
| 14 | ofi_l2_5s | 215950 | 0.0787 | 0.0853 | 6.10 | 0.0839 | 6.78 | -0.0013 |
| 15 | tr_signed_vol_1s | 233949 | 0.0575 | 0.0736 | 6.68 | 0.0374 | 6.80 | -0.0362 |
| 16 | tr_last_trade_sign | 233948 | 0.0564 | 0.0630 | 5.45 | 0.0587 | 5.52 | -0.0042 |
| 17 | tr_signed_vol_5s | 233949 | 0.0503 | 0.0625 | 3.86 | 0.0524 | 4.27 | -0.0101 |
| 18 | tr_buy_vol_5s | 233949 | 0.0409 | 0.0587 | 2.52 | 0.0522 | 3.09 | -0.0064 |
| 19 | ofi_integrated_5s | 215950 | 0.0529 | 0.0570 | 4.09 | 0.0556 | 4.75 | -0.0014 |
| 20 | qi_l2 | 233949 | 0.0429 | 0.0566 | 4.49 | 0.0627 | 5.44 | 0.0061 |

### AMZN / calendar / `label_y_mid_10s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 4181 | 0.2767 | 0.2990 | 8.96 | 0.2447 | 9.24 | -0.0543 |
| 2 | tr_aggr_imb_1s | 31444 | 0.2260 | 0.2475 | 9.87 | 0.2113 | 9.75 | -0.0361 |
| 3 | tr_aggr_imb_5s | 102345 | 0.1591 | 0.1693 | 8.18 | 0.1392 | 4.39 | -0.0300 |
| 4 | ofi_l1_5s | 215900 | 0.0640 | 0.1125 | 2.41 | 0.1049 | 1.16 | -0.0077 |
| 5 | tr_signed_vol_5s | 233899 | 0.0913 | 0.1116 | 4.82 | 0.0997 | 4.82 | -0.0119 |
| 6 | ofi_integrated_5s | 215900 | 0.1004 | 0.1115 | 5.27 | 0.1133 | 6.42 | 0.0019 |
| 7 | ofi_best_5s | 233899 | 0.0540 | 0.1093 | 2.23 | 0.1013 | 1.18 | -0.0081 |
| 8 | ofi_deep_5s | 233899 | 0.0893 | 0.1061 | 5.22 | 0.1055 | 6.04 | -0.0006 |
| 9 | ofi_l8_5s | 215900 | 0.0886 | 0.1057 | 3.88 | 0.1071 | 5.47 | 0.0015 |
| 10 | ofi_l9_5s | 215900 | 0.0836 | 0.1053 | 3.33 | 0.1077 | 4.95 | 0.0024 |
| 11 | ofi_l3_5s | 215900 | 0.0895 | 0.1031 | 4.09 | 0.1029 | 6.32 | -0.0002 |
| 12 | ofi_l6_5s | 215900 | 0.0748 | 0.1024 | 3.15 | 0.1054 | 5.13 | 0.0030 |
| 13 | tr_last_trade_sign | 233898 | 0.0898 | 0.1022 | 6.07 | 0.0922 | 5.90 | -0.0100 |
| 14 | ofi_l4_5s | 215900 | 0.0757 | 0.1014 | 3.43 | 0.1020 | 5.47 | 0.0006 |
| 15 | ofi_l2_5s | 215900 | 0.0876 | 0.1001 | 4.05 | 0.0960 | 5.71 | -0.0041 |
| 16 | ofi_l10_5s | 215900 | 0.0975 | 0.0989 | 6.26 | 0.1010 | 6.77 | 0.0021 |
| 17 | tr_signed_vol_1s | 233899 | 0.0743 | 0.0984 | 7.51 | 0.0599 | 6.97 | -0.0385 |
| 18 | ofi_l5_5s | 215900 | 0.0668 | 0.0977 | 3.04 | 0.0995 | 4.67 | 0.0017 |
| 19 | ofi_l7_5s | 215900 | 0.0789 | 0.0974 | 3.44 | 0.0994 | 5.07 | 0.0021 |
| 20 | tr_buy_vol_5s | 233899 | 0.0570 | 0.0833 | 2.26 | 0.0741 | 2.46 | -0.0092 |

### AMZN / calendar / `label_y_wmid_10s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | mp_wmid_dev_mid | 233899 | -0.3444 | -0.3297 | -24.97 | -0.3269 | -10.34 | 0.0028 |
| 2 | mp_dev_wmid | 215873 | 0.3328 | 0.3273 | 24.41 | 0.3268 | 11.56 | -0.0005 |
| 3 | qi_touch | 233899 | -0.2957 | -0.3044 | -22.73 | -0.3094 | -2.46 | -0.0050 |
| 4 | qi_l1 | 233899 | -0.2957 | -0.3044 | -22.73 | -0.3094 | -2.46 | -0.0050 |
| 5 | qi_sign | 233899 | -0.2585 | -0.2710 | -20.97 | -0.2571 | -19.92 | 0.0139 |
| 6 | mp_dev_mid | 215873 | -0.2342 | -0.2658 | -13.38 | -0.2799 | -7.94 | -0.0141 |
| 7 | st_depth_bid_touch | 233899 | -0.0683 | -0.2059 | -2.16 | -0.1686 | -1.47 | 0.0374 |
| 8 | tr_aggr_imb_100ms | 4181 | 0.2065 | 0.2054 | 7.47 | 0.1627 | 7.91 | -0.0428 |
| 9 | st_depth_ask_touch | 233899 | 0.1357 | 0.1834 | 7.69 | 0.1813 | 10.78 | -0.0021 |
| 10 | tr_aggr_imb_1s | 31444 | 0.1714 | 0.1720 | 7.96 | 0.1471 | 7.96 | -0.0249 |
| 11 | qi_depth_inv | 233899 | -0.1534 | -0.1527 | -10.52 | -0.1592 | -2.57 | -0.0066 |
| 12 | qi_depth_exp | 233899 | -0.1310 | -0.1257 | -9.09 | -0.1312 | -4.28 | -0.0054 |
| 13 | tr_aggr_imb_5s | 102345 | 0.1106 | 0.1089 | 6.08 | 0.0926 | 4.14 | -0.0163 |
| 14 | ofi_l2_5s | 215900 | 0.0851 | 0.0880 | 4.52 | 0.0901 | 6.34 | 0.0021 |
| 15 | tr_last_trade_sign | 233898 | 0.0721 | 0.0780 | 5.14 | 0.0756 | 5.31 | -0.0024 |
| 16 | tr_signed_vol_5s | 233899 | 0.0615 | 0.0717 | 3.68 | 0.0651 | 4.09 | -0.0066 |
| 17 | tr_signed_vol_1s | 233899 | 0.0536 | 0.0665 | 6.11 | 0.0427 | 5.94 | -0.0238 |
| 18 | ofi_integrated_5s | 215900 | 0.0623 | 0.0631 | 3.61 | 0.0667 | 4.43 | 0.0036 |
| 19 | ofi_l9_5s | 215900 | 0.0470 | 0.0600 | 2.30 | 0.0645 | 3.20 | 0.0045 |
| 20 | ofi_l3_5s | 215900 | 0.0701 | 0.0600 | 4.70 | 0.0639 | 5.32 | 0.0039 |

### AMZN / event / `label_y_mid_1s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 7653 | 0.4422 | 0.5222 | 32.91 | 0.4614 | 13.71 | -0.0608 |
| 2 | tr_aggr_imb_1s | 11707 | 0.3844 | 0.4328 | 33.79 | 0.3970 | 23.99 | -0.0358 |
| 3 | tr_signed_vol_100ms | 26719 | 0.3164 | 0.3426 | 19.60 | 0.2519 | 16.85 | -0.0907 |
| 4 | tr_signed_vol_1s | 26719 | 0.3497 | 0.3320 | 18.50 | 0.2756 | 6.24 | -0.0564 |
| 5 | tr_aggr_imb_5s | 18105 | 0.2916 | 0.3150 | 30.84 | 0.2949 | 1.36 | -0.0201 |
| 6 | ofi_integrated_1s | 25000 | 0.2544 | 0.2899 | 9.55 | 0.2818 | 16.16 | -0.0081 |
| 7 | ofi_deep_1s | 26719 | 0.2539 | 0.2814 | 10.58 | 0.2711 | 10.72 | -0.0104 |
| 8 | ofi_integrated_100ms | 25000 | 0.2316 | 0.2787 | 5.80 | 0.2667 | 12.87 | -0.0120 |
| 9 | ofi_l1_100ms | 25000 | 0.1155 | 0.2761 | 4.84 | 0.2338 | 4.28 | -0.0422 |
| 10 | ofi_l7_1s | 25000 | 0.2035 | 0.2753 | 5.86 | 0.2696 | 10.58 | -0.0057 |
| 11 | ofi_l4_1s | 25000 | 0.1633 | 0.2745 | 4.56 | 0.2587 | 6.86 | -0.0157 |
| 12 | ofi_l6_1s | 25000 | 0.1952 | 0.2742 | 5.79 | 0.2653 | 2.06 | -0.0089 |
| 13 | ofi_l5_1s | 25000 | 0.2008 | 0.2730 | 5.45 | 0.2622 | 8.06 | -0.0108 |
| 14 | tr_signed_vol_5s | 26719 | 0.2630 | 0.2717 | 15.69 | 0.2450 | 13.71 | -0.0266 |
| 15 | ofi_best_100ms | 26719 | 0.1178 | 0.2712 | 5.14 | 0.2278 | 4.45 | -0.0433 |
| 16 | ofi_l8_1s | 25000 | 0.2206 | 0.2705 | 6.42 | 0.2650 | 1.38 | -0.0056 |
| 17 | ofi_deep_100ms | 26719 | 0.2325 | 0.2703 | 6.88 | 0.2552 | 7.41 | -0.0151 |
| 18 | ofi_l1_1s | 25000 | 0.1352 | 0.2678 | 6.24 | 0.2374 | 6.88 | -0.0305 |
| 19 | ofi_raw_l7_1s | 26719 | 0.2034 | 0.2664 | 6.13 | 0.2588 | 11.00 | -0.0075 |
| 20 | ofi_l9_1s | 25000 | 0.2132 | 0.2661 | 6.49 | 0.2622 | 8.24 | -0.0039 |

### AMZN / event / `label_y_wmid_1s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | mp_wmid_dev_mid | 26719 | -0.3438 | -0.2801 | -38.38 | -0.2779 | -3.46 | 0.0022 |
| 2 | mp_dev_wmid | 24981 | 0.3270 | 0.2741 | 38.04 | 0.2752 | 3.70 | 0.0012 |
| 3 | qi_touch | 26719 | -0.2849 | -0.2541 | -38.79 | -0.2581 | -1.21 | -0.0040 |
| 4 | qi_l1 | 26719 | -0.2849 | -0.2541 | -38.79 | -0.2581 | -1.21 | -0.0040 |
| 5 | tr_aggr_imb_100ms | 7653 | 0.2642 | 0.2376 | 18.76 | 0.2126 | 13.14 | -0.0249 |
| 6 | qi_sign | 26719 | -0.2496 | -0.2311 | -35.20 | -0.2182 | -34.58 | 0.0129 |
| 7 | mp_dev_mid | 24981 | -0.2098 | -0.2151 | -11.18 | -0.2267 | -2.14 | -0.0116 |
| 8 | tr_aggr_imb_1s | 11707 | 0.2314 | 0.2085 | 19.87 | 0.1908 | 17.08 | -0.0177 |
| 9 | st_depth_bid_touch | 26719 | -0.0222 | -0.1726 | -2.21 | -0.1486 | 1.66 | 0.0240 |
| 10 | st_depth_ask_touch | 26719 | 0.1237 | 0.1608 | 9.43 | 0.1564 | 12.60 | -0.0044 |
| 11 | tr_signed_vol_1s | 26719 | 0.2480 | 0.1596 | 15.33 | 0.1384 | 6.98 | -0.0212 |
| 12 | tr_signed_vol_100ms | 26719 | 0.2202 | 0.1512 | 14.99 | 0.1175 | 13.21 | -0.0337 |
| 13 | tr_aggr_imb_5s | 18105 | 0.1670 | 0.1392 | 17.73 | 0.1340 | 1.35 | -0.0052 |
| 14 | ofi_l2_5s | 25000 | 0.1226 | 0.1296 | 6.15 | 0.1259 | 10.24 | -0.0037 |
| 15 | ofi_l2_1s | 25000 | 0.1856 | 0.1274 | 7.39 | 0.1254 | 7.62 | -0.0020 |
| 16 | ofi_raw_l2_1s | 26719 | 0.1836 | 0.1257 | 7.61 | 0.1246 | 7.43 | -0.0012 |
| 17 | tr_signed_vol_5s | 26719 | 0.1803 | 0.1243 | 12.89 | 0.1198 | 11.33 | -0.0045 |
| 18 | qi_depth_inv | 26719 | -0.1446 | -0.1225 | -17.55 | -0.1276 | -1.91 | -0.0052 |
| 19 | tr_sell_vol_1s | 26719 | -0.1186 | -0.1184 | -10.18 | -0.0937 | -0.80 | 0.0247 |
| 20 | tr_buy_vol_1s | 26719 | 0.2230 | 0.1132 | 12.46 | 0.0701 | 1.32 | -0.0431 |

### AMZN / event / `label_y_mid_2s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 7652 | 0.4481 | 0.5276 | 28.59 | 0.4673 | 13.98 | -0.0603 |
| 2 | tr_aggr_imb_1s | 11704 | 0.3933 | 0.4415 | 29.33 | 0.4014 | 22.58 | -0.0401 |
| 3 | tr_signed_vol_100ms | 26716 | 0.3037 | 0.3367 | 15.70 | 0.2438 | 13.83 | -0.0928 |
| 4 | tr_signed_vol_1s | 26716 | 0.3381 | 0.3317 | 16.57 | 0.2714 | 5.96 | -0.0603 |
| 5 | tr_aggr_imb_5s | 18102 | 0.2970 | 0.3225 | 26.52 | 0.2990 | 1.46 | -0.0235 |
| 6 | ofi_integrated_1s | 24997 | 0.2480 | 0.2823 | 7.92 | 0.2733 | 13.59 | -0.0090 |
| 7 | tr_signed_vol_5s | 26716 | 0.2625 | 0.2772 | 14.04 | 0.2457 | 12.60 | -0.0316 |
| 8 | ofi_deep_1s | 26716 | 0.2477 | 0.2730 | 8.74 | 0.2628 | 9.88 | -0.0102 |
| 9 | ofi_l6_1s | 24997 | 0.1956 | 0.2689 | 5.17 | 0.2599 | 2.06 | -0.0090 |
| 10 | ofi_integrated_100ms | 24997 | 0.2208 | 0.2669 | 5.05 | 0.2534 | 10.87 | -0.0135 |
| 11 | ofi_l4_1s | 24997 | 0.1568 | 0.2658 | 3.88 | 0.2494 | 6.51 | -0.0164 |
| 12 | ofi_l5_1s | 24997 | 0.1968 | 0.2651 | 4.79 | 0.2531 | 7.62 | -0.0120 |
| 13 | ofi_l7_1s | 24997 | 0.1935 | 0.2632 | 4.88 | 0.2567 | 9.21 | -0.0066 |
| 14 | ofi_l1_100ms | 24997 | 0.1059 | 0.2623 | 4.01 | 0.2198 | 3.98 | -0.0425 |
| 15 | ofi_l1_1s | 24997 | 0.1300 | 0.2613 | 5.16 | 0.2290 | 6.10 | -0.0323 |
| 16 | ofi_l8_1s | 24997 | 0.2136 | 0.2602 | 5.39 | 0.2539 | 1.38 | -0.0063 |
| 17 | ofi_l9_1s | 24997 | 0.2082 | 0.2601 | 5.55 | 0.2551 | 7.83 | -0.0050 |
| 18 | ofi_raw_l6_1s | 26716 | 0.1920 | 0.2589 | 5.32 | 0.2484 | 2.14 | -0.0105 |
| 19 | ofi_deep_100ms | 26716 | 0.2218 | 0.2580 | 5.94 | 0.2424 | 7.00 | -0.0156 |
| 20 | ofi_best_1s | 26716 | 0.1311 | 0.2564 | 5.46 | 0.2230 | 6.34 | -0.0335 |

### AMZN / event / `label_y_wmid_2s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | mp_wmid_dev_mid | 26716 | -0.3489 | -0.3010 | -35.59 | -0.2992 | -3.78 | 0.0018 |
| 2 | mp_dev_wmid | 24978 | 0.3326 | 0.2945 | 35.12 | 0.2973 | 4.08 | 0.0028 |
| 3 | tr_aggr_imb_100ms | 7652 | 0.3004 | 0.2744 | 18.65 | 0.2463 | 13.94 | -0.0282 |
| 4 | qi_touch | 26716 | -0.2889 | -0.2714 | -35.34 | -0.2757 | -1.20 | -0.0043 |
| 5 | qi_l1 | 26716 | -0.2889 | -0.2714 | -35.34 | -0.2757 | -1.20 | -0.0043 |
| 6 | qi_sign | 26716 | -0.2544 | -0.2483 | -32.45 | -0.2352 | -32.02 | 0.0131 |
| 7 | tr_aggr_imb_1s | 11704 | 0.2580 | 0.2327 | 19.09 | 0.2074 | 18.08 | -0.0254 |
| 8 | mp_dev_mid | 24978 | -0.2104 | -0.2323 | -10.85 | -0.2422 | -2.13 | -0.0099 |
| 9 | st_depth_bid_touch | 26716 | -0.0292 | -0.1744 | -2.26 | -0.1493 | 1.49 | 0.0251 |
| 10 | st_depth_ask_touch | 26716 | 0.1246 | 0.1721 | 8.98 | 0.1693 | 12.30 | -0.0028 |
| 11 | tr_signed_vol_1s | 26716 | 0.2480 | 0.1721 | 14.01 | 0.1450 | 6.53 | -0.0270 |
| 12 | tr_signed_vol_100ms | 26716 | 0.2212 | 0.1703 | 13.11 | 0.1283 | 11.73 | -0.0419 |
| 13 | tr_aggr_imb_5s | 18102 | 0.1860 | 0.1556 | 16.99 | 0.1454 | 1.51 | -0.0103 |
| 14 | ofi_l2_1s | 24997 | 0.1931 | 0.1425 | 7.36 | 0.1374 | 7.55 | -0.0051 |
| 15 | ofi_l2_5s | 24997 | 0.1385 | 0.1423 | 6.06 | 0.1343 | 9.37 | -0.0079 |
| 16 | tr_signed_vol_5s | 26716 | 0.1861 | 0.1400 | 11.65 | 0.1315 | 10.86 | -0.0085 |
| 17 | ofi_raw_l2_1s | 26716 | 0.1896 | 0.1382 | 7.55 | 0.1343 | 6.81 | -0.0039 |
| 18 | tr_buy_vol_1s | 26716 | 0.2161 | 0.1269 | 11.92 | 0.0766 | 1.30 | -0.0502 |
| 19 | tr_sell_vol_100ms | 26716 | -0.1243 | -0.1265 | -7.46 | -0.0870 | -6.48 | 0.0394 |
| 20 | ofi_integrated_5s | 24997 | 0.1443 | 0.1264 | 10.56 | 0.1247 | 8.10 | -0.0017 |

### AMZN / event / `label_y_mid_5s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 7644 | 0.4505 | 0.5060 | 22.69 | 0.4295 | 13.14 | -0.0765 |
| 2 | tr_aggr_imb_1s | 11695 | 0.3992 | 0.4371 | 23.27 | 0.3844 | 19.95 | -0.0527 |
| 3 | tr_aggr_imb_5s | 18093 | 0.3028 | 0.3251 | 20.50 | 0.2921 | 1.44 | -0.0330 |
| 4 | tr_signed_vol_1s | 26707 | 0.2913 | 0.3169 | 12.92 | 0.2542 | 5.31 | -0.0626 |
| 5 | tr_signed_vol_100ms | 26707 | 0.2664 | 0.3129 | 12.65 | 0.2258 | 12.60 | -0.0871 |
| 6 | tr_signed_vol_5s | 26707 | 0.2460 | 0.2739 | 11.16 | 0.2427 | 11.27 | -0.0312 |
| 7 | ofi_integrated_1s | 24988 | 0.2282 | 0.2620 | 5.86 | 0.2535 | 10.43 | -0.0085 |
| 8 | ofi_deep_1s | 26707 | 0.2219 | 0.2550 | 6.43 | 0.2460 | 7.99 | -0.0090 |
| 9 | ofi_l1_1s | 24988 | 0.1146 | 0.2525 | 3.41 | 0.2204 | 5.17 | -0.0321 |
| 10 | ofi_l6_1s | 24988 | 0.1734 | 0.2480 | 3.75 | 0.2387 | 2.53 | -0.0093 |
| 11 | ofi_best_1s | 26707 | 0.1136 | 0.2472 | 3.48 | 0.2147 | 5.38 | -0.0324 |
| 12 | ofi_raw_l1_1s | 26707 | 0.1136 | 0.2472 | 3.48 | 0.2147 | 5.38 | -0.0324 |
| 13 | ofi_l4_1s | 24988 | 0.1405 | 0.2457 | 2.82 | 0.2301 | 6.45 | -0.0155 |
| 14 | ofi_l5_1s | 24988 | 0.1748 | 0.2453 | 3.46 | 0.2341 | 6.62 | -0.0112 |
| 15 | ofi_l1_5s | 24988 | 0.1179 | 0.2446 | 3.96 | 0.2272 | 0.85 | -0.0175 |
| 16 | ofi_l9_1s | 24988 | 0.1923 | 0.2434 | 4.05 | 0.2380 | 7.48 | -0.0054 |
| 17 | tr_last_trade_sign | 26706 | 0.2246 | 0.2432 | 18.90 | 0.2165 | 18.66 | -0.0267 |
| 18 | ofi_integrated_5s | 24988 | 0.2164 | 0.2423 | 10.71 | 0.2423 | 8.94 | 0.0001 |
| 19 | tr_buy_vol_1s | 26707 | 0.2236 | 0.2420 | 9.59 | 0.1582 | 1.21 | -0.0839 |
| 20 | ofi_l3_1s | 24988 | 0.2182 | 0.2419 | 7.58 | 0.2210 | 8.56 | -0.0208 |

### AMZN / event / `label_y_wmid_5s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | mp_wmid_dev_mid | 26707 | -0.3490 | -0.3154 | -31.79 | -0.3099 | -3.38 | 0.0054 |
| 2 | tr_aggr_imb_100ms | 7644 | 0.3231 | 0.3150 | 16.29 | 0.2518 | 12.66 | -0.0631 |
| 3 | mp_dev_wmid | 24969 | 0.3295 | 0.3095 | 31.91 | 0.3076 | 3.60 | -0.0019 |
| 4 | qi_touch | 26707 | -0.2909 | -0.2844 | -29.69 | -0.2844 | -1.19 | -0.0000 |
| 5 | qi_l1 | 26707 | -0.2909 | -0.2844 | -29.69 | -0.2844 | -1.19 | -0.0000 |
| 6 | tr_aggr_imb_1s | 11695 | 0.2829 | 0.2759 | 16.75 | 0.2302 | 15.81 | -0.0457 |
| 7 | qi_sign | 26707 | -0.2511 | -0.2553 | -26.66 | -0.2358 | -25.62 | 0.0195 |
| 8 | mp_dev_mid | 24969 | -0.2171 | -0.2466 | -10.29 | -0.2497 | -2.06 | -0.0031 |
| 9 | tr_signed_vol_1s | 26707 | 0.2015 | 0.1960 | 9.76 | 0.1547 | 5.65 | -0.0412 |
| 10 | tr_aggr_imb_5s | 18093 | 0.2026 | 0.1880 | 14.33 | 0.1645 | 1.44 | -0.0235 |
| 11 | tr_signed_vol_100ms | 26707 | 0.1948 | 0.1876 | 11.19 | 0.1309 | 10.07 | -0.0567 |
| 12 | st_depth_bid_touch | 26707 | -0.0649 | -0.1848 | -3.70 | -0.1508 | -4.86 | 0.0341 |
| 13 | st_depth_ask_touch | 26707 | 0.1259 | 0.1743 | 7.53 | 0.1700 | 10.78 | -0.0043 |
| 14 | ofi_l2_1s | 24988 | 0.1891 | 0.1656 | 7.57 | 0.1558 | 7.56 | -0.0098 |
| 15 | ofi_l2_5s | 24988 | 0.1551 | 0.1633 | 5.75 | 0.1525 | 9.28 | -0.0107 |
| 16 | ofi_raw_l2_1s | 26707 | 0.1840 | 0.1627 | 7.80 | 0.1540 | 5.30 | -0.0087 |
| 17 | tr_signed_vol_5s | 26707 | 0.1687 | 0.1614 | 8.89 | 0.1414 | 9.11 | -0.0200 |
| 18 | tr_buy_vol_1s | 26707 | 0.1648 | 0.1543 | 7.72 | 0.0982 | 1.25 | -0.0560 |
| 19 | ofi_integrated_5s | 24988 | 0.1499 | 0.1459 | 8.48 | 0.1431 | 7.04 | -0.0028 |
| 20 | ofi_integrated_1s | 24988 | 0.1464 | 0.1433 | 4.98 | 0.1374 | 7.71 | -0.0060 |

### AMZN / event / `label_y_mid_10s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | tr_aggr_imb_100ms | 7630 | 0.3976 | 0.4370 | 16.75 | 0.3616 | 11.33 | -0.0754 |
| 2 | tr_aggr_imb_1s | 11676 | 0.3464 | 0.3729 | 16.51 | 0.3204 | 14.01 | -0.0525 |
| 3 | tr_aggr_imb_5s | 18074 | 0.2634 | 0.2815 | 13.94 | 0.2464 | 1.41 | -0.0351 |
| 4 | tr_signed_vol_1s | 26688 | 0.2322 | 0.2676 | 9.74 | 0.2133 | 5.11 | -0.0544 |
| 5 | tr_signed_vol_100ms | 26688 | 0.2182 | 0.2592 | 10.30 | 0.1834 | 10.67 | -0.0758 |
| 6 | tr_signed_vol_5s | 26688 | 0.2134 | 0.2403 | 8.40 | 0.2123 | 9.36 | -0.0280 |
| 7 | ofi_integrated_1s | 24969 | 0.2067 | 0.2367 | 6.36 | 0.2317 | 9.91 | -0.0050 |
| 8 | ofi_l1_1s | 24969 | 0.1032 | 0.2296 | 3.01 | 0.2033 | 4.65 | -0.0263 |
| 9 | ofi_deep_1s | 26688 | 0.1951 | 0.2267 | 6.78 | 0.2201 | 7.58 | -0.0066 |
| 10 | ofi_l1_5s | 24969 | 0.1256 | 0.2265 | 3.45 | 0.2139 | 0.87 | -0.0126 |
| 11 | ofi_l6_1s | 24969 | 0.1638 | 0.2241 | 3.78 | 0.2188 | 2.53 | -0.0053 |
| 12 | ofi_l5_1s | 24969 | 0.1595 | 0.2231 | 3.39 | 0.2162 | 6.28 | -0.0069 |
| 13 | ofi_best_1s | 26688 | 0.0981 | 0.2224 | 2.92 | 0.1965 | 4.82 | -0.0260 |
| 14 | ofi_raw_l1_1s | 26688 | 0.0981 | 0.2224 | 2.92 | 0.1965 | 4.82 | -0.0260 |
| 15 | ofi_integrated_5s | 24969 | 0.2019 | 0.2217 | 8.55 | 0.2258 | 7.68 | 0.0041 |
| 16 | ofi_l4_1s | 24969 | 0.1353 | 0.2216 | 2.80 | 0.2092 | 5.81 | -0.0124 |
| 17 | ofi_best_5s | 26688 | 0.1139 | 0.2205 | 3.28 | 0.2077 | 0.87 | -0.0128 |
| 18 | ofi_l9_1s | 24969 | 0.1713 | 0.2202 | 3.83 | 0.2174 | 6.24 | -0.0028 |
| 19 | ofi_l3_1s | 24969 | 0.1905 | 0.2191 | 7.17 | 0.2023 | 7.92 | -0.0168 |
| 20 | ofi_l9_5s | 24969 | 0.1785 | 0.2181 | 5.53 | 0.2213 | 2.72 | 0.0032 |

### AMZN / event / `label_y_wmid_10s`

| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | IC sadj | t_hac sadj | delta (sadj-raw) |
|---|---|---|---|---|---|---|---|---|
| 1 | mp_wmid_dev_mid | 26688 | -0.3195 | -0.2953 | -27.31 | -0.2888 | -3.40 | 0.0065 |
| 2 | mp_dev_wmid | 24950 | 0.3028 | 0.2869 | 25.96 | 0.2847 | 3.62 | -0.0022 |
| 3 | tr_aggr_imb_100ms | 7630 | 0.2770 | 0.2755 | 11.73 | 0.2131 | 9.10 | -0.0624 |
| 4 | qi_touch | 26688 | -0.2724 | -0.2705 | -23.66 | -0.2691 | -1.19 | 0.0014 |
| 5 | qi_l1 | 26688 | -0.2724 | -0.2705 | -23.66 | -0.2691 | -1.19 | 0.0014 |
| 6 | tr_aggr_imb_1s | 11676 | 0.2408 | 0.2365 | 11.73 | 0.1900 | 10.53 | -0.0465 |
| 7 | qi_sign | 26688 | -0.2294 | -0.2325 | -21.10 | -0.2113 | -20.20 | 0.0212 |
| 8 | mp_dev_mid | 24950 | -0.2025 | -0.2292 | -10.44 | -0.2329 | -2.09 | -0.0038 |
| 9 | st_depth_bid_touch | 26688 | -0.0401 | -0.1766 | -1.28 | -0.1422 | -4.18 | 0.0345 |
| 10 | tr_aggr_imb_5s | 18074 | 0.1845 | 0.1762 | 10.26 | 0.1488 | 1.42 | -0.0275 |
| 11 | tr_signed_vol_1s | 26688 | 0.1740 | 0.1708 | 8.01 | 0.1345 | 4.87 | -0.0364 |
| 12 | st_depth_ask_touch | 26688 | 0.1206 | 0.1612 | 6.91 | 0.1611 | 9.55 | -0.0002 |
| 13 | tr_signed_vol_100ms | 26688 | 0.1648 | 0.1611 | 8.18 | 0.1104 | 7.89 | -0.0507 |
| 14 | tr_signed_vol_5s | 26688 | 0.1592 | 0.1580 | 6.99 | 0.1419 | 7.84 | -0.0162 |
| 15 | ofi_l2_5s | 24969 | 0.1461 | 0.1542 | 4.38 | 0.1482 | 7.29 | -0.0060 |
| 16 | ofi_l2_1s | 24969 | 0.1654 | 0.1456 | 7.11 | 0.1381 | 6.88 | -0.0075 |
| 17 | ofi_raw_l2_1s | 26688 | 0.1610 | 0.1435 | 7.34 | 0.1359 | 5.17 | -0.0076 |
| 18 | ofi_integrated_5s | 24969 | 0.1467 | 0.1431 | 6.78 | 0.1468 | 6.37 | 0.0037 |
| 19 | ofi_deep_5s | 26688 | 0.1358 | 0.1400 | 6.62 | 0.1420 | 6.45 | 0.0020 |
| 20 | ofi_l9_5s | 24969 | 0.1235 | 0.1373 | 5.05 | 0.1401 | 2.80 | 0.0028 |

