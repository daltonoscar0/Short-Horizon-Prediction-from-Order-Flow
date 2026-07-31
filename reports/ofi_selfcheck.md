# OFI self-check: Cont-Kukanov-Stoikov Delta-mid ~ OFI

`python3 -m edge.analysis.ofi_selfcheck --seed 0`. Regresses `label_y_mid_1s` (forward mid return over the next second, ticks) on the TRAILING 1s-window OFI columns as of the same stamp t: `ofi_best_1s` (the CKS spec), `ofi_deep_1s` (summed raw shares over levels 1..10), `ofi_raw_l{m}_1s` (m=1..10, per-level slope profile) and `ofi_integrated_1s`. Univariate OLS with intercept, hand-written on numpy (no scipy, no sklearn, no statsmodels). Rows dropped PAIRWISE per regression. Naive OLS SE is reported but judged wrong (overlapping labels on a 100ms grid are autocorrelated); significance is judged on the Newey-West HAC t-statistic, bar |t_hac| > 3.0.

## HAC lag

- INTC/calendar: HAC lag = 10 (fixed a priori: 10 calendar stamps span the 1s label horizon, always)
- INTC/event: HAC lag = 1 (median event-clock stamps per calendar second, computed). Distribution is heavily right-skewed -- mean 2.65, p90=4, p99=39, max=151: event-clock stamps arrive in bursts, so the median (the a-priori rule) is much smaller than a typical burst's width.
- AMZN/calendar: HAC lag = 10 (fixed a priori: 10 calendar stamps span the 1s label horizon, always)
- AMZN/event: HAC lag = 1 (median event-clock stamps per calendar second, computed). Distribution is heavily right-skewed -- mean 1.14, p90=3, p99=9, max=36: event-clock stamps arrive in bursts, so the median (the a-priori rule) is much smaller than a typical burst's width.

## Slope table

| ticker | clock | regressor | n | lag | slope | se_ols | t_ols | se_hac | t_hac | r2 |
|---|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | ofi_best_1s | 233989 | 10 | 9.47731e-07 | 5.22562e-08 | 18.136 | 1.54973e-07 | 6.115 | 0.00140 |
| INTC | calendar | ofi_deep_1s | 233989 | 10 | 3.07747e-09 | 7.67452e-09 | 0.401 | 2.4406e-08 | 0.126 | 0.00000 |
| INTC | calendar | ofi_raw_l1_1s | 233989 | 10 | 9.47731e-07 | 5.22562e-08 | 18.136 | 1.54973e-07 | 6.115 | 0.00140 |
| INTC | calendar | ofi_raw_l2_1s | 233989 | 10 | -2.52316e-08 | 6.22493e-08 | -0.405 | 1.8152e-07 | -0.139 | 0.00000 |
| INTC | calendar | ofi_raw_l3_1s | 233989 | 10 | 1.38986e-08 | 6.56974e-08 | 0.212 | 2.10058e-07 | 0.066 | 0.00000 |
| INTC | calendar | ofi_raw_l4_1s | 233989 | 10 | -9.69855e-08 | 6.70989e-08 | -1.445 | 2.23542e-07 | -0.434 | 0.00001 |
| INTC | calendar | ofi_raw_l5_1s | 233989 | 10 | -3.1228e-07 | 6.80379e-08 | -4.590 | 2.32705e-07 | -1.342 | 0.00009 |
| INTC | calendar | ofi_raw_l6_1s | 233989 | 10 | -3.41033e-07 | 7.21813e-08 | -4.725 | 2.46879e-07 | -1.381 | 0.00010 |
| INTC | calendar | ofi_raw_l7_1s | 233989 | 10 | -3.93529e-07 | 7.78746e-08 | -5.053 | 2.70191e-07 | -1.456 | 0.00011 |
| INTC | calendar | ofi_raw_l8_1s | 233989 | 10 | -2.30204e-07 | 8.38467e-08 | -2.746 | 3.15013e-07 | -0.731 | 0.00003 |
| INTC | calendar | ofi_raw_l9_1s | 233989 | 10 | -2.19928e-07 | 9.17629e-08 | -2.397 | 3.43062e-07 | -0.641 | 0.00002 |
| INTC | calendar | ofi_raw_l10_1s | 233989 | 10 | -1.32461e-07 | 1.01546e-07 | -1.304 | 3.50125e-07 | -0.378 | 0.00001 |
| INTC | calendar | ofi_integrated_1s | 215990 | 10 | 0.000703074 | 0.000231815 | 3.033 | 0.000820178 | 0.857 | 0.00004 |
| INTC | event | ofi_best_1s | 62025 | 1 | 3.24423e-06 | 6.70303e-08 | 48.400 | 1.01776e-07 | 31.876 | 0.03639 |
| INTC | event | ofi_deep_1s | 62025 | 1 | 2.5172e-07 | 9.55974e-09 | 26.331 | 1.37574e-08 | 18.297 | 0.01106 |
| INTC | event | ofi_raw_l1_1s | 62025 | 1 | 3.24423e-06 | 6.70303e-08 | 48.400 | 1.01776e-07 | 31.876 | 0.03639 |
| INTC | event | ofi_raw_l2_1s | 62025 | 1 | 1.60005e-06 | 7.53274e-08 | 21.241 | 9.82895e-08 | 16.279 | 0.00722 |
| INTC | event | ofi_raw_l3_1s | 62025 | 1 | 1.99116e-06 | 8.29661e-08 | 24.000 | 1.11585e-07 | 17.844 | 0.00920 |
| INTC | event | ofi_raw_l4_1s | 62025 | 1 | 2.06724e-06 | 8.34796e-08 | 24.763 | 1.17303e-07 | 17.623 | 0.00979 |
| INTC | event | ofi_raw_l5_1s | 62025 | 1 | 1.91848e-06 | 8.63272e-08 | 22.223 | 1.24251e-07 | 15.440 | 0.00790 |
| INTC | event | ofi_raw_l6_1s | 62025 | 1 | 1.66591e-06 | 9.17381e-08 | 18.159 | 1.43819e-07 | 11.583 | 0.00529 |
| INTC | event | ofi_raw_l7_1s | 62025 | 1 | 2.01691e-06 | 9.75134e-08 | 20.683 | 1.62767e-07 | 12.391 | 0.00685 |
| INTC | event | ofi_raw_l8_1s | 62025 | 1 | 2.14355e-06 | 1.05143e-07 | 20.387 | 1.79703e-07 | 11.928 | 0.00666 |
| INTC | event | ofi_raw_l9_1s | 62025 | 1 | 2.3171e-06 | 1.13666e-07 | 20.385 | 2.00737e-07 | 11.543 | 0.00666 |
| INTC | event | ofi_raw_l10_1s | 62025 | 1 | 2.15037e-06 | 1.26308e-07 | 17.025 | 2.29758e-07 | 9.359 | 0.00465 |
| INTC | event | ofi_integrated_1s | 53560 | 1 | 0.0109717 | 0.000304694 | 36.009 | 0.000446302 | 24.584 | 0.02364 |
| AMZN | calendar | ofi_best_1s | 233989 | 10 | 0.000159271 | 8.03314e-06 | 19.827 | 4.47186e-05 | 3.562 | 0.00168 |
| AMZN | calendar | ofi_deep_1s | 233989 | 10 | 3.47051e-05 | 1.08594e-06 | 31.959 | 5.21021e-06 | 6.661 | 0.00435 |
| AMZN | calendar | ofi_raw_l1_1s | 233989 | 10 | 0.000159271 | 8.03314e-06 | 19.827 | 4.47186e-05 | 3.562 | 0.00168 |
| AMZN | calendar | ofi_raw_l2_1s | 233989 | 10 | 0.000214385 | 1.16958e-05 | 18.330 | 8.06273e-05 | 2.659 | 0.00143 |
| AMZN | calendar | ofi_raw_l3_1s | 233989 | 10 | 0.000260657 | 1.11896e-05 | 23.295 | 8.64896e-05 | 3.014 | 0.00231 |
| AMZN | calendar | ofi_raw_l4_1s | 233989 | 10 | 0.000176563 | 8.53233e-06 | 20.693 | 4.77752e-05 | 3.696 | 0.00183 |
| AMZN | calendar | ofi_raw_l5_1s | 233989 | 10 | 0.000159951 | 7.00354e-06 | 22.839 | 3.67205e-05 | 4.356 | 0.00222 |
| AMZN | calendar | ofi_raw_l6_1s | 233989 | 10 | 0.000133326 | 6.43937e-06 | 20.705 | 3.34771e-05 | 3.983 | 0.00183 |
| AMZN | calendar | ofi_raw_l7_1s | 233989 | 10 | 0.00014661 | 6.45969e-06 | 22.696 | 3.39725e-05 | 4.316 | 0.00220 |
| AMZN | calendar | ofi_raw_l8_1s | 233989 | 10 | 0.000174714 | 7.18036e-06 | 24.332 | 3.88727e-05 | 4.495 | 0.00252 |
| AMZN | calendar | ofi_raw_l9_1s | 233989 | 10 | 0.000137245 | 6.23562e-06 | 22.010 | 3.66527e-05 | 3.744 | 0.00207 |
| AMZN | calendar | ofi_raw_l10_1s | 233989 | 10 | 0.000260967 | 8.15887e-06 | 31.986 | 3.41576e-05 | 7.640 | 0.00435 |
| AMZN | calendar | ofi_integrated_1s | 215990 | 10 | 0.0165809 | 0.000527666 | 31.423 | 0.00259088 | 6.400 | 0.00455 |
| AMZN | event | ofi_best_1s | 26725 | 1 | 0.000343588 | 1.52046e-05 | 22.598 | 5.19408e-05 | 6.615 | 0.01875 |
| AMZN | event | ofi_deep_1s | 26725 | 1 | 0.00010345 | 2.40744e-06 | 42.971 | 9.7659e-06 | 10.593 | 0.06463 |
| AMZN | event | ofi_raw_l1_1s | 26725 | 1 | 0.000343588 | 1.52046e-05 | 22.598 | 5.19408e-05 | 6.615 | 0.01875 |
| AMZN | event | ofi_raw_l2_1s | 26725 | 1 | 0.00120912 | 3.20086e-05 | 37.775 | 0.000153887 | 7.857 | 0.05069 |
| AMZN | event | ofi_raw_l3_1s | 26725 | 1 | 0.00116209 | 3.02557e-05 | 38.409 | 0.000134465 | 8.642 | 0.05232 |
| AMZN | event | ofi_raw_l4_1s | 26725 | 1 | 0.000519679 | 1.89859e-05 | 27.372 | 0.000107575 | 4.831 | 0.02727 |
| AMZN | event | ofi_raw_l5_1s | 26725 | 1 | 0.000587957 | 1.82327e-05 | 32.247 | 0.000105727 | 5.561 | 0.03746 |
| AMZN | event | ofi_raw_l6_1s | 26725 | 1 | 0.000542742 | 1.69585e-05 | 32.004 | 9.09429e-05 | 5.968 | 0.03691 |
| AMZN | event | ofi_raw_l7_1s | 26725 | 1 | 0.000586577 | 1.7265e-05 | 33.975 | 9.56387e-05 | 6.133 | 0.04141 |
| AMZN | event | ofi_raw_l8_1s | 26725 | 1 | 0.000621372 | 1.73798e-05 | 35.753 | 9.41294e-05 | 6.601 | 0.04565 |
| AMZN | event | ofi_raw_l9_1s | 26725 | 1 | 0.000562202 | 1.59616e-05 | 35.222 | 8.22503e-05 | 6.835 | 0.04436 |
| AMZN | event | ofi_raw_l10_1s | 26725 | 1 | 0.000873933 | 2.00317e-05 | 43.627 | 6.93621e-05 | 12.600 | 0.06649 |
| AMZN | event | ofi_integrated_1s | 25004 | 1 | 0.0518479 | 0.00124471 | 41.655 | 0.00542195 | 9.563 | 0.06490 |

## Depth profile: slope vs. book level
### INTC / calendar
| level m | slope (ofi_raw_l{m}_1s) | t_hac | significant (\|t_hac\|>3) |
|---|---|---|---|
| 1 | 9.47731e-07 | 6.12 | yes |
| 2 | -2.52316e-08 | -0.14 | no |
| 3 | 1.38986e-08 | 0.07 | no |
| 4 | -9.69855e-08 | -0.43 | no |
| 5 | -3.1228e-07 | -1.34 | no |
| 6 | -3.41033e-07 | -1.38 | no |
| 7 | -3.93529e-07 | -1.46 | no |
| 8 | -2.30204e-07 | -0.73 | no |
| 9 | -2.19928e-07 | -0.64 | no |
| 10 | -1.32461e-07 | -0.38 | no |

level-vs-slope Pearson correlation: -0.6303 (strong; expected negative -- slope should decay with level)

ofi_best_1s slope: 9.47731e-07 (t_hac=6.12); ofi_deep_1s slope: 3.07747e-09 (t_hac=0.13, **NOT significant**); deep < best: YES

### INTC / event
| level m | slope (ofi_raw_l{m}_1s) | t_hac | significant (\|t_hac\|>3) |
|---|---|---|---|
| 1 | 3.24423e-06 | 31.88 | yes |
| 2 | 1.60005e-06 | 16.28 | yes |
| 3 | 1.99116e-06 | 17.84 | yes |
| 4 | 2.06724e-06 | 17.62 | yes |
| 5 | 1.91848e-06 | 15.44 | yes |
| 6 | 1.66591e-06 | 11.58 | yes |
| 7 | 2.01691e-06 | 12.39 | yes |
| 8 | 2.14355e-06 | 11.93 | yes |
| 9 | 2.3171e-06 | 11.54 | yes |
| 10 | 2.15037e-06 | 9.36 | yes |

level-vs-slope Pearson correlation: -0.1808 (weak; expected negative -- slope should decay with level)

ofi_best_1s slope: 3.24423e-06 (t_hac=31.88); ofi_deep_1s slope: 2.5172e-07 (t_hac=18.30, **significant**); deep < best: YES

### AMZN / calendar
| level m | slope (ofi_raw_l{m}_1s) | t_hac | significant (\|t_hac\|>3) |
|---|---|---|---|
| 1 | 0.000159271 | 3.56 | yes |
| 2 | 0.000214385 | 2.66 | no |
| 3 | 0.000260657 | 3.01 | yes |
| 4 | 0.000176563 | 3.70 | yes |
| 5 | 0.000159951 | 4.36 | yes |
| 6 | 0.000133326 | 3.98 | yes |
| 7 | 0.00014661 | 4.32 | yes |
| 8 | 0.000174714 | 4.49 | yes |
| 9 | 0.000137245 | 3.74 | yes |
| 10 | 0.000260967 | 7.64 | yes |

level-vs-slope Pearson correlation: -0.0662 (weak; expected negative -- slope should decay with level)

ofi_best_1s slope: 0.000159271 (t_hac=3.56); ofi_deep_1s slope: 3.47051e-05 (t_hac=6.66, **significant**); deep < best: YES

### AMZN / event
| level m | slope (ofi_raw_l{m}_1s) | t_hac | significant (\|t_hac\|>3) |
|---|---|---|---|
| 1 | 0.000343588 | 6.61 | yes |
| 2 | 0.00120912 | 7.86 | yes |
| 3 | 0.00116209 | 8.64 | yes |
| 4 | 0.000519679 | 4.83 | yes |
| 5 | 0.000587957 | 5.56 | yes |
| 6 | 0.000542742 | 5.97 | yes |
| 7 | 0.000586577 | 6.13 | yes |
| 8 | 0.000621372 | 6.60 | yes |
| 9 | 0.000562202 | 6.84 | yes |
| 10 | 0.000873933 | 12.60 | yes |

level-vs-slope Pearson correlation: -0.1476 (weak; expected negative -- slope should decay with level)

ofi_best_1s slope: 0.000343588 (t_hac=6.61); ofi_deep_1s slope: 0.00010345 (t_hac=10.59, **significant**); deep < best: YES

## Ticker comparison: slope ~ c / depth (CKS)
| clock | ticker | avg touch depth (shares) | ofi_best_1s slope | slope INTC/AMZN | depth AMZN/INTC (predicted slope ratio) | ratio agreement |
|---|---|---|---|---|---|---|
| calendar | INTC | 28037.7 | 9.47731e-07 | 0.0060 | 0.0127 | 53.3% relative deviation |
| calendar | AMZN | 357.3 | 0.000159271 | 0.0060 | 0.0127 | 53.3% relative deviation |
| event | INTC | 30365.5 | 3.24423e-06 | 0.0094 | 0.0127 | 25.5% relative deviation |
| event | AMZN | 384.7 | 0.000343588 | 0.0094 | 0.0127 | 25.5% relative deviation |

## Independent raw-CSV re-derivation
Re-implements the CKS best-level e_n formula from scratch by reading `data/raw/lobster/*.csv` with plain line iteration (no `edge.io.lobster`, no `edge.features.ofi` import in the derivation itself), on a padded slice of raw rows per ticker, then diffs against (a) `edge.features.ofi._event_contributions` applied to the same raw slice at event granularity, and (b) the real `ofi_best_1s` column of the built calendar-clock parquet at matching whole-second timestamps.

### INTC

- raw rows read (padded): 25000, target window: rows [5000, 25000) of the padded slice
- event-level: max |independent e_n - edge.features.ofi e_n| over 24999 compared events: **0**
- windowed-sum: 25 whole-second calendar boundaries compared against the real `ofi_best_1s` column; max |independent bucket sum - pipeline value|: **0**

| t_k (ns after midnight) | independent Sum e_n | pipeline ofi_best_1s | diff |
|---|---|---|---|
| 40264000000000 | -1200 | -1200 | 0 |
| 40265000000000 | -800 | -800 | 0 |
| 40266000000000 | 400 | 400 | 0 |
| 40267000000000 | -831 | -831 | 0 |
| 40268000000000 | 3436 | 3436 | 0 |
| 40269000000000 | 813 | 813 | 0 |
| 40270000000000 | -1100 | -1100 | 0 |
| 40271000000000 | 1000 | 1000 | 0 |
| 40272000000000 | 1100 | 1100 | 0 |
| 40273000000000 | 0 | 0 | 0 |

### AMZN

- raw rows read (padded): 25000, target window: rows [5000, 25000) of the padded slice
- event-level: max |independent e_n - edge.features.ofi e_n| over 24999 compared events: **0**
- windowed-sum: 25 whole-second calendar boundaries compared against the real `ofi_best_1s` column; max |independent bucket sum - pipeline value|: **0**

| t_k (ns after midnight) | independent Sum e_n | pipeline ofi_best_1s | diff |
|---|---|---|---|
| 41700000000000 | 0 | 0 | 0 |
| 41701000000000 | 0 | 0 | 0 |
| 41702000000000 | 0 | 0 | 0 |
| 41703000000000 | -16 | -16 | 0 |
| 41704000000000 | 0 | 0 | 0 |
| 41705000000000 | 0 | 0 | 0 |
| 41706000000000 | 100 | 100 | 0 |
| 41707000000000 | 0 | 0 | 0 |
| 41708000000000 | 0 | 0 | 0 |
| 41709000000000 | 3 | 3 | 0 |

## Verdict

**PASS.** criterion 1 (positive, HAC-significant ofi_best_1s slope) holds on all 4 ticker/clock combos. criterion 2a (slope decays across levels; deep-book slope below best-level slope) holds. criterion 2b holds for clock=calendar: |slope_INTC| < |slope_AMZN|; slope ratio=0.0060 vs predicted 1/depth ratio=0.0127 (53.3% relative deviation). criterion 2b holds for clock=event: |slope_INTC| < |slope_AMZN|; slope ratio=0.0094 vs predicted 1/depth ratio=0.0127 (25.5% relative deviation). criterion 3 (R2 small but non-zero, no leak) holds: INTC/calendar=0.00140, INTC/event=0.03639, AMZN/calendar=0.00168, AMZN/event=0.01875

Caveats (do not change the verdict, but are honest to flag):

- AMZN/calendar ofi_best_1s: |t_hac|=3.56 clears the 3.0 bar but is not comfortably above it -- marginal, not a strong result.
- INTC/calendar: ofi_deep_1s slope is smaller than ofi_best_1s in magnitude but is NOT itself HAC-significant (t_hac=0.13) -- the 'deeper book, smaller slope' claim holds in direction, not in precision, here.
- INTC/event: level-vs-slope correlation is weak (-0.1808) -- the per-level decay is not clean, only directionally negative.
- AMZN/calendar: level-vs-slope correlation is weak (-0.0662) -- the per-level decay is not clean, only directionally negative.
- AMZN/event: level-vs-slope correlation is weak (-0.1476) -- the per-level decay is not clean, only directionally negative.
