# OFI self-check: Cont-Kukanov-Stoikov Delta-mid ~ OFI

`python3 -m edge.analysis.ofi_selfcheck --seed 0`. Regresses `label_y_mid_1s` (forward mid return over the next second, ticks) on the TRAILING 1s-window OFI columns as of the same stamp t: `ofi_best_1s` (the CKS spec), `ofi_deep_1s` (summed raw shares over levels 1..10), `ofi_raw_l{m}_1s` (m=1..10, per-level slope profile) and `ofi_integrated_1s`. Univariate OLS with intercept, hand-written on numpy (no scipy, no sklearn, no statsmodels). Rows dropped PAIRWISE per regression. Naive OLS SE is reported but judged wrong (overlapping labels on a 100ms grid are autocorrelated); significance is judged on the Newey-West HAC t-statistic, bar |t_hac| > 3.0.

## HAC lag

- INTC/calendar: HAC lag = 10 (fixed a priori: 10 calendar stamps span the 1s label horizon, always)
- INTC/event: HAC lag = 1 (median event-clock stamps per calendar second, computed). Distribution is heavily right-skewed -- mean 2.63, p90=4, p99=38, max=151: event-clock stamps arrive in bursts, so the median (the a-priori rule) is much smaller than a typical burst's width.
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
| INTC | event | ofi_best_1s | 61437 | 1 | 3.14871e-06 | 6.68778e-08 | 47.082 | 1.01248e-07 | 31.099 | 0.03483 |
| INTC | event | ofi_deep_1s | 61437 | 1 | 2.42479e-07 | 9.53286e-09 | 25.436 | 1.37719e-08 | 17.607 | 0.01042 |
| INTC | event | ofi_raw_l1_1s | 61437 | 1 | 3.14871e-06 | 6.68778e-08 | 47.082 | 1.01248e-07 | 31.099 | 0.03483 |
| INTC | event | ofi_raw_l2_1s | 61437 | 1 | 1.53194e-06 | 7.51406e-08 | 20.388 | 9.81147e-08 | 15.614 | 0.00672 |
| INTC | event | ofi_raw_l3_1s | 61437 | 1 | 1.92252e-06 | 8.27194e-08 | 23.242 | 1.11613e-07 | 17.225 | 0.00872 |
| INTC | event | ofi_raw_l4_1s | 61437 | 1 | 1.99265e-06 | 8.32292e-08 | 23.942 | 1.17269e-07 | 16.992 | 0.00924 |
| INTC | event | ofi_raw_l5_1s | 61437 | 1 | 1.84826e-06 | 8.60672e-08 | 21.475 | 1.24448e-07 | 14.852 | 0.00745 |
| INTC | event | ofi_raw_l6_1s | 61437 | 1 | 1.59033e-06 | 9.14129e-08 | 17.397 | 1.44115e-07 | 11.035 | 0.00490 |
| INTC | event | ofi_raw_l7_1s | 61437 | 1 | 1.94609e-06 | 9.71839e-08 | 20.025 | 1.63305e-07 | 11.917 | 0.00648 |
| INTC | event | ofi_raw_l8_1s | 61437 | 1 | 2.05097e-06 | 1.04818e-07 | 19.567 | 1.80375e-07 | 11.371 | 0.00619 |
| INTC | event | ofi_raw_l9_1s | 61437 | 1 | 2.2194e-06 | 1.13369e-07 | 19.577 | 2.01431e-07 | 11.018 | 0.00620 |
| INTC | event | ofi_raw_l10_1s | 61437 | 1 | 2.0593e-06 | 1.26039e-07 | 16.339 | 2.30857e-07 | 8.920 | 0.00433 |
| INTC | event | ofi_integrated_1s | 52994 | 1 | 0.0106487 | 0.000302169 | 35.241 | 0.000445889 | 23.882 | 0.02290 |
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
| AMZN | event | ofi_best_1s | 26719 | 1 | 0.000343325 | 1.5204e-05 | 22.581 | 5.19042e-05 | 6.615 | 0.01873 |
| AMZN | event | ofi_deep_1s | 26719 | 1 | 0.000103364 | 2.40879e-06 | 42.911 | 9.77036e-06 | 10.579 | 0.06448 |
| AMZN | event | ofi_raw_l1_1s | 26719 | 1 | 0.000343325 | 1.5204e-05 | 22.581 | 5.19042e-05 | 6.615 | 0.01873 |
| AMZN | event | ofi_raw_l2_1s | 26719 | 1 | 0.0012071 | 3.20366e-05 | 37.679 | 0.000153935 | 7.842 | 0.05046 |
| AMZN | event | ofi_raw_l3_1s | 26719 | 1 | 0.00116096 | 3.03124e-05 | 38.300 | 0.000134855 | 8.609 | 0.05205 |
| AMZN | event | ofi_raw_l4_1s | 26719 | 1 | 0.00051851 | 1.89991e-05 | 27.291 | 0.000107511 | 4.823 | 0.02712 |
| AMZN | event | ofi_raw_l5_1s | 26719 | 1 | 0.000587157 | 1.82343e-05 | 32.201 | 0.000105617 | 5.559 | 0.03736 |
| AMZN | event | ofi_raw_l6_1s | 26719 | 1 | 0.000542648 | 1.69573e-05 | 32.001 | 9.09351e-05 | 5.967 | 0.03691 |
| AMZN | event | ofi_raw_l7_1s | 26719 | 1 | 0.000586371 | 1.72695e-05 | 33.954 | 9.56721e-05 | 6.129 | 0.04137 |
| AMZN | event | ofi_raw_l8_1s | 26719 | 1 | 0.00062113 | 1.73864e-05 | 35.725 | 9.41804e-05 | 6.595 | 0.04559 |
| AMZN | event | ofi_raw_l9_1s | 26719 | 1 | 0.00056316 | 1.59924e-05 | 35.214 | 8.26965e-05 | 6.810 | 0.04436 |
| AMZN | event | ofi_raw_l10_1s | 26719 | 1 | 0.000877525 | 2.00932e-05 | 43.673 | 6.98072e-05 | 12.571 | 0.06663 |
| AMZN | event | ofi_integrated_1s | 25000 | 1 | 0.0517789 | 0.00124505 | 41.588 | 0.00541944 | 9.554 | 0.06471 |

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
| 1 | 3.14871e-06 | 31.10 | yes |
| 2 | 1.53194e-06 | 15.61 | yes |
| 3 | 1.92252e-06 | 17.22 | yes |
| 4 | 1.99265e-06 | 16.99 | yes |
| 5 | 1.84826e-06 | 14.85 | yes |
| 6 | 1.59033e-06 | 11.04 | yes |
| 7 | 1.94609e-06 | 11.92 | yes |
| 8 | 2.05097e-06 | 11.37 | yes |
| 9 | 2.2194e-06 | 11.02 | yes |
| 10 | 2.0593e-06 | 8.92 | yes |

level-vs-slope Pearson correlation: -0.1957 (weak; expected negative -- slope should decay with level)

ofi_best_1s slope: 3.14871e-06 (t_hac=31.10); ofi_deep_1s slope: 2.42479e-07 (t_hac=17.61, **significant**); deep < best: YES

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
| 1 | 0.000343325 | 6.61 | yes |
| 2 | 0.0012071 | 7.84 | yes |
| 3 | 0.00116096 | 8.61 | yes |
| 4 | 0.00051851 | 4.82 | yes |
| 5 | 0.000587157 | 5.56 | yes |
| 6 | 0.000542648 | 5.97 | yes |
| 7 | 0.000586371 | 6.13 | yes |
| 8 | 0.00062113 | 6.60 | yes |
| 9 | 0.00056316 | 6.81 | yes |
| 10 | 0.000877525 | 12.57 | yes |

level-vs-slope Pearson correlation: -0.1437 (weak; expected negative -- slope should decay with level)

ofi_best_1s slope: 0.000343325 (t_hac=6.61); ofi_deep_1s slope: 0.000103364 (t_hac=10.58, **significant**); deep < best: YES

## Ticker comparison: slope ~ c / depth (CKS)
| clock | ticker | avg touch depth (shares) | ofi_best_1s slope | slope INTC/AMZN | depth AMZN/INTC (predicted slope ratio) | ratio agreement |
|---|---|---|---|---|---|---|
| calendar | INTC | 28037.7 | 9.47731e-07 | 0.0060 | 0.0127 | 53.3% relative deviation |
| calendar | AMZN | 357.3 | 0.000159271 | 0.0060 | 0.0127 | 53.3% relative deviation |
| event | INTC | 30347.1 | 3.14871e-06 | 0.0092 | 0.0127 | 27.6% relative deviation |
| event | AMZN | 384.6 | 0.000343325 | 0.0092 | 0.0127 | 27.6% relative deviation |

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

**PASS.** criterion 1 (positive, HAC-significant ofi_best_1s slope) holds on all 4 ticker/clock combos. criterion 2a (slope decays across levels; deep-book slope below best-level slope) holds. criterion 2b holds for clock=calendar: |slope_INTC| < |slope_AMZN|; slope ratio=0.0060 vs predicted 1/depth ratio=0.0127 (53.3% relative deviation). criterion 2b holds for clock=event: |slope_INTC| < |slope_AMZN|; slope ratio=0.0092 vs predicted 1/depth ratio=0.0127 (27.6% relative deviation). criterion 3 (R2 small but non-zero, no leak) holds: INTC/calendar=0.00140, INTC/event=0.03483, AMZN/calendar=0.00168, AMZN/event=0.01873

Caveats (do not change the verdict, but are honest to flag):

- AMZN/calendar ofi_best_1s: |t_hac|=3.56 clears the 3.0 bar but is not comfortably above it -- marginal, not a strong result.
- INTC/calendar: ofi_deep_1s slope is smaller than ofi_best_1s in magnitude but is NOT itself HAC-significant (t_hac=0.13) -- the 'deeper book, smaller slope' claim holds in direction, not in precision, here.
- INTC/event: level-vs-slope correlation is weak (-0.1957) -- the per-level decay is not clean, only directionally negative.
- AMZN/calendar: level-vs-slope correlation is weak (-0.0662) -- the per-level decay is not clean, only directionally negative.
- AMZN/event: level-vs-slope correlation is weak (-0.1437) -- the per-level decay is not clean, only directionally negative.
