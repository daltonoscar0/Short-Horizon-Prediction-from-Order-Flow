# Ladder under three evaluation protocols (Phase 2)

`python3 -m edge.harness.run --seed 0`

**A** shuffled k-fold (what most of the literature does) | **B** walk-forward, no purge, no embargo | **C** purged walk-forward with embargo (correct).

`A - C` is total optimism. `B - C` isolates the label-overlap component, since B is already strictly forward in time. `A - B` is the residual: training on the future outright. Every number is the distribution ACROSS FOLDS -- mean, median and worst -- never a pooled number alone (EVAL_PROTOCOL.md sec. 1).

Folds: 5 per protocol, identical count across protocols so the gap is not partly a training-set-size difference. Walk-forward test blocks are cut on WALL-CLOCK TIME, not stamp index. Embargo E = max(60s, 10h). Ridge lambda = 1.0, GBM 200 rounds, both frozen a priori.

R2 is computed against a ZERO forecast, never the test mean (EVAL_PROTOCOL.md sec. 2 names the test-mean version as the most common subtle leak in this literature). The `r2_test_mean` column is reported alongside purely to show the size of that gap.

## Optimism table (the headline)

Mean out-of-sample R2 vs zero, by protocol, with the two gaps.

| ticker | clock | h | rung | A | B | C | A-C (total) | B-C (overlap) | A-B (future) |
|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | 0.01243 | 0.01047 | 0.01049 | 0.00194 | -0.00002 | 0.00196 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | 0.00515 | 0.00261 | 0.00262 | 0.00252 | -0.00002 | 0.00254 |
| INTC | calendar | 1s | L2 OLS on OFI | 0.00204 | 0.00082 | 0.00090 | 0.00114 | -0.00008 | 0.00122 |
| INTC | calendar | 1s | L3 ridge, full set | 0.03830 | -0.00414 | -0.00450 | 0.04280 | 0.00036 | 0.04244 |
| INTC | calendar | 1s | L4 GBM, full set | 0.19103 | -0.31325 | -0.29876 | 0.48979 | -0.01450 | 0.50429 |
| INTC | calendar | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | 0.01880 | 0.01547 | 0.01545 | 0.00335 | 0.00002 | 0.00334 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | 0.00796 | 0.00372 | 0.00364 | 0.00432 | 0.00007 | 0.00424 |
| INTC | calendar | 2s | L2 OLS on OFI | 0.00377 | 0.00214 | 0.00230 | 0.00147 | -0.00016 | 0.00164 |
| INTC | calendar | 2s | L3 ridge, full set | 0.05512 | 0.00335 | 0.00646 | 0.04866 | -0.00311 | 0.05177 |
| INTC | calendar | 2s | L4 GBM, full set | 0.27262 | -0.42719 | -0.39280 | 0.66542 | -0.03439 | 0.69981 |
| INTC | calendar | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | 0.02909 | 0.02536 | 0.02528 | 0.00382 | 0.00008 | 0.00373 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | 0.01187 | 0.00674 | 0.00657 | 0.00530 | 0.00017 | 0.00513 |
| INTC | calendar | 5s | L2 OLS on OFI | 0.00510 | 0.00137 | 0.00180 | 0.00330 | -0.00044 | 0.00373 |
| INTC | calendar | 5s | L3 ridge, full set | 0.07428 | 0.00361 | 0.01496 | 0.05932 | -0.01135 | 0.07067 |
| INTC | calendar | 5s | L4 GBM, full set | 0.39939 | -0.47445 | -0.47902 | 0.87841 | 0.00457 | 0.87384 |
| INTC | calendar | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | 0.03421 | 0.03158 | 0.03138 | 0.00284 | 0.00020 | 0.00263 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | 0.01686 | 0.01164 | 0.01170 | 0.00517 | -0.00006 | 0.00523 |
| INTC | calendar | 10s | L2 OLS on OFI | 0.00512 | -0.00125 | -0.00076 | 0.00589 | -0.00048 | 0.00637 |
| INTC | calendar | 10s | L3 ridge, full set | 0.08080 | -0.04516 | -0.03035 | 0.11115 | -0.01481 | 0.12597 |
| INTC | calendar | 10s | L4 GBM, full set | 0.51843 | -0.71645 | -0.55226 | 1.07069 | -0.16419 | 1.23488 |
| INTC | event | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | event | 1s | L1 sign(queue imbalance) | 0.08522 | 0.07229 | 0.07189 | 0.01334 | 0.00041 | 0.01293 |
| INTC | event | 1s | L1m sign(microprice - mid) | 0.02543 | 0.01207 | 0.01149 | 0.01394 | 0.00058 | 0.01336 |
| INTC | event | 1s | L2 OLS on OFI | 0.05426 | 0.03360 | 0.03302 | 0.02124 | 0.00058 | 0.02066 |
| INTC | event | 1s | L3 ridge, full set | 0.24586 | 0.10548 | 0.09556 | 0.15031 | 0.00992 | 0.14038 |
| INTC | event | 1s | L4 GBM, full set | 0.61034 | 0.13540 | 0.12394 | 0.48640 | 0.01146 | 0.47494 |
| INTC | event | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | event | 2s | L1 sign(queue imbalance) | 0.07633 | 0.06300 | 0.06252 | 0.01381 | 0.00048 | 0.01333 |
| INTC | event | 2s | L1m sign(microprice - mid) | 0.03292 | 0.01605 | 0.01508 | 0.01784 | 0.00097 | 0.01687 |
| INTC | event | 2s | L2 OLS on OFI | 0.05735 | 0.01003 | 0.00901 | 0.04834 | 0.00102 | 0.04732 |
| INTC | event | 2s | L3 ridge, full set | 0.22465 | -0.00339 | -0.02568 | 0.25033 | 0.02229 | 0.22804 |
| INTC | event | 2s | L4 GBM, full set | 0.58587 | 0.07551 | -0.01489 | 0.60076 | 0.09040 | 0.51037 |
| INTC | event | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | event | 5s | L1 sign(queue imbalance) | 0.06262 | 0.05258 | 0.05194 | 0.01068 | 0.00064 | 0.01004 |
| INTC | event | 5s | L1m sign(microprice - mid) | 0.02576 | 0.01416 | 0.01322 | 0.01254 | 0.00094 | 0.01160 |
| INTC | event | 5s | L2 OLS on OFI | 0.04650 | 0.00216 | 0.00135 | 0.04516 | 0.00081 | 0.04435 |
| INTC | event | 5s | L3 ridge, full set | 0.19572 | 0.07397 | 0.07490 | 0.12082 | -0.00093 | 0.12175 |
| INTC | event | 5s | L4 GBM, full set | 0.58944 | -0.04786 | -0.09606 | 0.68549 | 0.04820 | 0.63729 |
| INTC | event | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | event | 10s | L1 sign(queue imbalance) | 0.04133 | 0.03765 | 0.03731 | 0.00401 | 0.00034 | 0.00367 |
| INTC | event | 10s | L1m sign(microprice - mid) | 0.03622 | 0.03295 | 0.03259 | 0.00363 | 0.00037 | 0.00327 |
| INTC | event | 10s | L2 OLS on OFI | 0.02957 | 0.00681 | 0.00669 | 0.02288 | 0.00012 | 0.02276 |
| INTC | event | 10s | L3 ridge, full set | 0.14448 | -0.03655 | -0.06129 | 0.20577 | 0.02474 | 0.18103 |
| INTC | event | 10s | L4 GBM, full set | 0.57606 | -0.22297 | -0.18977 | 0.76583 | -0.03319 | 0.79903 |
| AMZN | calendar | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | 0.00327 | 0.00284 | 0.00278 | 0.00048 | 0.00006 | 0.00043 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | 0.00224 | 0.00185 | 0.00186 | 0.00037 | -0.00001 | 0.00038 |
| AMZN | calendar | 1s | L2 OLS on OFI | 0.00262 | 0.00173 | 0.00162 | 0.00100 | 0.00011 | 0.00089 |
| AMZN | calendar | 1s | L3 ridge, full set | 0.02178 | -0.11120 | -0.11006 | 0.13184 | -0.00114 | 0.13298 |
| AMZN | calendar | 1s | L4 GBM, full set | 0.15848 | -0.24438 | -0.24707 | 0.40555 | 0.00269 | 0.40287 |
| AMZN | calendar | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | 0.00458 | 0.00363 | 0.00358 | 0.00100 | 0.00005 | 0.00095 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | 0.00318 | 0.00215 | 0.00216 | 0.00102 | -0.00001 | 0.00103 |
| AMZN | calendar | 2s | L2 OLS on OFI | 0.00286 | 0.00116 | 0.00123 | 0.00163 | -0.00007 | 0.00169 |
| AMZN | calendar | 2s | L3 ridge, full set | 0.02491 | -0.19931 | -0.18923 | 0.21414 | -0.01008 | 0.22422 |
| AMZN | calendar | 2s | L4 GBM, full set | 0.23202 | -0.29952 | -0.24307 | 0.47509 | -0.05644 | 0.53153 |
| AMZN | calendar | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | 0.00367 | 0.00197 | 0.00194 | 0.00173 | 0.00003 | 0.00170 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | 0.00209 | 0.00044 | 0.00045 | 0.00164 | -0.00000 | 0.00165 |
| AMZN | calendar | 5s | L2 OLS on OFI | 0.00190 | -0.00376 | -0.00311 | 0.00502 | -0.00065 | 0.00567 |
| AMZN | calendar | 5s | L3 ridge, full set | 0.02139 | -0.35888 | -0.34228 | 0.36367 | -0.01660 | 0.38027 |
| AMZN | calendar | 5s | L4 GBM, full set | 0.36496 | -0.26484 | -0.29512 | 0.66008 | 0.03028 | 0.62980 |
| AMZN | calendar | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | 0.00243 | 0.00070 | 0.00036 | 0.00207 | 0.00034 | 0.00173 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | 0.00123 | 0.00057 | 0.00050 | 0.00073 | 0.00008 | 0.00065 |
| AMZN | calendar | 10s | L2 OLS on OFI | 0.00238 | -0.00274 | -0.00165 | 0.00402 | -0.00109 | 0.00512 |
| AMZN | calendar | 10s | L3 ridge, full set | 0.01948 | -0.40838 | -0.39879 | 0.41827 | -0.00959 | 0.42786 |
| AMZN | calendar | 10s | L4 GBM, full set | 0.45048 | -0.40939 | -0.31450 | 0.76498 | -0.09489 | 0.85987 |
| AMZN | event | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | event | 1s | L1 sign(queue imbalance) | 0.01264 | 0.01044 | 0.01031 | 0.00233 | 0.00013 | 0.00220 |
| AMZN | event | 1s | L1m sign(microprice - mid) | 0.00752 | 0.00662 | 0.00663 | 0.00088 | -0.00002 | 0.00090 |
| AMZN | event | 1s | L2 OLS on OFI | 0.02062 | -0.15980 | -0.16339 | 0.18401 | 0.00359 | 0.18042 |
| AMZN | event | 1s | L3 ridge, full set | 0.15742 | -0.24116 | -0.25604 | 0.41346 | 0.01488 | 0.39858 |
| AMZN | event | 1s | L4 GBM, full set | 0.31205 | 0.05551 | 0.03373 | 0.27833 | 0.02179 | 0.25654 |
| AMZN | event | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | event | 2s | L1 sign(queue imbalance) | 0.01212 | 0.00941 | 0.00932 | 0.00281 | 0.00009 | 0.00271 |
| AMZN | event | 2s | L1m sign(microprice - mid) | 0.00782 | 0.00589 | 0.00597 | 0.00184 | -0.00009 | 0.00193 |
| AMZN | event | 2s | L2 OLS on OFI | 0.01904 | -0.11473 | -0.11977 | 0.13880 | 0.00504 | 0.13376 |
| AMZN | event | 2s | L3 ridge, full set | 0.14555 | -0.25556 | -0.25360 | 0.39915 | -0.00195 | 0.40111 |
| AMZN | event | 2s | L4 GBM, full set | 0.31271 | -0.00160 | -0.01083 | 0.32354 | 0.00923 | 0.31432 |
| AMZN | event | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | event | 5s | L1 sign(queue imbalance) | 0.01175 | 0.00886 | 0.00882 | 0.00293 | 0.00004 | 0.00289 |
| AMZN | event | 5s | L1m sign(microprice - mid) | 0.00699 | 0.00429 | 0.00433 | 0.00266 | -0.00005 | 0.00271 |
| AMZN | event | 5s | L2 OLS on OFI | 0.01387 | -0.13230 | -0.13271 | 0.14658 | 0.00041 | 0.14617 |
| AMZN | event | 5s | L3 ridge, full set | 0.11870 | -0.25986 | -0.26160 | 0.38031 | 0.00174 | 0.37856 |
| AMZN | event | 5s | L4 GBM, full set | 0.32297 | -0.00778 | -0.01544 | 0.33841 | 0.00766 | 0.33075 |
| AMZN | event | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | event | 10s | L1 sign(queue imbalance) | 0.00934 | 0.00809 | 0.00816 | 0.00118 | -0.00007 | 0.00124 |
| AMZN | event | 10s | L1m sign(microprice - mid) | 0.00540 | 0.00320 | 0.00332 | 0.00208 | -0.00013 | 0.00220 |
| AMZN | event | 10s | L2 OLS on OFI | 0.01082 | -0.08422 | -0.08338 | 0.09420 | -0.00084 | 0.09504 |
| AMZN | event | 10s | L3 ridge, full set | 0.08445 | -0.51414 | -0.55018 | 0.63463 | 0.03604 | 0.59859 |
| AMZN | event | 10s | L4 GBM, full set | 0.33824 | -0.07215 | -0.07161 | 0.40985 | -0.00053 | 0.41039 |

## Information coefficient, per protocol, with the worst fold

| ticker | clock | h | rung | protocol | IC mean | IC median | IC worst | folds | n_test median |
|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | A | n/a | n/a | n/a | 5 | 35897 |
| INTC | calendar | 1s | L0 zero | B | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 1s | L0 zero | C | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | A | 0.11812 | 0.12007 | 0.11189 | 5 | 35897 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | B | 0.11636 | 0.11122 | 0.08790 | 5 | 35362 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | C | 0.11636 | 0.11122 | 0.08790 | 5 | 35362 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | A | 0.09828 | 0.09973 | 0.08918 | 5 | 35897 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | B | 0.11514 | 0.11213 | 0.09968 | 5 | 35362 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | C | 0.11514 | 0.11213 | 0.09968 | 5 | 35362 |
| INTC | calendar | 1s | L2 OLS on OFI | A | 0.08132 | 0.08235 | 0.07356 | 5 | 35897 |
| INTC | calendar | 1s | L2 OLS on OFI | B | 0.07902 | 0.06866 | 0.04366 | 5 | 35362 |
| INTC | calendar | 1s | L2 OLS on OFI | C | 0.07902 | 0.06866 | 0.04366 | 5 | 35362 |
| INTC | calendar | 1s | L3 ridge, full set | A | 0.16989 | 0.17074 | 0.16388 | 5 | 35897 |
| INTC | calendar | 1s | L3 ridge, full set | B | 0.15273 | 0.15313 | 0.12566 | 5 | 35362 |
| INTC | calendar | 1s | L3 ridge, full set | C | 0.15271 | 0.15270 | 0.12640 | 5 | 35362 |
| INTC | calendar | 1s | L4 GBM, full set | A | 0.25605 | 0.25563 | 0.24928 | 5 | 35897 |
| INTC | calendar | 1s | L4 GBM, full set | B | 0.11667 | 0.12132 | 0.05921 | 5 | 35362 |
| INTC | calendar | 1s | L4 GBM, full set | C | 0.10924 | 0.10991 | 0.04619 | 5 | 35362 |
| INTC | calendar | 2s | L0 zero | A | n/a | n/a | n/a | 5 | 35894 |
| INTC | calendar | 2s | L0 zero | B | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 2s | L0 zero | C | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | A | 0.14650 | 0.14754 | 0.14119 | 5 | 35894 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | B | 0.14500 | 0.14071 | 0.11837 | 5 | 35362 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | C | 0.14500 | 0.14071 | 0.11837 | 5 | 35362 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | A | 0.11977 | 0.12317 | 0.10710 | 5 | 35894 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | B | 0.14022 | 0.13666 | 0.11058 | 5 | 35362 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | C | 0.14022 | 0.13666 | 0.11058 | 5 | 35362 |
| INTC | calendar | 2s | L2 OLS on OFI | A | 0.10018 | 0.10295 | 0.09387 | 5 | 35894 |
| INTC | calendar | 2s | L2 OLS on OFI | B | 0.09620 | 0.08418 | 0.06731 | 5 | 35362 |
| INTC | calendar | 2s | L2 OLS on OFI | C | 0.09620 | 0.08418 | 0.06731 | 5 | 35362 |
| INTC | calendar | 2s | L3 ridge, full set | A | 0.21236 | 0.21357 | 0.20543 | 5 | 35894 |
| INTC | calendar | 2s | L3 ridge, full set | B | 0.19173 | 0.19150 | 0.16704 | 5 | 35362 |
| INTC | calendar | 2s | L3 ridge, full set | C | 0.19200 | 0.19106 | 0.16711 | 5 | 35362 |
| INTC | calendar | 2s | L4 GBM, full set | A | 0.34952 | 0.35332 | 0.33982 | 5 | 35894 |
| INTC | calendar | 2s | L4 GBM, full set | B | 0.13510 | 0.12792 | 0.08421 | 5 | 35362 |
| INTC | calendar | 2s | L4 GBM, full set | C | 0.11784 | 0.11177 | 0.05732 | 5 | 35362 |
| INTC | calendar | 5s | L0 zero | A | n/a | n/a | n/a | 5 | 35884 |
| INTC | calendar | 5s | L0 zero | B | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 5s | L0 zero | C | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | A | 0.18967 | 0.18726 | 0.18507 | 5 | 35884 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | B | 0.18822 | 0.18548 | 0.16704 | 5 | 35362 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | C | 0.18822 | 0.18548 | 0.16704 | 5 | 35362 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | A | 0.12995 | 0.13012 | 0.12239 | 5 | 35884 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | B | 0.15127 | 0.15229 | 0.12508 | 5 | 35362 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | C | 0.15127 | 0.15229 | 0.12508 | 5 | 35362 |
| INTC | calendar | 5s | L2 OLS on OFI | A | 0.10036 | 0.10120 | 0.09726 | 5 | 35884 |
| INTC | calendar | 5s | L2 OLS on OFI | B | 0.09591 | 0.08662 | 0.06954 | 5 | 35362 |
| INTC | calendar | 5s | L2 OLS on OFI | C | 0.09591 | 0.08662 | 0.06954 | 5 | 35362 |
| INTC | calendar | 5s | L3 ridge, full set | A | 0.26397 | 0.26224 | 0.25707 | 5 | 35884 |
| INTC | calendar | 5s | L3 ridge, full set | B | 0.23522 | 0.23887 | 0.20721 | 5 | 35362 |
| INTC | calendar | 5s | L3 ridge, full set | C | 0.23577 | 0.23766 | 0.20717 | 5 | 35362 |
| INTC | calendar | 5s | L4 GBM, full set | A | 0.50008 | 0.50031 | 0.49329 | 5 | 35884 |
| INTC | calendar | 5s | L4 GBM, full set | B | 0.15246 | 0.13739 | 0.09839 | 5 | 35362 |
| INTC | calendar | 5s | L4 GBM, full set | C | 0.13383 | 0.12915 | 0.09009 | 5 | 35362 |
| INTC | calendar | 10s | L0 zero | A | n/a | n/a | n/a | 5 | 35878 |
| INTC | calendar | 10s | L0 zero | B | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 10s | L0 zero | C | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | A | 0.21211 | 0.21182 | 0.20563 | 5 | 35878 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | B | 0.21034 | 0.21831 | 0.16263 | 5 | 35362 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | C | 0.21034 | 0.21831 | 0.16263 | 5 | 35362 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | A | 0.13613 | 0.13958 | 0.12809 | 5 | 35878 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | B | 0.15727 | 0.16521 | 0.10211 | 5 | 35362 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | C | 0.15727 | 0.16521 | 0.10211 | 5 | 35362 |
| INTC | calendar | 10s | L2 OLS on OFI | A | 0.10192 | 0.10175 | 0.10058 | 5 | 35878 |
| INTC | calendar | 10s | L2 OLS on OFI | B | 0.09818 | 0.08891 | 0.08151 | 5 | 35362 |
| INTC | calendar | 10s | L2 OLS on OFI | C | 0.09818 | 0.08891 | 0.08151 | 5 | 35362 |
| INTC | calendar | 10s | L3 ridge, full set | A | 0.28737 | 0.28464 | 0.27597 | 5 | 35878 |
| INTC | calendar | 10s | L3 ridge, full set | B | 0.23697 | 0.23723 | 0.20762 | 5 | 35362 |
| INTC | calendar | 10s | L3 ridge, full set | C | 0.23695 | 0.23578 | 0.20885 | 5 | 35362 |
| INTC | calendar | 10s | L4 GBM, full set | A | 0.61967 | 0.62009 | 0.61147 | 5 | 35878 |
| INTC | calendar | 10s | L4 GBM, full set | B | 0.13556 | 0.12919 | 0.09793 | 5 | 35362 |
| INTC | calendar | 10s | L4 GBM, full set | C | 0.15652 | 0.18531 | 0.07234 | 5 | 35362 |
| INTC | event | 1s | L0 zero | A | n/a | n/a | n/a | 5 | 9418 |
| INTC | event | 1s | L0 zero | B | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 1s | L0 zero | C | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 1s | L1 sign(queue imbalance) | A | 0.31371 | 0.31427 | 0.30633 | 5 | 9418 |
| INTC | event | 1s | L1 sign(queue imbalance) | B | 0.29794 | 0.30619 | 0.22264 | 5 | 9151 |
| INTC | event | 1s | L1 sign(queue imbalance) | C | 0.29794 | 0.30619 | 0.22264 | 5 | 9151 |
| INTC | event | 1s | L1m sign(microprice - mid) | A | 0.15258 | 0.14529 | 0.14396 | 5 | 9418 |
| INTC | event | 1s | L1m sign(microprice - mid) | B | 0.15270 | 0.14578 | 0.11422 | 5 | 9151 |
| INTC | event | 1s | L1m sign(microprice - mid) | C | 0.15270 | 0.14578 | 0.11422 | 5 | 9151 |
| INTC | event | 1s | L2 OLS on OFI | A | 0.34838 | 0.34590 | 0.33632 | 5 | 9418 |
| INTC | event | 1s | L2 OLS on OFI | B | 0.33797 | 0.28736 | 0.23888 | 5 | 9151 |
| INTC | event | 1s | L2 OLS on OFI | C | 0.33797 | 0.28736 | 0.23888 | 5 | 9151 |
| INTC | event | 1s | L3 ridge, full set | A | 0.47900 | 0.47779 | 0.46827 | 5 | 9418 |
| INTC | event | 1s | L3 ridge, full set | B | 0.39756 | 0.40770 | 0.24145 | 5 | 9151 |
| INTC | event | 1s | L3 ridge, full set | C | 0.39444 | 0.40797 | 0.22545 | 5 | 9151 |
| INTC | event | 1s | L4 GBM, full set | A | 0.70086 | 0.69771 | 0.69085 | 5 | 9418 |
| INTC | event | 1s | L4 GBM, full set | B | 0.43521 | 0.42188 | 0.35893 | 5 | 9151 |
| INTC | event | 1s | L4 GBM, full set | C | 0.43517 | 0.42642 | 0.36484 | 5 | 9151 |
| INTC | event | 2s | L0 zero | A | n/a | n/a | n/a | 5 | 9414 |
| INTC | event | 2s | L0 zero | B | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 2s | L0 zero | C | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 2s | L1 sign(queue imbalance) | A | 0.30555 | 0.30664 | 0.29756 | 5 | 9414 |
| INTC | event | 2s | L1 sign(queue imbalance) | B | 0.29452 | 0.27675 | 0.23160 | 5 | 9151 |
| INTC | event | 2s | L1 sign(queue imbalance) | C | 0.29452 | 0.27675 | 0.23160 | 5 | 9151 |
| INTC | event | 2s | L1m sign(microprice - mid) | A | 0.16846 | 0.16526 | 0.16116 | 5 | 9414 |
| INTC | event | 2s | L1m sign(microprice - mid) | B | 0.16659 | 0.15345 | 0.12955 | 5 | 9151 |
| INTC | event | 2s | L1m sign(microprice - mid) | C | 0.16659 | 0.15345 | 0.12955 | 5 | 9151 |
| INTC | event | 2s | L2 OLS on OFI | A | 0.31944 | 0.31783 | 0.31022 | 5 | 9414 |
| INTC | event | 2s | L2 OLS on OFI | B | 0.29808 | 0.26659 | 0.22469 | 5 | 9151 |
| INTC | event | 2s | L2 OLS on OFI | C | 0.29808 | 0.26659 | 0.22469 | 5 | 9151 |
| INTC | event | 2s | L3 ridge, full set | A | 0.45781 | 0.45447 | 0.44585 | 5 | 9414 |
| INTC | event | 2s | L3 ridge, full set | B | 0.37458 | 0.34775 | 0.20284 | 5 | 9151 |
| INTC | event | 2s | L3 ridge, full set | C | 0.37239 | 0.34809 | 0.19174 | 5 | 9151 |
| INTC | event | 2s | L4 GBM, full set | A | 0.71317 | 0.71308 | 0.70658 | 5 | 9414 |
| INTC | event | 2s | L4 GBM, full set | B | 0.40903 | 0.37521 | 0.31441 | 5 | 9151 |
| INTC | event | 2s | L4 GBM, full set | C | 0.40718 | 0.37252 | 0.33030 | 5 | 9151 |
| INTC | event | 5s | L0 zero | A | n/a | n/a | n/a | 5 | 9397 |
| INTC | event | 5s | L0 zero | B | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 5s | L0 zero | C | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 5s | L1 sign(queue imbalance) | A | 0.28258 | 0.28393 | 0.27828 | 5 | 9397 |
| INTC | event | 5s | L1 sign(queue imbalance) | B | 0.28033 | 0.27322 | 0.22218 | 5 | 9151 |
| INTC | event | 5s | L1 sign(queue imbalance) | C | 0.28033 | 0.27322 | 0.22218 | 5 | 9151 |
| INTC | event | 5s | L1m sign(microprice - mid) | A | 0.16781 | 0.16793 | 0.16406 | 5 | 9397 |
| INTC | event | 5s | L1m sign(microprice - mid) | B | 0.17301 | 0.17625 | 0.10589 | 5 | 9151 |
| INTC | event | 5s | L1m sign(microprice - mid) | C | 0.17301 | 0.17625 | 0.10589 | 5 | 9151 |
| INTC | event | 5s | L2 OLS on OFI | A | 0.27255 | 0.27306 | 0.26239 | 5 | 9397 |
| INTC | event | 5s | L2 OLS on OFI | B | 0.26209 | 0.25921 | 0.13249 | 5 | 9151 |
| INTC | event | 5s | L2 OLS on OFI | C | 0.26209 | 0.25921 | 0.13249 | 5 | 9151 |
| INTC | event | 5s | L3 ridge, full set | A | 0.42700 | 0.42727 | 0.41626 | 5 | 9397 |
| INTC | event | 5s | L3 ridge, full set | B | 0.35498 | 0.36418 | 0.23149 | 5 | 9151 |
| INTC | event | 5s | L3 ridge, full set | C | 0.35431 | 0.36147 | 0.23030 | 5 | 9151 |
| INTC | event | 5s | L4 GBM, full set | A | 0.73978 | 0.73539 | 0.73325 | 5 | 9397 |
| INTC | event | 5s | L4 GBM, full set | B | 0.33122 | 0.32454 | 0.25645 | 5 | 9151 |
| INTC | event | 5s | L4 GBM, full set | C | 0.33786 | 0.32050 | 0.27394 | 5 | 9151 |
| INTC | event | 10s | L0 zero | A | n/a | n/a | n/a | 5 | 9365 |
| INTC | event | 10s | L0 zero | B | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 10s | L0 zero | C | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 10s | L1 sign(queue imbalance) | A | 0.24595 | 0.24632 | 0.24201 | 5 | 9365 |
| INTC | event | 10s | L1 sign(queue imbalance) | B | 0.24812 | 0.25484 | 0.10763 | 5 | 9151 |
| INTC | event | 10s | L1 sign(queue imbalance) | C | 0.24812 | 0.25484 | 0.10763 | 5 | 9151 |
| INTC | event | 10s | L1m sign(microprice - mid) | A | 0.16183 | 0.16349 | 0.14946 | 5 | 9365 |
| INTC | event | 10s | L1m sign(microprice - mid) | B | 0.17445 | 0.19160 | 0.09418 | 5 | 9151 |
| INTC | event | 10s | L1m sign(microprice - mid) | C | 0.17445 | 0.19160 | 0.09418 | 5 | 9151 |
| INTC | event | 10s | L2 OLS on OFI | A | 0.19501 | 0.19556 | 0.18020 | 5 | 9365 |
| INTC | event | 10s | L2 OLS on OFI | B | 0.20061 | 0.18890 | -0.01973 | 5 | 9151 |
| INTC | event | 10s | L2 OLS on OFI | C | 0.20061 | 0.18890 | -0.01973 | 5 | 9151 |
| INTC | event | 10s | L3 ridge, full set | A | 0.37286 | 0.37628 | 0.36004 | 5 | 9365 |
| INTC | event | 10s | L3 ridge, full set | B | 0.25088 | 0.29372 | 0.13204 | 5 | 9151 |
| INTC | event | 10s | L3 ridge, full set | C | 0.24714 | 0.29247 | 0.11348 | 5 | 9151 |
| INTC | event | 10s | L4 GBM, full set | A | 0.74198 | 0.73589 | 0.73002 | 5 | 9365 |
| INTC | event | 10s | L4 GBM, full set | B | 0.19462 | 0.18680 | 0.15195 | 5 | 9151 |
| INTC | event | 10s | L4 GBM, full set | C | 0.21135 | 0.21592 | 0.15050 | 5 | 9151 |
| AMZN | calendar | 1s | L0 zero | A | n/a | n/a | n/a | 5 | 31585 |
| AMZN | calendar | 1s | L0 zero | B | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 1s | L0 zero | C | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | A | 0.08354 | 0.08340 | 0.08012 | 5 | 31585 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | B | 0.07966 | 0.07221 | 0.05342 | 5 | 30378 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | C | 0.07966 | 0.07221 | 0.05342 | 5 | 30378 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | A | 0.06757 | 0.06713 | 0.06462 | 5 | 31585 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | B | 0.06656 | 0.06129 | 0.03474 | 5 | 30378 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | C | 0.06656 | 0.06129 | 0.03474 | 5 | 30378 |
| AMZN | calendar | 1s | L2 OLS on OFI | A | 0.06902 | 0.06915 | 0.05943 | 5 | 31585 |
| AMZN | calendar | 1s | L2 OLS on OFI | B | 0.06643 | 0.05687 | 0.03784 | 5 | 30378 |
| AMZN | calendar | 1s | L2 OLS on OFI | C | 0.06643 | 0.05687 | 0.03784 | 5 | 30378 |
| AMZN | calendar | 1s | L3 ridge, full set | A | 0.12826 | 0.12702 | 0.12602 | 5 | 31585 |
| AMZN | calendar | 1s | L3 ridge, full set | B | 0.10491 | 0.09480 | 0.05443 | 5 | 30378 |
| AMZN | calendar | 1s | L3 ridge, full set | C | 0.10537 | 0.09382 | 0.05990 | 5 | 30378 |
| AMZN | calendar | 1s | L4 GBM, full set | A | 0.29170 | 0.29056 | 0.28491 | 5 | 31585 |
| AMZN | calendar | 1s | L4 GBM, full set | B | 0.09052 | 0.09392 | 0.02868 | 5 | 30378 |
| AMZN | calendar | 1s | L4 GBM, full set | C | 0.08373 | 0.08775 | 0.03084 | 5 | 30378 |
| AMZN | calendar | 2s | L0 zero | A | n/a | n/a | n/a | 5 | 31584 |
| AMZN | calendar | 2s | L0 zero | B | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 2s | L0 zero | C | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | A | 0.10389 | 0.10451 | 0.10069 | 5 | 31584 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | B | 0.09912 | 0.09398 | 0.06196 | 5 | 30378 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | C | 0.09912 | 0.09398 | 0.06196 | 5 | 30378 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | A | 0.08455 | 0.08586 | 0.07681 | 5 | 31584 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | B | 0.08316 | 0.07452 | 0.04231 | 5 | 30378 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | C | 0.08316 | 0.07452 | 0.04231 | 5 | 30378 |
| AMZN | calendar | 2s | L2 OLS on OFI | A | 0.08315 | 0.08097 | 0.07967 | 5 | 31584 |
| AMZN | calendar | 2s | L2 OLS on OFI | B | 0.07969 | 0.07227 | 0.06248 | 5 | 30378 |
| AMZN | calendar | 2s | L2 OLS on OFI | C | 0.07969 | 0.07227 | 0.06248 | 5 | 30378 |
| AMZN | calendar | 2s | L3 ridge, full set | A | 0.15472 | 0.15290 | 0.15110 | 5 | 31584 |
| AMZN | calendar | 2s | L3 ridge, full set | B | 0.11698 | 0.10630 | 0.06075 | 5 | 30378 |
| AMZN | calendar | 2s | L3 ridge, full set | C | 0.11764 | 0.10546 | 0.06626 | 5 | 30378 |
| AMZN | calendar | 2s | L4 GBM, full set | A | 0.37309 | 0.37122 | 0.36935 | 5 | 31584 |
| AMZN | calendar | 2s | L4 GBM, full set | B | 0.10334 | 0.10745 | 0.04219 | 5 | 30378 |
| AMZN | calendar | 2s | L4 GBM, full set | C | 0.10157 | 0.09838 | 0.05344 | 5 | 30378 |
| AMZN | calendar | 5s | L0 zero | A | n/a | n/a | n/a | 5 | 31582 |
| AMZN | calendar | 5s | L0 zero | B | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 5s | L0 zero | C | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | A | 0.10303 | 0.10243 | 0.09757 | 5 | 31582 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | B | 0.03674 | 0.08788 | -0.11099 | 5 | 30378 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | C | 0.03674 | 0.08788 | -0.11099 | 5 | 30378 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | A | 0.07996 | 0.08180 | 0.07308 | 5 | 31582 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | B | 0.03259 | 0.06409 | -0.09173 | 5 | 30378 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | C | 0.03259 | 0.06409 | -0.09173 | 5 | 30378 |
| AMZN | calendar | 5s | L2 OLS on OFI | A | 0.08727 | 0.08845 | 0.08134 | 5 | 31582 |
| AMZN | calendar | 5s | L2 OLS on OFI | B | 0.08418 | 0.07543 | 0.06166 | 5 | 30378 |
| AMZN | calendar | 5s | L2 OLS on OFI | C | 0.08418 | 0.07543 | 0.06166 | 5 | 30378 |
| AMZN | calendar | 5s | L3 ridge, full set | A | 0.16044 | 0.15985 | 0.15199 | 5 | 31582 |
| AMZN | calendar | 5s | L3 ridge, full set | B | 0.10155 | 0.11989 | 0.00715 | 5 | 30378 |
| AMZN | calendar | 5s | L3 ridge, full set | C | 0.10103 | 0.11939 | 0.00851 | 5 | 30378 |
| AMZN | calendar | 5s | L4 GBM, full set | A | 0.49893 | 0.50104 | 0.48922 | 5 | 31582 |
| AMZN | calendar | 5s | L4 GBM, full set | B | 0.08939 | 0.09804 | -0.01232 | 5 | 30378 |
| AMZN | calendar | 5s | L4 GBM, full set | C | 0.08726 | 0.09686 | -0.02596 | 5 | 30378 |
| AMZN | calendar | 10s | L0 zero | A | n/a | n/a | n/a | 5 | 31570 |
| AMZN | calendar | 10s | L0 zero | B | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 10s | L0 zero | C | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | A | 0.08354 | 0.08240 | 0.08181 | 5 | 31570 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | B | 0.03326 | 0.08939 | -0.08797 | 5 | 30378 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | C | 0.03326 | 0.08939 | -0.08797 | 5 | 30378 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | A | 0.06249 | 0.06107 | 0.05795 | 5 | 31570 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | B | 0.02163 | 0.06432 | -0.09376 | 5 | 30378 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | C | 0.02163 | 0.06432 | -0.09376 | 5 | 30378 |
| AMZN | calendar | 10s | L2 OLS on OFI | A | 0.08623 | 0.08622 | 0.07622 | 5 | 31570 |
| AMZN | calendar | 10s | L2 OLS on OFI | B | 0.08449 | 0.07333 | 0.06503 | 5 | 30378 |
| AMZN | calendar | 10s | L2 OLS on OFI | C | 0.08449 | 0.07333 | 0.06503 | 5 | 30378 |
| AMZN | calendar | 10s | L3 ridge, full set | A | 0.15212 | 0.15250 | 0.14394 | 5 | 31570 |
| AMZN | calendar | 10s | L3 ridge, full set | B | 0.08279 | 0.09216 | -0.01069 | 5 | 30378 |
| AMZN | calendar | 10s | L3 ridge, full set | C | 0.08225 | 0.09362 | -0.01421 | 5 | 30378 |
| AMZN | calendar | 10s | L4 GBM, full set | A | 0.59054 | 0.59290 | 0.58034 | 5 | 31570 |
| AMZN | calendar | 10s | L4 GBM, full set | B | 0.04187 | 0.05269 | -0.05284 | 5 | 30378 |
| AMZN | calendar | 10s | L4 GBM, full set | C | 0.06937 | 0.08012 | -0.06343 | 5 | 30378 |
| AMZN | event | 1s | L0 zero | A | n/a | n/a | n/a | 5 | 4419 |
| AMZN | event | 1s | L0 zero | B | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 1s | L0 zero | C | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 1s | L1 sign(queue imbalance) | A | 0.15273 | 0.15069 | 0.14533 | 5 | 4419 |
| AMZN | event | 1s | L1 sign(queue imbalance) | B | 0.14802 | 0.14417 | 0.11171 | 5 | 3827 |
| AMZN | event | 1s | L1 sign(queue imbalance) | C | 0.14802 | 0.14417 | 0.11171 | 5 | 3827 |
| AMZN | event | 1s | L1m sign(microprice - mid) | A | 0.12332 | 0.11716 | 0.11299 | 5 | 4419 |
| AMZN | event | 1s | L1m sign(microprice - mid) | B | 0.12157 | 0.11310 | 0.07815 | 5 | 3827 |
| AMZN | event | 1s | L1m sign(microprice - mid) | C | 0.12157 | 0.11310 | 0.07815 | 5 | 3827 |
| AMZN | event | 1s | L2 OLS on OFI | A | 0.28444 | 0.28335 | 0.26945 | 5 | 4419 |
| AMZN | event | 1s | L2 OLS on OFI | B | 0.28334 | 0.28934 | 0.22546 | 5 | 3827 |
| AMZN | event | 1s | L2 OLS on OFI | C | 0.28334 | 0.28934 | 0.22546 | 5 | 3827 |
| AMZN | event | 1s | L3 ridge, full set | A | 0.33965 | 0.33566 | 0.32867 | 5 | 4419 |
| AMZN | event | 1s | L3 ridge, full set | B | 0.30654 | 0.32868 | 0.16077 | 5 | 3827 |
| AMZN | event | 1s | L3 ridge, full set | C | 0.30784 | 0.32804 | 0.16889 | 5 | 3827 |
| AMZN | event | 1s | L4 GBM, full set | A | 0.44894 | 0.45121 | 0.43846 | 5 | 4419 |
| AMZN | event | 1s | L4 GBM, full set | B | 0.32873 | 0.34359 | 0.22866 | 5 | 3827 |
| AMZN | event | 1s | L4 GBM, full set | C | 0.32859 | 0.34094 | 0.23883 | 5 | 3827 |
| AMZN | event | 2s | L0 zero | A | n/a | n/a | n/a | 5 | 4418 |
| AMZN | event | 2s | L0 zero | B | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 2s | L0 zero | C | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 2s | L1 sign(queue imbalance) | A | 0.15730 | 0.15478 | 0.14944 | 5 | 4418 |
| AMZN | event | 2s | L1 sign(queue imbalance) | B | 0.14887 | 0.14488 | 0.11676 | 5 | 3827 |
| AMZN | event | 2s | L1 sign(queue imbalance) | C | 0.14887 | 0.14488 | 0.11676 | 5 | 3827 |
| AMZN | event | 2s | L1m sign(microprice - mid) | A | 0.13242 | 0.13086 | 0.11968 | 5 | 4418 |
| AMZN | event | 2s | L1m sign(microprice - mid) | B | 0.12915 | 0.11849 | 0.09091 | 5 | 3827 |
| AMZN | event | 2s | L1m sign(microprice - mid) | C | 0.12915 | 0.11849 | 0.09091 | 5 | 3827 |
| AMZN | event | 2s | L2 OLS on OFI | A | 0.27524 | 0.27509 | 0.25715 | 5 | 4418 |
| AMZN | event | 2s | L2 OLS on OFI | B | 0.27142 | 0.26413 | 0.21293 | 5 | 3827 |
| AMZN | event | 2s | L2 OLS on OFI | C | 0.27142 | 0.26413 | 0.21293 | 5 | 3827 |
| AMZN | event | 2s | L3 ridge, full set | A | 0.34171 | 0.34068 | 0.32662 | 5 | 4418 |
| AMZN | event | 2s | L3 ridge, full set | B | 0.29418 | 0.30697 | 0.15402 | 5 | 3827 |
| AMZN | event | 2s | L3 ridge, full set | C | 0.29567 | 0.30653 | 0.16410 | 5 | 3827 |
| AMZN | event | 2s | L4 GBM, full set | A | 0.47130 | 0.47727 | 0.45452 | 5 | 4418 |
| AMZN | event | 2s | L4 GBM, full set | B | 0.32620 | 0.34490 | 0.22360 | 5 | 3827 |
| AMZN | event | 2s | L4 GBM, full set | C | 0.32886 | 0.35613 | 0.23174 | 5 | 3827 |
| AMZN | event | 5s | L0 zero | A | n/a | n/a | n/a | 5 | 4417 |
| AMZN | event | 5s | L0 zero | B | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 5s | L0 zero | C | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 5s | L1 sign(queue imbalance) | A | 0.14432 | 0.14335 | 0.12431 | 5 | 4417 |
| AMZN | event | 5s | L1 sign(queue imbalance) | B | 0.13762 | 0.13268 | 0.08315 | 5 | 3827 |
| AMZN | event | 5s | L1 sign(queue imbalance) | C | 0.13762 | 0.13268 | 0.08315 | 5 | 3827 |
| AMZN | event | 5s | L1m sign(microprice - mid) | A | 0.11832 | 0.12721 | 0.08873 | 5 | 4417 |
| AMZN | event | 5s | L1m sign(microprice - mid) | B | 0.11331 | 0.11272 | 0.05936 | 5 | 3827 |
| AMZN | event | 5s | L1m sign(microprice - mid) | C | 0.11331 | 0.11272 | 0.05936 | 5 | 3827 |
| AMZN | event | 5s | L2 OLS on OFI | A | 0.27183 | 0.27283 | 0.25556 | 5 | 4417 |
| AMZN | event | 5s | L2 OLS on OFI | B | 0.26818 | 0.28007 | 0.20927 | 5 | 3827 |
| AMZN | event | 5s | L2 OLS on OFI | C | 0.26818 | 0.28007 | 0.20927 | 5 | 3827 |
| AMZN | event | 5s | L3 ridge, full set | A | 0.32706 | 0.32927 | 0.29858 | 5 | 4417 |
| AMZN | event | 5s | L3 ridge, full set | B | 0.24890 | 0.30945 | 0.03374 | 5 | 3827 |
| AMZN | event | 5s | L3 ridge, full set | C | 0.24803 | 0.30884 | 0.03293 | 5 | 3827 |
| AMZN | event | 5s | L4 GBM, full set | A | 0.50800 | 0.51563 | 0.48183 | 5 | 4417 |
| AMZN | event | 5s | L4 GBM, full set | B | 0.30587 | 0.34324 | 0.12308 | 5 | 3827 |
| AMZN | event | 5s | L4 GBM, full set | C | 0.30057 | 0.33062 | 0.11193 | 5 | 3827 |
| AMZN | event | 10s | L0 zero | A | n/a | n/a | n/a | 5 | 4414 |
| AMZN | event | 10s | L0 zero | B | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 10s | L0 zero | C | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 10s | L1 sign(queue imbalance) | A | 0.12489 | 0.12479 | 0.09976 | 5 | 4414 |
| AMZN | event | 10s | L1 sign(queue imbalance) | B | 0.12363 | 0.12333 | 0.08287 | 5 | 3827 |
| AMZN | event | 10s | L1 sign(queue imbalance) | C | 0.12363 | 0.12333 | 0.08287 | 5 | 3827 |
| AMZN | event | 10s | L1m sign(microprice - mid) | A | 0.09980 | 0.10188 | 0.07827 | 5 | 4414 |
| AMZN | event | 10s | L1m sign(microprice - mid) | B | 0.07760 | 0.11986 | -0.05746 | 5 | 3827 |
| AMZN | event | 10s | L1m sign(microprice - mid) | C | 0.07760 | 0.11986 | -0.05746 | 5 | 3827 |
| AMZN | event | 10s | L2 OLS on OFI | A | 0.24503 | 0.24601 | 0.22618 | 5 | 4414 |
| AMZN | event | 10s | L2 OLS on OFI | B | 0.24040 | 0.23174 | 0.22637 | 5 | 3827 |
| AMZN | event | 10s | L2 OLS on OFI | C | 0.24040 | 0.23174 | 0.22637 | 5 | 3827 |
| AMZN | event | 10s | L3 ridge, full set | A | 0.28448 | 0.28323 | 0.25656 | 5 | 4414 |
| AMZN | event | 10s | L3 ridge, full set | B | 0.18799 | 0.23973 | -0.00390 | 5 | 3827 |
| AMZN | event | 10s | L3 ridge, full set | C | 0.18723 | 0.23994 | -0.00653 | 5 | 3827 |
| AMZN | event | 10s | L4 GBM, full set | A | 0.51913 | 0.51745 | 0.49473 | 5 | 4414 |
| AMZN | event | 10s | L4 GBM, full set | B | 0.23305 | 0.27116 | 0.04603 | 5 | 3827 |
| AMZN | event | 10s | L4 GBM, full set | C | 0.23707 | 0.27943 | 0.03996 | 5 | 3827 |

## Classification against the base rate

Balanced accuracy and MCC, printed next to the majority-class share. At 1s most labels are flat, so plain accuracy is meaningless (FEATURES.md sec. 7).

| ticker | clock | h | rung | protocol | bal. acc | MCC | base rate |
|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | A | 0.3333 | n/a | 0.9698 |
| INTC | calendar | 1s | L0 zero | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L0 zero | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.9698 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.9698 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L2 OLS on OFI | A | 0.3333 | n/a | 0.9698 |
| INTC | calendar | 1s | L2 OLS on OFI | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L2 OLS on OFI | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L3 ridge, full set | A | 0.3333 | -0.0007 | 0.9698 |
| INTC | calendar | 1s | L3 ridge, full set | B | 0.3341 | 0.0134 | 0.9700 |
| INTC | calendar | 1s | L3 ridge, full set | C | 0.3341 | 0.0130 | 0.9700 |
| INTC | calendar | 1s | L4 GBM, full set | A | 0.4178 | 0.2919 | 0.9698 |
| INTC | calendar | 1s | L4 GBM, full set | B | 0.3393 | 0.0252 | 0.9700 |
| INTC | calendar | 1s | L4 GBM, full set | C | 0.3427 | 0.0358 | 0.9700 |
| INTC | calendar | 2s | L0 zero | A | 0.3333 | n/a | 0.9425 |
| INTC | calendar | 2s | L0 zero | B | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L0 zero | C | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.9425 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.9425 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L2 OLS on OFI | A | 0.3333 | n/a | 0.9425 |
| INTC | calendar | 2s | L2 OLS on OFI | B | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L2 OLS on OFI | C | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L3 ridge, full set | A | 0.3341 | 0.0179 | 0.9425 |
| INTC | calendar | 2s | L3 ridge, full set | B | 0.3341 | 0.0066 | 0.9428 |
| INTC | calendar | 2s | L3 ridge, full set | C | 0.3340 | 0.0064 | 0.9428 |
| INTC | calendar | 2s | L4 GBM, full set | A | 0.4726 | 0.3780 | 0.9425 |
| INTC | calendar | 2s | L4 GBM, full set | B | 0.3547 | 0.0457 | 0.9428 |
| INTC | calendar | 2s | L4 GBM, full set | C | 0.3552 | 0.0570 | 0.9428 |
| INTC | calendar | 5s | L0 zero | A | 0.3333 | n/a | 0.8702 |
| INTC | calendar | 5s | L0 zero | B | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L0 zero | C | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.8702 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.8702 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L2 OLS on OFI | A | 0.3337 | 0.0177 | 0.8702 |
| INTC | calendar | 5s | L2 OLS on OFI | B | 0.3335 | 0.0148 | 0.8709 |
| INTC | calendar | 5s | L2 OLS on OFI | C | 0.3335 | 0.0148 | 0.8709 |
| INTC | calendar | 5s | L3 ridge, full set | A | 0.3368 | 0.0440 | 0.8702 |
| INTC | calendar | 5s | L3 ridge, full set | B | 0.3393 | 0.0426 | 0.8709 |
| INTC | calendar | 5s | L3 ridge, full set | C | 0.3389 | 0.0421 | 0.8709 |
| INTC | calendar | 5s | L4 GBM, full set | A | 0.5719 | 0.4756 | 0.8702 |
| INTC | calendar | 5s | L4 GBM, full set | B | 0.3790 | 0.0482 | 0.8709 |
| INTC | calendar | 5s | L4 GBM, full set | C | 0.3729 | 0.0544 | 0.8709 |
| INTC | calendar | 10s | L0 zero | A | 0.3333 | n/a | 0.7743 |
| INTC | calendar | 10s | L0 zero | B | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L0 zero | C | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.7743 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.7743 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L2 OLS on OFI | A | 0.3335 | 0.0141 | 0.7743 |
| INTC | calendar | 10s | L2 OLS on OFI | B | 0.3334 | 0.0082 | 0.7756 |
| INTC | calendar | 10s | L2 OLS on OFI | C | 0.3334 | 0.0082 | 0.7756 |
| INTC | calendar | 10s | L3 ridge, full set | A | 0.3428 | 0.0769 | 0.7743 |
| INTC | calendar | 10s | L3 ridge, full set | B | 0.3547 | 0.0613 | 0.7756 |
| INTC | calendar | 10s | L3 ridge, full set | C | 0.3542 | 0.0694 | 0.7756 |
| INTC | calendar | 10s | L4 GBM, full set | A | 0.6658 | 0.5696 | 0.7743 |
| INTC | calendar | 10s | L4 GBM, full set | B | 0.3673 | 0.0172 | 0.7756 |
| INTC | calendar | 10s | L4 GBM, full set | C | 0.3984 | 0.0663 | 0.7756 |
| INTC | event | 1s | L0 zero | A | 0.3333 | n/a | 0.9044 |
| INTC | event | 1s | L0 zero | B | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L0 zero | C | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.9044 |
| INTC | event | 1s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.9044 |
| INTC | event | 1s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L2 OLS on OFI | A | 0.3329 | 0.0025 | 0.9044 |
| INTC | event | 1s | L2 OLS on OFI | B | 0.3321 | 0.0013 | 0.9086 |
| INTC | event | 1s | L2 OLS on OFI | C | 0.3320 | 0.0010 | 0.9086 |
| INTC | event | 1s | L3 ridge, full set | A | 0.3737 | 0.1004 | 0.9044 |
| INTC | event | 1s | L3 ridge, full set | B | 0.3900 | 0.1033 | 0.9086 |
| INTC | event | 1s | L3 ridge, full set | C | 0.3927 | 0.1064 | 0.9086 |
| INTC | event | 1s | L4 GBM, full set | A | 0.6255 | 0.3464 | 0.9044 |
| INTC | event | 1s | L4 GBM, full set | B | 0.4711 | 0.1902 | 0.9086 |
| INTC | event | 1s | L4 GBM, full set | C | 0.4739 | 0.1911 | 0.9086 |
| INTC | event | 2s | L0 zero | A | 0.3333 | n/a | 0.8705 |
| INTC | event | 2s | L0 zero | B | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L0 zero | C | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.8705 |
| INTC | event | 2s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.8705 |
| INTC | event | 2s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L2 OLS on OFI | A | 0.3339 | 0.0026 | 0.8705 |
| INTC | event | 2s | L2 OLS on OFI | B | 0.3317 | -0.0111 | 0.8762 |
| INTC | event | 2s | L2 OLS on OFI | C | 0.3316 | -0.0113 | 0.8762 |
| INTC | event | 2s | L3 ridge, full set | A | 0.3922 | 0.1480 | 0.8705 |
| INTC | event | 2s | L3 ridge, full set | B | 0.4206 | 0.1619 | 0.8762 |
| INTC | event | 2s | L3 ridge, full set | C | 0.4229 | 0.1634 | 0.8762 |
| INTC | event | 2s | L4 GBM, full set | A | 0.6155 | 0.3716 | 0.8705 |
| INTC | event | 2s | L4 GBM, full set | B | 0.4496 | 0.1842 | 0.8762 |
| INTC | event | 2s | L4 GBM, full set | C | 0.4520 | 0.1709 | 0.8762 |
| INTC | event | 5s | L0 zero | A | 0.3333 | n/a | 0.7898 |
| INTC | event | 5s | L0 zero | B | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L0 zero | C | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.7898 |
| INTC | event | 5s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.7898 |
| INTC | event | 5s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L2 OLS on OFI | A | 0.3399 | 0.0442 | 0.7898 |
| INTC | event | 5s | L2 OLS on OFI | B | 0.3332 | 0.0074 | 0.7979 |
| INTC | event | 5s | L2 OLS on OFI | C | 0.3333 | 0.0076 | 0.7979 |
| INTC | event | 5s | L3 ridge, full set | A | 0.4123 | 0.1788 | 0.7898 |
| INTC | event | 5s | L3 ridge, full set | B | 0.4180 | 0.1608 | 0.7979 |
| INTC | event | 5s | L3 ridge, full set | C | 0.4192 | 0.1608 | 0.7979 |
| INTC | event | 5s | L4 GBM, full set | A | 0.6315 | 0.4054 | 0.7898 |
| INTC | event | 5s | L4 GBM, full set | B | 0.4267 | 0.1301 | 0.7979 |
| INTC | event | 5s | L4 GBM, full set | C | 0.4226 | 0.1171 | 0.7979 |
| INTC | event | 10s | L0 zero | A | 0.3333 | n/a | 0.7094 |
| INTC | event | 10s | L0 zero | B | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L0 zero | C | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.7094 |
| INTC | event | 10s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.7094 |
| INTC | event | 10s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L2 OLS on OFI | A | 0.3347 | 0.0141 | 0.7094 |
| INTC | event | 10s | L2 OLS on OFI | B | 0.3338 | 0.0230 | 0.7222 |
| INTC | event | 10s | L2 OLS on OFI | C | 0.3338 | 0.0232 | 0.7222 |
| INTC | event | 10s | L3 ridge, full set | A | 0.3886 | 0.1463 | 0.7094 |
| INTC | event | 10s | L3 ridge, full set | B | 0.3859 | 0.1064 | 0.7222 |
| INTC | event | 10s | L3 ridge, full set | C | 0.3878 | 0.1091 | 0.7222 |
| INTC | event | 10s | L4 GBM, full set | A | 0.6514 | 0.4423 | 0.7094 |
| INTC | event | 10s | L4 GBM, full set | B | 0.3882 | 0.0750 | 0.7222 |
| INTC | event | 10s | L4 GBM, full set | C | 0.3878 | 0.0750 | 0.7222 |
| AMZN | calendar | 1s | L0 zero | A | 0.3333 | n/a | 0.7777 |
| AMZN | calendar | 1s | L0 zero | B | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L0 zero | C | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.7777 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.7777 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L2 OLS on OFI | A | 0.3345 | 0.0218 | 0.7777 |
| AMZN | calendar | 1s | L2 OLS on OFI | B | 0.3338 | 0.0209 | 0.7795 |
| AMZN | calendar | 1s | L2 OLS on OFI | C | 0.3338 | 0.0207 | 0.7795 |
| AMZN | calendar | 1s | L3 ridge, full set | A | 0.3579 | 0.0947 | 0.7777 |
| AMZN | calendar | 1s | L3 ridge, full set | B | 0.3552 | 0.0675 | 0.7795 |
| AMZN | calendar | 1s | L3 ridge, full set | C | 0.3562 | 0.0695 | 0.7795 |
| AMZN | calendar | 1s | L4 GBM, full set | A | 0.5031 | 0.2005 | 0.7777 |
| AMZN | calendar | 1s | L4 GBM, full set | B | 0.3902 | 0.0547 | 0.7795 |
| AMZN | calendar | 1s | L4 GBM, full set | C | 0.3883 | 0.0540 | 0.7795 |
| AMZN | calendar | 2s | L0 zero | A | 0.3333 | n/a | 0.6497 |
| AMZN | calendar | 2s | L0 zero | B | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L0 zero | C | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.6497 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.6497 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L2 OLS on OFI | A | 0.3359 | 0.0366 | 0.6497 |
| AMZN | calendar | 2s | L2 OLS on OFI | B | 0.3355 | 0.0312 | 0.6523 |
| AMZN | calendar | 2s | L2 OLS on OFI | C | 0.3354 | 0.0355 | 0.6523 |
| AMZN | calendar | 2s | L3 ridge, full set | A | 0.3720 | 0.1069 | 0.6497 |
| AMZN | calendar | 2s | L3 ridge, full set | B | 0.3596 | 0.0658 | 0.6523 |
| AMZN | calendar | 2s | L3 ridge, full set | C | 0.3641 | 0.0715 | 0.6523 |
| AMZN | calendar | 2s | L4 GBM, full set | A | 0.5197 | 0.2131 | 0.6497 |
| AMZN | calendar | 2s | L4 GBM, full set | B | 0.3774 | 0.0448 | 0.6523 |
| AMZN | calendar | 2s | L4 GBM, full set | C | 0.3835 | 0.0567 | 0.6523 |
| AMZN | calendar | 5s | L0 zero | A | 0.3333 | n/a | 0.4303 |
| AMZN | calendar | 5s | L0 zero | B | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L0 zero | C | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.4303 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.4303 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L2 OLS on OFI | A | 0.3363 | 0.0361 | 0.4303 |
| AMZN | calendar | 5s | L2 OLS on OFI | B | 0.3378 | 0.0345 | 0.4341 |
| AMZN | calendar | 5s | L2 OLS on OFI | C | 0.3374 | 0.0332 | 0.4341 |
| AMZN | calendar | 5s | L3 ridge, full set | A | 0.3833 | 0.1046 | 0.4303 |
| AMZN | calendar | 5s | L3 ridge, full set | B | 0.3610 | 0.0514 | 0.4341 |
| AMZN | calendar | 5s | L3 ridge, full set | C | 0.3637 | 0.0558 | 0.4341 |
| AMZN | calendar | 5s | L4 GBM, full set | A | 0.5199 | 0.2564 | 0.4303 |
| AMZN | calendar | 5s | L4 GBM, full set | B | 0.3647 | 0.0454 | 0.4341 |
| AMZN | calendar | 5s | L4 GBM, full set | C | 0.3554 | 0.0351 | 0.4341 |
| AMZN | calendar | 10s | L0 zero | A | 0.3333 | n/a | 0.3938 |
| AMZN | calendar | 10s | L0 zero | B | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L0 zero | C | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.3938 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.3938 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L2 OLS on OFI | A | 0.3374 | 0.0290 | 0.3938 |
| AMZN | calendar | 10s | L2 OLS on OFI | B | 0.3389 | 0.0280 | 0.3922 |
| AMZN | calendar | 10s | L2 OLS on OFI | C | 0.3387 | 0.0276 | 0.3922 |
| AMZN | calendar | 10s | L3 ridge, full set | A | 0.3754 | 0.0765 | 0.3938 |
| AMZN | calendar | 10s | L3 ridge, full set | B | 0.3493 | 0.0269 | 0.3922 |
| AMZN | calendar | 10s | L3 ridge, full set | C | 0.3502 | 0.0287 | 0.3922 |
| AMZN | calendar | 10s | L4 GBM, full set | A | 0.5174 | 0.3098 | 0.3938 |
| AMZN | calendar | 10s | L4 GBM, full set | B | 0.3408 | 0.0097 | 0.3922 |
| AMZN | calendar | 10s | L4 GBM, full set | C | 0.3473 | 0.0257 | 0.3922 |
| AMZN | event | 1s | L0 zero | A | 0.3333 | n/a | 0.5564 |
| AMZN | event | 1s | L0 zero | B | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L0 zero | C | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.5564 |
| AMZN | event | 1s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.5564 |
| AMZN | event | 1s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L2 OLS on OFI | A | 0.3570 | 0.1059 | 0.5564 |
| AMZN | event | 1s | L2 OLS on OFI | B | 0.3629 | 0.0966 | 0.5621 |
| AMZN | event | 1s | L2 OLS on OFI | C | 0.3634 | 0.0969 | 0.5621 |
| AMZN | event | 1s | L3 ridge, full set | A | 0.4966 | 0.2391 | 0.5564 |
| AMZN | event | 1s | L3 ridge, full set | B | 0.4716 | 0.1965 | 0.5621 |
| AMZN | event | 1s | L3 ridge, full set | C | 0.4745 | 0.1994 | 0.5621 |
| AMZN | event | 1s | L4 GBM, full set | A | 0.5401 | 0.2592 | 0.5564 |
| AMZN | event | 1s | L4 GBM, full set | B | 0.4840 | 0.1836 | 0.5621 |
| AMZN | event | 1s | L4 GBM, full set | C | 0.4788 | 0.1750 | 0.5621 |
| AMZN | event | 2s | L0 zero | A | 0.3333 | n/a | 0.4538 |
| AMZN | event | 2s | L0 zero | B | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L0 zero | C | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.4538 |
| AMZN | event | 2s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.4538 |
| AMZN | event | 2s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L2 OLS on OFI | A | 0.3625 | 0.1182 | 0.4538 |
| AMZN | event | 2s | L2 OLS on OFI | B | 0.3676 | 0.1155 | 0.4617 |
| AMZN | event | 2s | L2 OLS on OFI | C | 0.3685 | 0.1160 | 0.4617 |
| AMZN | event | 2s | L3 ridge, full set | A | 0.4813 | 0.2185 | 0.4538 |
| AMZN | event | 2s | L3 ridge, full set | B | 0.4535 | 0.1768 | 0.4617 |
| AMZN | event | 2s | L3 ridge, full set | C | 0.4583 | 0.1831 | 0.4617 |
| AMZN | event | 2s | L4 GBM, full set | A | 0.5174 | 0.2481 | 0.4538 |
| AMZN | event | 2s | L4 GBM, full set | B | 0.4606 | 0.1708 | 0.4617 |
| AMZN | event | 2s | L4 GBM, full set | C | 0.4585 | 0.1681 | 0.4617 |
| AMZN | event | 5s | L0 zero | A | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L0 zero | B | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L0 zero | C | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L1 sign(queue imbalance) | A | 0.3888 | 0.0959 | 0.3687 |
| AMZN | event | 5s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L2 OLS on OFI | A | 0.3630 | 0.1033 | 0.3687 |
| AMZN | event | 5s | L2 OLS on OFI | B | 0.3652 | 0.0937 | 0.3687 |
| AMZN | event | 5s | L2 OLS on OFI | C | 0.3652 | 0.0937 | 0.3687 |
| AMZN | event | 5s | L3 ridge, full set | A | 0.4636 | 0.2004 | 0.3687 |
| AMZN | event | 5s | L3 ridge, full set | B | 0.4281 | 0.1466 | 0.3687 |
| AMZN | event | 5s | L3 ridge, full set | C | 0.4271 | 0.1457 | 0.3687 |
| AMZN | event | 5s | L4 GBM, full set | A | 0.4972 | 0.2659 | 0.3687 |
| AMZN | event | 5s | L4 GBM, full set | B | 0.4346 | 0.1647 | 0.3687 |
| AMZN | event | 5s | L4 GBM, full set | C | 0.4311 | 0.1605 | 0.3687 |
| AMZN | event | 10s | L0 zero | A | 0.3333 | n/a | 0.4304 |
| AMZN | event | 10s | L0 zero | B | 0.3333 | n/a | 0.4243 |
| AMZN | event | 10s | L0 zero | C | 0.3333 | n/a | 0.4243 |
| AMZN | event | 10s | L1 sign(queue imbalance) | A | 0.3786 | 0.0902 | 0.4304 |
| AMZN | event | 10s | L1 sign(queue imbalance) | B | 0.3527 | 0.0944 | 0.4243 |
| AMZN | event | 10s | L1 sign(queue imbalance) | C | 0.3643 | 0.1013 | 0.4243 |
| AMZN | event | 10s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.4304 |
| AMZN | event | 10s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.4243 |
| AMZN | event | 10s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.4243 |
| AMZN | event | 10s | L2 OLS on OFI | A | 0.3630 | 0.0830 | 0.4304 |
| AMZN | event | 10s | L2 OLS on OFI | B | 0.3680 | 0.0855 | 0.4243 |
| AMZN | event | 10s | L2 OLS on OFI | C | 0.3692 | 0.0871 | 0.4243 |
| AMZN | event | 10s | L3 ridge, full set | A | 0.4384 | 0.1675 | 0.4304 |
| AMZN | event | 10s | L3 ridge, full set | B | 0.3913 | 0.0976 | 0.4243 |
| AMZN | event | 10s | L3 ridge, full set | C | 0.3921 | 0.0997 | 0.4243 |
| AMZN | event | 10s | L4 GBM, full set | A | 0.4801 | 0.2785 | 0.4304 |
| AMZN | event | 10s | L4 GBM, full set | B | 0.3934 | 0.1203 | 0.4243 |
| AMZN | event | 10s | L4 GBM, full set | C | 0.3969 | 0.1253 | 0.4243 |

## Per-fold detail under protocol C (the honest ladder)

| ticker | clock | h | rung | per-fold IC | per-fold R2 |
|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | 0.1286, 0.1112, 0.1112, 0.0879, 0.1429 | 0.01461, 0.01076, 0.00984, 0.00078, 0.01647 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | 0.1029, 0.0997, 0.1164, 0.1121, 0.1446 | 0.00706, 0.00149, 0.00027, -0.00198, 0.00627 |
| INTC | calendar | 1s | L2 OLS on OFI | 0.0437, 0.1073, 0.0687, 0.0681, 0.1074 | -0.00502, 0.00065, 0.00343, 0.00186, 0.00359 |
| INTC | calendar | 1s | L3 ridge, full set | 0.1264, 0.1527, 0.1610, 0.1442, 0.1793 | -0.10278, 0.02237, 0.02246, 0.01064, 0.02480 |
| INTC | calendar | 1s | L4 GBM, full set | 0.0462, 0.1253, 0.0969, 0.1099, 0.1679 | -0.61783, -0.23991, -0.40002, -0.22304, -0.01298 |
| INTC | calendar | 2s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | 0.1510, 0.1407, 0.1384, 0.1184, 0.1765 | 0.01775, 0.01718, 0.01377, 0.00322, 0.02531 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | 0.1293, 0.1106, 0.1390, 0.1367, 0.1855 | 0.00992, 0.00180, -0.00049, -0.00302, 0.00999 |
| INTC | calendar | 2s | L2 OLS on OFI | 0.0673, 0.1128, 0.0842, 0.0722, 0.1445 | -0.00357, -0.00024, 0.00469, 0.00350, 0.00711 |
| INTC | calendar | 2s | L3 ridge, full set | 0.1851, 0.1973, 0.1911, 0.1671, 0.2194 | -0.07008, 0.03373, 0.01654, 0.00862, 0.04350 |
| INTC | calendar | 2s | L4 GBM, full set | 0.0573, 0.1689, 0.1058, 0.1118, 0.1453 | -0.52388, -0.31649, -0.77358, -0.23455, -0.11551 |
| INTC | calendar | 5s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | 0.1855, 0.1759, 0.1890, 0.1670, 0.2237 | 0.02827, 0.02421, 0.02579, 0.01121, 0.03691 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | 0.1523, 0.1252, 0.1251, 0.1816, 0.1722 | 0.01719, 0.01001, -0.00119, -0.00444, 0.01126 |
| INTC | calendar | 5s | L2 OLS on OFI | 0.0916, 0.0866, 0.0807, 0.0695, 0.1511 | -0.00510, 0.00068, 0.00498, -0.00011, 0.00857 |
| INTC | calendar | 5s | L3 ridge, full set | 0.2432, 0.2377, 0.2357, 0.2072, 0.2551 | -0.05288, 0.04603, 0.01387, 0.02126, 0.04652 |
| INTC | calendar | 5s | L4 GBM, full set | 0.0901, 0.2079, 0.1292, 0.0919, 0.1501 | -0.85018, -0.33557, -0.61923, -0.45286, -0.13725 |
| INTC | calendar | 10s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | 0.2205, 0.1626, 0.2183, 0.1969, 0.2534 | 0.04611, 0.01478, 0.03453, 0.01890, 0.04257 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | 0.1709, 0.1418, 0.1021, 0.2064, 0.1652 | 0.03266, 0.02314, -0.00445, -0.00546, 0.01261 |
| INTC | calendar | 10s | L2 OLS on OFI | 0.0873, 0.0889, 0.0815, 0.0895, 0.1437 | -0.01316, 0.00449, 0.00611, -0.00643, 0.00517 |
| INTC | calendar | 10s | L3 ridge, full set | 0.2088, 0.2358, 0.2516, 0.2323, 0.2563 | -0.22274, -0.01417, 0.01348, 0.03282, 0.03886 |
| INTC | calendar | 10s | L4 GBM, full set | 0.2013, 0.2162, 0.1853, 0.0723, 0.1075 | -0.64975, -0.60012, -0.60030, -0.46695, -0.44420 |
| INTC | event | 1s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | event | 1s | L1 sign(queue imbalance) | 0.2676, 0.3172, 0.2226, 0.3062, 0.3760 | 0.04044, 0.08738, 0.03826, 0.08099, 0.11236 |
| INTC | event | 1s | L1m sign(microprice - mid) | 0.1142, 0.1458, 0.1858, 0.1245, 0.1932 | -0.03355, 0.04073, 0.04422, -0.01914, 0.02521 |
| INTC | event | 1s | L2 OLS on OFI | 0.2676, 0.2874, 0.2389, 0.4841, 0.4119 | -0.10001, 0.03107, 0.06759, 0.11362, 0.05281 |
| INTC | event | 1s | L3 ridge, full set | 0.2255, 0.4080, 0.3115, 0.4663, 0.5610 | -0.17961, 0.16152, -0.02345, 0.27913, 0.24019 |
| INTC | event | 1s | L4 GBM, full set | 0.4264, 0.4514, 0.3648, 0.3802, 0.5529 | 0.12930, 0.13217, -0.13405, 0.25216, 0.24012 |
| INTC | event | 2s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | event | 2s | L1 sign(queue imbalance) | 0.2574, 0.2768, 0.2316, 0.3191, 0.3878 | 0.02875, 0.05789, 0.04415, 0.07057, 0.11123 |
| INTC | event | 2s | L1m sign(microprice - mid) | 0.1305, 0.1296, 0.1999, 0.1535, 0.2195 | -0.07220, 0.03156, 0.05144, 0.02551, 0.03910 |
| INTC | event | 2s | L2 OLS on OFI | 0.2247, 0.2666, 0.2380, 0.3455, 0.4156 | -0.24215, 0.02861, 0.08300, 0.10139, 0.07420 |
| INTC | event | 2s | L3 ridge, full set | 0.1917, 0.3481, 0.3268, 0.4802, 0.5151 | -0.71271, 0.07472, 0.01175, 0.34329, 0.15458 |
| INTC | event | 2s | L4 GBM, full set | 0.3725, 0.3594, 0.3303, 0.4542, 0.5195 | -0.05678, -0.09934, -0.38329, 0.32539, 0.13958 |
| INTC | event | 5s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | event | 5s | L1 sign(queue imbalance) | 0.2376, 0.2222, 0.2732, 0.3303, 0.3384 | 0.01898, 0.03126, 0.05355, 0.07550, 0.08042 |
| INTC | event | 5s | L1m sign(microprice - mid) | 0.1464, 0.1059, 0.2242, 0.1762, 0.2122 | -0.05779, 0.02747, 0.05536, 0.01861, 0.02245 |
| INTC | event | 5s | L2 OLS on OFI | 0.2090, 0.1325, 0.2592, 0.3427, 0.3671 | -0.19464, -0.03448, 0.06980, 0.09092, 0.07514 |
| INTC | event | 5s | L3 ridge, full set | 0.2303, 0.3084, 0.3615, 0.4262, 0.4452 | -0.17795, 0.07897, 0.04330, 0.29428, 0.13591 |
| INTC | event | 5s | L4 GBM, full set | 0.3592, 0.3132, 0.2739, 0.4224, 0.3205 | 0.04163, -0.30606, -0.38051, 0.20525, -0.04060 |
| INTC | event | 10s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | event | 10s | L1 sign(queue imbalance) | 0.2283, 0.1076, 0.2548, 0.3325, 0.3173 | 0.03591, -0.01394, 0.04732, 0.05596, 0.06133 |
| INTC | event | 10s | L1m sign(microprice - mid) | 0.1926, 0.0942, 0.2033, 0.1905, 0.1916 | 0.02932, 0.05183, 0.03486, 0.02179, 0.02514 |
| INTC | event | 10s | L2 OLS on OFI | 0.1732, -0.0197, 0.1889, 0.3762, 0.2845 | -0.03849, -0.03319, 0.02138, 0.04470, 0.03906 |
| INTC | event | 10s | L3 ridge, full set | 0.1135, 0.1935, 0.2925, 0.2980, 0.3383 | -0.57695, 0.01104, 0.06226, 0.16223, 0.03498 |
| INTC | event | 10s | L4 GBM, full set | 0.1505, 0.1844, 0.2159, 0.2437, 0.2622 | -0.23255, -0.22985, -0.25579, 0.03903, -0.26970 |
| AMZN | calendar | 1s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | 0.0534, 0.0722, 0.0769, 0.0643, 0.1315 | -0.00004, 0.00196, 0.00360, 0.00165, 0.00674 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | 0.0347, 0.0577, 0.0665, 0.0613, 0.1125 | -0.00018, 0.00090, 0.00230, 0.00176, 0.00453 |
| AMZN | calendar | 1s | L2 OLS on OFI | 0.0535, 0.0569, 0.0378, 0.0849, 0.0990 | -0.00303, 0.00145, 0.00142, 0.00286, 0.00540 |
| AMZN | calendar | 1s | L3 ridge, full set | 0.0599, 0.0807, 0.1222, 0.0938, 0.1703 | -0.61686, 0.01459, 0.01081, 0.00449, 0.03669 |
| AMZN | calendar | 1s | L4 GBM, full set | 0.0308, 0.0797, 0.0960, 0.0877, 0.1244 | -0.44976, -0.17126, -0.26822, -0.28347, -0.06263 |
| AMZN | calendar | 2s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | 0.0620, 0.0958, 0.0940, 0.0798, 0.1641 | -0.00048, 0.00191, 0.00548, 0.00163, 0.00938 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | 0.0423, 0.0817, 0.0745, 0.0721, 0.1452 | -0.00059, 0.00052, 0.00315, 0.00122, 0.00650 |
| AMZN | calendar | 2s | L2 OLS on OFI | 0.0625, 0.0723, 0.0630, 0.0772, 0.1235 | -0.00582, 0.00176, 0.00174, 0.00304, 0.00543 |
| AMZN | calendar | 2s | L3 ridge, full set | 0.0663, 0.0729, 0.1460, 0.1055, 0.1976 | -1.01246, 0.01208, 0.01433, 0.00503, 0.03486 |
| AMZN | calendar | 2s | L4 GBM, full set | 0.0534, 0.0984, 0.0931, 0.1120, 0.1510 | -0.32637, -0.19378, -0.31822, -0.30027, -0.07672 |
| AMZN | calendar | 5s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | -0.0473, -0.1110, 0.1063, 0.0879, 0.1478 | -0.00002, -0.00060, 0.00369, 0.00012, 0.00651 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | -0.0276, -0.0917, 0.0865, 0.0641, 0.1317 | 0.00021, -0.00231, 0.00127, -0.00084, 0.00389 |
| AMZN | calendar | 5s | L2 OLS on OFI | 0.0617, 0.0754, 0.0940, 0.0697, 0.1201 | -0.02497, 0.00157, 0.00335, 0.00238, 0.00211 |
| AMZN | calendar | 5s | L3 ridge, full set | 0.0085, 0.0173, 0.1647, 0.1194, 0.1953 | -1.74843, -0.00399, 0.02675, -0.00277, 0.01707 |
| AMZN | calendar | 5s | L4 GBM, full set | -0.0260, 0.0665, 0.0969, 0.1188, 0.1801 | -0.36947, -0.29998, -0.31440, -0.41983, -0.07193 |
| AMZN | calendar | 10s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | -0.0360, -0.0880, 0.1032, 0.0894, 0.0977 | -0.00498, -0.00017, 0.00214, 0.00105, 0.00377 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | -0.0171, -0.0938, 0.0851, 0.0643, 0.0696 | -0.00035, -0.00085, 0.00120, 0.00020, 0.00228 |
| AMZN | calendar | 10s | L2 OLS on OFI | 0.0650, 0.0679, 0.1086, 0.0733, 0.1076 | -0.01848, 0.00228, 0.00369, 0.00278, 0.00150 |
| AMZN | calendar | 10s | L3 ridge, full set | 0.0051, -0.0142, 0.1810, 0.0936, 0.1457 | -1.99638, -0.01720, 0.02167, -0.00394, 0.00187 |
| AMZN | calendar | 10s | L4 GBM, full set | -0.0634, 0.0569, 0.0801, 0.1551, 0.1182 | -0.34410, -0.63704, -0.39032, -0.05152, -0.14954 |
| AMZN | event | 1s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | event | 1s | L1 sign(queue imbalance) | 0.1205, 0.1442, 0.1674, 0.1117, 0.1963 | 0.00513, 0.00667, 0.01418, 0.00551, 0.02007 |
| AMZN | event | 1s | L1m sign(microprice - mid) | 0.0916, 0.1131, 0.1542, 0.0781, 0.1708 | 0.00476, 0.00117, 0.01299, 0.00258, 0.01167 |
| AMZN | event | 1s | L2 OLS on OFI | 0.2255, 0.3373, 0.2627, 0.2893, 0.3018 | -0.91517, 0.01234, 0.02622, 0.03559, 0.02404 |
| AMZN | event | 1s | L3 ridge, full set | 0.1689, 0.3349, 0.3280, 0.3086, 0.3987 | -1.85499, 0.14010, 0.09625, 0.12115, 0.21728 |
| AMZN | event | 1s | L4 GBM, full set | 0.2388, 0.3409, 0.3463, 0.3777, 0.3393 | -0.13188, 0.14319, -0.07361, 0.08285, 0.14808 |
| AMZN | event | 2s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | event | 2s | L1 sign(queue imbalance) | 0.1176, 0.1452, 0.1449, 0.1168, 0.2199 | 0.00334, 0.00510, 0.01238, 0.00382, 0.02195 |
| AMZN | event | 2s | L1m sign(microprice - mid) | 0.0938, 0.1185, 0.1441, 0.0909, 0.1985 | 0.00289, 0.00145, 0.01084, 0.00113, 0.01356 |
| AMZN | event | 2s | L2 OLS on OFI | 0.2129, 0.3130, 0.2590, 0.2641, 0.3081 | -0.69322, 0.01330, 0.02338, 0.03750, 0.02020 |
| AMZN | event | 2s | L3 ridge, full set | 0.1641, 0.3065, 0.3185, 0.2975, 0.3918 | -1.73519, 0.12437, 0.02678, 0.12281, 0.19321 |
| AMZN | event | 2s | L4 GBM, full set | 0.2317, 0.3565, 0.3249, 0.3750, 0.3561 | -0.09895, 0.11364, -0.23306, 0.03981, 0.12440 |
| AMZN | event | 5s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | event | 5s | L1 sign(queue imbalance) | 0.0832, 0.1327, 0.1588, 0.0994, 0.2141 | 0.00260, 0.00506, 0.01441, 0.00140, 0.02063 |
| AMZN | event | 5s | L1m sign(microprice - mid) | 0.0628, 0.1127, 0.1420, 0.0594, 0.1897 | 0.00102, 0.00167, 0.00903, -0.00225, 0.01221 |
| AMZN | event | 5s | L2 OLS on OFI | 0.2093, 0.2801, 0.2933, 0.2574, 0.3009 | -0.74040, 0.00957, 0.02149, 0.02937, 0.01644 |
| AMZN | event | 5s | L3 ridge, full set | 0.0329, 0.2393, 0.3213, 0.3088, 0.3377 | -1.72691, 0.10849, 0.09849, 0.09852, 0.11340 |
| AMZN | event | 5s | L4 GBM, full set | 0.1119, 0.3306, 0.3537, 0.3790, 0.3276 | -0.21142, 0.03659, 0.03190, -0.01063, 0.07633 |
| AMZN | event | 10s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | event | 10s | L1 sign(queue imbalance) | 0.0829, 0.1233, 0.1413, 0.1043, 0.1664 | 0.00420, 0.00263, 0.01074, 0.00560, 0.01762 |
| AMZN | event | 10s | L1m sign(microprice - mid) | -0.0575, 0.1199, 0.1231, 0.0584, 0.1441 | -0.00141, 0.00089, 0.00681, 0.00022, 0.01010 |
| AMZN | event | 10s | L2 OLS on OFI | 0.2286, 0.2317, 0.2575, 0.2264, 0.2578 | -0.47679, 0.01072, 0.01799, 0.02526, 0.00593 |
| AMZN | event | 10s | L3 ridge, full set | -0.0065, 0.1853, 0.2552, 0.2622, 0.2399 | -2.96714, 0.07538, 0.07001, 0.07016, 0.00068 |
| AMZN | event | 10s | L4 GBM, full set | 0.0400, 0.2794, 0.2933, 0.3418, 0.2309 | -0.20858, -0.08628, 0.01830, 0.04505, -0.12656 |

## Reading notes

- **L1 is the control.** It has no fitted parameters beyond a one-parameter rescale, so its `A-C` gap is a floor on what the harness itself contributes. A large L1 gap would mean the decomposition is measuring the protocols rather than leakage.
- **The 1s label is not one-second alpha.** The phase-1 adversary measured that OFI's entire content decays inside 100 ms (rho 0.458 over the first 100 ms against 0.027 over the remaining 900 ms). Read these as short-horizon numbers.
- **Fold row counts vary by design.** Test blocks are equal in DURATION, not in sample count, and rungs using trade features lose rows where a window holds no trades. Both are reported rather than absorbed.
