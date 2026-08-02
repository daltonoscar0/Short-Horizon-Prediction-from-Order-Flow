# Ladder under three evaluation protocols (Phase 2)

`python3 -m edge.harness.run --seed 0 --tickers INTC,AMZN --clocks calendar,event --horizons 1s,2s,5s,10s --embargo-sweep --in-sample-scale`

**A** shuffled k-fold (what most of the literature does) | **B** walk-forward, no purge, no embargo | **C** purged walk-forward with embargo (correct) | **D** control: C's test blocks, trained on everything outside them, past AND future, no purge.

> **Corrected 2026-08-02 after the phase-2 red team**
> (`reports/adversary/phase2-ladder-2026-08-02.md`). Two changes to how these numbers are produced, and one to what they are called:
> 1. **Calibration is now causal.** The through-origin rescale is fitted on a held-out last 20% of each fold's training rows, not in-sample. The previous table fitted it on the same rows as the model, which for a high-capacity rung returns almost no shrinkage and makes the out-of-sample amplitude far too large. That, not leakage, is what produced the large negative protocol-C R2 in the earlier draft.
> 2. **The artifact floor is measured, not assumed.** A FROZEN predictor -- fitted once on the whole day, scored unchanged under every protocol -- still shows a nonzero A-C gap, because A's fold mean is the sum(y^2)-weighted average of the day while C's is unweighted. Any rung not clearing that floor has measured nothing.
> 3. **`B - C` is not the label-overlap component** and is no longer labelled as one. Under expanding walk-forward, purge+embargo can only remove the last h+E of a multi-hour training set (0.6-1.1% of rows). Label overlap lives in `A - D`.

`A - C` is total optimism. `A - D` is fold geometry plus the temporal ADJACENCY of training rows, which is where label overlap actually sits. `D - C` is training on the future in disjoint blocks. `B - C` is the purge/embargo effect at a walk-forward boundary. Every number is the distribution ACROSS FOLDS -- mean, median and worst -- never a pooled number alone (EVAL_PROTOCOL.md sec. 1).

Folds: 5 per protocol. Equal fold COUNT does not equalise training-set size -- A trains on roughly 1.8x C's median and 8.7x C's first fold -- but a size-matched control reproduces the unmatched result to within 0.01 R2, so size is not the driver. Walk-forward test blocks are cut on WALL-CLOCK TIME, not stamp index. Embargo E = max(60s, 10h). Ridge lambda = 1.0, GBM 200 rounds, both frozen a priori.

R2 is computed against a ZERO forecast, never the test mean (EVAL_PROTOCOL.md sec. 2 names the test-mean version as the most common subtle leak in this literature).

## Optimism table (the headline)

Out-of-sample R2 vs zero. `A-C med` is the gap recomputed from per-fold MEDIANS and `A-C ex0` drops fold 0 -- both because 45 of 80 rows in the earlier draft lost more than half their gap when fold 0 was dropped, and the mean is the number that gets quoted. `floor` is the frozen-predictor artifact gap for that cell.

| ticker | clock | h | rung | A | B | C | D | A-C | A-C med | A-C ex0 | A-D | D-C | B-C | floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00136 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | 0.01240 | 0.00914 | 0.00938 | 0.00407 | 0.00302 | 0.00354 | 0.00411 | 0.00833 | -0.00531 | -0.00023 | 0.00136 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | 0.00504 | 0.00380 | 0.00402 | 0.00393 | 0.00102 | 0.00221 | 0.00173 | 0.00111 | -0.00009 | -0.00023 | 0.00136 |
| INTC | calendar | 1s | L2 OLS on OFI | 0.00201 | 0.00072 | 0.00084 | 0.00191 | 0.00117 | 0.00052 | -0.00016 | 0.00009 | 0.00107 | -0.00012 | 0.00136 |
| INTC | calendar | 1s | L3 ridge, full set | 0.03815 | 0.01955 | 0.02148 | 0.02774 | 0.01667 | 0.01480 | 0.01238 | 0.01042 | 0.00626 | -0.00192 | 0.00136 |
| INTC | calendar | 1s | L4 GBM, full set | 0.18316 | 0.02167 | 0.02134 | 0.00942 | 0.16182 | 0.16887 | 0.15881 | 0.17374 | -0.01192 | 0.00033 | 0.00136 |
| INTC | calendar | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00214 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | 0.01880 | 0.01266 | 0.01277 | 0.00466 | 0.00602 | 0.00720 | 0.00711 | 0.01414 | -0.00811 | -0.00011 | 0.00214 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | 0.00799 | 0.00566 | 0.00551 | 0.00575 | 0.00249 | 0.00352 | 0.00336 | 0.00224 | 0.00025 | 0.00016 | 0.00214 |
| INTC | calendar | 2s | L2 OLS on OFI | 0.00377 | 0.00076 | 0.00098 | 0.00315 | 0.00279 | 0.00078 | 0.00022 | 0.00062 | 0.00217 | -0.00021 | 0.00214 |
| INTC | calendar | 2s | L3 ridge, full set | 0.05514 | 0.03158 | 0.03094 | 0.03755 | 0.02421 | 0.02103 | 0.02125 | 0.01759 | 0.00662 | 0.00064 | 0.00214 |
| INTC | calendar | 2s | L4 GBM, full set | 0.26221 | 0.02628 | 0.02231 | 0.02574 | 0.23990 | 0.24165 | 0.23745 | 0.23647 | 0.00343 | 0.00398 | 0.00214 |
| INTC | calendar | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00256 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | 0.02910 | 0.02167 | 0.02094 | 0.00912 | 0.00816 | 0.00701 | 0.00871 | 0.01998 | -0.01182 | 0.00074 | 0.00256 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | 0.01185 | 0.00608 | 0.00465 | 0.01046 | 0.00720 | 0.00629 | 0.00722 | 0.00140 | 0.00581 | 0.00143 | 0.00256 |
| INTC | calendar | 5s | L2 OLS on OFI | 0.00504 | 0.00105 | 0.00139 | 0.00169 | 0.00365 | 0.00381 | 0.00206 | 0.00335 | 0.00030 | -0.00034 | 0.00256 |
| INTC | calendar | 5s | L3 ridge, full set | 0.07427 | 0.04198 | 0.04175 | 0.05340 | 0.03253 | 0.03666 | 0.03215 | 0.02088 | 0.01165 | 0.00023 | 0.00256 |
| INTC | calendar | 5s | L4 GBM, full set | 0.38623 | 0.02139 | 0.01900 | 0.02179 | 0.36722 | 0.37543 | 0.37018 | 0.36443 | 0.00279 | 0.00238 | 0.00256 |
| INTC | calendar | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00205 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | 0.03422 | 0.02822 | 0.02886 | 0.00971 | 0.00536 | 0.00138 | 0.00799 | 0.02452 | -0.01915 | -0.00064 | 0.00205 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | 0.01697 | 0.00724 | 0.00886 | 0.01538 | 0.00811 | 0.00872 | 0.01215 | 0.00159 | 0.00652 | -0.00162 | 0.00205 |
| INTC | calendar | 10s | L2 OLS on OFI | 0.00516 | -0.00362 | -0.00110 | 0.00205 | 0.00626 | 0.00498 | 0.00455 | 0.00311 | 0.00315 | -0.00252 | 0.00205 |
| INTC | calendar | 10s | L3 ridge, full set | 0.08075 | 0.04054 | 0.04132 | 0.05832 | 0.03942 | 0.04009 | 0.03974 | 0.02243 | 0.01700 | -0.00078 | 0.00205 |
| INTC | calendar | 10s | L4 GBM, full set | 0.50687 | 0.00666 | 0.01508 | 0.01679 | 0.49178 | 0.49081 | 0.49540 | 0.49008 | 0.00170 | -0.00843 | 0.00205 |
| INTC | event | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00824 |
| INTC | event | 1s | L1 sign(queue imbalance) | 0.08527 | 0.07142 | 0.06712 | 0.05595 | 0.01815 | 0.00817 | 0.01244 | 0.02932 | -0.01118 | 0.00430 | 0.00824 |
| INTC | event | 1s | L1m sign(microprice - mid) | 0.02504 | 0.01069 | 0.00483 | 0.02070 | 0.02021 | 0.01022 | 0.00924 | 0.00434 | 0.01587 | 0.00586 | 0.00824 |
| INTC | event | 1s | L2 OLS on OFI | 0.05383 | 0.04157 | 0.03614 | 0.05217 | 0.01769 | 0.00736 | -0.00851 | 0.00167 | 0.01602 | 0.00543 | 0.00824 |
| INTC | event | 1s | L3 ridge, full set | 0.24608 | 0.08747 | 0.07120 | 0.19156 | 0.17488 | 0.11903 | 0.07479 | 0.05452 | 0.12036 | 0.01627 | 0.00824 |
| INTC | event | 1s | L4 GBM, full set | 0.60133 | 0.18853 | 0.19124 | 0.22910 | 0.41009 | 0.43713 | 0.39998 | 0.37223 | 0.03786 | -0.00271 | 0.00824 |
| INTC | event | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00751 |
| INTC | event | 2s | L1 sign(queue imbalance) | 0.07639 | 0.06496 | 0.06140 | 0.03751 | 0.01499 | 0.02392 | 0.00792 | 0.03888 | -0.02389 | 0.00355 | 0.00751 |
| INTC | event | 2s | L1m sign(microprice - mid) | 0.03291 | 0.01650 | 0.00838 | 0.03105 | 0.02453 | 0.00849 | 0.00201 | 0.00186 | 0.02267 | 0.00812 | 0.00751 |
| INTC | event | 2s | L2 OLS on OFI | 0.05722 | 0.00362 | -0.00248 | 0.05301 | 0.05970 | -0.01377 | -0.00568 | 0.00421 | 0.05549 | 0.00609 | 0.00751 |
| INTC | event | 2s | L3 ridge, full set | 0.22456 | -0.01665 | -0.04504 | 0.16667 | 0.26960 | 0.12445 | 0.07662 | 0.05789 | 0.21171 | 0.02839 | 0.00751 |
| INTC | event | 2s | L4 GBM, full set | 0.57661 | 0.15641 | 0.12804 | 0.18529 | 0.44858 | 0.48037 | 0.42158 | 0.39132 | 0.05725 | 0.02838 | 0.00751 |
| INTC | event | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00235 |
| INTC | event | 5s | L1 sign(queue imbalance) | 0.06245 | 0.05837 | 0.05564 | 0.02826 | 0.00681 | 0.01348 | 0.00466 | 0.03419 | -0.02738 | 0.00273 | 0.00235 |
| INTC | event | 5s | L1m sign(microprice - mid) | 0.02575 | 0.01075 | 0.00650 | 0.02694 | 0.01925 | 0.01159 | 0.00470 | -0.00119 | 0.02043 | 0.00424 | 0.00235 |
| INTC | event | 5s | L2 OLS on OFI | 0.04718 | 0.02174 | 0.01633 | 0.03389 | 0.03085 | 0.00934 | -0.00628 | 0.01329 | 0.01756 | 0.00541 | 0.00235 |
| INTC | event | 5s | L3 ridge, full set | 0.19623 | 0.11309 | 0.11026 | 0.14693 | 0.08597 | 0.08472 | 0.04649 | 0.04930 | 0.03667 | 0.00283 | 0.00235 |
| INTC | event | 5s | L4 GBM, full set | 0.58354 | 0.09492 | 0.08382 | 0.12357 | 0.49972 | 0.50606 | 0.48835 | 0.45997 | 0.03975 | 0.01110 | 0.00235 |
| INTC | event | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | -0.00173 |
| INTC | event | 10s | L1 sign(queue imbalance) | 0.04140 | 0.04008 | 0.03708 | 0.01411 | 0.00432 | -0.00391 | 0.00774 | 0.02729 | -0.02297 | 0.00300 | -0.00173 |
| INTC | event | 10s | L1m sign(microprice - mid) | 0.03614 | 0.02870 | 0.02851 | 0.03468 | 0.00763 | 0.00670 | 0.00820 | 0.00146 | 0.00617 | 0.00019 | -0.00173 |
| INTC | event | 10s | L2 OLS on OFI | 0.02897 | 0.01918 | 0.00595 | 0.01814 | 0.02302 | 0.00378 | 0.00502 | 0.01083 | 0.01219 | 0.01323 | -0.00173 |
| INTC | event | 10s | L3 ridge, full set | 0.14468 | 0.06067 | 0.03501 | 0.08739 | 0.10967 | 0.08244 | 0.06036 | 0.05729 | 0.05238 | 0.02566 | -0.00173 |
| INTC | event | 10s | L4 GBM, full set | 0.56782 | 0.03882 | 0.03237 | 0.04543 | 0.53545 | 0.55259 | 0.52622 | 0.52239 | 0.01306 | 0.00645 | -0.00173 |
| AMZN | calendar | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00021 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | 0.00328 | 0.00209 | 0.00111 | -0.00234 | 0.00218 | 0.00186 | 0.00056 | 0.00562 | -0.00344 | 0.00098 | 0.00021 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | 0.00224 | 0.00053 | -0.00008 | -0.00161 | 0.00233 | 0.00053 | 0.00066 | 0.00386 | -0.00153 | 0.00061 | 0.00021 |
| AMZN | calendar | 1s | L2 OLS on OFI | 0.00260 | -0.00244 | -0.00140 | -0.00032 | 0.00399 | 0.00470 | 0.00316 | 0.00292 | 0.00108 | -0.00104 | 0.00021 |
| AMZN | calendar | 1s | L3 ridge, full set | 0.02166 | 0.00817 | 0.00728 | 0.00845 | 0.01438 | 0.01908 | 0.01222 | 0.01321 | 0.00117 | 0.00088 | 0.00021 |
| AMZN | calendar | 1s | L4 GBM, full set | 0.15646 | 0.01220 | 0.01283 | 0.01764 | 0.14363 | 0.14673 | 0.14037 | 0.13882 | 0.00481 | -0.00063 | 0.00021 |
| AMZN | calendar | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00034 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | 0.00460 | 0.00281 | 0.00231 | -0.00650 | 0.00229 | 0.00427 | 0.00122 | 0.01110 | -0.00881 | 0.00050 | 0.00034 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | 0.00321 | 0.00071 | 0.00071 | -0.00609 | 0.00249 | 0.00393 | 0.00205 | 0.00929 | -0.00680 | -0.00000 | 0.00034 |
| AMZN | calendar | 2s | L2 OLS on OFI | 0.00269 | -0.00258 | -0.00223 | 0.00059 | 0.00492 | 0.00536 | 0.00305 | 0.00210 | 0.00282 | -0.00035 | 0.00034 |
| AMZN | calendar | 2s | L3 ridge, full set | 0.02474 | 0.00845 | 0.00793 | 0.01182 | 0.01681 | 0.02264 | 0.01538 | 0.01292 | 0.00389 | 0.00052 | 0.00034 |
| AMZN | calendar | 2s | L4 GBM, full set | 0.22795 | 0.01052 | 0.00984 | 0.01287 | 0.21811 | 0.22423 | 0.21469 | 0.21508 | 0.00303 | 0.00068 | 0.00034 |
| AMZN | calendar | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00027 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | 0.00357 | 0.00030 | -0.00009 | -0.00612 | 0.00366 | 0.00379 | 0.00362 | 0.00969 | -0.00603 | 0.00040 | 0.00027 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | 0.00206 | -0.00395 | -0.00403 | -0.00729 | 0.00609 | 0.00400 | 0.00661 | 0.00934 | -0.00326 | 0.00008 | 0.00027 |
| AMZN | calendar | 5s | L2 OLS on OFI | 0.00139 | -0.00552 | -0.01052 | 0.00076 | 0.01191 | 0.00184 | -0.00063 | 0.00063 | 0.01128 | 0.00500 | 0.00027 |
| AMZN | calendar | 5s | L3 ridge, full set | 0.02141 | 0.00178 | 0.00375 | 0.00887 | 0.01767 | 0.02117 | 0.01358 | 0.01255 | 0.00512 | -0.00197 | 0.00027 |
| AMZN | calendar | 5s | L4 GBM, full set | 0.35629 | 0.01140 | 0.00611 | 0.01713 | 0.35017 | 0.35228 | 0.34390 | 0.33916 | 0.01101 | 0.00529 | 0.00027 |
| AMZN | calendar | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | -0.00001 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | 0.00235 | 0.00115 | -0.00190 | -0.00006 | 0.00425 | 0.00214 | 0.00240 | 0.00241 | 0.00184 | 0.00305 | -0.00001 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | 0.00121 | -0.00241 | -0.00367 | 0.00042 | 0.00488 | 0.00139 | 0.00575 | 0.00079 | 0.00409 | 0.00126 | -0.00001 |
| AMZN | calendar | 10s | L2 OLS on OFI | 0.00223 | -0.01647 | -0.02712 | 0.00105 | 0.02934 | 0.00155 | 0.00154 | 0.00118 | 0.02816 | 0.01064 | -0.00001 |
| AMZN | calendar | 10s | L3 ridge, full set | 0.01952 | -0.00442 | -0.04047 | 0.00373 | 0.06000 | 0.01977 | 0.01434 | 0.01580 | 0.04420 | 0.03605 | -0.00001 |
| AMZN | calendar | 10s | L4 GBM, full set | 0.44615 | 0.00230 | -0.00028 | 0.00839 | 0.44643 | 0.44451 | 0.44153 | 0.43776 | 0.00867 | 0.00259 | -0.00001 |
| AMZN | event | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00203 |
| AMZN | event | 1s | L1 sign(queue imbalance) | 0.01222 | 0.00978 | 0.00890 | 0.00275 | 0.00333 | 0.00673 | 0.00113 | 0.00947 | -0.00614 | 0.00089 | 0.00203 |
| AMZN | event | 1s | L1m sign(microprice - mid) | 0.00729 | 0.00444 | 0.00435 | 0.00221 | 0.00294 | 0.00237 | 0.00274 | 0.00508 | -0.00214 | 0.00010 | 0.00203 |
| AMZN | event | 1s | L2 OLS on OFI | 0.01868 | -0.04609 | -0.04789 | -0.07718 | 0.06656 | 0.01258 | 0.05714 | 0.09586 | -0.02929 | 0.00180 | 0.00203 |
| AMZN | event | 1s | L3 ridge, full set | 0.15593 | 0.02830 | 0.02225 | 0.08997 | 0.13368 | 0.12091 | 0.06635 | 0.06596 | 0.06771 | 0.00605 | 0.00203 |
| AMZN | event | 1s | L4 GBM, full set | 0.30326 | 0.13693 | 0.14245 | 0.13922 | 0.16081 | 0.15672 | 0.14014 | 0.16404 | -0.00322 | -0.00551 | 0.00203 |
| AMZN | event | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00233 |
| AMZN | event | 2s | L1 sign(queue imbalance) | 0.01198 | 0.00813 | 0.00829 | -0.00423 | 0.00369 | 0.00774 | 0.00210 | 0.01621 | -0.01252 | -0.00016 | 0.00233 |
| AMZN | event | 2s | L1m sign(microprice - mid) | 0.00783 | 0.00258 | 0.00327 | -0.00444 | 0.00456 | 0.00381 | 0.00390 | 0.01226 | -0.00771 | -0.00069 | 0.00233 |
| AMZN | event | 2s | L2 OLS on OFI | 0.01856 | -0.05957 | -0.05684 | -0.08532 | 0.07540 | 0.03428 | 0.08460 | 0.10388 | -0.02848 | -0.00273 | 0.00233 |
| AMZN | event | 2s | L3 ridge, full set | 0.14374 | 0.04969 | 0.04654 | 0.08295 | 0.09720 | 0.13004 | 0.07525 | 0.06079 | 0.03640 | 0.00315 | 0.00233 |
| AMZN | event | 2s | L4 GBM, full set | 0.30335 | 0.12758 | 0.12936 | 0.13056 | 0.17399 | 0.15943 | 0.15937 | 0.17278 | 0.00121 | -0.00178 | 0.00233 |
| AMZN | event | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00194 |
| AMZN | event | 5s | L1 sign(queue imbalance) | 0.01076 | 0.00420 | 0.00543 | 0.00033 | 0.00533 | 0.00638 | 0.00295 | 0.01043 | -0.00510 | -0.00123 | 0.00194 |
| AMZN | event | 5s | L1m sign(microprice - mid) | 0.00573 | -0.00311 | -0.00121 | -0.00538 | 0.00694 | 0.00538 | 0.00662 | 0.01111 | -0.00417 | -0.00190 | 0.00194 |
| AMZN | event | 5s | L2 OLS on OFI | 0.01295 | -0.05642 | -0.05820 | -0.06039 | 0.07115 | 0.01044 | 0.03922 | 0.07334 | -0.00219 | 0.00177 | 0.00194 |
| AMZN | event | 5s | L3 ridge, full set | 0.11792 | 0.06052 | 0.06061 | 0.08106 | 0.05732 | 0.05393 | 0.04008 | 0.03686 | 0.02045 | -0.00009 | 0.00194 |
| AMZN | event | 5s | L4 GBM, full set | 0.31183 | 0.11251 | 0.10640 | 0.12471 | 0.20542 | 0.19720 | 0.18303 | 0.18711 | 0.01831 | 0.00610 | 0.00194 |
| AMZN | event | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00000 | 0.00068 |
| AMZN | event | 10s | L1 sign(queue imbalance) | 0.00902 | 0.00574 | 0.00522 | 0.00652 | 0.00380 | 0.00391 | 0.00097 | 0.00249 | 0.00130 | 0.00052 | 0.00068 |
| AMZN | event | 10s | L1m sign(microprice - mid) | 0.00478 | -0.00153 | -0.00078 | 0.00042 | 0.00555 | 0.00429 | 0.00360 | 0.00436 | 0.00119 | -0.00075 | 0.00068 |
| AMZN | event | 10s | L2 OLS on OFI | 0.00664 | -0.09036 | -0.09707 | -0.04227 | 0.10371 | 0.00039 | 0.03922 | 0.04891 | 0.05480 | 0.00671 | 0.00068 |
| AMZN | event | 10s | L3 ridge, full set | 0.08317 | 0.01900 | 0.00602 | 0.03168 | 0.07715 | 0.06524 | 0.04893 | 0.05149 | 0.02566 | 0.01298 | 0.00068 |
| AMZN | event | 10s | L4 GBM, full set | 0.33055 | 0.05682 | 0.04603 | 0.07686 | 0.28452 | 0.29902 | 0.27892 | 0.25370 | 0.03082 | 0.01078 | 0.00068 |

## The artifact floor (frozen predictor, never refitted)

One `sign(queue imbalance)` model fitted once on every usable row of the day, then scored UNCHANGED under each protocol. Nothing is refitted between protocols, so every difference below is a pure fold-partition and aggregation artifact and none of it is leakage. This is the number the fitted rungs' gaps must be read against.

| ticker | clock | h | A R2 | B R2 | C R2 | D R2 | A-C (artifact) | A IC | C IC | A-C IC |
|---|---|---|---|---|---|---|---|---|---|---|
| AMZN | calendar | 10s | 0.00248 | 0.00249 | 0.00249 | 0.00249 | -0.00001 | 0.08354 | 0.08287 | 0.00068 |
| AMZN | calendar | 1s | 0.00327 | 0.00307 | 0.00307 | 0.00307 | 0.00021 | 0.08354 | 0.07966 | 0.00388 |
| AMZN | calendar | 2s | 0.00460 | 0.00426 | 0.00426 | 0.00426 | 0.00034 | 0.10389 | 0.09912 | 0.00477 |
| AMZN | calendar | 5s | 0.00371 | 0.00344 | 0.00344 | 0.00344 | 0.00027 | 0.10303 | 0.10006 | 0.00297 |
| AMZN | event | 10s | 0.00955 | 0.00886 | 0.00886 | 0.00886 | 0.00068 | 0.12489 | 0.12363 | 0.00126 |
| AMZN | event | 1s | 0.01268 | 0.01066 | 0.01066 | 0.01066 | 0.00203 | 0.15273 | 0.14802 | 0.00470 |
| AMZN | event | 2s | 0.01215 | 0.00982 | 0.00982 | 0.00982 | 0.00233 | 0.15730 | 0.14887 | 0.00844 |
| AMZN | event | 5s | 0.01180 | 0.00986 | 0.00986 | 0.00986 | 0.00194 | 0.14432 | 0.13762 | 0.00670 |
| INTC | calendar | 10s | 0.03423 | 0.03218 | 0.03218 | 0.03218 | 0.00205 | 0.21211 | 0.21034 | 0.00177 |
| INTC | calendar | 1s | 0.01246 | 0.01110 | 0.01110 | 0.01110 | 0.00136 | 0.11812 | 0.11636 | 0.00177 |
| INTC | calendar | 2s | 0.01883 | 0.01669 | 0.01669 | 0.01669 | 0.00214 | 0.14650 | 0.14500 | 0.00151 |
| INTC | calendar | 5s | 0.02912 | 0.02656 | 0.02656 | 0.02656 | 0.00256 | 0.18967 | 0.18822 | 0.00145 |
| INTC | event | 10s | 0.04144 | 0.04317 | 0.04317 | 0.04317 | -0.00173 | 0.24595 | 0.24812 | -0.00217 |
| INTC | event | 1s | 0.08528 | 0.07704 | 0.07704 | 0.07704 | 0.00824 | 0.31371 | 0.29794 | 0.01577 |
| INTC | event | 2s | 0.07639 | 0.06888 | 0.06888 | 0.06888 | 0.00751 | 0.30555 | 0.29452 | 0.01102 |
| INTC | event | 5s | 0.06274 | 0.06039 | 0.06039 | 0.06039 | 0.00235 | 0.28258 | 0.28033 | 0.00225 |

## Superseded: the in-sample calibration

What the earlier draft reported. The rescale is fitted on the same rows as the model, so a high-capacity rung's out-of-sample amplitude is far too large and its protocol-C R2 goes sharply negative. Kept so the size of the correction is visible rather than asserted.

| ticker | clock | h | rung | A (in-sample) | C (in-sample) | A-C (in-sample) | A-C (causal) |
|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | 0.01243 | 0.01049 | 0.00194 | 0.00302 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | 0.00515 | 0.00262 | 0.00252 | 0.00102 |
| INTC | calendar | 1s | L2 OLS on OFI | 0.00204 | 0.00090 | 0.00114 | 0.00117 |
| INTC | calendar | 1s | L3 ridge, full set | 0.03830 | -0.00450 | 0.04280 | 0.01667 |
| INTC | calendar | 1s | L4 GBM, full set | 0.19103 | -0.29876 | 0.48979 | 0.16182 |
| INTC | calendar | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | 0.01880 | 0.01545 | 0.00335 | 0.00602 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | 0.00796 | 0.00364 | 0.00432 | 0.00249 |
| INTC | calendar | 2s | L2 OLS on OFI | 0.00377 | 0.00230 | 0.00147 | 0.00279 |
| INTC | calendar | 2s | L3 ridge, full set | 0.05512 | 0.00646 | 0.04866 | 0.02421 |
| INTC | calendar | 2s | L4 GBM, full set | 0.27262 | -0.39280 | 0.66542 | 0.23990 |
| INTC | calendar | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | 0.02909 | 0.02528 | 0.00382 | 0.00816 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | 0.01187 | 0.00657 | 0.00530 | 0.00720 |
| INTC | calendar | 5s | L2 OLS on OFI | 0.00510 | 0.00180 | 0.00330 | 0.00365 |
| INTC | calendar | 5s | L3 ridge, full set | 0.07428 | 0.01496 | 0.05932 | 0.03253 |
| INTC | calendar | 5s | L4 GBM, full set | 0.39939 | -0.47902 | 0.87841 | 0.36722 |
| INTC | calendar | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | 0.03421 | 0.03138 | 0.00284 | 0.00536 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | 0.01686 | 0.01170 | 0.00517 | 0.00811 |
| INTC | calendar | 10s | L2 OLS on OFI | 0.00512 | -0.00076 | 0.00589 | 0.00626 |
| INTC | calendar | 10s | L3 ridge, full set | 0.08080 | -0.03035 | 0.11115 | 0.03942 |
| INTC | calendar | 10s | L4 GBM, full set | 0.51843 | -0.55226 | 1.07069 | 0.49178 |
| INTC | event | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | event | 1s | L1 sign(queue imbalance) | 0.08522 | 0.07189 | 0.01334 | 0.01815 |
| INTC | event | 1s | L1m sign(microprice - mid) | 0.02543 | 0.01149 | 0.01394 | 0.02021 |
| INTC | event | 1s | L2 OLS on OFI | 0.05426 | 0.03302 | 0.02124 | 0.01769 |
| INTC | event | 1s | L3 ridge, full set | 0.24586 | 0.09556 | 0.15031 | 0.17488 |
| INTC | event | 1s | L4 GBM, full set | 0.61034 | 0.12394 | 0.48640 | 0.41009 |
| INTC | event | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | event | 2s | L1 sign(queue imbalance) | 0.07633 | 0.06252 | 0.01381 | 0.01499 |
| INTC | event | 2s | L1m sign(microprice - mid) | 0.03292 | 0.01508 | 0.01784 | 0.02453 |
| INTC | event | 2s | L2 OLS on OFI | 0.05735 | 0.00901 | 0.04834 | 0.05970 |
| INTC | event | 2s | L3 ridge, full set | 0.22465 | -0.02568 | 0.25033 | 0.26960 |
| INTC | event | 2s | L4 GBM, full set | 0.58587 | -0.01489 | 0.60076 | 0.44858 |
| INTC | event | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | event | 5s | L1 sign(queue imbalance) | 0.06262 | 0.05194 | 0.01068 | 0.00681 |
| INTC | event | 5s | L1m sign(microprice - mid) | 0.02576 | 0.01322 | 0.01254 | 0.01925 |
| INTC | event | 5s | L2 OLS on OFI | 0.04650 | 0.00135 | 0.04516 | 0.03085 |
| INTC | event | 5s | L3 ridge, full set | 0.19572 | 0.07490 | 0.12082 | 0.08597 |
| INTC | event | 5s | L4 GBM, full set | 0.58944 | -0.09606 | 0.68549 | 0.49972 |
| INTC | event | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| INTC | event | 10s | L1 sign(queue imbalance) | 0.04133 | 0.03731 | 0.00401 | 0.00432 |
| INTC | event | 10s | L1m sign(microprice - mid) | 0.03622 | 0.03259 | 0.00363 | 0.00763 |
| INTC | event | 10s | L2 OLS on OFI | 0.02957 | 0.00669 | 0.02288 | 0.02302 |
| INTC | event | 10s | L3 ridge, full set | 0.14448 | -0.06129 | 0.20577 | 0.10967 |
| INTC | event | 10s | L4 GBM, full set | 0.57606 | -0.18977 | 0.76583 | 0.53545 |
| AMZN | calendar | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | 0.00327 | 0.00278 | 0.00048 | 0.00218 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | 0.00224 | 0.00186 | 0.00037 | 0.00233 |
| AMZN | calendar | 1s | L2 OLS on OFI | 0.00262 | 0.00162 | 0.00100 | 0.00399 |
| AMZN | calendar | 1s | L3 ridge, full set | 0.02178 | -0.11006 | 0.13184 | 0.01438 |
| AMZN | calendar | 1s | L4 GBM, full set | 0.15848 | -0.24707 | 0.40555 | 0.14363 |
| AMZN | calendar | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | 0.00458 | 0.00358 | 0.00100 | 0.00229 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | 0.00318 | 0.00216 | 0.00102 | 0.00249 |
| AMZN | calendar | 2s | L2 OLS on OFI | 0.00286 | 0.00123 | 0.00163 | 0.00492 |
| AMZN | calendar | 2s | L3 ridge, full set | 0.02491 | -0.18923 | 0.21414 | 0.01681 |
| AMZN | calendar | 2s | L4 GBM, full set | 0.23202 | -0.24307 | 0.47509 | 0.21811 |
| AMZN | calendar | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | 0.00367 | 0.00194 | 0.00173 | 0.00366 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | 0.00209 | 0.00045 | 0.00164 | 0.00609 |
| AMZN | calendar | 5s | L2 OLS on OFI | 0.00190 | -0.00311 | 0.00502 | 0.01191 |
| AMZN | calendar | 5s | L3 ridge, full set | 0.02139 | -0.34228 | 0.36367 | 0.01767 |
| AMZN | calendar | 5s | L4 GBM, full set | 0.36496 | -0.29512 | 0.66008 | 0.35017 |
| AMZN | calendar | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | 0.00243 | 0.00036 | 0.00207 | 0.00425 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | 0.00123 | 0.00050 | 0.00073 | 0.00488 |
| AMZN | calendar | 10s | L2 OLS on OFI | 0.00238 | -0.00165 | 0.00402 | 0.02934 |
| AMZN | calendar | 10s | L3 ridge, full set | 0.01948 | -0.39879 | 0.41827 | 0.06000 |
| AMZN | calendar | 10s | L4 GBM, full set | 0.45048 | -0.31450 | 0.76498 | 0.44643 |
| AMZN | event | 1s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | event | 1s | L1 sign(queue imbalance) | 0.01264 | 0.01031 | 0.00233 | 0.00333 |
| AMZN | event | 1s | L1m sign(microprice - mid) | 0.00752 | 0.00663 | 0.00088 | 0.00294 |
| AMZN | event | 1s | L2 OLS on OFI | 0.02062 | -0.16339 | 0.18401 | 0.06656 |
| AMZN | event | 1s | L3 ridge, full set | 0.15742 | -0.25604 | 0.41346 | 0.13368 |
| AMZN | event | 1s | L4 GBM, full set | 0.31205 | 0.03373 | 0.27833 | 0.16081 |
| AMZN | event | 2s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | event | 2s | L1 sign(queue imbalance) | 0.01212 | 0.00932 | 0.00281 | 0.00369 |
| AMZN | event | 2s | L1m sign(microprice - mid) | 0.00782 | 0.00597 | 0.00184 | 0.00456 |
| AMZN | event | 2s | L2 OLS on OFI | 0.01904 | -0.11977 | 0.13880 | 0.07540 |
| AMZN | event | 2s | L3 ridge, full set | 0.14555 | -0.25360 | 0.39915 | 0.09720 |
| AMZN | event | 2s | L4 GBM, full set | 0.31271 | -0.01083 | 0.32354 | 0.17399 |
| AMZN | event | 5s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | event | 5s | L1 sign(queue imbalance) | 0.01175 | 0.00882 | 0.00293 | 0.00533 |
| AMZN | event | 5s | L1m sign(microprice - mid) | 0.00699 | 0.00433 | 0.00266 | 0.00694 |
| AMZN | event | 5s | L2 OLS on OFI | 0.01387 | -0.13271 | 0.14658 | 0.07115 |
| AMZN | event | 5s | L3 ridge, full set | 0.11870 | -0.26160 | 0.38031 | 0.05732 |
| AMZN | event | 5s | L4 GBM, full set | 0.32297 | -0.01544 | 0.33841 | 0.20542 |
| AMZN | event | 10s | L0 zero | 0.00000 | 0.00000 | 0.00000 | 0.00000 |
| AMZN | event | 10s | L1 sign(queue imbalance) | 0.00934 | 0.00816 | 0.00118 | 0.00380 |
| AMZN | event | 10s | L1m sign(microprice - mid) | 0.00540 | 0.00332 | 0.00208 | 0.00555 |
| AMZN | event | 10s | L2 OLS on OFI | 0.01082 | -0.08338 | 0.09420 | 0.10371 |
| AMZN | event | 10s | L3 ridge, full set | 0.08445 | -0.55018 | 0.63463 | 0.07715 |
| AMZN | event | 10s | L4 GBM, full set | 0.33824 | -0.07161 | 0.40985 | 0.28452 |

## Information coefficient, per protocol, with the worst fold

| ticker | clock | h | rung | protocol | IC mean | IC median | IC worst | folds | n_test median |
|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | A | n/a | n/a | n/a | 5 | 35897 |
| INTC | calendar | 1s | L0 zero | B | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 1s | L0 zero | C | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 1s | L0 zero | D | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | A | 0.11812 | 0.12007 | 0.11189 | 5 | 35897 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | B | 0.11636 | 0.11122 | 0.08790 | 5 | 35362 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | C | 0.11636 | 0.11122 | 0.08790 | 5 | 35362 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | D | 0.11636 | 0.11122 | 0.08790 | 5 | 35362 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | A | 0.09828 | 0.09973 | 0.08918 | 5 | 35897 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | B | 0.11514 | 0.11213 | 0.09968 | 5 | 35362 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | C | 0.11514 | 0.11213 | 0.09968 | 5 | 35362 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | D | 0.11514 | 0.11213 | 0.09968 | 5 | 35362 |
| INTC | calendar | 1s | L2 OLS on OFI | A | 0.08132 | 0.08235 | 0.07356 | 5 | 35897 |
| INTC | calendar | 1s | L2 OLS on OFI | B | 0.07902 | 0.06866 | 0.04366 | 5 | 35362 |
| INTC | calendar | 1s | L2 OLS on OFI | C | 0.07902 | 0.06866 | 0.04366 | 5 | 35362 |
| INTC | calendar | 1s | L2 OLS on OFI | D | 0.07902 | 0.06866 | 0.04366 | 5 | 35362 |
| INTC | calendar | 1s | L3 ridge, full set | A | 0.16948 | 0.17038 | 0.16338 | 5 | 35897 |
| INTC | calendar | 1s | L3 ridge, full set | B | 0.14953 | 0.14900 | 0.12420 | 5 | 35362 |
| INTC | calendar | 1s | L3 ridge, full set | C | 0.14959 | 0.14881 | 0.12463 | 5 | 35362 |
| INTC | calendar | 1s | L3 ridge, full set | D | 0.16140 | 0.16082 | 0.14502 | 5 | 35362 |
| INTC | calendar | 1s | L4 GBM, full set | A | 0.25312 | 0.25465 | 0.24263 | 5 | 35897 |
| INTC | calendar | 1s | L4 GBM, full set | B | 0.12928 | 0.12531 | 0.11199 | 5 | 35362 |
| INTC | calendar | 1s | L4 GBM, full set | C | 0.13160 | 0.12805 | 0.10996 | 5 | 35362 |
| INTC | calendar | 1s | L4 GBM, full set | D | 0.12867 | 0.12163 | 0.10672 | 5 | 35362 |
| INTC | calendar | 2s | L0 zero | A | n/a | n/a | n/a | 5 | 35894 |
| INTC | calendar | 2s | L0 zero | B | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 2s | L0 zero | C | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 2s | L0 zero | D | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | A | 0.14650 | 0.14754 | 0.14119 | 5 | 35894 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | B | 0.14500 | 0.14071 | 0.11837 | 5 | 35362 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | C | 0.14500 | 0.14071 | 0.11837 | 5 | 35362 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | D | 0.14500 | 0.14071 | 0.11837 | 5 | 35362 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | A | 0.11977 | 0.12317 | 0.10710 | 5 | 35894 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | B | 0.14022 | 0.13666 | 0.11058 | 5 | 35362 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | C | 0.14022 | 0.13666 | 0.11058 | 5 | 35362 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | D | 0.14022 | 0.13666 | 0.11058 | 5 | 35362 |
| INTC | calendar | 2s | L2 OLS on OFI | A | 0.10018 | 0.10295 | 0.09387 | 5 | 35894 |
| INTC | calendar | 2s | L2 OLS on OFI | B | 0.09620 | 0.08418 | 0.06731 | 5 | 35362 |
| INTC | calendar | 2s | L2 OLS on OFI | C | 0.09620 | 0.08418 | 0.06731 | 5 | 35362 |
| INTC | calendar | 2s | L2 OLS on OFI | D | 0.09620 | 0.08418 | 0.06731 | 5 | 35362 |
| INTC | calendar | 2s | L3 ridge, full set | A | 0.21202 | 0.21335 | 0.20485 | 5 | 35894 |
| INTC | calendar | 2s | L3 ridge, full set | B | 0.18637 | 0.18410 | 0.16365 | 5 | 35362 |
| INTC | calendar | 2s | L3 ridge, full set | C | 0.18619 | 0.18432 | 0.16340 | 5 | 35362 |
| INTC | calendar | 2s | L3 ridge, full set | D | 0.19674 | 0.19527 | 0.16899 | 5 | 35362 |
| INTC | calendar | 2s | L4 GBM, full set | A | 0.34598 | 0.34542 | 0.33963 | 5 | 35894 |
| INTC | calendar | 2s | L4 GBM, full set | B | 0.15080 | 0.14674 | 0.13804 | 5 | 35362 |
| INTC | calendar | 2s | L4 GBM, full set | C | 0.15393 | 0.14629 | 0.13863 | 5 | 35362 |
| INTC | calendar | 2s | L4 GBM, full set | D | 0.14850 | 0.16010 | 0.10920 | 5 | 35362 |
| INTC | calendar | 5s | L0 zero | A | n/a | n/a | n/a | 5 | 35884 |
| INTC | calendar | 5s | L0 zero | B | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 5s | L0 zero | C | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 5s | L0 zero | D | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | A | 0.18967 | 0.18726 | 0.18507 | 5 | 35884 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | B | 0.18822 | 0.18548 | 0.16704 | 5 | 35362 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | C | 0.18822 | 0.18548 | 0.16704 | 5 | 35362 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | D | 0.18822 | 0.18548 | 0.16704 | 5 | 35362 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | A | 0.12995 | 0.13012 | 0.12239 | 5 | 35884 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | B | 0.15127 | 0.15229 | 0.12508 | 5 | 35362 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | C | 0.15127 | 0.15229 | 0.12508 | 5 | 35362 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | D | 0.15127 | 0.15229 | 0.12508 | 5 | 35362 |
| INTC | calendar | 5s | L2 OLS on OFI | A | 0.10036 | 0.10120 | 0.09726 | 5 | 35884 |
| INTC | calendar | 5s | L2 OLS on OFI | B | 0.09591 | 0.08662 | 0.06954 | 5 | 35362 |
| INTC | calendar | 5s | L2 OLS on OFI | C | 0.09591 | 0.08662 | 0.06954 | 5 | 35362 |
| INTC | calendar | 5s | L2 OLS on OFI | D | 0.09591 | 0.08662 | 0.06954 | 5 | 35362 |
| INTC | calendar | 5s | L3 ridge, full set | A | 0.26364 | 0.26213 | 0.25633 | 5 | 35884 |
| INTC | calendar | 5s | L3 ridge, full set | B | 0.22900 | 0.22728 | 0.20713 | 5 | 35362 |
| INTC | calendar | 5s | L3 ridge, full set | C | 0.22826 | 0.22823 | 0.20630 | 5 | 35362 |
| INTC | calendar | 5s | L3 ridge, full set | D | 0.24192 | 0.24495 | 0.20870 | 5 | 35362 |
| INTC | calendar | 5s | L4 GBM, full set | A | 0.49479 | 0.49653 | 0.48625 | 5 | 35884 |
| INTC | calendar | 5s | L4 GBM, full set | B | 0.16407 | 0.16827 | 0.12965 | 5 | 35362 |
| INTC | calendar | 5s | L4 GBM, full set | C | 0.16979 | 0.17455 | 0.13974 | 5 | 35362 |
| INTC | calendar | 5s | L4 GBM, full set | D | 0.19088 | 0.18450 | 0.13467 | 5 | 35362 |
| INTC | calendar | 10s | L0 zero | A | n/a | n/a | n/a | 5 | 35878 |
| INTC | calendar | 10s | L0 zero | B | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 10s | L0 zero | C | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 10s | L0 zero | D | n/a | n/a | n/a | 5 | 35362 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | A | 0.21211 | 0.21182 | 0.20563 | 5 | 35878 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | B | 0.21034 | 0.21831 | 0.16263 | 5 | 35362 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | C | 0.21034 | 0.21831 | 0.16263 | 5 | 35362 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | D | 0.21034 | 0.21831 | 0.16263 | 5 | 35362 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | A | 0.13613 | 0.13958 | 0.12809 | 5 | 35878 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | B | 0.15727 | 0.16521 | 0.10211 | 5 | 35362 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | C | 0.15727 | 0.16521 | 0.10211 | 5 | 35362 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | D | 0.15727 | 0.16521 | 0.10211 | 5 | 35362 |
| INTC | calendar | 10s | L2 OLS on OFI | A | 0.10192 | 0.10175 | 0.10058 | 5 | 35878 |
| INTC | calendar | 10s | L2 OLS on OFI | B | 0.09818 | 0.08891 | 0.08151 | 5 | 35362 |
| INTC | calendar | 10s | L2 OLS on OFI | C | 0.09818 | 0.08891 | 0.08151 | 5 | 35362 |
| INTC | calendar | 10s | L2 OLS on OFI | D | 0.09818 | 0.08891 | 0.08151 | 5 | 35362 |
| INTC | calendar | 10s | L3 ridge, full set | A | 0.28718 | 0.28441 | 0.27530 | 5 | 35878 |
| INTC | calendar | 10s | L3 ridge, full set | B | 0.23016 | 0.22815 | 0.19292 | 5 | 35362 |
| INTC | calendar | 10s | L3 ridge, full set | C | 0.23159 | 0.23070 | 0.19197 | 5 | 35362 |
| INTC | calendar | 10s | L3 ridge, full set | D | 0.25763 | 0.25377 | 0.23397 | 5 | 35362 |
| INTC | calendar | 10s | L4 GBM, full set | A | 0.61674 | 0.61785 | 0.60812 | 5 | 35878 |
| INTC | calendar | 10s | L4 GBM, full set | B | 0.08444 | 0.13355 | -0.17204 | 5 | 35362 |
| INTC | calendar | 10s | L4 GBM, full set | C | 0.15223 | 0.16702 | 0.09530 | 5 | 35362 |
| INTC | calendar | 10s | L4 GBM, full set | D | 0.15982 | 0.20296 | 0.04818 | 5 | 35362 |
| INTC | event | 1s | L0 zero | A | n/a | n/a | n/a | 5 | 9418 |
| INTC | event | 1s | L0 zero | B | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 1s | L0 zero | C | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 1s | L0 zero | D | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 1s | L1 sign(queue imbalance) | A | 0.31371 | 0.31427 | 0.30633 | 5 | 9418 |
| INTC | event | 1s | L1 sign(queue imbalance) | B | 0.29794 | 0.30619 | 0.22264 | 5 | 9151 |
| INTC | event | 1s | L1 sign(queue imbalance) | C | 0.29794 | 0.30619 | 0.22264 | 5 | 9151 |
| INTC | event | 1s | L1 sign(queue imbalance) | D | 0.29794 | 0.30619 | 0.22264 | 5 | 9151 |
| INTC | event | 1s | L1m sign(microprice - mid) | A | 0.15258 | 0.14529 | 0.14396 | 5 | 9418 |
| INTC | event | 1s | L1m sign(microprice - mid) | B | 0.15270 | 0.14578 | 0.11422 | 5 | 9151 |
| INTC | event | 1s | L1m sign(microprice - mid) | C | 0.15270 | 0.14578 | 0.11422 | 5 | 9151 |
| INTC | event | 1s | L1m sign(microprice - mid) | D | 0.15270 | 0.14578 | 0.11422 | 5 | 9151 |
| INTC | event | 1s | L2 OLS on OFI | A | 0.34838 | 0.34590 | 0.33632 | 5 | 9418 |
| INTC | event | 1s | L2 OLS on OFI | B | 0.33797 | 0.28736 | 0.23888 | 5 | 9151 |
| INTC | event | 1s | L2 OLS on OFI | C | 0.33797 | 0.28736 | 0.23888 | 5 | 9151 |
| INTC | event | 1s | L2 OLS on OFI | D | 0.33797 | 0.28736 | 0.23888 | 5 | 9151 |
| INTC | event | 1s | L3 ridge, full set | A | 0.47949 | 0.47790 | 0.46875 | 5 | 9418 |
| INTC | event | 1s | L3 ridge, full set | B | 0.36331 | 0.33509 | 0.16526 | 5 | 9151 |
| INTC | event | 1s | L3 ridge, full set | C | 0.36451 | 0.33470 | 0.16374 | 5 | 9151 |
| INTC | event | 1s | L3 ridge, full set | D | 0.42304 | 0.41026 | 0.33498 | 5 | 9151 |
| INTC | event | 1s | L4 GBM, full set | A | 0.69868 | 0.69847 | 0.68717 | 5 | 9418 |
| INTC | event | 1s | L4 GBM, full set | B | 0.42315 | 0.41744 | 0.30774 | 5 | 9151 |
| INTC | event | 1s | L4 GBM, full set | C | 0.44413 | 0.41098 | 0.35886 | 5 | 9151 |
| INTC | event | 1s | L4 GBM, full set | D | 0.44225 | 0.40112 | 0.38729 | 5 | 9151 |
| INTC | event | 2s | L0 zero | A | n/a | n/a | n/a | 5 | 9414 |
| INTC | event | 2s | L0 zero | B | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 2s | L0 zero | C | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 2s | L0 zero | D | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 2s | L1 sign(queue imbalance) | A | 0.30555 | 0.30664 | 0.29756 | 5 | 9414 |
| INTC | event | 2s | L1 sign(queue imbalance) | B | 0.29452 | 0.27675 | 0.23160 | 5 | 9151 |
| INTC | event | 2s | L1 sign(queue imbalance) | C | 0.29452 | 0.27675 | 0.23160 | 5 | 9151 |
| INTC | event | 2s | L1 sign(queue imbalance) | D | 0.29452 | 0.27675 | 0.23160 | 5 | 9151 |
| INTC | event | 2s | L1m sign(microprice - mid) | A | 0.16846 | 0.16526 | 0.16116 | 5 | 9414 |
| INTC | event | 2s | L1m sign(microprice - mid) | B | 0.16659 | 0.15345 | 0.12955 | 5 | 9151 |
| INTC | event | 2s | L1m sign(microprice - mid) | C | 0.16659 | 0.15345 | 0.12955 | 5 | 9151 |
| INTC | event | 2s | L1m sign(microprice - mid) | D | 0.16659 | 0.15345 | 0.12955 | 5 | 9151 |
| INTC | event | 2s | L2 OLS on OFI | A | 0.31944 | 0.31783 | 0.31022 | 5 | 9414 |
| INTC | event | 2s | L2 OLS on OFI | B | 0.19144 | 0.23798 | -0.26659 | 5 | 9151 |
| INTC | event | 2s | L2 OLS on OFI | C | 0.19144 | 0.23798 | -0.26659 | 5 | 9151 |
| INTC | event | 2s | L2 OLS on OFI | D | 0.29808 | 0.26659 | 0.22469 | 5 | 9151 |
| INTC | event | 2s | L3 ridge, full set | A | 0.45810 | 0.45455 | 0.44613 | 5 | 9414 |
| INTC | event | 2s | L3 ridge, full set | B | 0.33510 | 0.31669 | 0.13219 | 5 | 9151 |
| INTC | event | 2s | L3 ridge, full set | C | 0.33535 | 0.32248 | 0.12662 | 5 | 9151 |
| INTC | event | 2s | L3 ridge, full set | D | 0.40015 | 0.35828 | 0.31261 | 5 | 9151 |
| INTC | event | 2s | L4 GBM, full set | A | 0.71011 | 0.71077 | 0.70158 | 5 | 9414 |
| INTC | event | 2s | L4 GBM, full set | B | 0.39017 | 0.36080 | 0.26269 | 5 | 9151 |
| INTC | event | 2s | L4 GBM, full set | C | 0.39830 | 0.37583 | 0.30386 | 5 | 9151 |
| INTC | event | 2s | L4 GBM, full set | D | 0.42588 | 0.40224 | 0.34766 | 5 | 9151 |
| INTC | event | 5s | L0 zero | A | n/a | n/a | n/a | 5 | 9397 |
| INTC | event | 5s | L0 zero | B | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 5s | L0 zero | C | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 5s | L0 zero | D | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 5s | L1 sign(queue imbalance) | A | 0.28258 | 0.28393 | 0.27828 | 5 | 9397 |
| INTC | event | 5s | L1 sign(queue imbalance) | B | 0.28033 | 0.27322 | 0.22218 | 5 | 9151 |
| INTC | event | 5s | L1 sign(queue imbalance) | C | 0.28033 | 0.27322 | 0.22218 | 5 | 9151 |
| INTC | event | 5s | L1 sign(queue imbalance) | D | 0.28033 | 0.27322 | 0.22218 | 5 | 9151 |
| INTC | event | 5s | L1m sign(microprice - mid) | A | 0.16781 | 0.16793 | 0.16406 | 5 | 9397 |
| INTC | event | 5s | L1m sign(microprice - mid) | B | 0.13065 | 0.17625 | -0.10589 | 5 | 9151 |
| INTC | event | 5s | L1m sign(microprice - mid) | C | 0.13065 | 0.17625 | -0.10589 | 5 | 9151 |
| INTC | event | 5s | L1m sign(microprice - mid) | D | 0.17301 | 0.17625 | 0.10589 | 5 | 9151 |
| INTC | event | 5s | L2 OLS on OFI | A | 0.27255 | 0.27306 | 0.26239 | 5 | 9397 |
| INTC | event | 5s | L2 OLS on OFI | B | 0.20910 | 0.25921 | -0.13249 | 5 | 9151 |
| INTC | event | 5s | L2 OLS on OFI | C | 0.20910 | 0.25921 | -0.13249 | 5 | 9151 |
| INTC | event | 5s | L2 OLS on OFI | D | 0.26209 | 0.25921 | 0.13249 | 5 | 9151 |
| INTC | event | 5s | L3 ridge, full set | A | 0.42725 | 0.42744 | 0.41784 | 5 | 9397 |
| INTC | event | 5s | L3 ridge, full set | B | 0.33203 | 0.35760 | 0.20466 | 5 | 9151 |
| INTC | event | 5s | L3 ridge, full set | C | 0.33391 | 0.36505 | 0.20329 | 5 | 9151 |
| INTC | event | 5s | L3 ridge, full set | D | 0.36600 | 0.37881 | 0.27072 | 5 | 9151 |
| INTC | event | 5s | L4 GBM, full set | A | 0.73640 | 0.73457 | 0.72958 | 5 | 9397 |
| INTC | event | 5s | L4 GBM, full set | B | 0.32139 | 0.30561 | 0.19692 | 5 | 9151 |
| INTC | event | 5s | L4 GBM, full set | C | 0.32440 | 0.29578 | 0.22385 | 5 | 9151 |
| INTC | event | 5s | L4 GBM, full set | D | 0.37574 | 0.34450 | 0.32800 | 5 | 9151 |
| INTC | event | 10s | L0 zero | A | n/a | n/a | n/a | 5 | 9365 |
| INTC | event | 10s | L0 zero | B | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 10s | L0 zero | C | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 10s | L0 zero | D | n/a | n/a | n/a | 5 | 9151 |
| INTC | event | 10s | L1 sign(queue imbalance) | A | 0.24595 | 0.24632 | 0.24201 | 5 | 9365 |
| INTC | event | 10s | L1 sign(queue imbalance) | B | 0.24812 | 0.25484 | 0.10763 | 5 | 9151 |
| INTC | event | 10s | L1 sign(queue imbalance) | C | 0.24812 | 0.25484 | 0.10763 | 5 | 9151 |
| INTC | event | 10s | L1 sign(queue imbalance) | D | 0.24812 | 0.25484 | 0.10763 | 5 | 9151 |
| INTC | event | 10s | L1m sign(microprice - mid) | A | 0.16183 | 0.16349 | 0.14946 | 5 | 9365 |
| INTC | event | 10s | L1m sign(microprice - mid) | B | 0.17445 | 0.19160 | 0.09418 | 5 | 9151 |
| INTC | event | 10s | L1m sign(microprice - mid) | C | 0.17445 | 0.19160 | 0.09418 | 5 | 9151 |
| INTC | event | 10s | L1m sign(microprice - mid) | D | 0.17445 | 0.19160 | 0.09418 | 5 | 9151 |
| INTC | event | 10s | L2 OLS on OFI | A | 0.19501 | 0.19556 | 0.18020 | 5 | 9365 |
| INTC | event | 10s | L2 OLS on OFI | B | 0.20061 | 0.18890 | -0.01973 | 5 | 9151 |
| INTC | event | 10s | L2 OLS on OFI | C | 0.20061 | 0.18890 | -0.01973 | 5 | 9151 |
| INTC | event | 10s | L2 OLS on OFI | D | 0.20061 | 0.18890 | -0.01973 | 5 | 9151 |
| INTC | event | 10s | L3 ridge, full set | A | 0.37314 | 0.37523 | 0.36049 | 5 | 9365 |
| INTC | event | 10s | L3 ridge, full set | B | 0.22138 | 0.27564 | 0.04926 | 5 | 9151 |
| INTC | event | 10s | L3 ridge, full set | C | 0.22420 | 0.28431 | 0.05124 | 5 | 9151 |
| INTC | event | 10s | L3 ridge, full set | D | 0.28644 | 0.31564 | 0.22188 | 5 | 9151 |
| INTC | event | 10s | L4 GBM, full set | A | 0.73731 | 0.72946 | 0.72728 | 5 | 9365 |
| INTC | event | 10s | L4 GBM, full set | B | 0.22196 | 0.22471 | 0.11917 | 5 | 9151 |
| INTC | event | 10s | L4 GBM, full set | C | 0.18045 | 0.21594 | 0.00489 | 5 | 9151 |
| INTC | event | 10s | L4 GBM, full set | D | 0.25696 | 0.23115 | 0.19330 | 5 | 9151 |
| AMZN | calendar | 1s | L0 zero | A | n/a | n/a | n/a | 5 | 31585 |
| AMZN | calendar | 1s | L0 zero | B | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 1s | L0 zero | C | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 1s | L0 zero | D | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | A | 0.08354 | 0.08340 | 0.08012 | 5 | 31585 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | B | 0.07966 | 0.07221 | 0.05342 | 5 | 30378 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | C | 0.07966 | 0.07221 | 0.05342 | 5 | 30378 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | D | 0.07966 | 0.07221 | 0.05342 | 5 | 30378 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | A | 0.06757 | 0.06713 | 0.06462 | 5 | 31585 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | B | 0.04348 | 0.06129 | -0.05770 | 5 | 30378 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | C | 0.04348 | 0.06129 | -0.05770 | 5 | 30378 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | D | 0.06656 | 0.06129 | 0.03474 | 5 | 30378 |
| AMZN | calendar | 1s | L2 OLS on OFI | A | 0.06902 | 0.06915 | 0.05943 | 5 | 31585 |
| AMZN | calendar | 1s | L2 OLS on OFI | B | 0.04502 | 0.05687 | -0.05352 | 5 | 30378 |
| AMZN | calendar | 1s | L2 OLS on OFI | C | 0.04502 | 0.05687 | -0.05352 | 5 | 30378 |
| AMZN | calendar | 1s | L2 OLS on OFI | D | 0.06643 | 0.05687 | 0.03784 | 5 | 30378 |
| AMZN | calendar | 1s | L3 ridge, full set | A | 0.12830 | 0.12783 | 0.12541 | 5 | 31585 |
| AMZN | calendar | 1s | L3 ridge, full set | B | 0.10404 | 0.10066 | 0.05916 | 5 | 30378 |
| AMZN | calendar | 1s | L3 ridge, full set | C | 0.10478 | 0.10081 | 0.06169 | 5 | 30378 |
| AMZN | calendar | 1s | L3 ridge, full set | D | 0.11457 | 0.09776 | 0.08986 | 5 | 30378 |
| AMZN | calendar | 1s | L4 GBM, full set | A | 0.28859 | 0.28575 | 0.28289 | 5 | 31585 |
| AMZN | calendar | 1s | L4 GBM, full set | B | 0.08811 | 0.08321 | 0.03039 | 5 | 30378 |
| AMZN | calendar | 1s | L4 GBM, full set | C | 0.08610 | 0.08820 | 0.01843 | 5 | 30378 |
| AMZN | calendar | 1s | L4 GBM, full set | D | 0.11582 | 0.11247 | 0.09605 | 5 | 30378 |
| AMZN | calendar | 2s | L0 zero | A | n/a | n/a | n/a | 5 | 31584 |
| AMZN | calendar | 2s | L0 zero | B | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 2s | L0 zero | C | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 2s | L0 zero | D | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | A | 0.10389 | 0.10451 | 0.10069 | 5 | 31584 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | B | 0.06081 | 0.07983 | -0.09576 | 5 | 30378 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | C | 0.06081 | 0.07983 | -0.09576 | 5 | 30378 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | D | 0.09912 | 0.09398 | 0.06196 | 5 | 30378 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | A | 0.08455 | 0.08586 | 0.07681 | 5 | 31584 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | B | 0.05047 | 0.07206 | -0.08174 | 5 | 30378 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | C | 0.05047 | 0.07206 | -0.08174 | 5 | 30378 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | D | 0.08316 | 0.07452 | 0.04231 | 5 | 30378 |
| AMZN | calendar | 2s | L2 OLS on OFI | A | 0.08315 | 0.08097 | 0.07967 | 5 | 31584 |
| AMZN | calendar | 2s | L2 OLS on OFI | B | 0.05470 | 0.07227 | -0.06248 | 5 | 30378 |
| AMZN | calendar | 2s | L2 OLS on OFI | C | 0.05470 | 0.07227 | -0.06248 | 5 | 30378 |
| AMZN | calendar | 2s | L2 OLS on OFI | D | 0.07969 | 0.07227 | 0.06248 | 5 | 30378 |
| AMZN | calendar | 2s | L3 ridge, full set | A | 0.15470 | 0.15205 | 0.15128 | 5 | 31584 |
| AMZN | calendar | 2s | L3 ridge, full set | B | 0.11796 | 0.11328 | 0.07060 | 5 | 30378 |
| AMZN | calendar | 2s | L3 ridge, full set | C | 0.11885 | 0.11349 | 0.07353 | 5 | 30378 |
| AMZN | calendar | 2s | L3 ridge, full set | D | 0.13741 | 0.12134 | 0.10637 | 5 | 30378 |
| AMZN | calendar | 2s | L4 GBM, full set | A | 0.37327 | 0.37453 | 0.36677 | 5 | 31584 |
| AMZN | calendar | 2s | L4 GBM, full set | B | 0.10277 | 0.09314 | 0.04059 | 5 | 30378 |
| AMZN | calendar | 2s | L4 GBM, full set | C | 0.10076 | 0.08996 | 0.03975 | 5 | 30378 |
| AMZN | calendar | 2s | L4 GBM, full set | D | 0.12782 | 0.11327 | 0.10347 | 5 | 30378 |
| AMZN | calendar | 5s | L0 zero | A | n/a | n/a | n/a | 5 | 31582 |
| AMZN | calendar | 5s | L0 zero | B | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 5s | L0 zero | C | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 5s | L0 zero | D | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | A | 0.10303 | 0.10243 | 0.09757 | 5 | 31582 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | B | 0.03674 | 0.08788 | -0.11099 | 5 | 30378 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | C | 0.05566 | 0.08788 | -0.11099 | 5 | 30378 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | D | 0.10006 | 0.10630 | 0.04729 | 5 | 30378 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | A | 0.07996 | 0.08180 | 0.07308 | 5 | 31582 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | B | 0.03259 | 0.06409 | -0.09173 | 5 | 30378 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | C | 0.04362 | 0.06409 | -0.09173 | 5 | 30378 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | D | 0.08031 | 0.08647 | 0.02758 | 5 | 30378 |
| AMZN | calendar | 5s | L2 OLS on OFI | A | 0.08727 | 0.08845 | 0.08134 | 5 | 31582 |
| AMZN | calendar | 5s | L2 OLS on OFI | B | 0.05952 | 0.07543 | -0.06166 | 5 | 30378 |
| AMZN | calendar | 5s | L2 OLS on OFI | C | 0.05952 | 0.07543 | -0.06166 | 5 | 30378 |
| AMZN | calendar | 5s | L2 OLS on OFI | D | 0.08418 | 0.07543 | 0.06166 | 5 | 30378 |
| AMZN | calendar | 5s | L3 ridge, full set | A | 0.16136 | 0.16217 | 0.15193 | 5 | 31582 |
| AMZN | calendar | 5s | L3 ridge, full set | B | 0.11046 | 0.11777 | 0.03873 | 5 | 30378 |
| AMZN | calendar | 5s | L3 ridge, full set | C | 0.11168 | 0.11796 | 0.04359 | 5 | 30378 |
| AMZN | calendar | 5s | L3 ridge, full set | D | 0.13658 | 0.11896 | 0.09731 | 5 | 30378 |
| AMZN | calendar | 5s | L4 GBM, full set | A | 0.49597 | 0.49742 | 0.48746 | 5 | 31582 |
| AMZN | calendar | 5s | L4 GBM, full set | B | 0.08929 | 0.11170 | -0.00327 | 5 | 30378 |
| AMZN | calendar | 5s | L4 GBM, full set | C | 0.07651 | 0.10479 | -0.02534 | 5 | 30378 |
| AMZN | calendar | 5s | L4 GBM, full set | D | 0.13194 | 0.14258 | 0.08211 | 5 | 30378 |
| AMZN | calendar | 10s | L0 zero | A | n/a | n/a | n/a | 5 | 31570 |
| AMZN | calendar | 10s | L0 zero | B | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 10s | L0 zero | C | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 10s | L0 zero | D | n/a | n/a | n/a | 5 | 30378 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | A | 0.08354 | 0.08240 | 0.08181 | 5 | 31570 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | B | 0.03326 | 0.08939 | -0.08797 | 5 | 30378 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | C | 0.03326 | 0.08939 | -0.08797 | 5 | 30378 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | D | 0.08287 | 0.08939 | 0.03604 | 5 | 30378 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | A | 0.06249 | 0.06107 | 0.05795 | 5 | 31570 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | B | 0.02845 | 0.06432 | -0.09376 | 5 | 30378 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | C | 0.02163 | 0.06432 | -0.09376 | 5 | 30378 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | D | 0.06596 | 0.06957 | 0.01705 | 5 | 30378 |
| AMZN | calendar | 10s | L2 OLS on OFI | A | 0.08623 | 0.08622 | 0.07622 | 5 | 31570 |
| AMZN | calendar | 10s | L2 OLS on OFI | B | 0.05847 | 0.07333 | -0.06503 | 5 | 30378 |
| AMZN | calendar | 10s | L2 OLS on OFI | C | 0.05847 | 0.07333 | -0.06503 | 5 | 30378 |
| AMZN | calendar | 10s | L2 OLS on OFI | D | 0.08449 | 0.07333 | 0.06503 | 5 | 30378 |
| AMZN | calendar | 10s | L3 ridge, full set | A | 0.15265 | 0.15392 | 0.14238 | 5 | 31570 |
| AMZN | calendar | 10s | L3 ridge, full set | B | 0.06442 | 0.07133 | -0.06861 | 5 | 30378 |
| AMZN | calendar | 10s | L3 ridge, full set | C | 0.06114 | 0.07018 | -0.07182 | 5 | 30378 |
| AMZN | calendar | 10s | L3 ridge, full set | D | 0.12284 | 0.10143 | 0.08699 | 5 | 30378 |
| AMZN | calendar | 10s | L4 GBM, full set | A | 0.59252 | 0.59536 | 0.58425 | 5 | 31570 |
| AMZN | calendar | 10s | L4 GBM, full set | B | 0.05808 | 0.09451 | -0.04419 | 5 | 30378 |
| AMZN | calendar | 10s | L4 GBM, full set | C | 0.03832 | 0.06609 | -0.06542 | 5 | 30378 |
| AMZN | calendar | 10s | L4 GBM, full set | D | 0.10902 | 0.11162 | 0.04717 | 5 | 30378 |
| AMZN | event | 1s | L0 zero | A | n/a | n/a | n/a | 5 | 4419 |
| AMZN | event | 1s | L0 zero | B | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 1s | L0 zero | C | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 1s | L0 zero | D | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 1s | L1 sign(queue imbalance) | A | 0.15273 | 0.15069 | 0.14533 | 5 | 4419 |
| AMZN | event | 1s | L1 sign(queue imbalance) | B | 0.14802 | 0.14417 | 0.11171 | 5 | 3827 |
| AMZN | event | 1s | L1 sign(queue imbalance) | C | 0.14802 | 0.14417 | 0.11171 | 5 | 3827 |
| AMZN | event | 1s | L1 sign(queue imbalance) | D | 0.14802 | 0.14417 | 0.11171 | 5 | 3827 |
| AMZN | event | 1s | L1m sign(microprice - mid) | A | 0.12332 | 0.11716 | 0.11299 | 5 | 4419 |
| AMZN | event | 1s | L1m sign(microprice - mid) | B | 0.07633 | 0.09158 | -0.11310 | 5 | 3827 |
| AMZN | event | 1s | L1m sign(microprice - mid) | C | 0.12157 | 0.11310 | 0.07815 | 5 | 3827 |
| AMZN | event | 1s | L1m sign(microprice - mid) | D | 0.12157 | 0.11310 | 0.07815 | 5 | 3827 |
| AMZN | event | 1s | L2 OLS on OFI | A | 0.28444 | 0.28335 | 0.26945 | 5 | 4419 |
| AMZN | event | 1s | L2 OLS on OFI | B | 0.28334 | 0.28934 | 0.22546 | 5 | 3827 |
| AMZN | event | 1s | L2 OLS on OFI | C | 0.28334 | 0.28934 | 0.22546 | 5 | 3827 |
| AMZN | event | 1s | L2 OLS on OFI | D | 0.28334 | 0.28934 | 0.22546 | 5 | 3827 |
| AMZN | event | 1s | L3 ridge, full set | A | 0.33676 | 0.33317 | 0.32336 | 5 | 4419 |
| AMZN | event | 1s | L3 ridge, full set | B | 0.29768 | 0.31746 | 0.15914 | 5 | 3827 |
| AMZN | event | 1s | L3 ridge, full set | C | 0.29785 | 0.31699 | 0.16032 | 5 | 3827 |
| AMZN | event | 1s | L3 ridge, full set | D | 0.32001 | 0.33202 | 0.21356 | 5 | 3827 |
| AMZN | event | 1s | L4 GBM, full set | A | 0.43835 | 0.43669 | 0.42658 | 5 | 4419 |
| AMZN | event | 1s | L4 GBM, full set | B | 0.33888 | 0.34372 | 0.23628 | 5 | 3827 |
| AMZN | event | 1s | L4 GBM, full set | C | 0.33378 | 0.34207 | 0.23649 | 5 | 3827 |
| AMZN | event | 1s | L4 GBM, full set | D | 0.35751 | 0.35139 | 0.29881 | 5 | 3827 |
| AMZN | event | 2s | L0 zero | A | n/a | n/a | n/a | 5 | 4418 |
| AMZN | event | 2s | L0 zero | B | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 2s | L0 zero | C | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 2s | L0 zero | D | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 2s | L1 sign(queue imbalance) | A | 0.15730 | 0.15478 | 0.14944 | 5 | 4418 |
| AMZN | event | 2s | L1 sign(queue imbalance) | B | 0.14887 | 0.14488 | 0.11676 | 5 | 3827 |
| AMZN | event | 2s | L1 sign(queue imbalance) | C | 0.14887 | 0.14488 | 0.11676 | 5 | 3827 |
| AMZN | event | 2s | L1 sign(queue imbalance) | D | 0.14887 | 0.14488 | 0.11676 | 5 | 3827 |
| AMZN | event | 2s | L1m sign(microprice - mid) | A | 0.13242 | 0.13086 | 0.11968 | 5 | 4418 |
| AMZN | event | 2s | L1m sign(microprice - mid) | B | 0.07876 | 0.11752 | -0.11849 | 5 | 3827 |
| AMZN | event | 2s | L1m sign(microprice - mid) | C | 0.12915 | 0.11849 | 0.09091 | 5 | 3827 |
| AMZN | event | 2s | L1m sign(microprice - mid) | D | 0.12915 | 0.11849 | 0.09091 | 5 | 3827 |
| AMZN | event | 2s | L2 OLS on OFI | A | 0.27524 | 0.27509 | 0.25715 | 5 | 4418 |
| AMZN | event | 2s | L2 OLS on OFI | B | 0.18625 | 0.26413 | -0.21293 | 5 | 3827 |
| AMZN | event | 2s | L2 OLS on OFI | C | 0.18625 | 0.26413 | -0.21293 | 5 | 3827 |
| AMZN | event | 2s | L2 OLS on OFI | D | 0.27142 | 0.26413 | 0.21293 | 5 | 3827 |
| AMZN | event | 2s | L3 ridge, full set | A | 0.34053 | 0.33886 | 0.32726 | 5 | 4418 |
| AMZN | event | 2s | L3 ridge, full set | B | 0.29550 | 0.29345 | 0.20876 | 5 | 3827 |
| AMZN | event | 2s | L3 ridge, full set | C | 0.29618 | 0.29549 | 0.20979 | 5 | 3827 |
| AMZN | event | 2s | L3 ridge, full set | D | 0.31840 | 0.32271 | 0.23464 | 5 | 3827 |
| AMZN | event | 2s | L4 GBM, full set | A | 0.46171 | 0.46250 | 0.44116 | 5 | 4418 |
| AMZN | event | 2s | L4 GBM, full set | B | 0.33937 | 0.35663 | 0.24751 | 5 | 3827 |
| AMZN | event | 2s | L4 GBM, full set | C | 0.33972 | 0.35826 | 0.25145 | 5 | 3827 |
| AMZN | event | 2s | L4 GBM, full set | D | 0.35723 | 0.36353 | 0.30923 | 5 | 3827 |
| AMZN | event | 5s | L0 zero | A | n/a | n/a | n/a | 5 | 4417 |
| AMZN | event | 5s | L0 zero | B | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 5s | L0 zero | C | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 5s | L0 zero | D | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 5s | L1 sign(queue imbalance) | A | 0.14432 | 0.14335 | 0.12431 | 5 | 4417 |
| AMZN | event | 5s | L1 sign(queue imbalance) | B | 0.10436 | 0.13268 | -0.08315 | 5 | 3827 |
| AMZN | event | 5s | L1 sign(queue imbalance) | C | 0.10436 | 0.13268 | -0.08315 | 5 | 3827 |
| AMZN | event | 5s | L1 sign(queue imbalance) | D | 0.13762 | 0.13268 | 0.08315 | 5 | 3827 |
| AMZN | event | 5s | L1m sign(microprice - mid) | A | 0.11832 | 0.12721 | 0.08873 | 5 | 4417 |
| AMZN | event | 5s | L1m sign(microprice - mid) | B | 0.04312 | 0.05936 | -0.11272 | 5 | 3827 |
| AMZN | event | 5s | L1m sign(microprice - mid) | C | 0.04312 | 0.05936 | -0.11272 | 5 | 3827 |
| AMZN | event | 5s | L1m sign(microprice - mid) | D | 0.11331 | 0.11272 | 0.05936 | 5 | 3827 |
| AMZN | event | 5s | L2 OLS on OFI | A | 0.27183 | 0.27283 | 0.25556 | 5 | 4417 |
| AMZN | event | 5s | L2 OLS on OFI | B | 0.18447 | 0.28007 | -0.20927 | 5 | 3827 |
| AMZN | event | 5s | L2 OLS on OFI | C | 0.18447 | 0.28007 | -0.20927 | 5 | 3827 |
| AMZN | event | 5s | L2 OLS on OFI | D | 0.26818 | 0.28007 | 0.20927 | 5 | 3827 |
| AMZN | event | 5s | L3 ridge, full set | A | 0.32590 | 0.33128 | 0.29836 | 5 | 4417 |
| AMZN | event | 5s | L3 ridge, full set | B | 0.25384 | 0.28367 | 0.11810 | 5 | 3827 |
| AMZN | event | 5s | L3 ridge, full set | C | 0.25238 | 0.28248 | 0.11738 | 5 | 3827 |
| AMZN | event | 5s | L3 ridge, full set | D | 0.29585 | 0.31877 | 0.19597 | 5 | 3827 |
| AMZN | event | 5s | L4 GBM, full set | A | 0.49502 | 0.50045 | 0.46619 | 5 | 4417 |
| AMZN | event | 5s | L4 GBM, full set | B | 0.31433 | 0.32152 | 0.18446 | 5 | 3827 |
| AMZN | event | 5s | L4 GBM, full set | C | 0.31211 | 0.32412 | 0.17640 | 5 | 3827 |
| AMZN | event | 5s | L4 GBM, full set | D | 0.33983 | 0.34064 | 0.25041 | 5 | 3827 |
| AMZN | event | 10s | L0 zero | A | n/a | n/a | n/a | 5 | 4414 |
| AMZN | event | 10s | L0 zero | B | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 10s | L0 zero | C | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 10s | L0 zero | D | n/a | n/a | n/a | 5 | 3827 |
| AMZN | event | 10s | L1 sign(queue imbalance) | A | 0.12489 | 0.12479 | 0.09976 | 5 | 4414 |
| AMZN | event | 10s | L1 sign(queue imbalance) | B | 0.09048 | 0.12333 | -0.08287 | 5 | 3827 |
| AMZN | event | 10s | L1 sign(queue imbalance) | C | 0.09048 | 0.12333 | -0.08287 | 5 | 3827 |
| AMZN | event | 10s | L1 sign(queue imbalance) | D | 0.12363 | 0.12333 | 0.08287 | 5 | 3827 |
| AMZN | event | 10s | L1m sign(microprice - mid) | A | 0.09980 | 0.10188 | 0.07827 | 5 | 4414 |
| AMZN | event | 10s | L1m sign(microprice - mid) | B | -0.01959 | -0.05746 | -0.12312 | 5 | 3827 |
| AMZN | event | 10s | L1m sign(microprice - mid) | C | 0.07760 | 0.11986 | -0.05746 | 5 | 3827 |
| AMZN | event | 10s | L1m sign(microprice - mid) | D | 0.10059 | 0.11986 | 0.05746 | 5 | 3827 |
| AMZN | event | 10s | L2 OLS on OFI | A | 0.24503 | 0.24601 | 0.22618 | 5 | 4414 |
| AMZN | event | 10s | L2 OLS on OFI | B | 0.14895 | 0.23174 | -0.22863 | 5 | 3827 |
| AMZN | event | 10s | L2 OLS on OFI | C | 0.14895 | 0.23174 | -0.22863 | 5 | 3827 |
| AMZN | event | 10s | L2 OLS on OFI | D | 0.24040 | 0.23174 | 0.22637 | 5 | 3827 |
| AMZN | event | 10s | L3 ridge, full set | A | 0.28031 | 0.28180 | 0.24945 | 5 | 4414 |
| AMZN | event | 10s | L3 ridge, full set | B | 0.19134 | 0.21294 | 0.06169 | 5 | 3827 |
| AMZN | event | 10s | L3 ridge, full set | C | 0.18881 | 0.21225 | 0.05893 | 5 | 3827 |
| AMZN | event | 10s | L3 ridge, full set | D | 0.24252 | 0.24063 | 0.19073 | 5 | 3827 |
| AMZN | event | 10s | L4 GBM, full set | A | 0.50674 | 0.50907 | 0.48431 | 5 | 4414 |
| AMZN | event | 10s | L4 GBM, full set | B | 0.22498 | 0.24483 | 0.12460 | 5 | 3827 |
| AMZN | event | 10s | L4 GBM, full set | C | 0.21981 | 0.23541 | 0.11206 | 5 | 3827 |
| AMZN | event | 10s | L4 GBM, full set | D | 0.26958 | 0.29111 | 0.16188 | 5 | 3827 |

## Classification against the base rate

Balanced accuracy and MCC, printed next to the majority-class share. At 1s most labels are flat, so plain accuracy is meaningless (FEATURES.md sec. 7).

| ticker | clock | h | rung | protocol | bal. acc | MCC | base rate |
|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | A | 0.3333 | n/a | 0.9698 |
| INTC | calendar | 1s | L0 zero | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L0 zero | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L0 zero | D | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.9698 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.9698 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L2 OLS on OFI | A | 0.3333 | n/a | 0.9698 |
| INTC | calendar | 1s | L2 OLS on OFI | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L2 OLS on OFI | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L2 OLS on OFI | D | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L3 ridge, full set | A | 0.3333 | -0.0007 | 0.9698 |
| INTC | calendar | 1s | L3 ridge, full set | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L3 ridge, full set | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L3 ridge, full set | D | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L4 GBM, full set | A | 0.3943 | 0.2506 | 0.9698 |
| INTC | calendar | 1s | L4 GBM, full set | B | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L4 GBM, full set | C | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 1s | L4 GBM, full set | D | 0.3333 | n/a | 0.9700 |
| INTC | calendar | 2s | L0 zero | A | 0.3333 | n/a | 0.9425 |
| INTC | calendar | 2s | L0 zero | B | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L0 zero | C | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L0 zero | D | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.9425 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.9425 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L2 OLS on OFI | A | 0.3333 | n/a | 0.9425 |
| INTC | calendar | 2s | L2 OLS on OFI | B | 0.3333 | -0.0041 | 0.9428 |
| INTC | calendar | 2s | L2 OLS on OFI | C | 0.3333 | -0.0041 | 0.9428 |
| INTC | calendar | 2s | L2 OLS on OFI | D | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L3 ridge, full set | A | 0.3340 | 0.0164 | 0.9425 |
| INTC | calendar | 2s | L3 ridge, full set | B | 0.3338 | 0.0273 | 0.9428 |
| INTC | calendar | 2s | L3 ridge, full set | C | 0.3337 | 0.0337 | 0.9428 |
| INTC | calendar | 2s | L3 ridge, full set | D | 0.3338 | 0.0176 | 0.9428 |
| INTC | calendar | 2s | L4 GBM, full set | A | 0.4508 | 0.3500 | 0.9425 |
| INTC | calendar | 2s | L4 GBM, full set | B | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L4 GBM, full set | C | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 2s | L4 GBM, full set | D | 0.3333 | n/a | 0.9428 |
| INTC | calendar | 5s | L0 zero | A | 0.3333 | n/a | 0.8702 |
| INTC | calendar | 5s | L0 zero | B | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L0 zero | C | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L0 zero | D | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.8702 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.8702 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L2 OLS on OFI | A | 0.3336 | 0.0162 | 0.8702 |
| INTC | calendar | 5s | L2 OLS on OFI | B | 0.3333 | -0.0052 | 0.8709 |
| INTC | calendar | 5s | L2 OLS on OFI | C | 0.3333 | -0.0052 | 0.8709 |
| INTC | calendar | 5s | L2 OLS on OFI | D | 0.3333 | -0.0027 | 0.8709 |
| INTC | calendar | 5s | L3 ridge, full set | A | 0.3367 | 0.0434 | 0.8702 |
| INTC | calendar | 5s | L3 ridge, full set | B | 0.3335 | 0.0048 | 0.8709 |
| INTC | calendar | 5s | L3 ridge, full set | C | 0.3335 | 0.0078 | 0.8709 |
| INTC | calendar | 5s | L3 ridge, full set | D | 0.3336 | 0.0085 | 0.8709 |
| INTC | calendar | 5s | L4 GBM, full set | A | 0.5583 | 0.4579 | 0.8702 |
| INTC | calendar | 5s | L4 GBM, full set | B | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L4 GBM, full set | C | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 5s | L4 GBM, full set | D | 0.3333 | n/a | 0.8709 |
| INTC | calendar | 10s | L0 zero | A | 0.3333 | n/a | 0.7743 |
| INTC | calendar | 10s | L0 zero | B | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L0 zero | C | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L0 zero | D | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.7743 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.7743 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L2 OLS on OFI | A | 0.3335 | 0.0141 | 0.7743 |
| INTC | calendar | 10s | L2 OLS on OFI | B | 0.3334 | 0.0020 | 0.7756 |
| INTC | calendar | 10s | L2 OLS on OFI | C | 0.3332 | -0.0077 | 0.7756 |
| INTC | calendar | 10s | L2 OLS on OFI | D | 0.3334 | 0.0162 | 0.7756 |
| INTC | calendar | 10s | L3 ridge, full set | A | 0.3423 | 0.0748 | 0.7743 |
| INTC | calendar | 10s | L3 ridge, full set | B | 0.3346 | 0.0178 | 0.7756 |
| INTC | calendar | 10s | L3 ridge, full set | C | 0.3346 | 0.0196 | 0.7756 |
| INTC | calendar | 10s | L3 ridge, full set | D | 0.3367 | 0.0361 | 0.7756 |
| INTC | calendar | 10s | L4 GBM, full set | A | 0.6589 | 0.5590 | 0.7743 |
| INTC | calendar | 10s | L4 GBM, full set | B | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L4 GBM, full set | C | 0.3333 | n/a | 0.7756 |
| INTC | calendar | 10s | L4 GBM, full set | D | 0.3333 | n/a | 0.7756 |
| INTC | event | 1s | L0 zero | A | 0.3333 | n/a | 0.9044 |
| INTC | event | 1s | L0 zero | B | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L0 zero | C | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L0 zero | D | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.9044 |
| INTC | event | 1s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.9044 |
| INTC | event | 1s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.9086 |
| INTC | event | 1s | L2 OLS on OFI | A | 0.3328 | -0.0007 | 0.9044 |
| INTC | event | 1s | L2 OLS on OFI | B | 0.3323 | -0.0079 | 0.9086 |
| INTC | event | 1s | L2 OLS on OFI | C | 0.3324 | 0.0023 | 0.9086 |
| INTC | event | 1s | L2 OLS on OFI | D | 0.3327 | -0.0032 | 0.9086 |
| INTC | event | 1s | L3 ridge, full set | A | 0.3741 | 0.1013 | 0.9044 |
| INTC | event | 1s | L3 ridge, full set | B | 0.3604 | 0.1060 | 0.9086 |
| INTC | event | 1s | L3 ridge, full set | C | 0.3569 | 0.0947 | 0.9086 |
| INTC | event | 1s | L3 ridge, full set | D | 0.3659 | 0.0871 | 0.9086 |
| INTC | event | 1s | L4 GBM, full set | A | 0.6147 | 0.3399 | 0.9044 |
| INTC | event | 1s | L4 GBM, full set | B | 0.3689 | 0.1936 | 0.9086 |
| INTC | event | 1s | L4 GBM, full set | C | 0.3718 | 0.1594 | 0.9086 |
| INTC | event | 1s | L4 GBM, full set | D | 0.3871 | 0.1878 | 0.9086 |
| INTC | event | 2s | L0 zero | A | 0.3333 | n/a | 0.8705 |
| INTC | event | 2s | L0 zero | B | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L0 zero | C | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L0 zero | D | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.8705 |
| INTC | event | 2s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.8705 |
| INTC | event | 2s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.8762 |
| INTC | event | 2s | L2 OLS on OFI | A | 0.3336 | 0.0006 | 0.8705 |
| INTC | event | 2s | L2 OLS on OFI | B | 0.3328 | -0.0020 | 0.8762 |
| INTC | event | 2s | L2 OLS on OFI | C | 0.3324 | -0.0048 | 0.8762 |
| INTC | event | 2s | L2 OLS on OFI | D | 0.3343 | 0.0031 | 0.8762 |
| INTC | event | 2s | L3 ridge, full set | A | 0.3926 | 0.1496 | 0.8705 |
| INTC | event | 2s | L3 ridge, full set | B | 0.3865 | 0.1545 | 0.8762 |
| INTC | event | 2s | L3 ridge, full set | C | 0.3890 | 0.1653 | 0.8762 |
| INTC | event | 2s | L3 ridge, full set | D | 0.3718 | 0.1163 | 0.8762 |
| INTC | event | 2s | L4 GBM, full set | A | 0.6084 | 0.3633 | 0.8705 |
| INTC | event | 2s | L4 GBM, full set | B | 0.3799 | 0.1755 | 0.8762 |
| INTC | event | 2s | L4 GBM, full set | C | 0.3754 | 0.1569 | 0.8762 |
| INTC | event | 2s | L4 GBM, full set | D | 0.3653 | 0.1407 | 0.8762 |
| INTC | event | 5s | L0 zero | A | 0.3333 | n/a | 0.7898 |
| INTC | event | 5s | L0 zero | B | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L0 zero | C | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L0 zero | D | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.7898 |
| INTC | event | 5s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.7898 |
| INTC | event | 5s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.7979 |
| INTC | event | 5s | L2 OLS on OFI | A | 0.3390 | 0.0386 | 0.7898 |
| INTC | event | 5s | L2 OLS on OFI | B | 0.3366 | 0.0493 | 0.7979 |
| INTC | event | 5s | L2 OLS on OFI | C | 0.3364 | 0.0477 | 0.7979 |
| INTC | event | 5s | L2 OLS on OFI | D | 0.3446 | 0.0530 | 0.7979 |
| INTC | event | 5s | L3 ridge, full set | A | 0.4119 | 0.1785 | 0.7898 |
| INTC | event | 5s | L3 ridge, full set | B | 0.3802 | 0.1554 | 0.7979 |
| INTC | event | 5s | L3 ridge, full set | C | 0.3803 | 0.1476 | 0.7979 |
| INTC | event | 5s | L3 ridge, full set | D | 0.3799 | 0.1273 | 0.7979 |
| INTC | event | 5s | L4 GBM, full set | A | 0.6289 | 0.4062 | 0.7898 |
| INTC | event | 5s | L4 GBM, full set | B | 0.3474 | 0.1037 | 0.7979 |
| INTC | event | 5s | L4 GBM, full set | C | 0.3476 | 0.0486 | 0.7979 |
| INTC | event | 5s | L4 GBM, full set | D | 0.3568 | 0.1018 | 0.7979 |
| INTC | event | 10s | L0 zero | A | 0.3333 | n/a | 0.7094 |
| INTC | event | 10s | L0 zero | B | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L0 zero | C | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L0 zero | D | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.7094 |
| INTC | event | 10s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.7094 |
| INTC | event | 10s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.7222 |
| INTC | event | 10s | L2 OLS on OFI | A | 0.3349 | 0.0161 | 0.7094 |
| INTC | event | 10s | L2 OLS on OFI | B | 0.3328 | -0.0353 | 0.7222 |
| INTC | event | 10s | L2 OLS on OFI | C | 0.3317 | -0.0345 | 0.7222 |
| INTC | event | 10s | L2 OLS on OFI | D | 0.3337 | 0.0328 | 0.7222 |
| INTC | event | 10s | L3 ridge, full set | A | 0.3881 | 0.1464 | 0.7094 |
| INTC | event | 10s | L3 ridge, full set | B | 0.3586 | 0.1011 | 0.7222 |
| INTC | event | 10s | L3 ridge, full set | C | 0.3681 | 0.1336 | 0.7222 |
| INTC | event | 10s | L3 ridge, full set | D | 0.3523 | 0.0602 | 0.7222 |
| INTC | event | 10s | L4 GBM, full set | A | 0.6455 | 0.4335 | 0.7094 |
| INTC | event | 10s | L4 GBM, full set | B | 0.3334 | 0.0083 | 0.7222 |
| INTC | event | 10s | L4 GBM, full set | C | 0.3344 | 0.0398 | 0.7222 |
| INTC | event | 10s | L4 GBM, full set | D | 0.3368 | 0.0544 | 0.7222 |
| AMZN | calendar | 1s | L0 zero | A | 0.3333 | n/a | 0.7777 |
| AMZN | calendar | 1s | L0 zero | B | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L0 zero | C | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L0 zero | D | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.7777 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.7777 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.7795 |
| AMZN | calendar | 1s | L2 OLS on OFI | A | 0.3346 | 0.0229 | 0.7777 |
| AMZN | calendar | 1s | L2 OLS on OFI | B | 0.3395 | 0.0433 | 0.7795 |
| AMZN | calendar | 1s | L2 OLS on OFI | C | 0.3393 | 0.0419 | 0.7795 |
| AMZN | calendar | 1s | L2 OLS on OFI | D | 0.3373 | 0.0342 | 0.7795 |
| AMZN | calendar | 1s | L3 ridge, full set | A | 0.3570 | 0.0945 | 0.7777 |
| AMZN | calendar | 1s | L3 ridge, full set | B | 0.3466 | 0.0697 | 0.7795 |
| AMZN | calendar | 1s | L3 ridge, full set | C | 0.3475 | 0.0719 | 0.7795 |
| AMZN | calendar | 1s | L3 ridge, full set | D | 0.3529 | 0.0790 | 0.7795 |
| AMZN | calendar | 1s | L4 GBM, full set | A | 0.4965 | 0.2028 | 0.7777 |
| AMZN | calendar | 1s | L4 GBM, full set | B | 0.3435 | 0.0520 | 0.7795 |
| AMZN | calendar | 1s | L4 GBM, full set | C | 0.3472 | 0.0610 | 0.7795 |
| AMZN | calendar | 1s | L4 GBM, full set | D | 0.3701 | 0.0899 | 0.7795 |
| AMZN | calendar | 2s | L0 zero | A | 0.3333 | n/a | 0.6497 |
| AMZN | calendar | 2s | L0 zero | B | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L0 zero | C | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L0 zero | D | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.6497 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | D | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.6497 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.6523 |
| AMZN | calendar | 2s | L2 OLS on OFI | A | 0.3362 | 0.0386 | 0.6497 |
| AMZN | calendar | 2s | L2 OLS on OFI | B | 0.3440 | 0.0562 | 0.6523 |
| AMZN | calendar | 2s | L2 OLS on OFI | C | 0.3438 | 0.0572 | 0.6523 |
| AMZN | calendar | 2s | L2 OLS on OFI | D | 0.3397 | 0.0493 | 0.6523 |
| AMZN | calendar | 2s | L3 ridge, full set | A | 0.3714 | 0.1064 | 0.6497 |
| AMZN | calendar | 2s | L3 ridge, full set | B | 0.3540 | 0.0692 | 0.6523 |
| AMZN | calendar | 2s | L3 ridge, full set | C | 0.3554 | 0.0778 | 0.6523 |
| AMZN | calendar | 2s | L3 ridge, full set | D | 0.3610 | 0.0910 | 0.6523 |
| AMZN | calendar | 2s | L4 GBM, full set | A | 0.5190 | 0.2139 | 0.6497 |
| AMZN | calendar | 2s | L4 GBM, full set | B | 0.3497 | 0.0326 | 0.6523 |
| AMZN | calendar | 2s | L4 GBM, full set | C | 0.3531 | 0.0488 | 0.6523 |
| AMZN | calendar | 2s | L4 GBM, full set | D | 0.3699 | 0.0711 | 0.6523 |
| AMZN | calendar | 5s | L0 zero | A | 0.3333 | n/a | 0.4303 |
| AMZN | calendar | 5s | L0 zero | B | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L0 zero | C | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L0 zero | D | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.4303 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | D | 0.3622 | 0.0501 | 0.4341 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.4303 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.4341 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | D | 0.3360 | 0.0243 | 0.4341 |
| AMZN | calendar | 5s | L2 OLS on OFI | A | 0.3381 | 0.0421 | 0.4303 |
| AMZN | calendar | 5s | L2 OLS on OFI | B | 0.3475 | 0.0620 | 0.4341 |
| AMZN | calendar | 5s | L2 OLS on OFI | C | 0.3474 | 0.0593 | 0.4341 |
| AMZN | calendar | 5s | L2 OLS on OFI | D | 0.3379 | 0.0361 | 0.4341 |
| AMZN | calendar | 5s | L3 ridge, full set | A | 0.3821 | 0.1058 | 0.4303 |
| AMZN | calendar | 5s | L3 ridge, full set | B | 0.3550 | 0.0540 | 0.4341 |
| AMZN | calendar | 5s | L3 ridge, full set | C | 0.3562 | 0.0615 | 0.4341 |
| AMZN | calendar | 5s | L3 ridge, full set | D | 0.3640 | 0.0886 | 0.4341 |
| AMZN | calendar | 5s | L4 GBM, full set | A | 0.5202 | 0.2571 | 0.4303 |
| AMZN | calendar | 5s | L4 GBM, full set | B | 0.3538 | 0.0660 | 0.4341 |
| AMZN | calendar | 5s | L4 GBM, full set | C | 0.3496 | 0.0359 | 0.4341 |
| AMZN | calendar | 5s | L4 GBM, full set | D | 0.3713 | 0.0831 | 0.4341 |
| AMZN | calendar | 10s | L0 zero | A | 0.3333 | n/a | 0.3938 |
| AMZN | calendar | 10s | L0 zero | B | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L0 zero | C | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L0 zero | D | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.3938 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | C | 0.3280 | -0.0474 | 0.3922 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | D | 0.3599 | 0.0595 | 0.3922 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.3938 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | B | 0.3273 | -0.0687 | 0.3922 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | C | 0.3273 | -0.0687 | 0.3922 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.3922 |
| AMZN | calendar | 10s | L2 OLS on OFI | A | 0.3375 | 0.0306 | 0.3938 |
| AMZN | calendar | 10s | L2 OLS on OFI | B | 0.3510 | 0.0445 | 0.3922 |
| AMZN | calendar | 10s | L2 OLS on OFI | C | 0.3508 | 0.0424 | 0.3922 |
| AMZN | calendar | 10s | L2 OLS on OFI | D | 0.3389 | 0.0333 | 0.3922 |
| AMZN | calendar | 10s | L3 ridge, full set | A | 0.3746 | 0.0761 | 0.3938 |
| AMZN | calendar | 10s | L3 ridge, full set | B | 0.3520 | 0.0410 | 0.3922 |
| AMZN | calendar | 10s | L3 ridge, full set | C | 0.3516 | 0.0397 | 0.3922 |
| AMZN | calendar | 10s | L3 ridge, full set | D | 0.3583 | 0.0617 | 0.3922 |
| AMZN | calendar | 10s | L4 GBM, full set | A | 0.5164 | 0.3096 | 0.3938 |
| AMZN | calendar | 10s | L4 GBM, full set | B | 0.3404 | 0.0182 | 0.3922 |
| AMZN | calendar | 10s | L4 GBM, full set | C | 0.3218 | -0.0223 | 0.3922 |
| AMZN | calendar | 10s | L4 GBM, full set | D | 0.3514 | 0.0443 | 0.3922 |
| AMZN | event | 1s | L0 zero | A | 0.3333 | n/a | 0.5564 |
| AMZN | event | 1s | L0 zero | B | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L0 zero | C | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L0 zero | D | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L1 sign(queue imbalance) | A | 0.3333 | n/a | 0.5564 |
| AMZN | event | 1s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L1 sign(queue imbalance) | D | 0.3793 | 0.0622 | 0.5621 |
| AMZN | event | 1s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.5564 |
| AMZN | event | 1s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L1m sign(microprice - mid) | D | 0.3333 | n/a | 0.5621 |
| AMZN | event | 1s | L2 OLS on OFI | A | 0.3625 | 0.1178 | 0.5564 |
| AMZN | event | 1s | L2 OLS on OFI | B | 0.4069 | 0.1717 | 0.5621 |
| AMZN | event | 1s | L2 OLS on OFI | C | 0.4092 | 0.1748 | 0.5621 |
| AMZN | event | 1s | L2 OLS on OFI | D | 0.3812 | 0.1362 | 0.5621 |
| AMZN | event | 1s | L3 ridge, full set | A | 0.4935 | 0.2404 | 0.5564 |
| AMZN | event | 1s | L3 ridge, full set | B | 0.4290 | 0.1714 | 0.5621 |
| AMZN | event | 1s | L3 ridge, full set | C | 0.4301 | 0.1705 | 0.5621 |
| AMZN | event | 1s | L3 ridge, full set | D | 0.4754 | 0.2226 | 0.5621 |
| AMZN | event | 1s | L4 GBM, full set | A | 0.5381 | 0.2631 | 0.5564 |
| AMZN | event | 1s | L4 GBM, full set | B | 0.4939 | 0.2130 | 0.5621 |
| AMZN | event | 1s | L4 GBM, full set | C | 0.4916 | 0.2057 | 0.5621 |
| AMZN | event | 1s | L4 GBM, full set | D | 0.4805 | 0.2359 | 0.5621 |
| AMZN | event | 2s | L0 zero | A | 0.3333 | n/a | 0.4538 |
| AMZN | event | 2s | L0 zero | B | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L0 zero | C | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L0 zero | D | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L1 sign(queue imbalance) | A | 0.3449 | 0.0779 | 0.4538 |
| AMZN | event | 2s | L1 sign(queue imbalance) | B | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L1 sign(queue imbalance) | C | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L1 sign(queue imbalance) | D | 0.3726 | 0.0615 | 0.4617 |
| AMZN | event | 2s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.4538 |
| AMZN | event | 2s | L1m sign(microprice - mid) | B | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L1m sign(microprice - mid) | C | 0.3333 | n/a | 0.4617 |
| AMZN | event | 2s | L1m sign(microprice - mid) | D | 0.3696 | 0.0654 | 0.4617 |
| AMZN | event | 2s | L2 OLS on OFI | A | 0.3630 | 0.1167 | 0.4538 |
| AMZN | event | 2s | L2 OLS on OFI | B | 0.3965 | 0.1146 | 0.4617 |
| AMZN | event | 2s | L2 OLS on OFI | C | 0.3979 | 0.1150 | 0.4617 |
| AMZN | event | 2s | L2 OLS on OFI | D | 0.3828 | 0.1402 | 0.4617 |
| AMZN | event | 2s | L3 ridge, full set | A | 0.4811 | 0.2195 | 0.4538 |
| AMZN | event | 2s | L3 ridge, full set | B | 0.4217 | 0.1873 | 0.4617 |
| AMZN | event | 2s | L3 ridge, full set | C | 0.4256 | 0.1634 | 0.4617 |
| AMZN | event | 2s | L3 ridge, full set | D | 0.4660 | 0.2127 | 0.4617 |
| AMZN | event | 2s | L4 GBM, full set | A | 0.5118 | 0.2393 | 0.4538 |
| AMZN | event | 2s | L4 GBM, full set | B | 0.4656 | 0.1812 | 0.4617 |
| AMZN | event | 2s | L4 GBM, full set | C | 0.4661 | 0.1818 | 0.4617 |
| AMZN | event | 2s | L4 GBM, full set | D | 0.4707 | 0.2101 | 0.4617 |
| AMZN | event | 5s | L0 zero | A | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L0 zero | B | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L0 zero | C | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L0 zero | D | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L1 sign(queue imbalance) | A | 0.3583 | 0.1069 | 0.3687 |
| AMZN | event | 5s | L1 sign(queue imbalance) | B | 0.3532 | 0.0825 | 0.3687 |
| AMZN | event | 5s | L1 sign(queue imbalance) | C | 0.3532 | 0.0825 | 0.3687 |
| AMZN | event | 5s | L1 sign(queue imbalance) | D | 0.3714 | 0.0794 | 0.3687 |
| AMZN | event | 5s | L1m sign(microprice - mid) | A | 0.3333 | n/a | 0.3687 |
| AMZN | event | 5s | L1m sign(microprice - mid) | B | 0.3381 | 0.0407 | 0.3687 |
| AMZN | event | 5s | L1m sign(microprice - mid) | C | 0.3381 | 0.0407 | 0.3687 |
| AMZN | event | 5s | L1m sign(microprice - mid) | D | 0.3625 | 0.0677 | 0.3687 |
| AMZN | event | 5s | L2 OLS on OFI | A | 0.3625 | 0.1006 | 0.3687 |
| AMZN | event | 5s | L2 OLS on OFI | B | 0.3866 | 0.1195 | 0.3687 |
| AMZN | event | 5s | L2 OLS on OFI | C | 0.3861 | 0.1176 | 0.3687 |
| AMZN | event | 5s | L2 OLS on OFI | D | 0.3776 | 0.1187 | 0.3687 |
| AMZN | event | 5s | L3 ridge, full set | A | 0.4631 | 0.1998 | 0.3687 |
| AMZN | event | 5s | L3 ridge, full set | B | 0.4109 | 0.1336 | 0.3687 |
| AMZN | event | 5s | L3 ridge, full set | C | 0.4130 | 0.1385 | 0.3687 |
| AMZN | event | 5s | L3 ridge, full set | D | 0.4431 | 0.1815 | 0.3687 |
| AMZN | event | 5s | L4 GBM, full set | A | 0.4923 | 0.2559 | 0.3687 |
| AMZN | event | 5s | L4 GBM, full set | B | 0.4247 | 0.1508 | 0.3687 |
| AMZN | event | 5s | L4 GBM, full set | C | 0.4227 | 0.1482 | 0.3687 |
| AMZN | event | 5s | L4 GBM, full set | D | 0.4477 | 0.1829 | 0.3687 |
| AMZN | event | 10s | L0 zero | A | 0.3333 | n/a | 0.4304 |
| AMZN | event | 10s | L0 zero | B | 0.3333 | n/a | 0.4243 |
| AMZN | event | 10s | L0 zero | C | 0.3333 | n/a | 0.4243 |
| AMZN | event | 10s | L0 zero | D | 0.3333 | n/a | 0.4243 |
| AMZN | event | 10s | L1 sign(queue imbalance) | A | 0.3786 | 0.0902 | 0.4304 |
| AMZN | event | 10s | L1 sign(queue imbalance) | B | 0.3426 | 0.0813 | 0.4243 |
| AMZN | event | 10s | L1 sign(queue imbalance) | C | 0.3629 | 0.0945 | 0.4243 |
| AMZN | event | 10s | L1 sign(queue imbalance) | D | 0.3689 | 0.0867 | 0.4243 |
| AMZN | event | 10s | L1m sign(microprice - mid) | A | 0.3465 | 0.0722 | 0.4304 |
| AMZN | event | 10s | L1m sign(microprice - mid) | B | 0.3374 | 0.0421 | 0.4243 |
| AMZN | event | 10s | L1m sign(microprice - mid) | C | 0.3374 | 0.0421 | 0.4243 |
| AMZN | event | 10s | L1m sign(microprice - mid) | D | 0.3605 | 0.0755 | 0.4243 |
| AMZN | event | 10s | L2 OLS on OFI | A | 0.3771 | 0.0962 | 0.4304 |
| AMZN | event | 10s | L2 OLS on OFI | B | 0.3566 | 0.0547 | 0.4243 |
| AMZN | event | 10s | L2 OLS on OFI | C | 0.3682 | 0.0666 | 0.4243 |
| AMZN | event | 10s | L2 OLS on OFI | D | 0.3613 | 0.0882 | 0.4243 |
| AMZN | event | 10s | L3 ridge, full set | A | 0.4352 | 0.1629 | 0.4304 |
| AMZN | event | 10s | L3 ridge, full set | B | 0.3798 | 0.0989 | 0.4243 |
| AMZN | event | 10s | L3 ridge, full set | C | 0.3776 | 0.0935 | 0.4243 |
| AMZN | event | 10s | L3 ridge, full set | D | 0.4025 | 0.1263 | 0.4243 |
| AMZN | event | 10s | L4 GBM, full set | A | 0.4767 | 0.2699 | 0.4304 |
| AMZN | event | 10s | L4 GBM, full set | B | 0.3862 | 0.1006 | 0.4243 |
| AMZN | event | 10s | L4 GBM, full set | C | 0.3827 | 0.1023 | 0.4243 |
| AMZN | event | 10s | L4 GBM, full set | D | 0.4045 | 0.1282 | 0.4243 |

## Per-fold detail under protocol C (the honest ladder)

| ticker | clock | h | rung | per-fold IC | per-fold R2 |
|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | 0.1286, 0.1112, 0.1112, 0.0879, 0.1429 | 0.01473, 0.00902, 0.00765, 0.00440, 0.01109 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | 0.1029, 0.0997, 0.1164, 0.1121, 0.1446 | 0.00807, 0.00328, 0.00232, 0.00232, 0.00414 |
| INTC | calendar | 1s | L2 OLS on OFI | 0.0437, 0.1073, 0.0687, 0.0681, 0.1074 | -0.00392, 0.00054, 0.00262, 0.00139, 0.00357 |
| INTC | calendar | 1s | L3 ridge, full set | 0.1246, 0.1488, 0.1495, 0.1440, 0.1810 | 0.00924, 0.02113, 0.02254, 0.03003, 0.02445 |
| INTC | calendar | 1s | L4 GBM, full set | 0.1100, 0.1310, 0.1280, 0.1230, 0.1659 | 0.01308, 0.01806, 0.01391, 0.02673, 0.03492 |
| INTC | calendar | 2s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | calendar | 2s | L1 sign(queue imbalance) | 0.1510, 0.1407, 0.1384, 0.1184, 0.1765 | 0.01711, 0.01160, 0.00929, 0.00848, 0.01739 |
| INTC | calendar | 2s | L1m sign(microprice - mid) | 0.1293, 0.1106, 0.1390, 0.1367, 0.1855 | 0.00994, 0.00444, 0.00312, 0.00371, 0.00631 |
| INTC | calendar | 2s | L2 OLS on OFI | 0.0673, 0.1128, 0.0842, 0.0722, 0.1445 | -0.00837, 0.00045, 0.00308, 0.00266, 0.00705 |
| INTC | calendar | 2s | L3 ridge, full set | 0.1667, 0.1942, 0.1843, 0.1634, 0.2224 | 0.02338, 0.03352, 0.02569, 0.03792, 0.03417 |
| INTC | calendar | 2s | L4 GBM, full set | 0.1386, 0.1700, 0.1436, 0.1463, 0.1711 | 0.01847, 0.02074, 0.01088, 0.02854, 0.03291 |
| INTC | calendar | 5s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | 0.1855, 0.1759, 0.1890, 0.1670, 0.2237 | 0.02186, 0.01636, 0.02409, 0.01577, 0.02661 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | 0.1523, 0.1252, 0.1251, 0.1816, 0.1722 | 0.00544, 0.00540, -0.00072, 0.00581, 0.00732 |
| INTC | calendar | 5s | L2 OLS on OFI | 0.0916, 0.0866, 0.0807, 0.0695, 0.1511 | -0.00283, 0.00138, 0.00395, -0.00156, 0.00598 |
| INTC | calendar | 5s | L3 ridge, full set | 0.2063, 0.2414, 0.2282, 0.2069, 0.2584 | 0.03834, 0.05290, 0.03579, 0.04687, 0.03484 |
| INTC | calendar | 5s | L4 GBM, full set | 0.1397, 0.1852, 0.1826, 0.1669, 0.1746 | 0.01590, 0.01595, 0.01630, 0.02751, 0.01934 |
| INTC | calendar | 10s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | calendar | 10s | L1 sign(queue imbalance) | 0.2205, 0.1626, 0.2183, 0.1969, 0.2534 | 0.03767, 0.01803, 0.03454, 0.02048, 0.03357 |
| INTC | calendar | 10s | L1m sign(microprice - mid) | 0.1709, 0.1418, 0.1021, 0.2064, 0.1652 | 0.02505, 0.00934, -0.00646, 0.00826, 0.00810 |
| INTC | calendar | 10s | L2 OLS on OFI | 0.0873, 0.0889, 0.0815, 0.0895, 0.1437 | -0.00650, 0.00093, 0.00465, -0.00517, 0.00060 |
| INTC | calendar | 10s | L3 ridge, full set | 0.1920, 0.2521, 0.2307, 0.2277, 0.2554 | 0.03808, 0.06061, 0.03902, 0.03497, 0.03394 |
| INTC | calendar | 10s | L4 GBM, full set | 0.1698, 0.2167, 0.1124, 0.1670, 0.0953 | 0.02093, 0.03466, 0.00109, 0.01530, 0.00345 |
| INTC | event | 1s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | event | 1s | L1 sign(queue imbalance) | 0.2676, 0.3172, 0.2226, 0.3062, 0.3760 | 0.04743, 0.07621, 0.02888, 0.08003, 0.10307 |
| INTC | event | 1s | L1m sign(microprice - mid) | 0.1142, 0.1458, 0.1858, 0.1245, 0.1932 | -0.03317, 0.02403, 0.04056, -0.02130, 0.01402 |
| INTC | event | 1s | L2 OLS on OFI | 0.2676, 0.2874, 0.2389, 0.4841, 0.4119 | -0.05861, 0.01029, 0.06435, 0.11395, 0.05074 |
| INTC | event | 1s | L3 ridge, full set | 0.1637, 0.3347, 0.3097, 0.4700, 0.5444 | -0.31261, 0.06199, 0.12589, 0.27034, 0.21038 |
| INTC | event | 1s | L4 GBM, full set | 0.4110, 0.5096, 0.3624, 0.3589, 0.5789 | 0.15765, 0.15884, 0.16273, 0.16666, 0.31029 |
| INTC | event | 2s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | event | 2s | L1 sign(queue imbalance) | 0.2574, 0.2768, 0.2316, 0.3191, 0.3878 | 0.03357, 0.05261, 0.04676, 0.06959, 0.10448 |
| INTC | event | 2s | L1m sign(microprice - mid) | 0.1305, 0.1296, 0.1999, 0.1535, 0.2195 | -0.07605, 0.01282, 0.04795, 0.02194, 0.03524 |
| INTC | event | 2s | L2 OLS on OFI | 0.2247, -0.2666, 0.2380, 0.3455, 0.4156 | -0.25704, -0.01065, 0.08006, 0.10425, 0.07099 |
| INTC | event | 2s | L3 ridge, full set | 0.1266, 0.2725, 0.3225, 0.4580, 0.4972 | -0.80172, 0.02811, 0.10169, 0.28712, 0.15957 |
| INTC | event | 2s | L4 GBM, full set | 0.3328, 0.3758, 0.3039, 0.4213, 0.5577 | 0.03269, 0.09661, 0.04878, 0.22767, 0.23444 |
| INTC | event | 5s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | event | 5s | L1 sign(queue imbalance) | 0.2376, 0.2222, 0.2732, 0.3303, 0.3384 | 0.04241, 0.03736, 0.04903, 0.07466, 0.07473 |
| INTC | event | 5s | L1m sign(microprice - mid) | 0.1464, -0.1059, 0.2242, 0.1762, 0.2122 | -0.04693, -0.00848, 0.05044, 0.01557, 0.02192 |
| INTC | event | 5s | L2 OLS on OFI | 0.2090, -0.1325, 0.2592, 0.3427, 0.3671 | -0.12755, -0.00153, 0.04218, 0.09688, 0.07168 |
| INTC | event | 5s | L3 ridge, full set | 0.2033, 0.2673, 0.3651, 0.4078, 0.4262 | -0.03788, 0.05830, 0.10942, 0.28076, 0.14072 |
| INTC | event | 5s | L4 GBM, full set | 0.2307, 0.2958, 0.3675, 0.2238, 0.5042 | 0.04801, 0.07874, 0.08855, 0.04850, 0.15532 |
| INTC | event | 10s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| INTC | event | 10s | L1 sign(queue imbalance) | 0.2283, 0.1076, 0.2548, 0.3325, 0.3173 | 0.04589, 0.00655, 0.01892, 0.05511, 0.05892 |
| INTC | event | 10s | L1m sign(microprice - mid) | 0.1926, 0.0942, 0.2033, 0.1905, 0.1916 | 0.03806, 0.03596, 0.01548, 0.02773, 0.02531 |
| INTC | event | 10s | L2 OLS on OFI | 0.1732, -0.0197, 0.1889, 0.3762, 0.2845 | -0.05933, 0.00051, 0.03348, 0.02704, 0.02808 |
| INTC | event | 10s | L3 ridge, full set | 0.0512, 0.1676, 0.2990, 0.2843, 0.3189 | -0.15714, 0.03031, 0.09156, 0.14835, 0.06196 |
| INTC | event | 10s | L4 GBM, full set | 0.0049, 0.2159, 0.2224, 0.0862, 0.3728 | 0.00224, 0.07607, 0.01168, 0.00468, 0.06719 |
| AMZN | calendar | 1s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | 0.0534, 0.0722, 0.0769, 0.0643, 0.1315 | -0.00596, 0.00044, 0.00371, 0.00155, 0.00578 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | 0.0347, -0.0577, 0.0665, 0.0613, 0.1125 | -0.00721, -0.00308, 0.00311, 0.00148, 0.00529 |
| AMZN | calendar | 1s | L2 OLS on OFI | -0.0535, 0.0569, 0.0378, 0.0849, 0.0990 | -0.00544, 0.00076, -0.00197, 0.00647, -0.00679 |
| AMZN | calendar | 1s | L3 ridge, full set | 0.0617, 0.0768, 0.1157, 0.1008, 0.1689 | -0.00354, 0.00153, 0.00126, 0.00874, 0.02842 |
| AMZN | calendar | 1s | L4 GBM, full set | 0.0184, 0.0740, 0.0946, 0.0882, 0.1552 | 0.00015, 0.00915, 0.01482, 0.00743, 0.03261 |
| AMZN | calendar | 2s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | calendar | 2s | L1 sign(queue imbalance) | 0.0620, -0.0958, 0.0940, 0.0798, 0.1641 | -0.00269, -0.00089, 0.00697, 0.00066, 0.00750 |
| AMZN | calendar | 2s | L1m sign(microprice - mid) | 0.0423, -0.0817, 0.0745, 0.0721, 0.1452 | -0.00151, -0.00619, 0.00533, -0.00079, 0.00673 |
| AMZN | calendar | 2s | L2 OLS on OFI | -0.0625, 0.0723, 0.0630, 0.0772, 0.1235 | -0.01005, 0.00080, -0.00507, 0.00600, -0.00282 |
| AMZN | calendar | 2s | L3 ridge, full set | 0.0735, 0.0757, 0.1293, 0.1135, 0.2023 | -0.00000, 0.00097, 0.00125, 0.00843, 0.02900 |
| AMZN | calendar | 2s | L4 GBM, full set | 0.0397, 0.0728, 0.0962, 0.0900, 0.2051 | 0.00037, 0.00716, 0.00228, 0.00491, 0.03448 |
| AMZN | calendar | 5s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | 0.0473, -0.1110, 0.1063, 0.0879, 0.1478 | -0.00002, -0.00747, 0.00543, -0.00156, 0.00315 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | 0.0276, -0.0917, 0.0865, 0.0641, 0.1317 | -0.00209, -0.01815, 0.00616, -0.00754, 0.00147 |
| AMZN | calendar | 5s | L2 OLS on OFI | -0.0617, 0.0754, 0.0940, 0.0697, 0.1201 | -0.05994, 0.00015, 0.00693, 0.00374, -0.00350 |
| AMZN | calendar | 5s | L3 ridge, full set | 0.0436, 0.0579, 0.1310, 0.1180, 0.2079 | -0.01277, 0.00006, 0.01853, -0.00498, 0.01790 |
| AMZN | calendar | 5s | L4 GBM, full set | -0.0253, 0.0191, 0.1048, 0.1133, 0.1708 | -0.01033, 0.00047, 0.00517, 0.00832, 0.02695 |
| AMZN | calendar | 10s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | calendar | 10s | L1 sign(queue imbalance) | -0.0360, -0.0880, 0.1032, 0.0894, 0.0977 | -0.00943, -0.00800, 0.00436, 0.00009, 0.00347 |
| AMZN | calendar | 10s | L1m sign(microprice - mid) | -0.0171, -0.0938, 0.0851, 0.0643, 0.0696 | -0.00030, -0.02108, 0.00494, -0.00457, 0.00266 |
| AMZN | calendar | 10s | L2 OLS on OFI | -0.0650, 0.0679, 0.1086, 0.0733, 0.1076 | -0.13764, 0.00046, 0.00160, 0.00407, -0.00407 |
| AMZN | calendar | 10s | L3 ridge, full set | 0.0294, -0.0718, 0.1230, 0.0702, 0.1550 | -0.22340, -0.00054, 0.01269, -0.00286, 0.01175 |
| AMZN | calendar | 10s | L4 GBM, full set | -0.0066, -0.0654, 0.0661, 0.0745, 0.1231 | -0.00771, -0.00677, 0.00430, 0.00611, 0.00266 |
| AMZN | event | 1s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | event | 1s | L1 sign(queue imbalance) | 0.1205, 0.1442, 0.1674, 0.1117, 0.1963 | 0.00187, 0.00609, 0.01425, 0.00521, 0.01706 |
| AMZN | event | 1s | L1m sign(microprice - mid) | 0.0916, 0.1131, 0.1542, 0.0781, 0.1708 | 0.00473, 0.00092, 0.00824, -0.00165, 0.00949 |
| AMZN | event | 1s | L2 OLS on OFI | 0.2255, 0.3373, 0.2627, 0.2893, 0.3018 | -0.07869, 0.00846, 0.02208, 0.11397, -0.30525 |
| AMZN | event | 1s | L3 ridge, full set | 0.1603, 0.3237, 0.3170, 0.3077, 0.3806 | -0.23902, 0.01283, 0.03835, 0.11310, 0.18600 |
| AMZN | event | 1s | L4 GBM, full set | 0.2365, 0.3146, 0.3421, 0.3699, 0.4059 | 0.05274, 0.14375, 0.12741, 0.17685, 0.21148 |
| AMZN | event | 2s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | event | 2s | L1 sign(queue imbalance) | 0.1176, 0.1452, 0.1449, 0.1168, 0.2199 | 0.00412, 0.00457, 0.01255, 0.00289, 0.01734 |
| AMZN | event | 2s | L1m sign(microprice - mid) | 0.0938, 0.1185, 0.1441, 0.0909, 0.1985 | 0.00262, 0.00081, 0.00804, -0.00437, 0.00925 |
| AMZN | event | 2s | L2 OLS on OFI | -0.2129, 0.3130, 0.2590, 0.2641, 0.3081 | -0.01665, 0.00804, -0.03200, 0.09480, -0.33839 |
| AMZN | event | 2s | L3 ridge, full set | 0.2098, 0.2955, 0.2932, 0.2959, 0.3866 | -0.03453, 0.01805, -0.03686, 0.10629, 0.17976 |
| AMZN | event | 2s | L4 GBM, full set | 0.2514, 0.3583, 0.3223, 0.3613, 0.4053 | 0.06711, 0.14018, 0.08042, 0.16295, 0.19612 |
| AMZN | event | 5s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | event | 5s | L1 sign(queue imbalance) | -0.0832, 0.1327, 0.1588, 0.0994, 0.2141 | -0.00402, 0.00435, 0.01702, -0.00379, 0.01360 |
| AMZN | event | 5s | L1m sign(microprice - mid) | -0.0628, -0.1127, 0.1420, 0.0594, 0.1897 | -0.00171, -0.00043, 0.00949, -0.01695, 0.00356 |
| AMZN | event | 5s | L2 OLS on OFI | -0.2093, 0.2801, 0.2933, 0.2574, 0.3009 | -0.18140, 0.00261, 0.04166, 0.09710, -0.25096 |
| AMZN | event | 5s | L3 ridge, full set | 0.1174, 0.2230, 0.2825, 0.2967, 0.3424 | -0.00524, 0.01355, 0.06710, 0.11490, 0.11272 |
| AMZN | event | 5s | L4 GBM, full set | 0.1764, 0.3153, 0.3241, 0.3676, 0.3772 | 0.01822, 0.08846, 0.11601, 0.16096, 0.14836 |
| AMZN | event | 10s | L0 zero | n/a, n/a, n/a, n/a, n/a | 0.00000, 0.00000, 0.00000, 0.00000, 0.00000 |
| AMZN | event | 10s | L1 sign(queue imbalance) | -0.0829, 0.1233, 0.1413, 0.1043, 0.1664 | -0.00773, 0.00226, 0.01122, 0.00420, 0.01615 |
| AMZN | event | 10s | L1m sign(microprice - mid) | -0.0575, 0.1199, 0.1231, 0.0584, 0.1441 | -0.00826, 0.00059, 0.00455, -0.00819, 0.00743 |
| AMZN | event | 10s | L2 OLS on OFI | -0.2286, 0.2317, 0.2575, 0.2264, 0.2578 | -0.35694, 0.00432, 0.01979, 0.06247, -0.21499 |
| AMZN | event | 10s | L3 ridge, full set | 0.0589, 0.1922, 0.2122, 0.2435, 0.2372 | -0.10602, 0.00191, 0.01875, 0.08921, 0.02626 |
| AMZN | event | 10s | L4 GBM, full set | 0.1121, 0.2354, 0.1988, 0.2642, 0.2886 | 0.00778, 0.03824, 0.00785, 0.10319, 0.07312 |

## Embargo sensitivity (protocol C)

The frozen headline is E = max(60s, 10h); these rows exist to show the result is not an artifact of that choice.

| ticker | clock | h | embargo (s) | rung | R2 mean | IC mean | n_train median |
|---|---|---|---|---|---|---|---|
| AMZN | calendar | 10s | 0 | L0 zero | 0.00000 | n/a | 81903 |
| AMZN | calendar | 10s | 0 | L1 sign(queue imbalance) | 0.00100 | 0.03326 | 81903 |
| AMZN | calendar | 10s | 0 | L1m sign(microprice - mid) | -0.00205 | 0.02845 | 81903 |
| AMZN | calendar | 10s | 0 | L2 OLS on OFI | -0.01569 | 0.05847 | 81903 |
| AMZN | calendar | 10s | 0 | L3 ridge, full set | -0.00651 | 0.06430 | 81903 |
| AMZN | calendar | 10s | 0 | L4 GBM, full set | 0.00047 | 0.02595 | 81903 |
| AMZN | calendar | 10s | 1 | L0 zero | 0.00000 | n/a | 81893 |
| AMZN | calendar | 10s | 1 | L1 sign(queue imbalance) | 0.00101 | 0.03326 | 81893 |
| AMZN | calendar | 10s | 1 | L1m sign(microprice - mid) | -0.00205 | 0.02845 | 81893 |
| AMZN | calendar | 10s | 1 | L2 OLS on OFI | -0.01543 | 0.05847 | 81893 |
| AMZN | calendar | 10s | 1 | L3 ridge, full set | -0.00704 | 0.06428 | 81893 |
| AMZN | calendar | 10s | 1 | L4 GBM, full set | 0.00055 | 0.02962 | 81893 |
| AMZN | calendar | 10s | 10 | L0 zero | 0.00000 | n/a | 81803 |
| AMZN | calendar | 10s | 10 | L1 sign(queue imbalance) | 0.00094 | 0.03326 | 81803 |
| AMZN | calendar | 10s | 10 | L1m sign(microprice - mid) | -0.00228 | 0.02845 | 81803 |
| AMZN | calendar | 10s | 10 | L2 OLS on OFI | -0.01710 | 0.05847 | 81803 |
| AMZN | calendar | 10s | 10 | L3 ridge, full set | -0.01092 | 0.06425 | 81803 |
| AMZN | calendar | 10s | 10 | L4 GBM, full set | 0.00013 | 0.03984 | 81803 |
| AMZN | calendar | 10s | 60 | L0 zero | 0.00000 | n/a | 81419 |
| AMZN | calendar | 10s | 60 | L1 sign(queue imbalance) | 0.00056 | 0.03326 | 81419 |
| AMZN | calendar | 10s | 60 | L1m sign(microprice - mid) | -0.00290 | 0.02845 | 81419 |
| AMZN | calendar | 10s | 60 | L2 OLS on OFI | -0.02521 | 0.05847 | 81419 |
| AMZN | calendar | 10s | 60 | L3 ridge, full set | -0.00039 | 0.06474 | 81419 |
| AMZN | calendar | 10s | 60 | L4 GBM, full set | 0.00296 | 0.04619 | 81419 |
| AMZN | calendar | 10s | 300 | L0 zero | 0.00000 | n/a | 79738 |
| AMZN | calendar | 10s | 300 | L1 sign(queue imbalance) | -0.00147 | 0.03326 | 79738 |
| AMZN | calendar | 10s | 300 | L1m sign(microprice - mid) | -0.00214 | 0.02163 | 79738 |
| AMZN | calendar | 10s | 300 | L2 OLS on OFI | 0.00116 | 0.08449 | 79738 |
| AMZN | calendar | 10s | 300 | L3 ridge, full set | 0.00511 | 0.05943 | 79738 |
| AMZN | calendar | 10s | 300 | L4 GBM, full set | 0.00463 | 0.08283 | 79738 |
| AMZN | calendar | 1s | 0 | L0 zero | 0.00000 | n/a | 81990 |
| AMZN | calendar | 1s | 0 | L1 sign(queue imbalance) | 0.00214 | 0.07966 | 81990 |
| AMZN | calendar | 1s | 0 | L1m sign(microprice - mid) | 0.00068 | 0.04348 | 81990 |
| AMZN | calendar | 1s | 0 | L2 OLS on OFI | -0.00248 | 0.04502 | 81990 |
| AMZN | calendar | 1s | 0 | L3 ridge, full set | 0.00815 | 0.10419 | 81990 |
| AMZN | calendar | 1s | 0 | L4 GBM, full set | 0.01186 | 0.08755 | 81990 |
| AMZN | calendar | 1s | 1 | L0 zero | 0.00000 | n/a | 81980 |
| AMZN | calendar | 1s | 1 | L1 sign(queue imbalance) | 0.00216 | 0.07966 | 81980 |
| AMZN | calendar | 1s | 1 | L1m sign(microprice - mid) | 0.00070 | 0.04348 | 81980 |
| AMZN | calendar | 1s | 1 | L2 OLS on OFI | -0.00243 | 0.04502 | 81980 |
| AMZN | calendar | 1s | 1 | L3 ridge, full set | 0.00815 | 0.10417 | 81980 |
| AMZN | calendar | 1s | 1 | L4 GBM, full set | 0.01229 | 0.08753 | 81980 |
| AMZN | calendar | 1s | 10 | L0 zero | 0.00000 | n/a | 81893 |
| AMZN | calendar | 1s | 10 | L1 sign(queue imbalance) | 0.00214 | 0.07966 | 81893 |
| AMZN | calendar | 1s | 10 | L1m sign(microprice - mid) | 0.00078 | 0.04348 | 81893 |
| AMZN | calendar | 1s | 10 | L2 OLS on OFI | -0.00214 | 0.04502 | 81893 |
| AMZN | calendar | 1s | 10 | L3 ridge, full set | 0.00805 | 0.10386 | 81893 |
| AMZN | calendar | 1s | 10 | L4 GBM, full set | 0.01255 | 0.08699 | 81893 |
| AMZN | calendar | 1s | 60 | L0 zero | 0.00000 | n/a | 81481 |
| AMZN | calendar | 1s | 60 | L1 sign(queue imbalance) | 0.00111 | 0.07966 | 81481 |
| AMZN | calendar | 1s | 60 | L1m sign(microprice - mid) | -0.00008 | 0.04348 | 81481 |
| AMZN | calendar | 1s | 60 | L2 OLS on OFI | -0.00140 | 0.04502 | 81481 |
| AMZN | calendar | 1s | 60 | L3 ridge, full set | 0.00728 | 0.10478 | 81481 |
| AMZN | calendar | 1s | 60 | L4 GBM, full set | 0.01283 | 0.08610 | 81481 |
| AMZN | calendar | 1s | 300 | L0 zero | 0.00000 | n/a | 79823 |
| AMZN | calendar | 1s | 300 | L1 sign(queue imbalance) | 0.00252 | 0.07966 | 79823 |
| AMZN | calendar | 1s | 300 | L1m sign(microprice - mid) | 0.00104 | 0.04348 | 79823 |
| AMZN | calendar | 1s | 300 | L2 OLS on OFI | -0.00072 | 0.04502 | 79823 |
| AMZN | calendar | 1s | 300 | L3 ridge, full set | -0.00069 | 0.09799 | 79823 |
| AMZN | calendar | 1s | 300 | L4 GBM, full set | 0.01048 | 0.08743 | 79823 |
| AMZN | calendar | 2s | 0 | L0 zero | 0.00000 | n/a | 81980 |
| AMZN | calendar | 2s | 0 | L1 sign(queue imbalance) | 0.00298 | 0.06081 | 81980 |
| AMZN | calendar | 2s | 0 | L1m sign(microprice - mid) | 0.00083 | 0.03355 | 81980 |
| AMZN | calendar | 2s | 0 | L2 OLS on OFI | -0.00260 | 0.05470 | 81980 |
| AMZN | calendar | 2s | 0 | L3 ridge, full set | 0.00843 | 0.11823 | 81980 |
| AMZN | calendar | 2s | 0 | L4 GBM, full set | 0.01223 | 0.10205 | 81980 |
| AMZN | calendar | 2s | 1 | L0 zero | 0.00000 | n/a | 81970 |
| AMZN | calendar | 2s | 1 | L1 sign(queue imbalance) | 0.00301 | 0.06081 | 81970 |
| AMZN | calendar | 2s | 1 | L1m sign(microprice - mid) | 0.00083 | 0.03355 | 81970 |
| AMZN | calendar | 2s | 1 | L2 OLS on OFI | -0.00268 | 0.05470 | 81970 |
| AMZN | calendar | 2s | 1 | L3 ridge, full set | 0.00842 | 0.11820 | 81970 |
| AMZN | calendar | 2s | 1 | L4 GBM, full set | 0.01232 | 0.10831 | 81970 |
| AMZN | calendar | 2s | 10 | L0 zero | 0.00000 | n/a | 81883 |
| AMZN | calendar | 2s | 10 | L1 sign(queue imbalance) | 0.00308 | 0.06081 | 81883 |
| AMZN | calendar | 2s | 10 | L1m sign(microprice - mid) | 0.00093 | 0.03355 | 81883 |
| AMZN | calendar | 2s | 10 | L2 OLS on OFI | -0.00246 | 0.05470 | 81883 |
| AMZN | calendar | 2s | 10 | L3 ridge, full set | 0.00835 | 0.11784 | 81883 |
| AMZN | calendar | 2s | 10 | L4 GBM, full set | 0.01163 | 0.10407 | 81883 |
| AMZN | calendar | 2s | 60 | L0 zero | 0.00000 | n/a | 81471 |
| AMZN | calendar | 2s | 60 | L1 sign(queue imbalance) | 0.00231 | 0.06081 | 81471 |
| AMZN | calendar | 2s | 60 | L1m sign(microprice - mid) | 0.00071 | 0.05047 | 81471 |
| AMZN | calendar | 2s | 60 | L2 OLS on OFI | -0.00223 | 0.05470 | 81471 |
| AMZN | calendar | 2s | 60 | L3 ridge, full set | 0.00793 | 0.11885 | 81471 |
| AMZN | calendar | 2s | 60 | L4 GBM, full set | 0.00984 | 0.10076 | 81471 |
| AMZN | calendar | 2s | 300 | L0 zero | 0.00000 | n/a | 79818 |
| AMZN | calendar | 2s | 300 | L1 sign(queue imbalance) | 0.00309 | 0.03603 | 79818 |
| AMZN | calendar | 2s | 300 | L1m sign(microprice - mid) | 0.00141 | 0.05047 | 79818 |
| AMZN | calendar | 2s | 300 | L2 OLS on OFI | -0.00043 | 0.07969 | 79818 |
| AMZN | calendar | 2s | 300 | L3 ridge, full set | 0.00418 | 0.09360 | 79818 |
| AMZN | calendar | 2s | 300 | L4 GBM, full set | 0.01075 | 0.09921 | 79818 |
| AMZN | calendar | 5s | 0 | L0 zero | 0.00000 | n/a | 81953 |
| AMZN | calendar | 5s | 0 | L1 sign(queue imbalance) | 0.00031 | 0.03674 | 81953 |
| AMZN | calendar | 5s | 0 | L1m sign(microprice - mid) | -0.00372 | 0.03259 | 81953 |
| AMZN | calendar | 5s | 0 | L2 OLS on OFI | -0.00609 | 0.05952 | 81953 |
| AMZN | calendar | 5s | 0 | L3 ridge, full set | 0.00152 | 0.11063 | 81953 |
| AMZN | calendar | 5s | 0 | L4 GBM, full set | 0.00968 | 0.06813 | 81953 |
| AMZN | calendar | 5s | 1 | L0 zero | 0.00000 | n/a | 81943 |
| AMZN | calendar | 5s | 1 | L1 sign(queue imbalance) | 0.00033 | 0.03674 | 81943 |
| AMZN | calendar | 5s | 1 | L1m sign(microprice - mid) | -0.00368 | 0.03259 | 81943 |
| AMZN | calendar | 5s | 1 | L2 OLS on OFI | -0.00605 | 0.05952 | 81943 |
| AMZN | calendar | 5s | 1 | L3 ridge, full set | 0.00142 | 0.11059 | 81943 |
| AMZN | calendar | 5s | 1 | L4 GBM, full set | 0.01059 | 0.07208 | 81943 |
| AMZN | calendar | 5s | 10 | L0 zero | 0.00000 | n/a | 81853 |
| AMZN | calendar | 5s | 10 | L1 sign(queue imbalance) | 0.00037 | 0.03674 | 81853 |
| AMZN | calendar | 5s | 10 | L1m sign(microprice - mid) | -0.00369 | 0.03259 | 81853 |
| AMZN | calendar | 5s | 10 | L2 OLS on OFI | -0.00750 | 0.05952 | 81853 |
| AMZN | calendar | 5s | 10 | L3 ridge, full set | 0.00036 | 0.11044 | 81853 |
| AMZN | calendar | 5s | 10 | L4 GBM, full set | 0.01052 | 0.07949 | 81853 |
| AMZN | calendar | 5s | 60 | L0 zero | 0.00000 | n/a | 81441 |
| AMZN | calendar | 5s | 60 | L1 sign(queue imbalance) | -0.00009 | 0.05566 | 81441 |
| AMZN | calendar | 5s | 60 | L1m sign(microprice - mid) | -0.00403 | 0.04362 | 81441 |
| AMZN | calendar | 5s | 60 | L2 OLS on OFI | -0.01052 | 0.05952 | 81441 |
| AMZN | calendar | 5s | 60 | L3 ridge, full set | 0.00375 | 0.11168 | 81441 |
| AMZN | calendar | 5s | 60 | L4 GBM, full set | 0.00611 | 0.07651 | 81441 |
| AMZN | calendar | 5s | 300 | L0 zero | 0.00000 | n/a | 79788 |
| AMZN | calendar | 5s | 300 | L1 sign(queue imbalance) | 0.00015 | 0.03674 | 79788 |
| AMZN | calendar | 5s | 300 | L1m sign(microprice - mid) | -0.00226 | 0.03259 | 79788 |
| AMZN | calendar | 5s | 300 | L2 OLS on OFI | 0.00189 | 0.08418 | 79788 |
| AMZN | calendar | 5s | 300 | L3 ridge, full set | -0.01076 | 0.10024 | 79788 |
| AMZN | calendar | 5s | 300 | L4 GBM, full set | 0.01071 | 0.09481 | 79788 |
| AMZN | event | 10s | 0 | L0 zero | 0.00000 | n/a | 11772 |
| AMZN | event | 10s | 0 | L1 sign(queue imbalance) | 0.00571 | 0.09048 | 11772 |
| AMZN | event | 10s | 0 | L1m sign(microprice - mid) | -0.00162 | -0.01959 | 11772 |
| AMZN | event | 10s | 0 | L2 OLS on OFI | -0.08975 | 0.14895 | 11772 |
| AMZN | event | 10s | 0 | L3 ridge, full set | 0.02022 | 0.19121 | 11772 |
| AMZN | event | 10s | 0 | L4 GBM, full set | 0.05391 | 0.22503 | 11772 |
| AMZN | event | 10s | 1 | L0 zero | 0.00000 | n/a | 11770 |
| AMZN | event | 10s | 1 | L1 sign(queue imbalance) | 0.00570 | 0.09048 | 11770 |
| AMZN | event | 10s | 1 | L1m sign(microprice - mid) | -0.00167 | -0.01959 | 11770 |
| AMZN | event | 10s | 1 | L2 OLS on OFI | -0.08961 | 0.14895 | 11770 |
| AMZN | event | 10s | 1 | L3 ridge, full set | 0.02046 | 0.19117 | 11770 |
| AMZN | event | 10s | 1 | L4 GBM, full set | 0.05195 | 0.22047 | 11770 |
| AMZN | event | 10s | 10 | L0 zero | 0.00000 | n/a | 11755 |
| AMZN | event | 10s | 10 | L1 sign(queue imbalance) | 0.00560 | 0.09048 | 11755 |
| AMZN | event | 10s | 10 | L1m sign(microprice - mid) | -0.00168 | -0.01959 | 11755 |
| AMZN | event | 10s | 10 | L2 OLS on OFI | -0.08966 | 0.14895 | 11755 |
| AMZN | event | 10s | 10 | L3 ridge, full set | 0.01986 | 0.19088 | 11755 |
| AMZN | event | 10s | 10 | L4 GBM, full set | 0.05603 | 0.22386 | 11755 |
| AMZN | event | 10s | 60 | L0 zero | 0.00000 | n/a | 11688 |
| AMZN | event | 10s | 60 | L1 sign(queue imbalance) | 0.00588 | 0.09048 | 11688 |
| AMZN | event | 10s | 60 | L1m sign(microprice - mid) | -0.00024 | 0.07760 | 11688 |
| AMZN | event | 10s | 60 | L2 OLS on OFI | -0.09074 | 0.14895 | 11688 |
| AMZN | event | 10s | 60 | L3 ridge, full set | 0.01614 | 0.19019 | 11688 |
| AMZN | event | 10s | 60 | L4 GBM, full set | 0.04962 | 0.22578 | 11688 |
| AMZN | event | 10s | 300 | L0 zero | 0.00000 | n/a | 11439 |
| AMZN | event | 10s | 300 | L1 sign(queue imbalance) | 0.00259 | 0.09048 | 11439 |
| AMZN | event | 10s | 300 | L1m sign(microprice - mid) | -0.00136 | 0.07760 | 11439 |
| AMZN | event | 10s | 300 | L2 OLS on OFI | -0.03566 | 0.14895 | 11439 |
| AMZN | event | 10s | 300 | L3 ridge, full set | 0.02515 | 0.17087 | 11439 |
| AMZN | event | 10s | 300 | L4 GBM, full set | 0.05750 | 0.23631 | 11439 |
| AMZN | event | 1s | 0 | L0 zero | 0.00000 | n/a | 11779 |
| AMZN | event | 1s | 0 | L1 sign(queue imbalance) | 0.00972 | 0.14802 | 11779 |
| AMZN | event | 1s | 0 | L1m sign(microprice - mid) | 0.00440 | 0.07633 | 11779 |
| AMZN | event | 1s | 0 | L2 OLS on OFI | -0.04742 | 0.28334 | 11779 |
| AMZN | event | 1s | 0 | L3 ridge, full set | 0.03018 | 0.29758 | 11779 |
| AMZN | event | 1s | 0 | L4 GBM, full set | 0.13756 | 0.33743 | 11779 |
| AMZN | event | 1s | 1 | L0 zero | 0.00000 | n/a | 11778 |
| AMZN | event | 1s | 1 | L1 sign(queue imbalance) | 0.00972 | 0.14802 | 11778 |
| AMZN | event | 1s | 1 | L1m sign(microprice - mid) | 0.00440 | 0.07633 | 11778 |
| AMZN | event | 1s | 1 | L2 OLS on OFI | -0.04742 | 0.28334 | 11778 |
| AMZN | event | 1s | 1 | L3 ridge, full set | 0.03018 | 0.29758 | 11778 |
| AMZN | event | 1s | 1 | L4 GBM, full set | 0.13751 | 0.33742 | 11778 |
| AMZN | event | 1s | 10 | L0 zero | 0.00000 | n/a | 11770 |
| AMZN | event | 1s | 10 | L1 sign(queue imbalance) | 0.00974 | 0.14802 | 11770 |
| AMZN | event | 1s | 10 | L1m sign(microprice - mid) | 0.00441 | 0.07633 | 11770 |
| AMZN | event | 1s | 10 | L2 OLS on OFI | -0.04745 | 0.28334 | 11770 |
| AMZN | event | 1s | 10 | L3 ridge, full set | 0.03004 | 0.29771 | 11770 |
| AMZN | event | 1s | 10 | L4 GBM, full set | 0.13890 | 0.34044 | 11770 |
| AMZN | event | 1s | 60 | L0 zero | 0.00000 | n/a | 11706 |
| AMZN | event | 1s | 60 | L1 sign(queue imbalance) | 0.00890 | 0.14802 | 11706 |
| AMZN | event | 1s | 60 | L1m sign(microprice - mid) | 0.00435 | 0.12157 | 11706 |
| AMZN | event | 1s | 60 | L2 OLS on OFI | -0.04789 | 0.28334 | 11706 |
| AMZN | event | 1s | 60 | L3 ridge, full set | 0.02225 | 0.29785 | 11706 |
| AMZN | event | 1s | 60 | L4 GBM, full set | 0.14245 | 0.33378 | 11706 |
| AMZN | event | 1s | 300 | L0 zero | 0.00000 | n/a | 11449 |
| AMZN | event | 1s | 300 | L1 sign(queue imbalance) | 0.01006 | 0.14802 | 11449 |
| AMZN | event | 1s | 300 | L1m sign(microprice - mid) | 0.00485 | 0.12157 | 11449 |
| AMZN | event | 1s | 300 | L2 OLS on OFI | -0.04575 | 0.28334 | 11449 |
| AMZN | event | 1s | 300 | L3 ridge, full set | 0.00842 | 0.29125 | 11449 |
| AMZN | event | 1s | 300 | L4 GBM, full set | 0.13833 | 0.34012 | 11449 |
| AMZN | event | 2s | 0 | L0 zero | 0.00000 | n/a | 11778 |
| AMZN | event | 2s | 0 | L1 sign(queue imbalance) | 0.00817 | 0.14887 | 11778 |
| AMZN | event | 2s | 0 | L1m sign(microprice - mid) | 0.00243 | 0.04426 | 11778 |
| AMZN | event | 2s | 0 | L2 OLS on OFI | -0.05901 | 0.18625 | 11778 |
| AMZN | event | 2s | 0 | L3 ridge, full set | 0.05001 | 0.29546 | 11778 |
| AMZN | event | 2s | 0 | L4 GBM, full set | 0.12848 | 0.34230 | 11778 |
| AMZN | event | 2s | 1 | L0 zero | 0.00000 | n/a | 11778 |
| AMZN | event | 2s | 1 | L1 sign(queue imbalance) | 0.00817 | 0.14887 | 11778 |
| AMZN | event | 2s | 1 | L1m sign(microprice - mid) | 0.00243 | 0.04426 | 11778 |
| AMZN | event | 2s | 1 | L2 OLS on OFI | -0.05902 | 0.18625 | 11778 |
| AMZN | event | 2s | 1 | L3 ridge, full set | 0.05001 | 0.29546 | 11778 |
| AMZN | event | 2s | 1 | L4 GBM, full set | 0.12897 | 0.34248 | 11778 |
| AMZN | event | 2s | 10 | L0 zero | 0.00000 | n/a | 11769 |
| AMZN | event | 2s | 10 | L1 sign(queue imbalance) | 0.00819 | 0.14887 | 11769 |
| AMZN | event | 2s | 10 | L1m sign(microprice - mid) | 0.00241 | 0.04426 | 11769 |
| AMZN | event | 2s | 10 | L2 OLS on OFI | -0.05905 | 0.18625 | 11769 |
| AMZN | event | 2s | 10 | L3 ridge, full set | 0.04994 | 0.29551 | 11769 |
| AMZN | event | 2s | 10 | L4 GBM, full set | 0.12661 | 0.33787 | 11769 |
| AMZN | event | 2s | 60 | L0 zero | 0.00000 | n/a | 11705 |
| AMZN | event | 2s | 60 | L1 sign(queue imbalance) | 0.00829 | 0.14887 | 11705 |
| AMZN | event | 2s | 60 | L1m sign(microprice - mid) | 0.00327 | 0.12915 | 11705 |
| AMZN | event | 2s | 60 | L2 OLS on OFI | -0.05684 | 0.18625 | 11705 |
| AMZN | event | 2s | 60 | L3 ridge, full set | 0.04654 | 0.29618 | 11705 |
| AMZN | event | 2s | 60 | L4 GBM, full set | 0.12936 | 0.33972 | 11705 |
| AMZN | event | 2s | 300 | L0 zero | 0.00000 | n/a | 11449 |
| AMZN | event | 2s | 300 | L1 sign(queue imbalance) | 0.00823 | 0.14887 | 11449 |
| AMZN | event | 2s | 300 | L1m sign(microprice - mid) | 0.00333 | 0.09165 | 11449 |
| AMZN | event | 2s | 300 | L2 OLS on OFI | -0.08692 | 0.27142 | 11449 |
| AMZN | event | 2s | 300 | L3 ridge, full set | 0.02583 | 0.28854 | 11449 |
| AMZN | event | 2s | 300 | L4 GBM, full set | 0.11833 | 0.33540 | 11449 |
| AMZN | event | 5s | 0 | L0 zero | 0.00000 | n/a | 11777 |
| AMZN | event | 5s | 0 | L1 sign(queue imbalance) | 0.00425 | 0.10436 | 11777 |
| AMZN | event | 5s | 0 | L1m sign(microprice - mid) | -0.00320 | 0.04312 | 11777 |
| AMZN | event | 5s | 0 | L2 OLS on OFI | -0.05503 | 0.18447 | 11777 |
| AMZN | event | 5s | 0 | L3 ridge, full set | 0.06089 | 0.25366 | 11777 |
| AMZN | event | 5s | 0 | L4 GBM, full set | 0.11011 | 0.30688 | 11777 |
| AMZN | event | 5s | 1 | L0 zero | 0.00000 | n/a | 11776 |
| AMZN | event | 5s | 1 | L1 sign(queue imbalance) | 0.00425 | 0.10436 | 11776 |
| AMZN | event | 5s | 1 | L1m sign(microprice - mid) | -0.00320 | 0.04312 | 11776 |
| AMZN | event | 5s | 1 | L2 OLS on OFI | -0.05503 | 0.18447 | 11776 |
| AMZN | event | 5s | 1 | L3 ridge, full set | 0.06089 | 0.25366 | 11776 |
| AMZN | event | 5s | 1 | L4 GBM, full set | 0.11326 | 0.31093 | 11776 |
| AMZN | event | 5s | 10 | L0 zero | 0.00000 | n/a | 11764 |
| AMZN | event | 5s | 10 | L1 sign(queue imbalance) | 0.00437 | 0.10436 | 11764 |
| AMZN | event | 5s | 10 | L1m sign(microprice - mid) | -0.00312 | 0.04312 | 11764 |
| AMZN | event | 5s | 10 | L2 OLS on OFI | -0.05650 | 0.18447 | 11764 |
| AMZN | event | 5s | 10 | L3 ridge, full set | 0.06085 | 0.25352 | 11764 |
| AMZN | event | 5s | 10 | L4 GBM, full set | 0.10967 | 0.30880 | 11764 |
| AMZN | event | 5s | 60 | L0 zero | 0.00000 | n/a | 11701 |
| AMZN | event | 5s | 60 | L1 sign(queue imbalance) | 0.00543 | 0.10436 | 11701 |
| AMZN | event | 5s | 60 | L1m sign(microprice - mid) | -0.00121 | 0.04312 | 11701 |
| AMZN | event | 5s | 60 | L2 OLS on OFI | -0.05820 | 0.18447 | 11701 |
| AMZN | event | 5s | 60 | L3 ridge, full set | 0.06061 | 0.25238 | 11701 |
| AMZN | event | 5s | 60 | L4 GBM, full set | 0.10640 | 0.31211 | 11701 |
| AMZN | event | 5s | 300 | L0 zero | 0.00000 | n/a | 11446 |
| AMZN | event | 5s | 300 | L1 sign(queue imbalance) | 0.00393 | 0.10436 | 11446 |
| AMZN | event | 5s | 300 | L1m sign(microprice - mid) | -0.00318 | 0.08821 | 11446 |
| AMZN | event | 5s | 300 | L2 OLS on OFI | -0.02886 | 0.26818 | 11446 |
| AMZN | event | 5s | 300 | L3 ridge, full set | 0.05548 | 0.21644 | 11446 |
| AMZN | event | 5s | 300 | L4 GBM, full set | 0.10115 | 0.23400 | 11446 |
| INTC | calendar | 10s | 0 | L0 zero | 0.00000 | n/a | 90235 |
| INTC | calendar | 10s | 0 | L1 sign(queue imbalance) | 0.02901 | 0.21034 | 90235 |
| INTC | calendar | 10s | 0 | L1m sign(microprice - mid) | 0.00828 | 0.15727 | 90235 |
| INTC | calendar | 10s | 0 | L2 OLS on OFI | -0.00351 | 0.09818 | 90235 |
| INTC | calendar | 10s | 0 | L3 ridge, full set | 0.04063 | 0.23025 | 90235 |
| INTC | calendar | 10s | 0 | L4 GBM, full set | 0.01186 | 0.16024 | 90235 |
| INTC | calendar | 10s | 1 | L0 zero | 0.00000 | n/a | 90225 |
| INTC | calendar | 10s | 1 | L1 sign(queue imbalance) | 0.02905 | 0.21034 | 90225 |
| INTC | calendar | 10s | 1 | L1m sign(microprice - mid) | 0.00834 | 0.15727 | 90225 |
| INTC | calendar | 10s | 1 | L2 OLS on OFI | -0.00349 | 0.09818 | 90225 |
| INTC | calendar | 10s | 1 | L3 ridge, full set | 0.04063 | 0.23026 | 90225 |
| INTC | calendar | 10s | 1 | L4 GBM, full set | 0.01019 | 0.15693 | 90225 |
| INTC | calendar | 10s | 10 | L0 zero | 0.00000 | n/a | 90135 |
| INTC | calendar | 10s | 10 | L1 sign(queue imbalance) | 0.02917 | 0.21034 | 90135 |
| INTC | calendar | 10s | 10 | L1m sign(microprice - mid) | 0.00852 | 0.15727 | 90135 |
| INTC | calendar | 10s | 10 | L2 OLS on OFI | -0.00349 | 0.09818 | 90135 |
| INTC | calendar | 10s | 10 | L3 ridge, full set | 0.04073 | 0.23034 | 90135 |
| INTC | calendar | 10s | 10 | L4 GBM, full set | -0.00020 | 0.08472 | 90135 |
| INTC | calendar | 10s | 60 | L0 zero | 0.00000 | n/a | 89668 |
| INTC | calendar | 10s | 60 | L1 sign(queue imbalance) | 0.02629 | 0.21034 | 89668 |
| INTC | calendar | 10s | 60 | L1m sign(microprice - mid) | 0.00511 | 0.15727 | 89668 |
| INTC | calendar | 10s | 60 | L2 OLS on OFI | -0.00194 | 0.09818 | 89668 |
| INTC | calendar | 10s | 60 | L3 ridge, full set | 0.04177 | 0.22997 | 89668 |
| INTC | calendar | 10s | 60 | L4 GBM, full set | -0.00884 | 0.10274 | 89668 |
| INTC | calendar | 10s | 300 | L0 zero | 0.00000 | n/a | 87457 |
| INTC | calendar | 10s | 300 | L1 sign(queue imbalance) | 0.03043 | 0.21034 | 87457 |
| INTC | calendar | 10s | 300 | L1m sign(microprice - mid) | 0.00899 | 0.15727 | 87457 |
| INTC | calendar | 10s | 300 | L2 OLS on OFI | -0.00306 | 0.06262 | 87457 |
| INTC | calendar | 10s | 300 | L3 ridge, full set | -0.00507 | 0.15788 | 87457 |
| INTC | calendar | 10s | 300 | L4 GBM, full set | -0.02912 | 0.12692 | 87457 |
| INTC | calendar | 1s | 0 | L0 zero | 0.00000 | n/a | 90323 |
| INTC | calendar | 1s | 0 | L1 sign(queue imbalance) | 0.00916 | 0.11636 | 90323 |
| INTC | calendar | 1s | 0 | L1m sign(microprice - mid) | 0.00379 | 0.11514 | 90323 |
| INTC | calendar | 1s | 0 | L2 OLS on OFI | 0.00073 | 0.07902 | 90323 |
| INTC | calendar | 1s | 0 | L3 ridge, full set | 0.01949 | 0.14943 | 90323 |
| INTC | calendar | 1s | 0 | L4 GBM, full set | 0.02254 | 0.13119 | 90323 |
| INTC | calendar | 1s | 1 | L0 zero | 0.00000 | n/a | 90313 |
| INTC | calendar | 1s | 1 | L1 sign(queue imbalance) | 0.00912 | 0.11636 | 90313 |
| INTC | calendar | 1s | 1 | L1m sign(microprice - mid) | 0.00376 | 0.11514 | 90313 |
| INTC | calendar | 1s | 1 | L2 OLS on OFI | 0.00070 | 0.07902 | 90313 |
| INTC | calendar | 1s | 1 | L3 ridge, full set | 0.01914 | 0.14942 | 90313 |
| INTC | calendar | 1s | 1 | L4 GBM, full set | 0.02281 | 0.13138 | 90313 |
| INTC | calendar | 1s | 10 | L0 zero | 0.00000 | n/a | 90225 |
| INTC | calendar | 1s | 10 | L1 sign(queue imbalance) | 0.00924 | 0.11636 | 90225 |
| INTC | calendar | 1s | 10 | L1m sign(microprice - mid) | 0.00408 | 0.11514 | 90225 |
| INTC | calendar | 1s | 10 | L2 OLS on OFI | 0.00066 | 0.07902 | 90225 |
| INTC | calendar | 1s | 10 | L3 ridge, full set | 0.02020 | 0.14953 | 90225 |
| INTC | calendar | 1s | 10 | L4 GBM, full set | 0.02154 | 0.12848 | 90225 |
| INTC | calendar | 1s | 60 | L0 zero | 0.00000 | n/a | 89755 |
| INTC | calendar | 1s | 60 | L1 sign(queue imbalance) | 0.00938 | 0.11636 | 89755 |
| INTC | calendar | 1s | 60 | L1m sign(microprice - mid) | 0.00402 | 0.11514 | 89755 |
| INTC | calendar | 1s | 60 | L2 OLS on OFI | 0.00084 | 0.07902 | 89755 |
| INTC | calendar | 1s | 60 | L3 ridge, full set | 0.02148 | 0.14959 | 89755 |
| INTC | calendar | 1s | 60 | L4 GBM, full set | 0.02134 | 0.13160 | 89755 |
| INTC | calendar | 1s | 300 | L0 zero | 0.00000 | n/a | 87538 |
| INTC | calendar | 1s | 300 | L1 sign(queue imbalance) | 0.00920 | 0.11636 | 87538 |
| INTC | calendar | 1s | 300 | L1m sign(microprice - mid) | 0.00471 | 0.11514 | 87538 |
| INTC | calendar | 1s | 300 | L2 OLS on OFI | 0.00146 | 0.07902 | 87538 |
| INTC | calendar | 1s | 300 | L3 ridge, full set | 0.02043 | 0.14718 | 87538 |
| INTC | calendar | 1s | 300 | L4 GBM, full set | 0.01273 | 0.12362 | 87538 |
| INTC | calendar | 2s | 0 | L0 zero | 0.00000 | n/a | 90313 |
| INTC | calendar | 2s | 0 | L1 sign(queue imbalance) | 0.01264 | 0.14500 | 90313 |
| INTC | calendar | 2s | 0 | L1m sign(microprice - mid) | 0.00563 | 0.14022 | 90313 |
| INTC | calendar | 2s | 0 | L2 OLS on OFI | 0.00071 | 0.09620 | 90313 |
| INTC | calendar | 2s | 0 | L3 ridge, full set | 0.03143 | 0.18628 | 90313 |
| INTC | calendar | 2s | 0 | L4 GBM, full set | 0.02461 | 0.14865 | 90313 |
| INTC | calendar | 2s | 1 | L0 zero | 0.00000 | n/a | 90303 |
| INTC | calendar | 2s | 1 | L1 sign(queue imbalance) | 0.01275 | 0.14500 | 90303 |
| INTC | calendar | 2s | 1 | L1m sign(microprice - mid) | 0.00584 | 0.14022 | 90303 |
| INTC | calendar | 2s | 1 | L2 OLS on OFI | 0.00066 | 0.09620 | 90303 |
| INTC | calendar | 2s | 1 | L3 ridge, full set | 0.03165 | 0.18629 | 90303 |
| INTC | calendar | 2s | 1 | L4 GBM, full set | 0.02434 | 0.14867 | 90303 |
| INTC | calendar | 2s | 10 | L0 zero | 0.00000 | n/a | 90215 |
| INTC | calendar | 2s | 10 | L1 sign(queue imbalance) | 0.01294 | 0.14500 | 90215 |
| INTC | calendar | 2s | 10 | L1m sign(microprice - mid) | 0.00619 | 0.14022 | 90215 |
| INTC | calendar | 2s | 10 | L2 OLS on OFI | 0.00065 | 0.09620 | 90215 |
| INTC | calendar | 2s | 10 | L3 ridge, full set | 0.03183 | 0.18637 | 90215 |
| INTC | calendar | 2s | 10 | L4 GBM, full set | 0.02548 | 0.15208 | 90215 |
| INTC | calendar | 2s | 60 | L0 zero | 0.00000 | n/a | 89745 |
| INTC | calendar | 2s | 60 | L1 sign(queue imbalance) | 0.01277 | 0.14500 | 89745 |
| INTC | calendar | 2s | 60 | L1m sign(microprice - mid) | 0.00551 | 0.14022 | 89745 |
| INTC | calendar | 2s | 60 | L2 OLS on OFI | 0.00098 | 0.09620 | 89745 |
| INTC | calendar | 2s | 60 | L3 ridge, full set | 0.03094 | 0.18619 | 89745 |
| INTC | calendar | 2s | 60 | L4 GBM, full set | 0.02231 | 0.15393 | 89745 |
| INTC | calendar | 2s | 300 | L0 zero | 0.00000 | n/a | 87535 |
| INTC | calendar | 2s | 300 | L1 sign(queue imbalance) | 0.01424 | 0.14500 | 87535 |
| INTC | calendar | 2s | 300 | L1m sign(microprice - mid) | 0.00697 | 0.14022 | 87535 |
| INTC | calendar | 2s | 300 | L2 OLS on OFI | 0.00302 | 0.09620 | 87535 |
| INTC | calendar | 2s | 300 | L3 ridge, full set | 0.02698 | 0.18198 | 87535 |
| INTC | calendar | 2s | 300 | L4 GBM, full set | 0.02113 | 0.14773 | 87535 |
| INTC | calendar | 5s | 0 | L0 zero | 0.00000 | n/a | 90284 |
| INTC | calendar | 5s | 0 | L1 sign(queue imbalance) | 0.02205 | 0.18822 | 90284 |
| INTC | calendar | 5s | 0 | L1m sign(microprice - mid) | 0.00677 | 0.15127 | 90284 |
| INTC | calendar | 5s | 0 | L2 OLS on OFI | 0.00097 | 0.09591 | 90284 |
| INTC | calendar | 5s | 0 | L3 ridge, full set | 0.04214 | 0.22901 | 90284 |
| INTC | calendar | 5s | 0 | L4 GBM, full set | 0.02107 | 0.16353 | 90284 |
| INTC | calendar | 5s | 1 | L0 zero | 0.00000 | n/a | 90275 |
| INTC | calendar | 5s | 1 | L1 sign(queue imbalance) | 0.02211 | 0.18822 | 90275 |
| INTC | calendar | 5s | 1 | L1m sign(microprice - mid) | 0.00690 | 0.15127 | 90275 |
| INTC | calendar | 5s | 1 | L2 OLS on OFI | 0.00097 | 0.09591 | 90275 |
| INTC | calendar | 5s | 1 | L3 ridge, full set | 0.04219 | 0.22902 | 90275 |
| INTC | calendar | 5s | 1 | L4 GBM, full set | 0.02404 | 0.17007 | 90275 |
| INTC | calendar | 5s | 10 | L0 zero | 0.00000 | n/a | 90185 |
| INTC | calendar | 5s | 10 | L1 sign(queue imbalance) | 0.02223 | 0.18822 | 90185 |
| INTC | calendar | 5s | 10 | L1m sign(microprice - mid) | 0.00717 | 0.15127 | 90185 |
| INTC | calendar | 5s | 10 | L2 OLS on OFI | 0.00096 | 0.09591 | 90185 |
| INTC | calendar | 5s | 10 | L3 ridge, full set | 0.04223 | 0.22902 | 90185 |
| INTC | calendar | 5s | 10 | L4 GBM, full set | 0.02152 | 0.16652 | 90185 |
| INTC | calendar | 5s | 60 | L0 zero | 0.00000 | n/a | 89715 |
| INTC | calendar | 5s | 60 | L1 sign(queue imbalance) | 0.02094 | 0.18822 | 89715 |
| INTC | calendar | 5s | 60 | L1m sign(microprice - mid) | 0.00465 | 0.15127 | 89715 |
| INTC | calendar | 5s | 60 | L2 OLS on OFI | 0.00139 | 0.09591 | 89715 |
| INTC | calendar | 5s | 60 | L3 ridge, full set | 0.04175 | 0.22826 | 89715 |
| INTC | calendar | 5s | 60 | L4 GBM, full set | 0.01900 | 0.16979 | 89715 |
| INTC | calendar | 5s | 300 | L0 zero | 0.00000 | n/a | 87505 |
| INTC | calendar | 5s | 300 | L1 sign(queue imbalance) | 0.02411 | 0.18822 | 87505 |
| INTC | calendar | 5s | 300 | L1m sign(microprice - mid) | 0.00618 | 0.15127 | 87505 |
| INTC | calendar | 5s | 300 | L2 OLS on OFI | 0.00185 | 0.09591 | 87505 |
| INTC | calendar | 5s | 300 | L3 ridge, full set | 0.02527 | 0.15036 | 87505 |
| INTC | calendar | 5s | 300 | L4 GBM, full set | 0.01158 | 0.11501 | 87505 |
| INTC | event | 10s | 0 | L0 zero | 0.00000 | n/a | 25327 |
| INTC | event | 10s | 0 | L1 sign(queue imbalance) | 0.04007 | 0.24812 | 25327 |
| INTC | event | 10s | 0 | L1m sign(microprice - mid) | 0.03092 | 0.17445 | 25327 |
| INTC | event | 10s | 0 | L2 OLS on OFI | 0.01531 | 0.20061 | 25327 |
| INTC | event | 10s | 0 | L3 ridge, full set | 0.04447 | 0.22231 | 25327 |
| INTC | event | 10s | 0 | L4 GBM, full set | 0.03936 | 0.22003 | 25327 |
| INTC | event | 10s | 1 | L0 zero | 0.00000 | n/a | 25323 |
| INTC | event | 10s | 1 | L1 sign(queue imbalance) | 0.04007 | 0.24812 | 25323 |
| INTC | event | 10s | 1 | L1m sign(microprice - mid) | 0.03090 | 0.17445 | 25323 |
| INTC | event | 10s | 1 | L2 OLS on OFI | 0.01532 | 0.20061 | 25323 |
| INTC | event | 10s | 1 | L3 ridge, full set | 0.04448 | 0.22231 | 25323 |
| INTC | event | 10s | 1 | L4 GBM, full set | 0.04007 | 0.22105 | 25323 |
| INTC | event | 10s | 10 | L0 zero | 0.00000 | n/a | 25279 |
| INTC | event | 10s | 10 | L1 sign(queue imbalance) | 0.03989 | 0.24812 | 25279 |
| INTC | event | 10s | 10 | L1m sign(microprice - mid) | 0.03127 | 0.17445 | 25279 |
| INTC | event | 10s | 10 | L2 OLS on OFI | 0.01283 | 0.20061 | 25279 |
| INTC | event | 10s | 10 | L3 ridge, full set | 0.04039 | 0.22228 | 25279 |
| INTC | event | 10s | 10 | L4 GBM, full set | 0.03992 | 0.20831 | 25279 |
| INTC | event | 10s | 60 | L0 zero | 0.00000 | n/a | 25030 |
| INTC | event | 10s | 60 | L1 sign(queue imbalance) | 0.03652 | 0.24812 | 25030 |
| INTC | event | 10s | 60 | L1m sign(microprice - mid) | 0.02774 | 0.17445 | 25030 |
| INTC | event | 10s | 60 | L2 OLS on OFI | 0.00553 | 0.20061 | 25030 |
| INTC | event | 10s | 60 | L3 ridge, full set | 0.03461 | 0.22667 | 25030 |
| INTC | event | 10s | 60 | L4 GBM, full set | 0.03434 | 0.18794 | 25030 |
| INTC | event | 10s | 300 | L0 zero | 0.00000 | n/a | 24430 |
| INTC | event | 10s | 300 | L1 sign(queue imbalance) | 0.03591 | 0.24812 | 24430 |
| INTC | event | 10s | 300 | L1m sign(microprice - mid) | 0.02563 | 0.17445 | 24430 |
| INTC | event | 10s | 300 | L2 OLS on OFI | 0.00738 | 0.20061 | 24430 |
| INTC | event | 10s | 300 | L3 ridge, full set | 0.04503 | 0.22950 | 24430 |
| INTC | event | 10s | 300 | L4 GBM, full set | 0.04579 | 0.18611 | 24430 |
| INTC | event | 1s | 0 | L0 zero | 0.00000 | n/a | 25343 |
| INTC | event | 1s | 0 | L1 sign(queue imbalance) | 0.07141 | 0.29794 | 25343 |
| INTC | event | 1s | 0 | L1m sign(microprice - mid) | 0.01067 | 0.15270 | 25343 |
| INTC | event | 1s | 0 | L2 OLS on OFI | 0.04161 | 0.33797 | 25343 |
| INTC | event | 1s | 0 | L3 ridge, full set | 0.08689 | 0.36322 | 25343 |
| INTC | event | 1s | 0 | L4 GBM, full set | 0.18900 | 0.42134 | 25343 |
| INTC | event | 1s | 1 | L0 zero | 0.00000 | n/a | 25341 |
| INTC | event | 1s | 1 | L1 sign(queue imbalance) | 0.07124 | 0.29794 | 25341 |
| INTC | event | 1s | 1 | L1m sign(microprice - mid) | 0.01050 | 0.15270 | 25341 |
| INTC | event | 1s | 1 | L2 OLS on OFI | 0.04200 | 0.33797 | 25341 |
| INTC | event | 1s | 1 | L3 ridge, full set | 0.08526 | 0.36292 | 25341 |
| INTC | event | 1s | 1 | L4 GBM, full set | 0.18339 | 0.42418 | 25341 |
| INTC | event | 1s | 10 | L0 zero | 0.00000 | n/a | 25323 |
| INTC | event | 1s | 10 | L1 sign(queue imbalance) | 0.07079 | 0.29794 | 25323 |
| INTC | event | 1s | 10 | L1m sign(microprice - mid) | 0.00902 | 0.15270 | 25323 |
| INTC | event | 1s | 10 | L2 OLS on OFI | 0.04068 | 0.33797 | 25323 |
| INTC | event | 1s | 10 | L3 ridge, full set | 0.07663 | 0.36263 | 25323 |
| INTC | event | 1s | 10 | L4 GBM, full set | 0.18200 | 0.41834 | 25323 |
| INTC | event | 1s | 60 | L0 zero | 0.00000 | n/a | 25071 |
| INTC | event | 1s | 60 | L1 sign(queue imbalance) | 0.06712 | 0.29794 | 25071 |
| INTC | event | 1s | 60 | L1m sign(microprice - mid) | 0.00483 | 0.15270 | 25071 |
| INTC | event | 1s | 60 | L2 OLS on OFI | 0.03614 | 0.33797 | 25071 |
| INTC | event | 1s | 60 | L3 ridge, full set | 0.07120 | 0.36451 | 25071 |
| INTC | event | 1s | 60 | L4 GBM, full set | 0.19124 | 0.44413 | 25071 |
| INTC | event | 1s | 300 | L0 zero | 0.00000 | n/a | 24437 |
| INTC | event | 1s | 300 | L1 sign(queue imbalance) | 0.06993 | 0.29794 | 24437 |
| INTC | event | 1s | 300 | L1m sign(microprice - mid) | 0.00881 | 0.15270 | 24437 |
| INTC | event | 1s | 300 | L2 OLS on OFI | 0.04597 | 0.33797 | 24437 |
| INTC | event | 1s | 300 | L3 ridge, full set | 0.07324 | 0.35742 | 24437 |
| INTC | event | 1s | 300 | L4 GBM, full set | 0.19377 | 0.43197 | 24437 |
| INTC | event | 2s | 0 | L0 zero | 0.00000 | n/a | 25341 |
| INTC | event | 2s | 0 | L1 sign(queue imbalance) | 0.06478 | 0.29452 | 25341 |
| INTC | event | 2s | 0 | L1m sign(microprice - mid) | 0.01639 | 0.16659 | 25341 |
| INTC | event | 2s | 0 | L2 OLS on OFI | 0.00476 | 0.19144 | 25341 |
| INTC | event | 2s | 0 | L3 ridge, full set | -0.01969 | 0.33473 | 25341 |
| INTC | event | 2s | 0 | L4 GBM, full set | 0.14492 | 0.38622 | 25341 |
| INTC | event | 2s | 1 | L0 zero | 0.00000 | n/a | 25339 |
| INTC | event | 2s | 1 | L1 sign(queue imbalance) | 0.06452 | 0.29452 | 25339 |
| INTC | event | 2s | 1 | L1m sign(microprice - mid) | 0.01489 | 0.16659 | 25339 |
| INTC | event | 2s | 1 | L2 OLS on OFI | 0.00451 | 0.19144 | 25339 |
| INTC | event | 2s | 1 | L3 ridge, full set | -0.03160 | 0.33399 | 25339 |
| INTC | event | 2s | 1 | L4 GBM, full set | 0.13564 | 0.39306 | 25339 |
| INTC | event | 2s | 10 | L0 zero | 0.00000 | n/a | 25321 |
| INTC | event | 2s | 10 | L1 sign(queue imbalance) | 0.06429 | 0.29452 | 25321 |
| INTC | event | 2s | 10 | L1m sign(microprice - mid) | 0.01445 | 0.16659 | 25321 |
| INTC | event | 2s | 10 | L2 OLS on OFI | 0.00361 | 0.19144 | 25321 |
| INTC | event | 2s | 10 | L3 ridge, full set | -0.03333 | 0.33411 | 25321 |
| INTC | event | 2s | 10 | L4 GBM, full set | 0.13540 | 0.38089 | 25321 |
| INTC | event | 2s | 60 | L0 zero | 0.00000 | n/a | 25070 |
| INTC | event | 2s | 60 | L1 sign(queue imbalance) | 0.06140 | 0.29452 | 25070 |
| INTC | event | 2s | 60 | L1m sign(microprice - mid) | 0.00838 | 0.16659 | 25070 |
| INTC | event | 2s | 60 | L2 OLS on OFI | -0.00248 | 0.19144 | 25070 |
| INTC | event | 2s | 60 | L3 ridge, full set | -0.04504 | 0.33535 | 25070 |
| INTC | event | 2s | 60 | L4 GBM, full set | 0.12804 | 0.39830 | 25070 |
| INTC | event | 2s | 300 | L0 zero | 0.00000 | n/a | 24437 |
| INTC | event | 2s | 300 | L1 sign(queue imbalance) | 0.06959 | 0.29452 | 24437 |
| INTC | event | 2s | 300 | L1m sign(microprice - mid) | 0.01984 | 0.16659 | 24437 |
| INTC | event | 2s | 300 | L2 OLS on OFI | 0.03292 | 0.29808 | 24437 |
| INTC | event | 2s | 300 | L3 ridge, full set | -0.02208 | 0.32393 | 24437 |
| INTC | event | 2s | 300 | L4 GBM, full set | 0.14771 | 0.37110 | 24437 |
| INTC | event | 5s | 0 | L0 zero | 0.00000 | n/a | 25334 |
| INTC | event | 5s | 0 | L1 sign(queue imbalance) | 0.05814 | 0.28033 | 25334 |
| INTC | event | 5s | 0 | L1m sign(microprice - mid) | 0.00993 | 0.13065 | 25334 |
| INTC | event | 5s | 0 | L2 OLS on OFI | 0.02243 | 0.20910 | 25334 |
| INTC | event | 5s | 0 | L3 ridge, full set | 0.11371 | 0.33181 | 25334 |
| INTC | event | 5s | 0 | L4 GBM, full set | 0.08916 | 0.31638 | 25334 |
| INTC | event | 5s | 1 | L0 zero | 0.00000 | n/a | 25334 |
| INTC | event | 5s | 1 | L1 sign(queue imbalance) | 0.05814 | 0.28033 | 25334 |
| INTC | event | 5s | 1 | L1m sign(microprice - mid) | 0.00992 | 0.13065 | 25334 |
| INTC | event | 5s | 1 | L2 OLS on OFI | 0.02243 | 0.20910 | 25334 |
| INTC | event | 5s | 1 | L3 ridge, full set | 0.11370 | 0.33184 | 25334 |
| INTC | event | 5s | 1 | L4 GBM, full set | 0.08940 | 0.31526 | 25334 |
| INTC | event | 5s | 10 | L0 zero | 0.00000 | n/a | 25314 |
| INTC | event | 5s | 10 | L1 sign(queue imbalance) | 0.05793 | 0.28033 | 25314 |
| INTC | event | 5s | 10 | L1m sign(microprice - mid) | 0.00946 | 0.13065 | 25314 |
| INTC | event | 5s | 10 | L2 OLS on OFI | 0.02115 | 0.20910 | 25314 |
| INTC | event | 5s | 10 | L3 ridge, full set | 0.11261 | 0.33214 | 25314 |
| INTC | event | 5s | 10 | L4 GBM, full set | 0.09403 | 0.31985 | 25314 |
| INTC | event | 5s | 60 | L0 zero | 0.00000 | n/a | 25058 |
| INTC | event | 5s | 60 | L1 sign(queue imbalance) | 0.05564 | 0.28033 | 25058 |
| INTC | event | 5s | 60 | L1m sign(microprice - mid) | 0.00650 | 0.13065 | 25058 |
| INTC | event | 5s | 60 | L2 OLS on OFI | 0.01633 | 0.20910 | 25058 |
| INTC | event | 5s | 60 | L3 ridge, full set | 0.11026 | 0.33391 | 25058 |
| INTC | event | 5s | 60 | L4 GBM, full set | 0.08382 | 0.32440 | 25058 |
| INTC | event | 5s | 300 | L0 zero | 0.00000 | n/a | 24433 |
| INTC | event | 5s | 300 | L1 sign(queue imbalance) | 0.06128 | 0.28033 | 24433 |
| INTC | event | 5s | 300 | L1m sign(microprice - mid) | 0.01406 | 0.13065 | 24433 |
| INTC | event | 5s | 300 | L2 OLS on OFI | 0.03393 | 0.26209 | 24433 |
| INTC | event | 5s | 300 | L3 ridge, full set | 0.11253 | 0.32785 | 24433 |
| INTC | event | 5s | 300 | L4 GBM, full set | 0.07769 | 0.31567 | 24433 |

## Reading notes

- **L1 is the control.** It has no fitted parameters beyond a one-parameter rescale, so its `A-C` gap is a floor on what the harness itself contributes. A large L1 gap would mean the decomposition is measuring the protocols rather than leakage.
- **The 1s label is not one-second alpha.** The phase-1 adversary measured that OFI's entire content decays inside 100 ms (rho 0.458 over the first 100 ms against 0.027 over the remaining 900 ms). Read these as short-horizon numbers.
- **Fold row counts vary by design.** Test blocks are equal in DURATION, not in sample count, and rungs using trade features lose rows where a window holds no trades. Both are reported rather than absorbed.
