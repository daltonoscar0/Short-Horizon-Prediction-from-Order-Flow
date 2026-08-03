# The frozen-predictor artifact floor (Phase 2)

`python3 -m edge.harness.floors --seed 0 --tickers INTC,AMZN --clocks calendar,event --horizons 1s,2s,5s,10s`

Each row is ONE model, fitted once on every usable row of the day, then scored under each protocol with **only the through-origin scale refitted per fold, under that protocol's own holdout rule** -- exactly the treatment the fitted rungs get. The raw model cannot have learned anything from any split, so every difference below is artifact and **none of it can be leakage**. A rung's `A - C` gap has to clear the floor measured AT ITS OWN MAGNITUDE.

> **Corrected 2026-08-03 after the third red-team pass.** The previous version of this table froze the SCALE as well as the model. That made it blind to the largest channel: under protocol A the calibration holdout is a random 20% of the training rows and is exchangeable with A's random test fold, while under B/C/D it is the last 20% in time and is not. A floor measured with that channel held fixed is not a bound on a rung exposed to it, and it was published as one -- the R2 floors were understated 2-6x and the AUC floor's worst cell was quoted as +0.0053 when the like-for-like figure is an order of magnitude larger. The frozen-scale column is retained beside the corrected one: it is the pure partition artifact, a real but different quantity, and the difference between the columns is the calibration channel.

The floor is the interaction between the aggregation asymmetry -- A's fold mean approximates the `sum(y^2)`-weighted pooled metric, C's is an unweighted average over contiguous segments of very unequal `sum(y^2)` -- and the predictor's own variance. It therefore SCALES WITH THE PREDICTOR, which is why one magnitude does not bound another and why the earlier practice of quoting the L1 floor beside an L4 gap was wrong.

| ticker | clock | h | frozen at | A R2 | B R2 | C R2 | D R2 | **A-C floor** | A-C (frozen scale) | A IC | C IC | A-C IC floor |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L1 | 0.01240 | 0.00914 | 0.00938 | 0.00407 | **0.00302** | 0.00136 | 0.11812 | 0.11636 | 0.00177 |
| INTC | calendar | 1s | L3 | 0.03908 | 0.03482 | 0.03484 | 0.02956 | **0.00425** | 0.00207 | 0.17081 | 0.17063 | 0.00018 |
| INTC | calendar | 1s | L4 | 0.25091 | 0.23235 | 0.23174 | 0.23443 | **0.01917** | 0.01615 | 0.26947 | 0.26407 | 0.00540 |
| INTC | calendar | 2s | L1 | 0.01880 | 0.01266 | 0.01277 | 0.00466 | **0.00602** | 0.00214 | 0.14650 | 0.14500 | 0.00151 |
| INTC | calendar | 2s | L3 | 0.05580 | 0.04802 | 0.04802 | 0.03449 | **0.00777** | 0.00379 | 0.21309 | 0.21283 | 0.00026 |
| INTC | calendar | 2s | L4 | 0.32036 | 0.29830 | 0.29831 | 0.29781 | **0.02205** | 0.02098 | 0.36256 | 0.35550 | 0.00707 |
| INTC | calendar | 5s | L1 | 0.02910 | 0.02167 | 0.02094 | 0.00912 | **0.00816** | 0.00256 | 0.18967 | 0.18822 | 0.00145 |
| INTC | calendar | 5s | L3 | 0.07498 | 0.06478 | 0.06477 | 0.04244 | **0.01021** | 0.00531 | 0.26484 | 0.26449 | 0.00035 |
| INTC | calendar | 5s | L4 | 0.43721 | 0.40799 | 0.40777 | 0.40823 | **0.02944** | 0.02692 | 0.51142 | 0.50233 | 0.00909 |
| INTC | calendar | 10s | L1 | 0.03422 | 0.02822 | 0.02886 | 0.00971 | **0.00536** | 0.00205 | 0.21211 | 0.21034 | 0.00177 |
| INTC | calendar | 10s | L3 | 0.08145 | 0.07289 | 0.07367 | 0.04262 | **0.00778** | 0.00473 | 0.28827 | 0.28661 | 0.00166 |
| INTC | calendar | 10s | L4 | 0.54614 | 0.51474 | 0.51619 | 0.50760 | **0.02995** | 0.02710 | 0.63156 | 0.61859 | 0.01297 |
| INTC | event | 1s | L1 | 0.08527 | 0.07142 | 0.06712 | 0.05595 | **0.01815** | 0.00824 | 0.31371 | 0.29794 | 0.01577 |
| INTC | event | 1s | L3 | 0.24806 | 0.23011 | 0.22715 | 0.24017 | **0.02092** | 0.00714 | 0.48065 | 0.45948 | 0.02117 |
| INTC | event | 1s | L4 | 0.65054 | 0.64404 | 0.64333 | 0.64436 | **0.00721** | 0.00576 | 0.71640 | 0.70770 | 0.00870 |
| INTC | event | 2s | L1 | 0.07639 | 0.06496 | 0.06140 | 0.03751 | **0.01499** | 0.00751 | 0.30555 | 0.29452 | 0.01102 |
| INTC | event | 2s | L3 | 0.22722 | 0.21571 | 0.21595 | 0.22724 | **0.01126** | 0.00153 | 0.46055 | 0.44304 | 0.01751 |
| INTC | event | 2s | L4 | 0.62682 | 0.62090 | 0.62071 | 0.62047 | **0.00611** | 0.00563 | 0.73114 | 0.72174 | 0.00940 |
| INTC | event | 5s | L1 | 0.06245 | 0.05837 | 0.05564 | 0.02826 | **0.00681** | 0.00235 | 0.28258 | 0.28033 | 0.00225 |
| INTC | event | 5s | L3 | 0.19817 | 0.19043 | 0.19139 | 0.19515 | **0.00678** | 0.00321 | 0.42977 | 0.41666 | 0.01312 |
| INTC | event | 5s | L4 | 0.62974 | 0.61165 | 0.61170 | 0.61205 | **0.01804** | 0.01737 | 0.76047 | 0.75223 | 0.00824 |
| INTC | event | 10s | L1 | 0.04140 | 0.04008 | 0.03708 | 0.01411 | **0.00432** | -0.00173 | 0.24595 | 0.24812 | -0.00217 |
| INTC | event | 10s | L3 | 0.14663 | 0.13943 | 0.13814 | 0.14554 | **0.00849** | 0.00038 | 0.37556 | 0.35719 | 0.01837 |
| INTC | event | 10s | L4 | 0.61916 | 0.59577 | 0.59502 | 0.59751 | **0.02413** | 0.02124 | 0.76634 | 0.75553 | 0.01082 |
| AMZN | calendar | 1s | L1 | 0.00328 | 0.00209 | 0.00111 | -0.00234 | **0.00218** | 0.00021 | 0.08354 | 0.07966 | 0.00388 |
| AMZN | calendar | 1s | L3 | 0.02241 | 0.01366 | 0.01504 | 0.01723 | **0.00737** | 0.00171 | 0.12940 | 0.12520 | 0.00420 |
| AMZN | calendar | 1s | L4 | 0.22430 | 0.21242 | 0.21335 | 0.21640 | **0.01094** | 0.00791 | 0.32314 | 0.31823 | 0.00491 |
| AMZN | calendar | 2s | L1 | 0.00460 | 0.00281 | 0.00231 | -0.00650 | **0.00229** | 0.00034 | 0.10389 | 0.06081 | 0.04308 |
| AMZN | calendar | 2s | L3 | 0.02581 | 0.01369 | 0.01382 | 0.02066 | **0.01199** | 0.00192 | 0.15643 | 0.09812 | 0.05831 |
| AMZN | calendar | 2s | L4 | 0.29091 | 0.27110 | 0.27177 | 0.27979 | **0.01914** | 0.01081 | 0.40405 | 0.39602 | 0.00802 |
| AMZN | calendar | 5s | L1 | 0.00357 | 0.00030 | -0.00009 | -0.00612 | **0.00366** | 0.00027 | 0.10303 | 0.05566 | 0.04737 |
| AMZN | calendar | 5s | L3 | 0.02246 | 0.01196 | 0.01139 | 0.02146 | **0.01107** | 0.00022 | 0.16359 | 0.11225 | 0.05134 |
| AMZN | calendar | 5s | L4 | 0.41703 | 0.39218 | 0.39353 | 0.39564 | **0.02350** | 0.01299 | 0.53024 | 0.52081 | 0.00943 |
| AMZN | calendar | 10s | L1 | 0.00235 | 0.00115 | -0.00190 | -0.00006 | **0.00425** | -0.00001 | 0.08354 | 0.03326 | 0.05028 |
| AMZN | calendar | 10s | L3 | 0.02073 | 0.00205 | 0.00701 | 0.01952 | **0.01371** | -0.00053 | 0.15608 | 0.10704 | 0.04905 |
| AMZN | calendar | 10s | L4 | 0.50045 | 0.47720 | 0.48126 | 0.48133 | **0.01920** | 0.01091 | 0.62445 | 0.61733 | 0.00713 |
| AMZN | event | 1s | L1 | 0.01222 | 0.00978 | 0.00890 | 0.00275 | **0.00333** | 0.00203 | 0.15273 | 0.14802 | 0.00470 |
| AMZN | event | 1s | L3 | 0.16206 | 0.11523 | 0.11926 | 0.13279 | **0.04280** | 0.02127 | 0.34254 | 0.33658 | 0.00596 |
| AMZN | event | 1s | L4 | 0.44979 | 0.42513 | 0.42631 | 0.42702 | **0.02349** | 0.01780 | 0.54142 | 0.53200 | 0.00942 |
| AMZN | event | 2s | L1 | 0.01198 | 0.00813 | 0.00829 | -0.00423 | **0.00369** | 0.00233 | 0.15730 | 0.14887 | 0.00844 |
| AMZN | event | 2s | L3 | 0.14937 | 0.08548 | 0.09130 | 0.12352 | **0.05806** | 0.01865 | 0.34490 | 0.33724 | 0.00766 |
| AMZN | event | 2s | L4 | 0.44361 | 0.40773 | 0.40970 | 0.41631 | **0.03391** | 0.02279 | 0.56041 | 0.54911 | 0.01130 |
| AMZN | event | 5s | L1 | 0.01076 | 0.00420 | 0.00543 | 0.00033 | **0.00533** | 0.00194 | 0.14432 | 0.10436 | 0.03996 |
| AMZN | event | 5s | L3 | 0.12207 | 0.09220 | 0.09407 | 0.11590 | **0.02801** | 0.00059 | 0.33003 | 0.21798 | 0.11205 |
| AMZN | event | 5s | L4 | 0.45293 | 0.43682 | 0.43788 | 0.43703 | **0.01505** | 0.01436 | 0.60132 | 0.58668 | 0.01463 |
| AMZN | event | 10s | L1 | 0.00902 | 0.00574 | 0.00522 | 0.00652 | **0.00380** | 0.00068 | 0.12489 | 0.09048 | 0.03441 |
| AMZN | event | 10s | L3 | 0.08805 | 0.04850 | 0.05000 | 0.07066 | **0.03805** | -0.00339 | 0.28710 | 0.17183 | 0.11527 |
| AMZN | event | 10s | L4 | 0.47033 | 0.45835 | 0.45873 | 0.45813 | **0.01160** | 0.01225 | 0.62365 | 0.61430 | 0.00935 |

## The same floor for AUC and the calibration slope

Added 2026-08-03 with those two metrics. A gap in a new metric needs a floor measured in THAT metric.

**The AUC floor is not small because rank statistics are immune to the aggregation asymmetry.** An earlier draft of `FINDINGS.md` claimed that, and it is wrong: protocol A's per-fold AUC is a whole-day POOLED AUC while C's is a WITHIN-BLOCK AUC, and any frozen score with a time-varying offset makes those two estimands diverge with nothing refitted at all -- the red team drove the floor to -0.137 that way. It is small HERE because these particular frozen predictors are temporally homogeneous. Read it as a measurement on this data, not as a property of the metric.

`neg C` counts protocol-C folds where the refitted scale came out NEGATIVE, which flips the ranking and sends AUC to 1-AUC; it is 0 under A everywhere. `A-C AUC raw` is the same floor on the raw score, immune to that sign by construction.

| ticker | clock | h | frozen at | A AUC | C AUC | **A-C AUC floor** | A-C AUC raw | neg C | A calib | C calib | A-C calib |
|---|---|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L1 | 0.8205 | 0.8178 | **0.0026** | 0.0026 | 0 | 0.9769 | 1.3491 | -0.3722 |
| INTC | calendar | 1s | L3 | 0.9145 | 0.9184 | **-0.0038** | -0.0038 | 0 | 0.9331 | 1.2496 | -0.3165 |
| INTC | calendar | 1s | L4 | 0.9960 | 0.9950 | **0.0010** | 0.0010 | 0 | 0.9938 | 1.0454 | -0.0516 |
| INTC | calendar | 2s | L1 | 0.7928 | 0.7905 | **0.0023** | 0.0023 | 0 | 0.9958 | 1.4527 | -0.4568 |
| INTC | calendar | 2s | L3 | 0.8976 | 0.8989 | **-0.0014** | -0.0014 | 0 | 0.9715 | 1.3087 | -0.3372 |
| INTC | calendar | 2s | L4 | 0.9959 | 0.9953 | **0.0006** | 0.0006 | 0 | 1.0209 | 1.0128 | 0.0081 |
| INTC | calendar | 5s | L1 | 0.7519 | 0.7507 | **0.0012** | 0.0012 | 0 | 1.0064 | 1.4497 | -0.4433 |
| INTC | calendar | 5s | L3 | 0.8604 | 0.8629 | **-0.0025** | -0.0025 | 0 | 0.9711 | 1.3454 | -0.3743 |
| INTC | calendar | 5s | L4 | 0.9949 | 0.9937 | **0.0012** | 0.0012 | 0 | 1.0112 | 0.9816 | 0.0296 |
| INTC | calendar | 10s | L1 | 0.7155 | 0.7133 | **0.0022** | 0.0022 | 0 | 1.0206 | 1.2349 | -0.2143 |
| INTC | calendar | 10s | L3 | 0.8164 | 0.8172 | **-0.0008** | -0.0008 | 0 | 0.9862 | 1.2890 | -0.3028 |
| INTC | calendar | 10s | L4 | 0.9936 | 0.9932 | **0.0004** | 0.0004 | 0 | 1.0175 | 0.9588 | 0.0587 |
| INTC | event | 1s | L1 | 0.8451 | 0.8437 | **0.0014** | 0.0014 | 0 | 0.9991 | 1.1718 | -0.1726 |
| INTC | event | 1s | L3 | 0.9514 | 0.9471 | **0.0043** | 0.0043 | 0 | 0.9700 | 1.0853 | -0.1152 |
| INTC | event | 1s | L4 | 0.9987 | 0.9981 | **0.0006** | 0.0006 | 0 | 0.9975 | 0.9905 | 0.0069 |
| INTC | event | 2s | L1 | 0.8005 | 0.8069 | **-0.0064** | -0.0064 | 0 | 1.0330 | 1.2284 | -0.1954 |
| INTC | event | 2s | L3 | 0.9197 | 0.9174 | **0.0023** | 0.0023 | 0 | 0.9363 | 1.0570 | -0.1207 |
| INTC | event | 2s | L4 | 0.9976 | 0.9969 | **0.0007** | 0.0007 | 0 | 0.9992 | 0.9891 | 0.0101 |
| INTC | event | 5s | L1 | 0.7473 | 0.7598 | **-0.0125** | -0.0125 | 0 | 0.9829 | 1.2414 | -0.2586 |
| INTC | event | 5s | L3 | 0.8810 | 0.8773 | **0.0037** | 0.0037 | 0 | 0.9411 | 1.0973 | -0.1562 |
| INTC | event | 5s | L4 | 0.9947 | 0.9934 | **0.0013** | 0.0013 | 0 | 1.0170 | 1.0077 | 0.0094 |
| INTC | event | 10s | L1 | 0.7002 | 0.7093 | **-0.0091** | -0.0091 | 0 | 1.0136 | 1.8551 | -0.8415 |
| INTC | event | 10s | L3 | 0.8327 | 0.8348 | **-0.0021** | -0.0021 | 0 | 0.9425 | 1.1465 | -0.2041 |
| INTC | event | 10s | L4 | 0.9900 | 0.9878 | **0.0022** | 0.0022 | 0 | 1.0117 | 1.0042 | 0.0075 |
| AMZN | calendar | 1s | L1 | 0.5781 | 0.5740 | **0.0042** | 0.0042 | 0 | 1.1007 | 3.2424 | -2.1417 |
| AMZN | calendar | 1s | L3 | 0.6398 | 0.6383 | **0.0015** | 0.0015 | 0 | 1.0366 | 10.6818 | -9.6452 |
| AMZN | calendar | 1s | L4 | 0.8368 | 0.8352 | **0.0015** | 0.0015 | 0 | 0.9525 | 0.9906 | -0.0380 |
| AMZN | calendar | 2s | L1 | 0.5784 | 0.5480 | **0.0304** | 0.0043 | 1 | 1.1145 | -0.2069 | 1.3214 |
| AMZN | calendar | 2s | L3 | 0.6339 | 0.5878 | **0.0461** | 0.0035 | 1 | 1.0015 | 0.2650 | 0.7365 |
| AMZN | calendar | 2s | L4 | 0.8331 | 0.8300 | **0.0031** | 0.0031 | 0 | 0.9713 | 1.0002 | -0.0289 |
| AMZN | calendar | 5s | L1 | 0.5667 | 0.5355 | **0.0313** | 0.0030 | 1 | 1.3391 | 1.8715 | -0.5324 |
| AMZN | calendar | 5s | L3 | 0.6103 | 0.5753 | **0.0351** | 0.0032 | 1 | 1.0554 | 0.7221 | 0.3333 |
| AMZN | calendar | 5s | L4 | 0.8392 | 0.8355 | **0.0037** | 0.0037 | 0 | 1.0093 | 0.9994 | 0.0099 |
| AMZN | calendar | 10s | L1 | 0.5515 | 0.5192 | **0.0324** | 0.0011 | 2 | 1.5965 | 0.9711 | 0.6254 |
| AMZN | calendar | 10s | L3 | 0.5898 | 0.5624 | **0.0275** | -0.0008 | 1 | 1.0281 | 0.5478 | 0.4803 |
| AMZN | calendar | 10s | L4 | 0.8453 | 0.8434 | **0.0019** | 0.0019 | 0 | 1.0167 | 0.9842 | 0.0325 |
| AMZN | event | 1s | L1 | 0.6088 | 0.6052 | **0.0036** | 0.0036 | 0 | 1.0308 | 1.4318 | -0.4010 |
| AMZN | event | 1s | L3 | 0.7504 | 0.7496 | **0.0008** | 0.0008 | 0 | 1.0277 | 1.3322 | -0.3045 |
| AMZN | event | 1s | L4 | 0.8826 | 0.8813 | **0.0012** | 0.0012 | 0 | 1.0138 | 1.0518 | -0.0380 |
| AMZN | event | 2s | L1 | 0.6021 | 0.5968 | **0.0053** | 0.0053 | 0 | 1.0462 | 1.6740 | -0.6279 |
| AMZN | event | 2s | L3 | 0.7282 | 0.7255 | **0.0027** | 0.0027 | 0 | 0.9853 | 1.6068 | -0.6215 |
| AMZN | event | 2s | L4 | 0.8638 | 0.8614 | **0.0024** | 0.0024 | 0 | 0.9763 | 1.0351 | -0.0589 |
| AMZN | event | 5s | L1 | 0.5882 | 0.5588 | **0.0294** | 0.0045 | 1 | 1.3469 | 1.4759 | -0.1290 |
| AMZN | event | 5s | L3 | 0.6960 | 0.6315 | **0.0645** | -0.0005 | 1 | 0.9922 | -1.0963 | 2.0885 |
| AMZN | event | 5s | L4 | 0.8475 | 0.8430 | **0.0045** | 0.0045 | 0 | 0.9733 | 0.9992 | -0.0259 |
| AMZN | event | 10s | L1 | 0.5701 | 0.5469 | **0.0232** | 0.0006 | 1 | 1.2381 | 0.7976 | 0.4404 |
| AMZN | event | 10s | L3 | 0.6593 | 0.6001 | **0.0593** | -0.0009 | 1 | 0.9572 | 0.0768 | 0.8805 |
| AMZN | event | 10s | L4 | 0.8320 | 0.8301 | **0.0020** | 0.0020 | 0 | 0.9666 | 1.0002 | -0.0337 |

The `A-C calib` column is a MEASUREMENT OF THE CALIBRATION DESIGN, not of the protocols. Under A the holdout is exchangeable with A's test fold, so A's slope is driven toward 1.0 for any predictor -- these frozen rows, which cannot leak, reproduce the fitted rungs' published A-vs-C slope split to three decimals. Do not read it as leakage.

## The floor over four fit windows -- and the WORST one is the one to quote

Added 2026-08-03 after the fourth red-team pass. Every floor above fits the frozen model on the WHOLE DAY, and that is the choice that MINIMISES the floor. At L4's magnitude it does so mechanically: a whole-day GBM scores AUC ~0.996, a ceiling where two protocols have almost no room to differ, while the rung it is meant to bound scores ~0.72 under C. Refitting the same frozen model on half the day puts its accuracy in the rungs' operating range and the floor grows several-fold. A margin quoted against the whole-day floor is therefore an overstatement, and the earlier '98x its floor' claim was exactly that.

`max` is the worst floor across fit windows for that cell. **Quote margins against `max`.**

| ticker | clock | h | frozen at | metric | whole | first 50% | last 50% | random 50% | **max** |
|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L1 | r2 | 0.00302 | 0.00302 | 0.00302 | 0.00302 | **0.00302** |
| INTC | calendar | 1s | L1 | auc_raw | 0.0026 | 0.0026 | 0.0026 | 0.0026 | **0.0026** |
| INTC | calendar | 1s | L3 | r2 | 0.00425 | 0.00120 | 0.00234 | 0.00396 | **0.00425** |
| INTC | calendar | 1s | L3 | auc_raw | -0.0038 | -0.0063 | -0.0176 | -0.0037 | **-0.0037** |
| INTC | calendar | 1s | L4 | r2 | 0.01917 | 0.01943 | -0.03712 | 0.01592 | **0.01943** |
| INTC | calendar | 1s | L4 | auc_raw | 0.0010 | 0.0253 | 0.0157 | 0.0008 | **0.0253** |
| INTC | calendar | 2s | L1 | r2 | 0.00602 | 0.00602 | 0.00602 | 0.00602 | **0.00602** |
| INTC | calendar | 2s | L1 | auc_raw | 0.0023 | 0.0023 | 0.0023 | 0.0023 | **0.0023** |
| INTC | calendar | 2s | L3 | r2 | 0.00777 | 0.00403 | 0.00502 | 0.00811 | **0.00811** |
| INTC | calendar | 2s | L3 | auc_raw | -0.0014 | -0.0059 | -0.0151 | -0.0015 | **-0.0014** |
| INTC | calendar | 2s | L4 | r2 | 0.02205 | 0.02038 | -0.03099 | 0.02056 | **0.02205** |
| INTC | calendar | 2s | L4 | auc_raw | 0.0006 | 0.0327 | 0.0233 | 0.0002 | **0.0327** |
| INTC | calendar | 5s | L1 | r2 | 0.00816 | 0.00816 | 0.00816 | 0.00816 | **0.00816** |
| INTC | calendar | 5s | L1 | auc_raw | 0.0012 | 0.0012 | 0.0012 | 0.0012 | **0.0012** |
| INTC | calendar | 5s | L3 | r2 | 0.01021 | 0.00424 | 0.00947 | 0.01009 | **0.01021** |
| INTC | calendar | 5s | L3 | auc_raw | -0.0025 | -0.0045 | -0.0138 | -0.0026 | **-0.0025** |
| INTC | calendar | 5s | L4 | r2 | 0.02944 | 0.03380 | -0.00953 | 0.02738 | **0.03380** |
| INTC | calendar | 5s | L4 | auc_raw | 0.0012 | 0.0355 | 0.0309 | 0.0013 | **0.0355** |
| INTC | calendar | 10s | L1 | r2 | 0.00536 | 0.00536 | 0.00536 | 0.00536 | **0.00536** |
| INTC | calendar | 10s | L1 | auc_raw | 0.0022 | 0.0022 | 0.0022 | 0.0022 | **0.0022** |
| INTC | calendar | 10s | L3 | r2 | 0.00778 | -0.00082 | 0.00761 | 0.00735 | **0.00778** |
| INTC | calendar | 10s | L3 | auc_raw | -0.0008 | -0.0080 | -0.0127 | -0.0013 | **-0.0008** |
| INTC | calendar | 10s | L4 | r2 | 0.02995 | -0.02758 | -0.00179 | 0.03043 | **0.03043** |
| INTC | calendar | 10s | L4 | auc_raw | 0.0004 | 0.0383 | 0.0353 | 0.0004 | **0.0383** |
| INTC | event | 1s | L1 | r2 | 0.01815 | 0.01815 | 0.01815 | 0.01815 | **0.01815** |
| INTC | event | 1s | L1 | auc_raw | 0.0014 | 0.0014 | 0.0014 | 0.0014 | **0.0014** |
| INTC | event | 1s | L3 | r2 | 0.02092 | 0.02041 | 0.01316 | 0.02187 | **0.02187** |
| INTC | event | 1s | L3 | auc_raw | 0.0043 | -0.0076 | 0.0059 | 0.0044 | **0.0059** |
| INTC | event | 1s | L4 | r2 | 0.00721 | 0.00713 | 0.03376 | 0.00169 | **0.03376** |
| INTC | event | 1s | L4 | auc_raw | 0.0006 | 0.0001 | 0.0198 | 0.0003 | **0.0198** |
| INTC | event | 2s | L1 | r2 | 0.01499 | 0.01499 | 0.01499 | 0.01499 | **0.01499** |
| INTC | event | 2s | L1 | auc_raw | -0.0064 | -0.0064 | -0.0064 | -0.0064 | **-0.0064** |
| INTC | event | 2s | L3 | r2 | 0.01126 | 0.01399 | 0.01366 | 0.01193 | **0.01399** |
| INTC | event | 2s | L3 | auc_raw | 0.0023 | -0.0090 | 0.0029 | 0.0025 | **0.0029** |
| INTC | event | 2s | L4 | r2 | 0.00611 | -0.00371 | 0.03163 | 0.00406 | **0.03163** |
| INTC | event | 2s | L4 | auc_raw | 0.0007 | -0.0010 | 0.0281 | 0.0007 | **0.0281** |
| INTC | event | 5s | L1 | r2 | 0.00681 | 0.00681 | 0.00681 | 0.00681 | **0.00681** |
| INTC | event | 5s | L1 | auc_raw | -0.0125 | -0.0125 | -0.0125 | -0.0125 | **-0.0125** |
| INTC | event | 5s | L3 | r2 | 0.00678 | -0.00235 | 0.01353 | 0.00846 | **0.01353** |
| INTC | event | 5s | L3 | auc_raw | 0.0037 | -0.0153 | 0.0066 | 0.0037 | **0.0066** |
| INTC | event | 5s | L4 | r2 | 0.01804 | -0.04524 | 0.05621 | 0.01560 | **0.05621** |
| INTC | event | 5s | L4 | auc_raw | 0.0013 | 0.0210 | 0.0361 | 0.0014 | **0.0361** |
| INTC | event | 10s | L1 | r2 | 0.00432 | 0.00432 | 0.00432 | 0.00432 | **0.00432** |
| INTC | event | 10s | L1 | auc_raw | -0.0091 | -0.0091 | -0.0091 | -0.0091 | **-0.0091** |
| INTC | event | 10s | L3 | r2 | 0.00849 | -0.00869 | 0.01696 | 0.01120 | **0.01696** |
| INTC | event | 10s | L3 | auc_raw | -0.0021 | -0.0426 | 0.0036 | -0.0009 | **0.0036** |
| INTC | event | 10s | L4 | r2 | 0.02413 | 0.02619 | 0.01783 | 0.02650 | **0.02650** |
| INTC | event | 10s | L4 | auc_raw | 0.0022 | 0.0046 | 0.0367 | 0.0012 | **0.0367** |
| AMZN | calendar | 1s | L1 | r2 | 0.00218 | 0.00218 | 0.00218 | 0.00218 | **0.00218** |
| AMZN | calendar | 1s | L1 | auc_raw | 0.0042 | 0.0042 | 0.0042 | 0.0042 | **0.0042** |
| AMZN | calendar | 1s | L3 | r2 | 0.00737 | 0.00746 | 0.00494 | 0.00746 | **0.00746** |
| AMZN | calendar | 1s | L3 | auc_raw | 0.0015 | -0.0103 | 0.0009 | 0.0023 | **0.0023** |
| AMZN | calendar | 1s | L4 | r2 | 0.01094 | -0.00108 | -0.01228 | 0.00804 | **0.01094** |
| AMZN | calendar | 1s | L4 | auc_raw | 0.0015 | 0.0068 | 0.0128 | 0.0019 | **0.0128** |
| AMZN | calendar | 2s | L1 | r2 | 0.00229 | 0.00229 | 0.00229 | 0.00229 | **0.00229** |
| AMZN | calendar | 2s | L1 | auc_raw | 0.0043 | 0.0043 | 0.0043 | 0.0043 | **0.0043** |
| AMZN | calendar | 2s | L3 | r2 | 0.01199 | 0.01193 | 0.00823 | 0.01206 | **0.01206** |
| AMZN | calendar | 2s | L3 | auc_raw | 0.0035 | -0.0101 | 0.0032 | 0.0033 | **0.0035** |
| AMZN | calendar | 2s | L4 | r2 | 0.01914 | -0.00107 | -0.00541 | 0.01136 | **0.01914** |
| AMZN | calendar | 2s | L4 | auc_raw | 0.0031 | 0.0079 | 0.0143 | 0.0019 | **0.0143** |
| AMZN | calendar | 5s | L1 | r2 | 0.00366 | 0.00366 | 0.00366 | 0.00366 | **0.00366** |
| AMZN | calendar | 5s | L1 | auc_raw | 0.0030 | 0.0030 | 0.0030 | 0.0030 | **0.0030** |
| AMZN | calendar | 5s | L3 | r2 | 0.01107 | 0.01183 | 0.00086 | 0.01149 | **0.01183** |
| AMZN | calendar | 5s | L3 | auc_raw | 0.0032 | -0.0101 | 0.0027 | 0.0029 | **0.0032** |
| AMZN | calendar | 5s | L4 | r2 | 0.02350 | 0.00054 | -0.00559 | 0.01174 | **0.02350** |
| AMZN | calendar | 5s | L4 | auc_raw | 0.0037 | 0.0080 | 0.0168 | 0.0025 | **0.0168** |
| AMZN | calendar | 10s | L1 | r2 | 0.00425 | 0.00425 | 0.00425 | 0.00425 | **0.00425** |
| AMZN | calendar | 10s | L1 | auc_raw | 0.0011 | 0.0011 | 0.0011 | 0.0011 | **0.0011** |
| AMZN | calendar | 10s | L3 | r2 | 0.01371 | 0.01363 | -0.00528 | 0.01590 | **0.01590** |
| AMZN | calendar | 10s | L3 | auc_raw | -0.0008 | -0.0091 | -0.0004 | -0.0012 | **-0.0004** |
| AMZN | calendar | 10s | L4 | r2 | 0.01920 | 0.01711 | -0.02168 | 0.01293 | **0.01920** |
| AMZN | calendar | 10s | L4 | auc_raw | 0.0019 | 0.0196 | 0.0199 | 0.0014 | **0.0199** |
| AMZN | event | 1s | L1 | r2 | 0.00333 | 0.00333 | 0.00333 | 0.00333 | **0.00333** |
| AMZN | event | 1s | L1 | auc_raw | 0.0036 | 0.0036 | 0.0036 | 0.0036 | **0.0036** |
| AMZN | event | 1s | L3 | r2 | 0.04280 | 0.03183 | 0.03428 | 0.04294 | **0.04294** |
| AMZN | event | 1s | L3 | auc_raw | 0.0008 | -0.0070 | -0.0050 | 0.0006 | **0.0008** |
| AMZN | event | 1s | L4 | r2 | 0.02349 | 0.00171 | 0.01628 | 0.02256 | **0.02349** |
| AMZN | event | 1s | L4 | auc_raw | 0.0012 | -0.0047 | 0.0080 | 0.0009 | **0.0080** |
| AMZN | event | 2s | L1 | r2 | 0.00369 | 0.00369 | 0.00369 | 0.00369 | **0.00369** |
| AMZN | event | 2s | L1 | auc_raw | 0.0053 | 0.0053 | 0.0053 | 0.0053 | **0.0053** |
| AMZN | event | 2s | L3 | r2 | 0.05806 | 0.05760 | 0.04112 | 0.05730 | **0.05806** |
| AMZN | event | 2s | L3 | auc_raw | 0.0027 | -0.0013 | 0.0020 | 0.0025 | **0.0027** |
| AMZN | event | 2s | L4 | r2 | 0.03391 | 0.03662 | -0.00053 | 0.02216 | **0.03662** |
| AMZN | event | 2s | L4 | auc_raw | 0.0024 | 0.0003 | 0.0105 | -0.0002 | **0.0105** |
| AMZN | event | 5s | L1 | r2 | 0.00533 | 0.00533 | 0.00533 | 0.00533 | **0.00533** |
| AMZN | event | 5s | L1 | auc_raw | 0.0045 | 0.0045 | 0.0045 | 0.0045 | **0.0045** |
| AMZN | event | 5s | L3 | r2 | 0.02801 | 0.03611 | -0.00653 | 0.02542 | **0.03611** |
| AMZN | event | 5s | L3 | auc_raw | -0.0005 | -0.0009 | 0.0012 | -0.0005 | **0.0012** |
| AMZN | event | 5s | L4 | r2 | 0.01505 | 0.00736 | -0.00142 | 0.01045 | **0.01505** |
| AMZN | event | 5s | L4 | auc_raw | 0.0045 | 0.0002 | 0.0126 | 0.0026 | **0.0126** |
| AMZN | event | 10s | L1 | r2 | 0.00380 | 0.00380 | 0.00380 | 0.00380 | **0.00380** |
| AMZN | event | 10s | L1 | auc_raw | 0.0006 | 0.0006 | 0.0006 | 0.0006 | **0.0006** |
| AMZN | event | 10s | L3 | r2 | 0.03805 | 0.05374 | -0.02013 | 0.04144 | **0.05374** |
| AMZN | event | 10s | L3 | auc_raw | -0.0009 | -0.0024 | 0.0016 | -0.0022 | **0.0016** |
| AMZN | event | 10s | L4 | r2 | 0.01160 | 0.01276 | -0.04079 | 0.01694 | **0.01694** |
| AMZN | event | 10s | L4 | auc_raw | 0.0020 | 0.0041 | 0.0150 | 0.0002 | **0.0150** |
