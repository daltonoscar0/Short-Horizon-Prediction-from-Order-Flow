# FINDINGS

The claim, the evidence, the failure modes, and what would change my mind.

Everything here is one trading day, 2012-06-21, INTC and AMZN, LOBSTER level-10. Read
[Limitations](#5-limitations-that-bound-every-claim-above) before quoting any number.

---

## 1. The claim

**Leaky cross-validation buys optimism in proportion to model capacity, and what it buys is
training rows *adjacent* to the test rows, not the future.**

Median out-of-sample R² against a zero forecast, across all 96 (ticker × clock × horizon)
cells, with the same features, rows, folds and seed under every protocol:

| rung | params | A − C (total) | *artifact floor* | A − D (adjacency) | D − C (future) | B − C (purge) | C (honest) | C worst cell |
|---|---|---|---|---|---|---|---|---|
| L0 zero | 0 | 0.0000 |, | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| L1 `sign(queue imbalance)` | 0 | +0.0043 | *+0.0043* | +0.0126 | −0.0085 | +0.0006 | +0.0091 | −0.0019 |
| L1m `sign(microprice − mid)` | 0 | +0.0058 | *+0.0043* | +0.0030 | +0.0007 | +0.0001 | +0.0042 | −0.0040 |
| L2 OLS on OFI | 2 | +0.0204 | *+0.0043* | +0.0032 | +0.0030 | +0.0018 | −0.0018 | −0.0971 |
| L3 ridge, full set | 28 | +0.0587 | *+0.0112* | +0.0296 | +0.0231 | +0.0019 | **+0.0266** | −0.0450 |
| L4 GBM, full set | many | **+0.3173** | *+0.0192* | **+0.2964** | +0.0067 | +0.0025 | **+0.0273** | −0.0003 |

> **These numbers replace an earlier draft of this file.** That draft reported L4's gap as
> +0.5453 and its honest R² as −0.2164, and concluded that the ladder inverts under protocol C.
> Both were artifacts of an in-sample calibration; see §3. The gap is real and still large, but
> it is roughly a third to a half of what was first published, and the inversion did not
> survive at all. The superseded columns remain in `reports/ladder.md`.
>
> **The `A − C` column was suspected of bundling a second mechanism. It does not.** A first
> control (`A′`) attributed about half of L4's gap to the harness's own calibration rule; a
> clean control built afterwards showed that attribution was itself confounded, with the
> model's fit rows held fixed the rule is worth **exactly zero** on raw AUC. §1.1a carries both
> controls and the correction.

### 1.1 The control that makes the table readable, and why L1 was not it

The gap needs a floor: some of any A-versus-C difference is the protocols not being comparable
rather than leakage. An earlier draft used L1 as that floor, on the grounds that it has "no
fitted parameters beyond a one-parameter rescale". **The parenthetical was doing all the work.**
That rescale *is* fitted, so L1's gap mixes a partition artifact with a refit effect, and the
red team showed the two together account for the entire L1/L1m/L2 gap.

The correct control is **frozen**: one `sign(queue imbalance)` model fitted once on every usable
row of the day, then scored *unchanged* under each protocol. Nothing is refitted between
protocols, so every difference is a pure partition artifact and none of it can be leakage.

It is not zero:

```
INTC calendar 1s   frozen sign(qi)   A=+0.01246  C=+0.01110   A-C = +0.00136
INTC calendar 10s  frozen GBM        A=+0.54618  C=+0.51908   A-C = +0.02710
```

and on the scale-free metric it reproduced the old L1 row **exactly**, frozen A−C IC
of +0.00177 against the published L1 row's +0.00176 (INTC cal 1s), +0.01577 against +0.01577
(INTC event 1s). That equality is the proof that the L1/L1m/L2 rows contained *no leakage
signal at all*.

**The mechanism** is fold aggregation, not leakage. Protocol A's five folds are random subsets
of one universe, so its fold mean approximates the `Σy²`-weighted pooled metric. Protocol C's
folds are contiguous time segments with very unequal `Σy²` (measured shares at INTC calendar
1s: 0.188 / 0.260 / 0.158 / 0.108 / 0.285), so its fold mean is *unweighted*. Mean-over-folds
estimates two different quantities. The SS-weighted average of C's per-fold R² equals A's fold
mean to five decimals.

Every headline cell now ships with its frozen-predictor floor beside it. A rung whose gap does
not clear its floor has measured nothing.

**This control was itself insufficient, and §1.1a is the repair.** Freezing the model is not
enough when the harness also fits a *scale* per fold under a protocol-dependent rule. The
numbers quoted immediately above (+0.00136, +0.02710) are the frozen-*scale* variant, which
measures the partition artifact alone; the floors in the table above are the like-for-like ones,
with the scale refitted per fold exactly as the rungs get it, and they are 2-6× larger.

### 1.1a Two channels inside `A − C`, and the claim restated on a scale-free metric

**A second thing was wrong with the floor, and it was found on 2026-08-03 by the metrics added
that same day.** The floor above froze the *scale* along with the model. But the harness fits
each rung's output scale on a held-out slice of its training rows, and the rule for choosing
that slice is **protocol-dependent**: a random 20% under A, the last 20% in time under B/C/D.
Under A that slice is *exchangeable with A's random test fold*; under walk-forward it is not.
Freezing the scale switched that channel off inside the control while leaving every measured
rung exposed to it, a control that could not see the thing it was controlling for.

The first attempt to separate the two was to re-run protocol A under C's positional rule (`A′`).
It is **superseded**, the numbers are kept because the correction that followed is only legible
beside them:

| rung | `A − C` | `A′ − C`, *rule "held constant"* | *floor* (like-for-like) |
|---|---|---|---|
| L1 `sign(queue imbalance)` | +0.0043 | −0.0021 | *+0.0043* |
| L3 ridge, full set | +0.0587 | +0.0542 | *+0.0112* |
| L4 GBM, full set | **+0.3173** | +0.1379 | *+0.0192* |

**`A′` is not a clean control, and it turned out to be measuring the wrong thing entirely.**
`Rung.fit` fits the raw model on the *complement* of the calibration holdout, so changing the
rule also changes which rows the model trains on, under `A′` the GBM sees the first 80% of the
day in time and never sees its tail. `A′` is a temporally handicapped model, not the same model
calibrated differently.

**The clean control** (`--holdout-control`, 2026-08-03) fixes that by fitting the raw model on
the *same seeded random 80%* of each fold's training rows in every arm, and moving only the rows
the scale is fitted on, accepting that the scale rows then overlap the model's.

**That overlap has a cost, and it is not symmetric.** It makes each arm's calibration partly
in-sample, the artifact killed on 2026-08-02, and the bias grows with capacity per training
row, so it lands almost entirely on the walk-forward side: A's arm is unharmed (R² 0.18378
against the main ladder's 0.18316 at INTC calendar 1s) while C's is destroyed (−0.25953 against
+0.02134). Under permuted labels the construction manufactures an R² gap of **+0.062 ± 0.058**
out of pure noise, against **+0.010 ± 0.009** for the main ladder. An earlier draft of this
paragraph said the overlap was identical in every arm and the differences therefore clean; that
is false for R² and it is withdrawn. **Only the rank columns of this control are usable**, where
the permutation null is the same as the main ladder's and unbiased.

On raw-score AUC it settles the question exactly, because raw AUC is a rank statistic on the raw
model and therefore *cannot see the scale at all*. With the model held fixed, the calibration
rule moves it by **0.000000 across all 96 rows** of the control table. That is an identity rather
than a measurement, and it is the point:

| rung | uncontrolled `A − C` | `A′ − C` (confounded) | **clean split-only** | cells positive |
|---|---|---|---|---|
| L1 `sign(queue imbalance)` | +0.0022 | +0.0023 | **+0.0023** | 13 / 16, = its own floor |
| L1m `sign(microprice − mid)` | −0.0018 | −0.0018 | **−0.0018** |, |
| L2 OLS on OFI | −0.0002 | −0.0002 | **−0.0002** |, |
| L3 ridge, full set | +0.0409 | +0.0283 | **+0.0409** | **16 / 16** |
| L4 GBM, full set | +0.1740 | +0.1371 | **+0.1913** | **16 / 16** |

**This reverses an attribution published earlier the same day.** `A′ − C` shrank L4's raw-AUC gap
from +0.1740 to +0.1371 and that shrink was written up as the calibration rule's contribution. It
cannot have been, with the model held fixed the rule is worth exactly nothing on this metric. All
of it was the model's lost training window. The split-only gap is **+0.1913**, *larger* than the
uncontrolled figure, positive in 16 of 16 cells, ranging +0.0657 to +0.3132.

So "about half of L4's gap is the calibration rule" is withdrawn. The R² version of the same
statistic is withdrawn with it: under permuted labels this construction manufactures an R² gap of
+0.062 ± 0.058 out of pure noise (against +0.010 ± 0.009 for the main ladder), and a constant
predictor scores +0.0082 on it. The decoupled arm's R² columns measure the overlap, not the rule.

**The obvious objection is that the common model-fit rule is itself a free parameter, chosen
after seeing the old one shrink the gap.** It does not survive a sweep. The same L4 `A − C`
raw-AUC gap across all 16 cells under six different common model-fit rules (measured by the fifth
red team, independently):

| common model-fit rule | L4 median gap | min cell | cells positive |
|---|---|---|---|
| each protocol's own coupled rule (the headline `A − C`) | +0.1740 | +0.0493 | 16 / 16 |
| `first 80%` positional, *this is the superseded `A′ − C`* | **+0.1371** | +0.0297 | 16 / 16 |
| **`random 80%`, seed 0, the published clean control** | **+0.1913** | +0.0657 | 16 / 16 |
| `random 80%`, a different seed | +0.1927 | +0.0649 | 16 / 16 |
| `last 80%` positional | +0.1977 | +0.0672 | 16 / 16 |
| `all 100%`, no model holdout at all | +0.2010 | +0.0649 | 16 / 16 |

Five of six land at +0.17-+0.20. The *only* rule producing a materially smaller number is the one
that withholds the day's tail from protocol A, i.e. that removes the adjacency channel under
study from the arm being measured. Diagnostically, C's level barely moves across rules (0.706 to
0.717) while A's does (0.9059 headline, 0.8556 under `first 80%`). `A′` was low because it
crippled A. The narrowest correct statement is therefore a range, not a point:
**+0.19 to +0.20 under every model-fit rule that does not strip A of its own adjacency**, with
+0.137 reachable only by stripping it. The 80% fraction was swept too (50/80/90/95/99/100%) and
the residual dependence runs *against* the published number, not for it.

One caveat that must travel with the R² columns of that control table, and it is a familiar one:
**the overlap makes each arm's scale partly in-sample**, which is the calibration killed on
2026-08-02, and it bites here exactly as it did then, under the walk-forward arms a
high-capacity rung's amplitude is far too large and its R²-vs-zero goes sharply negative. The
overlap lands asymmetrically, on the walk-forward side, where capacity per training row is
highest, so **neither the levels nor the differences of that table are usable**, and none of it
is quoted. Raw AUC has no amplitude in it and is free of this, which is the third independent
reason this project ended up quoting a rank metric.

**On directional AUC**, which has no amplitude in it at all, up versus down on the rows the
deadband calls directional, scored on the **raw** output:

| rung | AUC under A | AUC under C | A − C | **clean split-only** | cells positive | *floor* |
|---|---|---|---|---|---|---|
| L0 zero | 0.5000 | 0.5000 | 0.0000 | 0.0000 | 0 / 16 |, |
| L1 `sign(queue imbalance)` | 0.6545 | 0.6573 | +0.0022 | +0.0023 | 13 / 16 | *+0.0031* |
| L1m `sign(microprice − mid)` | 0.5880 | 0.5892 | **−0.0018** | −0.0018 | 6 / 16 | *+0.0031* |
| L2 OLS on OFI | 0.6546 | 0.6567 | **−0.0002** | −0.0002 | 7 / 16 | *+0.0031* |
| L3 ridge, full set | 0.7811 | **0.7319** | +0.0409 | **+0.0409** | **16 / 16** | *+0.0025* |
| L4 GBM, full set | 0.9059 | 0.7122 | **+0.1740** | **+0.1913** | **16 / 16** | *+0.0014* |

**Raw, because the rescale is rank-preserving only up to sign.** The fitted scale comes out
negative in 27 of 400 protocol-C folds and 0 of 400 protocol-A folds; a negative scale flips the
ranking and sends AUC to 1−AUC. On the rescaled column that channel alone manufactured the
unfitted rungs' entire apparent gap, L1m read +0.0035 and L2 +0.0078 where the raw figures are
negative. The fitted rungs are untouched (L4 identical to four decimals), so the capacity result
survives and the "smooth progression across all six rungs" reading does not.

**What this table supports, stated as narrowly as it deserves.** The two controls compose, the
raw score removes the rescale's sign, the decoupled arm removes the calibration rule, and both
fitted rungs survive both, while all three unfitted rungs sit at zero to three decimals.

But the margin must be quoted against the **worst** floor, not the published one. Every floor
here fits the frozen model on the whole day, which is the choice that *minimises* it: at L4's
magnitude a whole-day GBM scores AUC ≈0.996, a **ceiling** where two protocols have almost no
room to differ, against the ≈0.72 of the rung it is meant to bound. Refit the frozen model on
half the day and the same floor grows several-fold, to +0.0383 in the worst cell.

| rung | clean split-only raw AUC | worst-window floor | margin | cells positive |
|---|---|---|---|---|
| L3 ridge | +0.0409 | up to +0.0066 | **12.5×** (min cell **7.3×**) | 16 / 16 |
| L4 GBM | +0.1913 | up to +0.0383 | **8.1×** (min cell **3.4×**) | **16 / 16** |

**L4 survives at 8.1× its worst-case floor, median of per-cell ratios, worst cell 3.4×, not the
98× an earlier version of this section claimed** against a floor measured at that ceiling. That is capacity-scaling optimism,
confirmed independently of the R² machinery, of the Σy²-weighting artifact, and of the
calibration rule, and it is the version to quote. What it does **not** support: a five-point
progression, and it supports L3 far more weakly than L4. On the **clean** arm, L4's gap is
positive in **80 of 80** folds with a worst fold of **+0.0157** and 64/64 excluding fold 0; L3's
is positive in only **55 of 80**, worst **−0.0829**. The "16 / 16 cells" column is true for both
and hides that difference entirely. (An earlier draft parked this caveat on the superseded `A′`
arm; recomputed on the clean one L3 is identical and L4 is slightly better.)

One number that bounds all of this: the per-cell shuffled-label null for the raw-AUC gap has
**sd ≈ 0.05**, so L4's minimum cell (+0.0657) is only about 1.3 sd from noise. It is the *fold
count*, 80 of 80, that carries the result, not any single cell.

An earlier version of this caveat compared L3's *raw* gap against its *rescaled* floor and
concluded it fell below the floor in 4 of 16 cells. That was the same metric-mismatch the
2026-08-02 pass killed the report over, pointed the other way, it under-claimed. Raw against
raw, L3 clears in 16 of 16 at a minimum ratio of 5.6.

**And a mechanism claimed here has been withdrawn.** An earlier draft said the AUC floor is flat
*because* a rank statistic has no analogue of the Σy² weighting asymmetry, that was the stated
reason for restating the headline on AUC at all. It is wrong. Protocol A's per-fold AUC is a
whole-day **pooled** AUC while C's is a **within-block** AUC; any frozen score with a
time-varying offset makes those estimands diverge with nothing refitted, and the red team drove
the floor to −0.137 that way. The AUC floor is small *here* because these particular frozen
predictors are temporally homogeneous. It is a measurement on this data, not a property of the
metric, and on other data it would have to be re-measured rather than assumed.

### 1.1b The alternative explanation, and the fold that rules it out

The standing objection to everything above is that protocol C's early folds simply train on
less data and on a narrower slice of the day. "Optimism scales with capacity" and "small
training sets hurt high-capacity models most" predict the same table, and on a single session
they are not separable by argument. They are separable by one measurement.

Across protocol C's five folds, A′'s training set shrinks from 7.79× C's to **1.06×** C's:

| fold | A′ / C training rows | L4 median `A′ − C` raw AUC |
|---|---|---|
| 0 | 7.79 | +0.1940 |
| 1 | 2.88 | +0.1845 |
| 2 | 1.77 | +0.1305 |
| 3 | 1.31 | +0.1158 |
| **4** | **1.06** | **+0.0940** |

At fold 4 the two protocols train on effectively the same number of rows and the shuffled
protocol still ranks up-moves above down-moves **9.4 AUC points** better. The gap shrinks with
the size ratio, so size is part of the story, but it does not go to zero when the ratio does.

Protocol D says the same thing from the other direction: handing the model the entire day
outside its test block, past *and* future, buys L4 **+0.0067 R²** (`D − C`). If coverage of the
day's regimes were the mechanism, D would capture most of the gap. It captures 2% of it.

Two caveats travel with the table. A median of per-cell differences is **not** the difference of
the medians, L1's median A sits just below its median C while 13 of 16 individual cells show a
positive gap, so the gap column is the median of per-cell gaps and the A and C columns are
reported rather than subtracted. And **every rung's worst fold is below 0.5** (L1 0.4294, L3
0.4552, L4 0.4725): in every configuration there is at least one block where the ranking is
inverted.

### 1.2 Where the optimism actually comes from, and the claim this replaces

An earlier draft of this file read the near-zero `B − C` column as "label overlap contributes
nothing, so purging barely matters". **That was wrong**, and the phase-2 red team disproved it
with a control the harness did not have. The corrected account:

`B − C` is near zero for an *arithmetic* reason, not a substantive one. Under expanding
walk-forward, purge and embargo can only remove the last `h + E` of a training set that spans
hours, 578 of 90,333 rows (0.64%) at INTC calendar 1s, 274 of 25,345 (1.1%) on the event
clock. A column that can touch 1% of the training set is not a measurement of label overlap in
general; it measures label overlap *at a walk-forward boundary*, where it is negligible by
construction.

To find where the optimism lives, the harness now runs **protocol D**: C's test blocks exactly,
but trained on everything outside them, past *and* future, unpurged. D holds the test
universe, the fold shapes and the fold aggregation fixed and varies only "may the model see the
future". The decomposition that gives:

Median R² across all 96 cells, from this repo's own run:

| rung | `A − C` total | `A − D` adjacency | `D − C` future | adjacency share |
|---|---|---|---|---|
| L1 `sign(queue imbalance)` | +0.0043 | +0.0126 | −0.0085 |, |
| L3 ridge | +0.0587 | +0.0296 | +0.0231 | 56% |
| L4 GBM | **+0.3173** | **+0.2964** | +0.0067 | **93%** |

Handing a model the entire future in blocks *disjoint* from the test set buys the GBM almost
nothing (+0.007). Nearly all of protocol A's advantage, 93% of it, comes from training rows
that are *temporally adjacent* to test rows. That is label overlap, and it sits **inside the
`A − B` column**, not in `B − C`.

The split is capacity-dependent, which is itself informative: for the 28-parameter ridge the
future is worth nearly as much as adjacency (44%), while the GBM's advantage is almost purely
memorisation of neighbouring rows. A model with enough capacity to memorise does not need the
future; it needs a near-duplicate.

So the reviewer-facing advice is the opposite of what the earlier draft said: label overlap is
the dominant mechanism. It is simply invisible in `B − C` under this fold geometry.

### 1.3 Under the correct protocol the ladder does not invert, it flattens

**A claim was withdrawn here.** An earlier draft said L3 and L4 go negative under C and that
the parameter-free `sign(queue imbalance)` is the best rung. That was the in-sample calibration
artifact (§3), and it does not survive the fix. What the corrected numbers say:

- L4 (+0.0273) and L3 (+0.0266) both beat L1 (+0.0091) on median protocol-C R², and **L4 beats
  L1 in 13 of 16 cells**. L4's worst cell (−0.0003) is better than L1's (−0.0019).
- On the scale-free metrics the order changes again, and **two of them now agree**: median
  protocol-C IC runs **L3 +0.2262 > L4 +0.1751 > L1 +0.1465**, and median protocol-C AUC runs
  **L3 0.7319 > L4 0.7122 > L1 0.6573**, with L4 beating L3 in only 6 of 16 cells against L3
  beating L1 in 16 of 16. The ridge, not the GBM, is the best-ranked rung, a conclusion
  R²-vs-zero reverses, and the disagreement is itself the point: R² rewards getting the
  amplitude right, rank metrics reward getting the order right, and the GBM wins the first
  while the ridge wins the second.
- Median balanced accuracy runs L4 0.3926 > L3 0.3760 > L1 **0.3333, chance exactly**, because
  a scaled sign predictor never clears the 0.5-tick deadband (§4). This is why AUC was worth
  adding: it is the only classification column that carries directional information for the
  four unfitted rungs, since it ranks the raw prediction instead of thresholding it.
- The calibration slope is reported because `EVAL_PROTOCOL.md` §2 asks for it, and **nothing
  here rests on it.** A draft of this file read protocol A's near-1.0 slopes as a leakage
  signature; that was withdrawn the same day. Under A the scale's holdout is exchangeable with
  A's test fold, so the slope goes to ≈1.0 mechanically for any predictor, a frozen model that
  cannot leak reproduces the split to three decimals, and holding the rule constant reverses it
  for L4. It measures the calibration design, not the protocol.

So the capacity is worth something, and the honest answer to PLAN.md's "does anything beat
`sign(queue imbalance)`?" is **yes, on prediction**, by about 3× in R² and by a smaller margin
in rank correlation.

The more useful framing is the size of what is left. L4's honest median R² of **0.0273** is
**12.6×** smaller than the 0.3434 protocol A advertises for the same model on the same rows;
for L3 the ratio is 3.1×. A reader of the shuffled-k-fold number would form an expectation an
order of magnitude too high for the high-capacity model, and, this is the part that matters
for model selection, **the inflation is not uniform across rungs**, so the shuffled protocol
also reorders the ladder. That, not an inversion, is the finding.

Split by clock, because a calendar-time result on a quiet instrument is partly measuring the
clock, and by tick regime, because that drives which features work:

Median protocol-C R², by regime:

| rung | INTC calendar | INTC event | AMZN calendar | AMZN event |
|---|---|---|---|---|
| L1 `sign(queue imbalance)` | +0.0169 | +0.0585 | +0.0005 | +0.0069 |
| L3 ridge | +0.0361 | +0.0531 | +0.0055 | +0.0344 |
| L4 GBM | +0.0202 | **+0.1059** | +0.0080 | **+0.1179** |

Two things this exposes that the pooled medians hide. **The event clock is where the fitted
rungs earn their keep**, L4 is 5× better on INTC's event clock than its calendar clock, which
is the expected place for queue-driven microstructure and is consistent with the phase-1
finding that OFI's content decays inside 100 ms. And **on INTC's calendar clock the GBM
(+0.0202) barely clears the parameter-free baseline (+0.0169) while the ridge (+0.0361) beats
both**, so "the linear baseline is fine" is the right answer in that regime, and the wrong one
on the event clock, where L4 is 6× L3.

## 2. After costs, nothing pays, including omniscience

Break-even cost is reported on the **gross** (mid-to-mid) series, so it is inclusive of every
cost: the total per-side cost the signal can bear before it stops paying. INTC calendar 1s,
1ms latency, protocol C:

| predictor | gross ticks/trade | 95% CI | worst fold | break-even (ticks) |
|---|---|---|---|---|
| L1 `sign(queue imbalance)` | 0.0213 | [0.0185, 0.0245] | 0.0123 | 0.0106 |
| L3 ridge | 0.0200 | [0.0175, 0.0229] | 0.0129 | 0.0100 |
| L4 GBM | 0.0176 | [0.0145, 0.0206] | 0.0092 | 0.0088 |
| `CTRL always short` | 0.0027 | [−0.0000, 0.0055] | −0.0019 | 0.0014 |
| `CTRL always long` | −0.0027 | [−0.0055, 0.0000] | −0.0049 | −0.0014 |
| **`ORACLE` perfect foresight** | 0.9442 | [0.9175, 0.9717] | 0.9180 | 0.4721 |

- **The signal is real but tiny.** L1's CI excludes the constant controls, so it is measuring
  microstructure rather than a directional drift. It can pay **0.0106 ticks** per side against
  an actual cost of roughly **0.92 ticks** (half of a 1.234-tick mean spread, plus the assumed
  0.30-tick take fee). It covers about **1%** of what it costs to trade it.
- **Perfect foresight loses money.** The oracle trades the realised sign of the 1s mid move and
  still ends the session **−$458**. When an omniscient trader is underwater, the cost model -
  not the predictor, is the binding constraint, and no rung on this ladder could have paid.
- **The GBM is not clearly the best rung even on gross PnL, but the evidence is weaker than an
  earlier draft of this line claimed.** Diebold-Mariano on per-trade gross PnL puts L4 below L3
  on INTC calendar 1s by −0.0025 ticks, **DM 1.713, p = 0.087**, and on AMZN calendar 1s the
  sign runs the other way at **p = 0.088**. Neither clears 5%, so the honest reading is *no
  significant difference in either direction*, on top of the nested-model caveat in §4 that
  makes these p-values indicative anyway. The earlier draft quoted p = 0.016 and p = 0.74 from
  a superseded cost run and drew a directional conclusion the current artifact does not
  support. The break-even ordering (L1 0.0106 > L3 0.0100 > L4 0.0088) is unaffected and is
  what the section actually rests on.

**PLAN.md's question was: does anything beat `sign(queue imbalance)` after costs?** On this
data the answer is **no**, and the more interesting half is that nothing beats the spread
either. "The linear baseline is fine" was an acceptable answer; the measured answer is that the
*unfitted* baseline is fine, and none of it is tradable at these horizons under an aggressive
fill model.

## 3. What the red team changed

Eleven red-team passes ran against the artifacts, read-only
(`reports/adversary/`). The ones that changed a reported number:

- **The large negative protocol-C R² was a calibration artifact, not a finding.** The
  through-origin rescale was fitted on the same rows as the model. For a high-capacity rung
  whose in-sample predictions are near-perfect that returns almost no shrinkage, so the
  out-of-sample amplitude is far too large and R²-vs-zero explodes negative. It never bit under
  protocol A, because A's "out-of-sample" rows are the temporal neighbours of its training
  rows, so it inflated the *measured optimism*, not merely the model. With a held-out scale,
  INTC calendar 1s L4 goes from C = −0.2988 to **+0.0213** and its A−C gap from 0.4898 to
  **0.1618**. Between 20% and 95% of most reported gaps was this.
- **The `B − C` column was labelled with the wrong mechanism** (§1.2), and an earlier draft of
  this document drew a reviewer-facing conclusion from the mislabel. Protocol D was added to
  measure the right thing.
- **L1 was not a sufficient control** (§1.1). The frozen-predictor floor replaced it.
- **45 of 80 rows lost more than half their gap when fold 0 was dropped.** The headline table
  now prints the median-based and ex-fold-0 gaps beside the mean. **Most of that fragility was
  itself downstream of the calibration artifact**, fold 0 trains on ~1/6 of the day, where an
  in-sample-calibrated ridge was amplitude-catastrophic. After the fix the L4 gap barely moves
  (+0.3173 mean -> +0.3114 ex-fold-0) and L3 moves moderately (+0.0587 -> +0.0399), but **L2 is
  still a fold-0 story** (+0.0204 -> +0.0026, an 87% drop) and should not be quoted from the
  mean alone.

- **The phase-3 headline column was model-invariant, and the red team killed it.** The original
  break-even column was computed on the *net* series, making it incremental to a spread already
  crossed. A constant always-long predictor scored −0.8072 ticks against a whole-ladder range
  of [−0.8025, −0.7702], *inside* it. The column was measuring the spread to three significant
  figures for every rung. It is now computed on the gross series, the controls ship as rows,
  and the old column is retained and demoted rather than deleted.
- **The confidence intervals were mislabelled** by a factor of `n_trades` in 120 of 120 rows -
  a CI for the mean per trade printed beside the total PnL.
- **The deflated Sharpe was a tautology.** On the net series every rung has a per-trade Sharpe
  near −7.6, and the DSR is 0.000 at every trial count from 1 to 10¹². Now reported on gross,
  where it discriminates: 1.000 for L1 against 0.060 for the always-short control on INTC
  calendar 1s at 1 ms.
- **`mp_micro` was Stoikov's first-order term G¹, not the microprice.** The landing matrix and
  outer recursion were never built. Both added; the recursion does not converge on this data
  (|eig(B)| = 0.9906), so `MICRO_ITERS = 6` is part of the feature's definition.
- **The event clock violated its own `≤ t` contract** on 10.48% of INTC stamps, putting a mid
  move that elapsed in zero wall-clock time inside `label_y_mid_*`.
- **A leaking full-sample PCA fit was invisible to every test**, because the only multi-block
  fixture was one level deep.
- **Positions could straddle fold boundaries.** The decision-spacing watermark reset per fold,
  so 4 positions on INTC calendar 1s opened 800-1000 ms apart against a required 1001 ms -
  immaterial to the number, but it falsified the module's stated invariant.

## 4. The boring explanations, and whether they survive

Proposed first, per the project's own rules. Each was run, not argued.

| explanation | status |
|---|---|
| Off-by-one in message/orderbook alignment | **Rejected.** Row *k* is the state after message *k*, re-verified from the raw CSVs by two independent passes (a type-3 deletion of 100 shares at row 5 reduces displayed size 400 -> 300). |
| Scaler or PCA fit on the full sample | **Rejected.** Three `.fit`-like sites, all on training rows only; a test greps for the patterns a full-sample fit leaves behind. Six PCA mutations now caught. |
| R² computed against the test mean | **Rejected.** `r2_vs_zero` is `1 − SSE/Σy²`; the test-mean variant is carried as a separate diagnostic column to show the size of the gap. |
| The correlations are lookahead | **Rejected.** Label shuffle collapses every feature to \|ρ\| < 0.01, and a forward time-shift *improves* the model monotonically, feeding features from *t*+1 event takes L4's protocol-C R² from 0.0213 to 0.0703 on INTC calendar 1s, which a pipeline already reading the future could not do. |
| The gap is protocol geometry, not leakage | **Partly conceded, and now measured rather than argued.** Protocol A's test universe is restricted to match B/C so A is not additionally tested on the market open. But a frozen predictor still shows A−C of +0.0014 to +0.0271 from fold aggregation alone (§1.1), which is the whole of the L1/L1m/L2 gap. Every cell now reports that floor beside its gap. |
| The gap is a training-set-size difference | **Rejected empirically, not rhetorically.** A trains on ~1.8× C's median and 8.7× C's first fold, so equal fold *count* does not settle it, the earlier draft's claim that it did was a non-sequitur. A size-matched control (C's test blocks, training randomly subsampled to exactly C's per-fold size) reproduces the unmatched result to within 0.01 R² in every cell. |
| The large negative C-protocol R² is a finding | **Rejected, it was a calibration artifact** (§3). Fixed, and the superseded columns are retained in `reports/ladder.md` so the size of the correction is visible. |
| Accuracy is class imbalance | **Conceded, the classification columns are uninformative** and are labelled as such. 96% of INTC 1s labels are flat; a scaled sign predictor lands entirely in the flat class, giving L1 a balanced accuracy of exactly 1/3 and an undefined MCC. The regression metrics are the informative ones. |
| The gross-PnL ordering is one-day drift | **Partially conceded.** A constant short earns +0.0989 gross ticks/trade on AMZN event 5s against L1's 0.3097, about **32%** of it is a one-day directional drift. The constant controls are in the table so this is visible per row. |
| The DM p-values are exact | **Conceded.** Adjacent rungs are largely nested (L3's feature set contains L2's), where standard DM is undersized. `EVAL_PROTOCOL.md` asks for the Giacomini-White variant; it is not implemented, and the p-values are labelled indicative. |
| The embargo value was chosen to flatter the result | **Rejected.** E = max(60s, 10h) is `EVAL_PROTOCOL.md`'s own default, frozen by rule. The sweep over E ∈ {0, 1s, 10s, 60s, 300s} shows R² decaying monotonically and flattening between 60s and 300s (INTC event 1s L1: 0.07228 → 0.07189 → 0.07186). |

## 5. Limitations that bound every claim above

- **One trading day, two names.** Folds are intraday blocks. The same-day-vs-next-day leak
  check **cannot be run at all**, recorded as unavailable, not as passed. §1.2's conclusion
  about purging is the one most likely to change with day-separated folds.
- **2012 vintage.** Market structure has moved on.
- **The fee schedule is an unverified assumption** ($0.0030/$0.0020). The gross-based break-even
  column exists so a reader can substitute their own.
- **No market impact model**, which biases PnL upward.
- **Every fill is priced at the touch for 100 shares.** `walk_the_book` is unit-tested but never
  called in the headline path.
- **The trading deadband (0 ticks) was set after seeing protocol-C output.** The reasoning is
  a-priori, but the trigger was an empty table. Disclosed in the frozen-decisions table rather
  than presented as clean.
- **The calibration holdout fraction (80/20) is a new free parameter, introduced 2026-08-02
  in response to the red team.** It was not frozen before the problem was known, and no
  sensitivity sweep over the fraction has been run. It is a defensible default and the
  direction of the fix is not in doubt, an in-sample scale cannot shrink a model that has
  fitted its own training rows, but the *magnitude* of every corrected gap depends on a
  number chosen after seeing that the old one was broken. Disclosed rather than presented as
  frozen, on the same footing as the phase-3 deadband.
- **Under protocol A the holdout is a seeded random 20%, not the last 20%**, and this is now
  measured, not waved at. A's training rows are not time-ordered, so a positional split would
  not be a held-out slice in any meaningful sense, but the consequence is that A calibrates on
  rows exchangeable with its own test fold and the walk-forward protocols do not. It is **48% of
  L4's `A − C` gap and all of L1's** (§1.1a). An earlier draft of this section claimed the
  frozen-predictor floor bounds it "since the frozen predictor has no calibration at all", the
  frozen predictor *did* have one and it was frozen too, which is exactly why the floor could
  not see this. Both the decomposition and the like-for-like floor now ship.
- **The calibration slope is unstable at the cell level.** Reported 2026-08-03, but its
  denominator is the variance of the prediction, so a low-variance fold sends it to ±40. Read
  the medians; the per-cell ranges are in `reports/ladder.md` and L3's is [−4.74, 43.20].
- **Not every emitted column has a hand-computed literal test** (~35 OFI level×window cells, 12
  label columns, 5 state columns are pinned only structurally).

## 6. What would change my mind

Specific results that would flip a specific conclusion.

0. **The gap is intraday burst duplication, and there is now a positive signature pointing that
   way.** Across protocol-A fold counts k ∈ {2,3,5,10} the raw-AUC gap is essentially **flat**
   (+0.0650 -> +0.0676) while the R² gap rises smoothly. That is what a channel looks like when it
   *saturates* as soon as each test row has one training neighbour, which is precisely what burst
   duplication would do, and not what "more training data leaks more" would do. 32% of INTC
   event-clock stamps sit within 0.5 ms of the previous one. Neither protocol D, nor the fold-4
   size match, nor any floor in this repo separates that hypothesis from the one §1 argues for.
   Day-separated folds would; one session cannot. **This is the strongest surviving alternative
   explanation and it was found by a red team, not by me.**
1. **Day-separated folds collapse the L4 gap.** If, on twenty sessions with folds cut on day
   boundaries, L4's A−C gap falls to near the frozen-predictor floor, then what this repo
   measured is intraday burst duplication rather than leakage in general, and §1 should be
   rewritten to say so. *This is the single most likely way the headline is wrong.*
2. **The gap survives a different calibration holdout.** The 80/20 split was chosen after the
   problem was found (§5). If sweeping the fraction, or replacing the held-out scale with a
   nested inner CV, moves the corrected gaps materially, then the magnitudes in §1 are a
   function of my repair rather than of the data, and only the IC columns should be quoted.
3. **The frozen-predictor floor accounts for more of the L4 gap on other data.** The floor is
   an artifact of unequal `Σy²` across contiguous time segments. On a calmer day, or with
   folds balanced on `Σy²` rather than on duration, it could be much larger relative to the
   measured gap. Reporting A−C against an `Σy²`-weighted C would settle it directly.
4. **`B − C` becomes large on day-separated folds.** §1.2 argues B−C is near zero for an
   arithmetic reason, purge can only touch ~1% of an expanding training set. Overnight gaps
   and day-boundary folds change that geometry entirely. If B−C grows to a material fraction
   of A−C on multi-day data, the correct advice to a reviewer changes again.
5. **The 13-of-16 margin does not survive more days.** L4 beats L1 under protocol C in 13 of
   16 cells, which is what withdrew the inversion claim, but 16 cells on one session is a
   thin basis for a ladder ordering. The three losing cells are all **INTC at the longer
   horizons** (calendar 5s and 10s, event 10s), i.e. the large-tick name where the queue
   signal is strongest and the GBM has least to add. If that pattern is real it is a statement
   about tick regime × horizon, not about capacity; if the margin goes to chance across twenty
   sessions, then "the fitted rungs beat the unfitted baseline" is a single-session result and
   §1.3 should claim only that the inversion was an artifact, without putting an ordering in
   its place.
6. **A passive fill model makes something profitable.** Every conclusion in §2 is conditional on
   aggressive fills. If posting at the touch with an explicit queue-position model turns L1's
   0.0106 ticks into something above cost, §2's conclusion is about the fill assumption rather
   than about the signal.
7. **The oracle turns a profit at some horizon.** The perfect-foresight bound losing money is
   what makes §2 a statement about the cost model. If the oracle is profitable at 30s or 60s,
   the horizon was the wrong choice, not the ladder.
8. **A trained order-flow LM beats the handcrafted set under C with all three controls passing.**
   That was the original question, and this harness exists to answer it the moment a checkpoint
   exists, provided it was not pretrained on the test days (`FUTURE.md` §2).
