VERDICT: SURVIVED WITH CAVEATS

# Phase 2 red team, fourth pass, the 2026-08-03 corrections

Reviewed: `reports/{ladder,artifact_floor,costs}.md`, `RESULTS.md`, `FINDINGS.md`, `README.md`,
`METHOD.md`, `INTERVIEW.md`, `DECISIONS.md`, `docs/spec/PLAN.md`, `Makefile`,
`src/edge/harness/{run,floors,models,metrics,splits}.py`, `src/edge/costs/run.py`,
`tests/test_score_fold.py`, `tests/test_pnl.py`. Read-only on `src/`, `tests/` and every `.md`
except this file. Everything below was run with `PYTHONPATH=src .venv-edge/bin/python3`
(3.14.5 / numpy 2.5.1 / lightgbm 4.7.0); scripts in `scratch/redteam/`.

**The corrections are real and the surviving headline is not wrong.** I rebuilt the pipeline
with my own driver and reproduce `A' - C` = +0.1379 (L4) and +0.0542 (L3) on R2, +0.1371 and
+0.0283 on raw AUC, cell by cell. It clears every floor I could construct, in 16 of 16 cells,
and it survives at fold 4 where A' and C train on the same number of rows, which is the
strongest thing in this repo and is not stated anywhere in it.

**What does not survive**: the "98x its floor" margin, the claim that `A' - C` holds anything
constant except the rule's *name*, the claim that the cost re-run left the unfitted rungs
unchanged, and three quoted phase-3 numbers in `FINDINGS.md` that no current artifact produces.
None of that moves the headline. All of it is reportable and one item flips a significance
statement.

---

## The eight lines of attack

### 1. Is the new floor like-for-like? Partly. It is not a bound, and the AUC version is measured at a ceiling.

The floor freezes the raw model and refits only the scale; the rungs refit both. So a third
channel, the raw model's own refitting on protocol-dependent rows, is outside the control by
construction, and no frozen predictor can bound it. I attacked it two ways.

**(a) The floor depends on where the frozen model is fitted, and "the whole day" is the choice
that minimises it.** Same construction, same per-fold scale refit, only the fit window moves
(`scratch/redteam/floors2.py`, 16 cells x 3 magnitudes x 4 windows):

| frozen at | fit window | R2 floor (med) | R2 floor (max cell) | raw-AUC floor (med) | raw-AUC floor (max cell) |
|---|---|---|---|---|---|
| L4 | whole day (**published**) | **+0.0192** | +0.0339 | **+0.0014** | +0.0045 |
| L4 | first 50% | +0.0072 | +0.0366 | **+0.0073** | **+0.0383** |
| L4 | random 50% | +0.0148 | +0.0297 | +0.0006 | +0.0029 |
| L4 | last 50% | −0.0036 | +0.0562 | **+0.0183** | +0.0367 |
| L3 | whole day (**published**) | +0.0112 | +0.0581 | +0.0001 | +0.0043 |
| L3 | last 50% | +0.0079 | +0.0411 | +0.0014 | +0.0066 |

The published raw-AUC floor of +0.0014 at L4's magnitude is not a property of the partition. A
whole-day-fitted GBM scores AUC **0.9960** on INTC calendar 1s, a ceiling, where two protocols
have almost no room to differ. The rung it is supposed to bound scores **0.72** under C. Refit
the frozen model on half the day, so its accuracy is in the rungs' operating range, and the same
floor is 5-13x larger at the median and **+0.0383** in the worst cell. **The "98x" and "11x"
margins in `RESULTS.md`, `FINDINGS.md` §1.1a and `README.md` are an artifact of that ceiling.**

**(b) The strongest floor I can build, and the headline still clears it.** Taking, per cell, the
**maximum over all four fit windows** at the rung's own magnitude:

| rung | R2 `A'-C` vs worst-variant floor | AUC `A'-C` vs worst-variant floor |
|---|---|---|
| L1 | **below floor in 16/16** (median −0.0021 vs +0.0043) | exactly at floor (ratio 1.00), for L1 the control *is* L1 |
| L3 | clears 16/16, median 4.0x, **min cell 1.1x** (margin +0.0027) | clears 16/16, median 18.5x, min cell 5.6x |
| L4 | clears 16/16, median 5.9x, min cell 2.6x | clears 16/16, median **6.9x**, min cell **1.5x** |

So: L4 survives, at roughly **7x**, not 98x. L3's R2 margin is one cell away from nothing.
L1 behaves exactly as a null control must. I also re-measured the floor with protocol A on the
positional rule, the configuration `A' - C` is actually measured in, and it moves L4 by
0.0002 and L3 by 0.0004, so that particular mismatch is immaterial (`scratch/redteam/floor_ap.py`).

### 2. The `A' - C` control. Not like-for-like, and "48% of the gap is the rule" is one of at least three answers.

`ladder.md` and every doc say "Split, folds, seed, model and metrics are unchanged; only the
calibration rule moves." **That is false for L3 and L4.** `Rung.fit` fits the raw model on
`~scale_holdout`, so changing the rule changes *which rows the model trains on*. Under A the
model sees a random 80% of the day; under A' it sees the first 80% **in time** and never sees the
day's tail. A' is a temporally handicapped model, not the same model.

The 2x2 that settles it (medians over 16 cells; `scratch/redteam/run_main.py`):

| L4 GBM | C on its own rule | C on A's random rule (`C'`) |
|---|---|---|
| **A on its own random rule** | +0.3173 (published) | **+0.5223** |
| **A' on C's positional rule** | **+0.1379** (published) | +0.3865 |

Both off-diagonal cells hold the rule constant. One says the rule inflates the gap by 57%; the
other says the rule **masks** 65% of it. `C - C'` is +0.2019: each protocol's default rule is the
one that flatters it, so the published `A - C` sits in the middle of a bracket
**[+0.138, +0.522]**, not at the top of it. L3: [+0.054, +0.183].

A third rule-matched control isolates the mechanism. `A''` = protocol A with the calibration
holdout as the **first** 20% in time, so the slice is contiguous and non-exchangeable exactly as
under C, but the model keeps the day's tail (`scratch/redteam/headrule.py`):

| rung, median over 16 cells | `A - C` | `A' - C` (tail hold) | `A'' - C` (head hold) |
|---|---|---|---|
| L1 R2 | +0.0043 | **−0.0021** | **+0.0034** |
| L4 R2 | +0.3173 | +0.1379 | +0.1255 |
| L3 raw AUC | +0.0409 | +0.0282 | **+0.0399** |
| L4 raw AUC | +0.1740 | +0.1371 | **+0.1727** |

Two things fall out. On **R2** the shrink survives the slice being moved to the other end of the
day, so the report's attribution is broadly right there. On **raw AUC** it does not: `A''` gives
back essentially the whole uncontrolled gap, which means the AUC shrink under `A'` is the
model's lost training window, **not** the calibration rule. And L1, where nothing but the slice
can move, keeps 79% of its gap under the head rule (+0.0034 of +0.0043), so
"`A' - C` = −0.0021, all of L1's gap was the rule" is a statement about one arbitrary slice.
L1's conclusion (it never clears its floor) holds under both; the quoted decomposition does not.

**The right control**, if one is wanted, is a rule that changes the calibration slice while
holding the model's fit rows fixed, e.g. fit the raw model on the same random 80% under both
arms and move only the scale rows, accepting the overlap. It is not implemented. Failing that,
the defensible statement is a **bracket**, not a share: the split-only gap for L4 lies in
[+0.13, +0.52] on R2 and [+0.137, +0.190] on raw AUC. The AUC bracket is tight, which is one more
reason AUC is the right metric to quote.

### 3. The raw-AUC claim. True at the median, false per cell; and raw AUC is legitimate for every rung here.

"The fitted rungs are untouched by it (L4 identical to four decimals, L3 −0.0010)" is a median
statement presented as a general one. Per cell, three cells move materially:

| cell | rung | rescaled `A-C` | raw `A-C` | neg-scale folds under C |
|---|---|---|---|---|
| AMZN calendar 10s | L3 | +0.0584 | **+0.0405** | 1 |
| AMZN calendar 10s | L4 | +0.3054 | +0.2944 | 1 |
| AMZN calendar 5s | L4 | +0.2610 | +0.2561 | 1 |

"That channel alone produced the unfitted rungs' apparent gaps" (`ladder.md` reading note) is
also overstated: L1's gap goes +0.0032 -> +0.0022, not to zero. It is exactly right for L1m
(+0.0035 -> −0.0018) and L2 (+0.0078 -> −0.0002), which is what `RESULTS.md` names.

Is raw AUC legitimate across rungs? Yes here, and for a reason worth stating: AUC is
rank-invariant, so for L1/L1m it is literally unchanged by any positive rescale, and for **L2**
the raw score is a monotone function of a single regressor, so its raw AUC is invariant to the
fit as well, which is why L2's `A-C` and `A'-C` raw columns are identical to four decimals in
all 16 cells. That is a consistency check the report does not claim and passes. Scoring AUC on
the raw score does *not* change what is compared, because AUC never sees the units.

Counts verified independently from the classification table: 27 negative-scale folds under C, 0
under A, over 400 (16 cells x 5 rungs x 5 folds). **32 under B**, true, unremarkable, and
mentioned nowhere.

### 4. The surviving headline. I could not kill it.

Independent reproduction first: my own driver gives INTC calendar 1s L4 per-fold R2 under A of
{0.18693, 0.17267, 0.18913, 0.18009, 0.18697} (mean 0.18316) and under C 0.02134, the shipped
numbers exactly. All 96 headline rows, 48 floor rows and the 96-row control table reproduce.

- **Training-set size / regime coverage, rejected empirically.** Per fold, the median `A' - C`
  raw-AUC gap for L4 runs +0.1940, +0.1845, +0.1305, +0.1158, **+0.0940**, while A'/C's
  training-row ratio runs 7.79, 2.88, 1.77, 1.31, **1.06**. At fold 4 the two protocols train on
  effectively the same number of rows and A' still ranks 9.4 AUC points better. Protocol D says
  the same thing from the other side: `D - C` = +0.0067 R2 for L4, i.e. handing the model the
  entire day outside the test block buys almost nothing. **This is the single best argument in
  the project against the regime-coverage alternative and it is not made anywhere.**
- **Fold-0 dominance, L4 no, L3 yes.** L4's `A' - C` raw AUC is positive in **79 of 80** folds
  (worst −0.0079); dropping fold 0 moves the median +0.1371 -> +0.1273. L3 is positive in only
  **55 of 80** folds (worst −0.0954), its per-fold medians run +0.0815, +0.0551, +0.0069,
  **−0.0174**, −0.0032, and dropping fold 0 halves it (+0.0282 -> +0.0131). **L3's AUC gap is a
  small-training-fold story.** The reports quote "16 / 16 cells" for both rungs, which hides
  that difference, and there is no per-fold or worst-fold column for `AUC raw` anywhere -
  a RULES.md rule 4 miss on the number the report says to quote.
- **Aggregation choice.** L4 raw AUC: median-of-per-cell-gaps +0.1371, difference-of-medians
  +0.1440, ex-fold-0 +0.1273. L3: +0.0282 / +0.0149 / +0.0131. L4 is aggregation-invariant; L3
  loses half under two of the three.
- **Standard probes, all pass** (`scratch/redteam/probes.py`, INTC cal 1s and AMZN event 5s):
  labels shuffled -> AUC 0.4951-0.5137 under both protocols, |gaps| <= 0.0077, R2 gaps <= 0.0005;
  features shifted +1 event -> L4 C R2 0.0213 -> 0.0703 and 0.1064 -> 0.1847, AUC 0.857 -> 0.884,
  so the leak probe fires through the identical code path; `A' - C` at k = 2/3/5/10 is smooth and
  monotone for L4 (R2 +0.065/+0.088/+0.094/+0.098; AUC +0.091/+0.101/+0.108/+0.130) and
  sign-changing noise for L1; a constant predictor scores exactly 0.00000 R2 and 0.5000 AUC under
  A, A', C and D, gaps vanish.
- **GBM hyperparameters** are frozen a priori and never swept, so I cannot say how much of L4's
  behaviour under C is those settings. Unchanged from the previous pass.

### 5. The costs change. Verified, and the claim that nothing else moved is false.

L4's INTC calendar 1s break-even 0.0080 -> 0.0088 and L3's 0.0098 -> 0.0100: **confirmed**
against `git show HEAD:reports/costs.md`. The direction is explicable, the causal holdout
refits the raw model on 80% of the training rows, so the traded sign changes, and the move is
not signed consistently (L3 at INTC calendar 5s goes the other way, 0.0347 -> 0.0316).

But `RESULTS.md` trial 22 and `DECISIONS.md` both assert "the unfitted rungs and all three
controls are unchanged to four decimals, which is the check that nothing else moved."
**18 of 72 unfitted-rung rows moved**, several materially:

| cell / rung / latency | be total, old -> new |
|---|---|
| INTC event 5s / L1m / 1ms | 0.0256 -> **0.0149** (−42%) |
| AMZN calendar 1s / L1m / 1ms | 0.0459 -> **0.0319** (−30%) |
| AMZN event 5s / L1 / 10ms | 0.1280 -> 0.1495 (+17%) |

The mechanism is the one the source file itself names: with a 0-tick deadband the traded
direction is `sign(scale) * sign(raw)`, and the causal scale can come out **negative**. AMZN
event 5s L1 fold 0 flips from **−0.0173 to +0.0173** gross ticks/trade, a whole fold's
direction inverted on an unfitted rung. The controls' gross columns are indeed unchanged; their
**DSR** moved in 17 of 24 rows (0.061 -> 0.060, 0.150 -> 0.147), which is the trial count
2,056 -> 2,152, not the calibration. Either way the sentence offered as "the check" is wrong.

**Does a `FINDINGS.md` §2 conclusion depend on it? Yes, and §2 was not regenerated.** Three
numbers there come from artifacts that no longer exist:

| `FINDINGS.md` says | current `reports/costs.md` |
|---|---|
| §2: DM puts L4 below L3 on INTC cal 1s, "mean diff −0.0036, DM +2.410, **p = 0.016**" | −0.0025, DM **1.713**, **p = 0.0868** |
| §2: "indistinguishable from it on AMZN calendar 1s (**p = 0.74**)" | DM −1.708, **p = 0.0877**, sign favouring **L4** |
| §3: DSR "1.000 for L1, **0.126** for the always-short control" | no always-short DSR anywhere near 0.126 (range 0.009-0.229) |
| §4: constant short "+0.1265 on AMZN event 5s against L1's 0.3083, about **41%**" | 0.0989 against 0.3097 = **32%**; neither figure matches the previous run either |
| §4: "L2 gross **0.0071** -> 0.0128 at +1 -> 0.0345 at +10" | L2 INTC cal 1s gross is now 0.0074 |

The break-even conclusion ("nothing pays", L1 0.0106 > L3 0.0100 > L4 0.0088 against ~0.92 ticks
of cost) is unaffected and reproduces exactly. The **evidential support** for "the GBM is not the
best rung even on gross PnL" is materially weaker than the sentence claims: p = 0.087, not 0.016,
and the AMZN cell it cites as a null now leans the other way.

### 6. Documentation vs artifacts.

Everything in the phase-2 tables reproduces. I re-derived from `reports/ladder.md`: the C medians
(L1 +0.0091, L3 +0.0266, L4 +0.0273), worst cells, IC medians (+0.1465 / +0.2262 / +0.1751),
L4 > L1 in 13/16, L4 > L3 in 6/16 on AUC and 10/16 on R2, the 12.6x ratio (0.3434 / 0.0273), the
raw-AUC medians and the 13/6/7/16/16 positive-cell counts, and the floor medians
+0.0043 / +0.0112 / +0.0192 and +0.0031 / +0.0025 / +0.0014. `ladder.md` and
`artifact_floor.md` agree on all 48 floor rows. The regeneration was **surgical**: against
`git show HEAD:reports/ladder.md`, exactly 96 cells changed (one per headline row, the `floor`
column) and **0** of the ~1,440 other shared cells across the in-sample, IC/calibration,
classification, per-fold-C and 480-row embargo tables. That is the check the run needed and it
passes. Every withdrawn claim I searched for is marked withdrawn everywhere it appears.

Defects found:

1. **The published reproduce command does not reproduce the published report.** `ladder.md`'s own
   header correctly echoes `--holdout-control`, but `make ladder`, `README.md` and `METHOD.md`
   all spell the command **without** it. Running any of the three regenerates `ladder.md`
   *minus the control table that the headline `A' - C` numbers come from*. RULES.md rule 7.
2. `README.md` says `make test # 392 tests`; the suite is **400** (`DECISIONS.md` says 400).
3. The stale phase-3 numbers in §5 above.
4. "48%" and "15%" are medians of per-cell ratios, while "+0.3173 -> +0.1379" is a ratio of
   medians (56.5%). Both are defensible, they are not the same statistic, and they sit in the
   same table row.
5. `FINDINGS.md`/`RESULTS.md` caveat "in 4 of 16 cells L3's gap is below the floor measured at its
   own magnitude" compares a **raw** gap against a **rescaled** floor. Raw against raw, L3 clears
   in **16 of 16** with a minimum ratio of 5.6. The caveat errs toward under-claiming, but it is
   the same metric-mismatch the 2026-08-02 pass killed the report over, pointed the other way.

### 7. The tests. The new ones bite. The floor has no test at all.

I mutated a copy of `src/` and ran the full suite each time (`scratch/redteam/mut.py`):

| mutation | result |
|---|---|
| `auc_raw` computed on the rescaled prediction | **CAUGHT** (1 failure) |
| A gets the positional rule too | **CAUGHT** (2) |
| `r2_vs_zero` against the test mean | **CAUGHT** (2) |
| purge dropped from protocol C | **CAUGHT** (25) |
| `frozen_floor(refit_scale=...)` default flipped to `False` | **SURVIVED** |
| floor's scale fitted on all training rows instead of the holdout | **SURVIVED** |
| floor uses A's holdout rule under every protocol | **SURVIVED** |
| `_floor_key` maps every rung to the L1 floor | **SURVIVED** |
| `frozen_floor_multi` measures only the L1 magnitude | **SURVIVED** |
| `score_fold` silently ignores its `holdout_rule` argument | **SURVIVED** |
| `edge.costs.run` reverted to the in-sample rescale | **SURVIVED** |

Five separate reversions of the 2026-08-03 floor correction, and the exact regression the
2026-08-02 pass killed the report for, all pass 400 tests. There is **no test anywhere** that
touches `frozen_floor`, `frozen_floor_multi`, `refit_scale` or `_floor_key`, the code that
produces the number every gap in the project is judged against. And `score_fold` ignoring
`holdout_rule` would silently turn the entire `A' - C` control into a duplicate of `A` with no
test noticing; the existing test covers `scale_holdout_mask` in isolation, not the wiring, which
is precisely the distinction `test_score_fold.py`'s own docstring says it exists to cover.
`tests/test_pnl.py`'s new case is good and does bite (removing `scale_holdout=None` from
`_ConstantDirection.fit` fails it).

### 8. Trial count. Internally consistent; the unlogged freedom is the metric and control axis.

2,152 = 2,056 + 96, and 96 = 16 cells x 6 rungs matches the log's own convention (a "fit" is one
(cell, protocol, rung), five folds). The +0 rows are defensible: 18, 19, 21 and 22 re-score
existing configurations. No 2026-08-03 work is committed, so I could not reconcile against git;
nothing in the working tree corresponds to a configuration that is not logged. Two frictions:
trial 19 is superseded and counted +0 while trials 5 and 9 are superseded and counted in full -
opposite conventions, both conservative in different directions; and the genuinely uncounted
degrees of freedom are the *choices*, not the fits, directional vs pooled AUC, raw vs rescaled
AUC, and (new today) **which** rule to standardise on for the control, where I show at least
three answers exist and the reported one is the smallest.

---

## The four standing items not covered above

**Row alignment, CLEAN, verified from the raw CSVs with no `edge.*` import**
(`scratch/redteam/align.py`, 30,000 messages each). Under "orderbook row *k* is the state **after**
message *k*": INTC 14,360 type-1 adds show the exact `+size` delta against 1,706 under the
off-by-one convention; AMZN 5,469 against 196; type-3 deletions 13,373 and 5,235 correct with
**0** mismatches.

**Lookahead, none on the harness path.** The only `rolling`/`min_periods` code is
`normalize.rolling_zscore`, whose window is `[t-window, t)`, strictly causal, excludes *t*, and
no `L3_FEATURES` column is a `_sadj` z-score, so it is not on the path at all. No `.iloc[i+`, no
`shift`, no centred window anywhere in `src/`.

**Scaler on the full sample, no.** `L3Ridge._fit_raw` fits `mu_`/`sigma_` inside the fold on
`X[~hold]`. The only `.fit(` call sites outside the rungs are `harness/run.py` and
`costs/run.py`, both on training rows. `frozen_floor` fits on the whole day **deliberately and
identically for every protocol**, which is the point of a floor.

**Base rate, disclosed and correct.** Balanced accuracy is exactly 0.3333 for L0/L1/L1m under
every protocol (chance on three classes) against a median base rate of 0.7489, because a rescaled
sign predictor never clears the 0.5-tick deadband. L3 0.3566, L4 0.3486. `METHOD.md` §6,
`ladder.md` and `FINDINGS.md` §1.3 all say to read those rows as null. **Deadband tuning is a
rule, not hindsight**: alpha = 0.5 ticks frozen 2026-07-31 in `DECISIONS.md` and `labels.py`,
sourced to `FEATURES.md` §7, a day before any phase-2 number. Protocols differ in nothing but the
split and the disclosed calibration rule: the only `protocol ==` branches in `src/` outside
`splits.py` are the one line in `scale_holdout_mask`.

---

## The narrowest defensible version of the claim

> On one session of two names, with features, rows, folds, seed, hyperparameters and metric code
> held identical, shuffled k-fold buys a GBM about **+0.14 R2** and about **+0.137 of directional
> AUC on the raw score** over purged walk-forward, and buys a parameter-free rung nothing. On R2
> that figure is the low end of a **[+0.13, +0.52]** bracket over calibration-rule choices; on
> raw AUC the bracket is a tight **[+0.137, +0.190]**, which is why AUC is the metric to quote.
> The AUC gap is positive in 79 of 80 protocol-C folds and clears the largest frozen-predictor
> floor I could construct by **~7x at the median and 1.5x in the worst cell**, not 98x. It
> survives at the fold where both protocols train on the same number of rows, and protocol D
> shows that handing a model the whole day outside the test block buys +0.007. The ridge's gap is
> real but is a small-training-fold story: positive in 55 of 80 folds and halved by dropping
> fold 0.

## What has to change (none of it blocks the run)

1. Regenerate the numbers in `FINDINGS.md` §2, §3 and §4 from the current `reports/costs.md`, and
   restate the DM claim at p = 0.087.
2. Delete or correct "the unfitted rungs and all three controls are unchanged to four decimals"
   in `RESULTS.md` trial 22 and `DECISIONS.md`. 18 of 72 unfitted rows moved and one fold's
   direction inverted.
3. Add `--holdout-control` to `make ladder`, `README.md` and `METHOD.md`. Fix "392 tests".
4. Write the tests for `frozen_floor` / `refit_scale` / `_floor_key` and one asserting
   `score_fold` actually forwards `holdout_rule`.
5. Requalify the AUC margins: quote ~7x against a floor measured at the rungs' operating point,
   not 98x against one measured at AUC 0.996.
6. Replace "only the calibration rule moves" with the truth (the rule also moves the model's fit
   rows) and replace "48% of the gap is the rule" with the bracket.
7. Give `AUC raw` a per-fold distribution and a worst-fold column, and say that L3's AUC gap is
   concentrated in the two smallest-training folds.

## Most likely remaining explanation that is not "the effect is real"

**Intraday burst duplication, which is adjacency measured on one day.** `A - D` = +0.296 is
called adjacency and the fold-4 result rules out training-set size, but on a single session
"training rows temporally adjacent to test rows" and "training rows that are near-duplicates of
test rows because 32% of event stamps sit within 0.5 ms of the previous one" are the same rows.
The repo already names this as its own most likely failure (`FINDINGS.md` §6.1) and it is the one
alternative that neither D, nor the fold-4 size match, nor any floor here can separate.

## What I could not check and why

- **Cross-day generality.** One date, two names. Day-boundary folds would change what the purge
  removes and are untestable here.
- **A clean rule-constant control.** Separating the calibration slice from the model's fit rows
  requires a variant that is not implemented and that I may not add.
- **GBM hyperparameters.** Frozen a priori, never swept.
- **Whether `.edge-runs/` corroborates the trial log.** No 2026-08-03 work is committed, so the
  log is the only record; I verified internal consistency, not provenance.
- **The fee schedule and the 1.234-tick mean spread** behind "~0.92 ticks of cost", assumptions,
  stated as such, not verifiable from this repo.
