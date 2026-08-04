VERDICT: SURVIVED WITH CAVEATS

# Phase 2 red team, fifth pass — the decoupled calibration control (06802c5)

Reviewed: `src/edge/harness/{run,models,splits,metrics,floors}.py`, `src/edge/costs/run.py`,
`tests/test_decoupled_control.py`, `reports/{ladder,artifact_floor,costs}.md`, `RESULTS.md`,
`FINDINGS.md`, `README.md`, `METHOD.md`, `INTERVIEW.md`, `DECISIONS.md`, `FUTURE.md`,
`docs/spec/PLAN.md`, `Makefile`, and the two prior passes for context only. Read-only on `src/`,
`tests/` and every `.md` but this one. Everything below run with
`PYTHONPATH=src .venv-edge/bin/python3` (3.14 / numpy 2.5.1 / lightgbm); scripts in
`scratch/adv5/`. Baseline suite: **429 passed**, as claimed.

**The headline number is right and it is more robust than the report argues.** I reproduced
`+0.1913` cell by cell with my own driver and my own AUC implementation, verified the raw-AUC
identity at machine precision (`max|ΔAUC_raw| = 0.000e+00`, with the scale genuinely moving), and
then ran the sweep the report does not contain: **the same A−C gap under five different common
model-fit rules across all 16 cells.** Four of the five agree (+0.191 / +0.193 / +0.198 / +0.201);
only `first 80% positional` — the rule that strips protocol A of the day's tail, i.e. of the
adjacency channel under study — gives the low +0.137. The withdrawal of "half the gap is the
calibration rule" is correct, and the bracket really does collapse toward the top. I also
computed the per-fold distribution the report omits: L4's clean gap is positive in **80 of 80**
folds, worst **+0.0157**, ex-fold-0 64/64 — better than the superseded arm's 79/80.

**What does not survive**: the sentence that makes the control legitimate on R² ("the overlap is
identical in every arm, so a DIFFERENCE between arms is still clean" — a label-permutation null
puts the decoupled R² gap at **+0.062 ± 0.058** where the main construction's is +0.010 ± 0.009);
the "the rule is worth 0.0116 on R², about 4%" figure built on it; L3's per-fold caveat, which was
moved onto the superseded arm and holds identically on the clean one; the "~10×/~21×" margins,
which are ratios of medians where the median of per-cell ratios is 8.1×/12.5×; L1's "below its
floor", which is vacuous because L1's gap *is* its floor in 16/16 cells; and the tests, four of
whose most important mutations pass 429 tests.

---

## 1. Is the zero real, or vacuous? Real, mechanically forced, and it does not license the whole conclusion on its own

**Verified.** `auc_raw = _directional_auc(y_te, rung._raw(X_all[te]))` and `_raw` is a pure
function of the parameters fitted on `X[fit_rows], y[fit_rows]`. With `model_mask` independent of
`holdout_rule`, `fit_rows` is byte-identical across arms, so raw AUC cannot move. Measured, not
assumed (`scratch/adv5/identity.py`, 2 cells × 2 protocols × 5 rungs × 5 folds):

```
max |AUC_raw(random rule) - AUC_raw(positional rule)| = 0.000e+00   (every row)
max |scale(random) - scale(positional)|               = 0.02 .. 10.4 (the scale does move)
```

So the control is not "nothing moved". But the reported quantity is zero **by construction**, and
the report says so, which is honest. The harder question is what the zero licenses.

**It does not, by itself, license "the whole `A′−C` shrink was the lost training window."** That
statement is derivable from the definition of `auc_raw` alone, with no new code: in the coupled
construction the rule's *only* channel into raw AUC is the model's fit rows, because the scale is
invisible to a rank statistic. The previous pass reached the same conclusion empirically with the
head-rule variant. The genuinely new content of this commit is not the zero — it is the *choice
of a new common model-fit rule*, and the number that falls out of it.

**The alternative explanation the control cannot rule out, and which I therefore tested myself:
the model-fit rule is a new free parameter, chosen after seeing that the old one shrank the gap.**
`A′−C` was already model-fit-rule-*matched* (both arms fit the first 80% positional of their own
training rows); it is the *headline* `A−C` that is rule-inconsistent (A's model gets a random 80%,
C's a contiguous first 80%). So the commit does not remove a choice, it relocates it. I ran the
sweep (`scratch/adv5/sweep16.py`, all 16 cells, 6 rules, L1/L3/L4, ~39 min):

| common model-fit rule | L4 median A−C raw AUC | mean | min cell | 16-cell positive |
|---|---|---|---|---|
| headline (each protocol's own coupled rule) | +0.1740 | +0.1786 | +0.0493 | 16/16 |
| `first 80%` positional (= the superseded `A′−C`) | **+0.1371** | +0.1416 | +0.0297 | 16/16 |
| `random 80%` seed 0 (= **the published clean control**) | **+0.1913** | +0.1864 | +0.0657 | 16/16 |
| `random 80%` different seed | +0.1927 | +0.1858 | +0.0649 | 16/16 |
| `last 80%` positional | +0.1977 | +0.1855 | +0.0672 | 16/16 |
| `all 100%` (no model holdout at all) | +0.2010 | +0.1904 | +0.0649 | 16/16 |

L3: +0.0409 / +0.0282 / +0.0409 / +0.0407 / +0.0313 / +0.0411. L1: +0.0022 under every rule (it
has no model). The published `+0.1913` and `+0.1371` both reproduce exactly.

**This defuses the objection rather than confirming it.** The gap is flat in the fit fraction and
in the draw, and the *only* rule that produces a materially smaller number is the one that
withholds the day's tail from A. Diagnosing why: the C arm's level barely moves across rules
(median 0.706–0.717); the A arm's does (0.9059 headline, 0.9048 random-80, **0.8556** first-80).
So `A′` was low because it crippled A, exactly as the commit message says. The narrowest correct
statement is therefore *not* a point estimate: **+0.19 to +0.20 over every model-fit rule that
does not remove A's adjacency to its own test rows**, with +0.137 attainable only by removing it.
The report should say that instead of "the bracket collapses"; the bracket narrows, and it does so
only under an argument the report never makes.

## 2. The accepted overlap. Not symmetric, and one quoted number depends on it

The claim under test: "The overlap is identical in every arm, so a DIFFERENCE between arms is
still clean" (`ladder.md`, `RESULTS.md`, `FINDINGS.md`, `METHOD.md`, `DECISIONS.md`).

The overlap *fraction* is identical (~0.8 in every arm, by independent draws). Its *effect* is
not, and the report's own table shows it: for INTC calendar 1s L4, the A arm is unharmed by the
overlap (`A_rand` R² 0.18378 against the headline ladder's A of 0.18316) while the C arm is
destroyed by it (`C_pos` R² **−0.25953** against the headline ladder's C of +0.02134). The
in-sample scale bias grows as the model's capacity per training row grows, and C trains on
1/1.8 of A's rows at the median and 1/8.7 at fold 0 — so the accepted cost lands almost entirely
on one side of every difference.

Measured directly with a label-permutation null (training labels shuffled within fold, 8 seeds,
INTC event 1s, L4 — `scratch/adv5/null2.py`):

| construction | shuffled-label null, raw-AUC gap | shuffled-label null, **R² gap** |
|---|---|---|
| main ladder (coupled, own rules) | −0.0020 ± 0.0510 | **+0.0098 ± 0.0092** |
| decoupled arm (`model_mask`) | −0.0022 ± 0.0540 / +0.0252 ± 0.0715 | **+0.0622 ± 0.0575** |

On raw AUC the two constructions have the same, unbiased null — the identity protects it. **On R²
the decoupled construction manufactures a +0.06 gap out of pure noise, six times the main
construction's bias and six times its spread.** So "the differences are clean" is true only for the
rank columns, and the report states it unqualifiedly for the whole table.

**One quoted number depends on the contaminated levels.** "On R² the rule is worth a median of
**0.0116** for L4 — about 4% of its gap" (`RESULTS.md`, `FINDINGS.md`, `METHOD.md`,
`DECISIONS.md`) is the `rule spread` column, `(A_pos−C_pos) − (A_rand−C_rand)`, which is dominated
by the difference of two heavily contaminated C arms (−0.25953 vs −0.24818 in the cell above). Two
further problems with it:

- **It is a numerator from the contaminated table over a denominator from the clean ladder.**
  0.0116 / 0.3173. That is the same class of metric mismatch this project has now made twice.
- **It is at its own null.** The identical statistic for rungs with nothing to fit: L1 +0.0046,
  L1m +0.0022, L2 +0.0066 (medians, from `ladder.md`), and a constant +1 predictor I inserted
  scores **+0.0082** on INTC event 1s (`scratch/adv5/nullgap.py`). L4's 0.0116 is within a factor
  of 1.5–2 of a predictor with no information. "The rule is worth 4%" should be withdrawn and
  replaced with "on R² the rule is not distinguishable from the null for this statistic".
- Separately, `ladder.md`'s "**their spread is what the calibration rule is worth**" is wrong as
  English: the spread is a difference-in-differences, i.e. the *interaction* of rule and protocol.
  A rule worth +0.05 under both protocols would show a spread of zero.

## 3. `model_fit_mask` is protocol-independent — and the 80% does not matter

The mask is `rng(2000 + 17·seed + fold_id).choice(n_train, 0.8·n_train)`. Under A and C, `tr`
differs, so it is the same *rule* on different rows — correct, and the honest description. It
reintroduces nothing that I can find: the mask depends only on `(n_train, fold_id, seed)`, so
within a protocol it is identical across arms (verified at machine precision, §1), and it is drawn
from a stream independent of `scale_holdout_mask`'s (1000+fold_id vs 2000+…), so the overlap is
the designed ~80% rather than an accident.

**The 80% is arbitrary and it does not matter.** Swept on INTC calendar 1s at 50/80/90/95/99/100%
and on all 16 cells at 80/100%: L4's gap runs +0.1769 (rand 50%) → +0.1847 (rand 80%) → +0.1874
(rand 99%) → +0.1931 (all) in that cell, and +0.1913 → +0.2010 across 16 cells. Monotone, flat,
and the direction of the residual dependence works *against* the reported number, not for it.
`FUTURE.md` flags "the 80% is arbitrary, no sweep"; the sweep exists now and it is benign.

## 4. Why the split-only number went UP, and whether it is comparable

It is comparable, and the increase is explained, but not by the explanation the report gives.

`+0.1740` is measured with a **mixed** model-fit rule (A's model on a random 80%, C's on a
contiguous first 80%, because each arm's model is the complement of its own calibration holdout).
`+0.1913` is measured with a **matched** rule (random 80% in both). Per §1, the C arm's level is
nearly rule-invariant (0.7122 headline vs 0.7171 random-80 at the median) while A's is not — so
the move from +0.1740 to +0.1913 is A's arm being handed back a marginally better model, plus
per-cell reshuffling. It is not "a confound removed"; it is a different, defensible, and as it
turns out barely-consequential normalisation. The four matched rules that do not cripple A all
land at +0.19–+0.20, and the headline +0.1740 sits just below them. **The comparison is fair; the
framing "larger than the uncontrolled figure, so the headline was conservative" happens to be
right, and is right for a reason the report does not give.**

## 5. The floors. Same-class mismatch, twice, plus a channel no floor can bound

- **Metric class is right this time**: raw gap against raw floor, at the rung's own magnitude, per
  cell. That was the previous pass's item 5 and it is fixed.
- **The statistic class is not.** "median ~10×" (L4) and "median ~21×" (L3) are *ratios of
  medians* (0.1913/0.0198 = 9.7; 0.0409/0.0019 = 21.5). The **median of per-cell ratios** is
  **8.06×** (L4, min cell 3.42×) and **12.50×** (L3, min cell 7.32×) — recomputed from
  `artifact_floor.md`'s own `max`-over-fit-windows column, cell by cell
  (`scratch/adv5/floors.py`). The previous pass flagged exactly this ("48% is a median of ratios,
  56.5% a ratio of medians, and they sit in the same row"); it has recurred. The min-cell *ratio*
  has also been silently replaced by a min-cell *absolute margin* ("min cell +0.0657"), which is
  the weaker statement dressed as the stronger one.
- **The floor is computed under the non-decoupled construction** (`frozen_floor` never takes a
  `model_mask`; its model is fitted once, never per fold). That was already true of the headline,
  so it is not new — but the decoupled gap is measured with the model refit on a per-fold random
  80%, and holding the protocol fixed at C while moving only that rule moves C's raw AUC by up to
  **0.051** in a cell (INTC calendar 1s: 0.7994 at all-100% vs 0.8575 at first-80%). That is 1.3×
  the worst-cell floor and is 100% model-refitting, which no frozen predictor can bound.
  `FUTURE.md` §7 names this; `README.md` and `RESULTS.md` quote the ~10× margin without it.
- **L1's floor comparison is vacuous.** L1's raw-AUC gap and L1's raw-AUC floor are the *same 16
  numbers* — ratio 1.000 in 16 of 16 cells, verified. "L1 +0.0023 against a worst-case floor of
  +0.0053, below floor" (README, INTERVIEW) is a median of a set compared against the max of the
  same set. The underlying fact (L1's gap is indistinguishable from a frozen predictor's) is
  correct and important; the presentation of it as a control that *passes* is circular.

## 6. Tests. The two claimed properties are pinned inside one function and nowhere else

`tests/test_decoupled_control.py` is dated before the branch and does bite for function-level
regressions. But two of its five property-1 tests are **tautological**:
`test_the_mask_is_identical_under_every_rule` and `..._under_every_protocol` call
`model_fit_mask(n, fold_id, seed)` with identical arguments and compare the result to itself; the
loop variable `rule` appears only in the error message. They cannot fail for any deterministic
implementation. The end-to-end test builds `self._fold("A")` twice, so it cannot see protocol
dependence either.

Mutation battery against a copy of `src/`, full 429-test suite each time (`scratch/adv5/mut.py`):

| mutation | result |
|---|---|
| `model_fit_mask` **varies by protocol** at the call site (the exact confound the control exists to remove) | **SURVIVED** |
| model mask made **positional** at the call site (the lost-training-window confound, restored verbatim) | **SURVIVED** |
| `DECOUPLED_ARMS["C_pos"]` silently points at protocol **A** (headline becomes A−A) | **SURVIVED** |
| `DECOUPLED_ARMS["A_rand"]` silently positional | **SURVIVED** |
| the four arms run with `decouple_scale=False` (silently the confounded arm) | **SURVIVED** |
| `model_fit_mask` itself positional | CAUGHT (1) |
| `model_fit_mask` fraction 0.8 → 0.5 | CAUGHT (1) |
| `model_fit_mask` ignores `fold_id` | CAUGHT (1) |
| `score_fold` ignores `decouple_scale` | CAUGHT (1) |
| `Rung.fit` ignores `model_mask` | CAUGHT (1) |

So: every regression *inside* `model_fit_mask` is caught; **every regression in how it is wired,
including the two the control was built to prevent, is not.** The guard test
(`test_without_the_control_the_model_DOES_move_with_the_rule`) does do what it claims — it
asserts the coupled path's raw AUC moves with the rule, and it is not vacuous — but it is run on
protocol A only, which is why the protocol-dependence mutation escapes. One assertion on
`fold.protocol` not appearing in the mask's inputs, and one on `DECOUPLED_ARMS`, would close it.

## 7. Documentation vs artifacts. Every quoted number reproduces; two withdrawn claims still stand

Reproduced from `reports/ladder.md` by re-parsing the tables (`scratch/adv5/parse_ladder.py`):
clean split-only medians L4 **+0.1913** (mean +0.1864, min +0.0657, max +0.3132, 16/16 positive),
L3 **+0.0409**, L1 **+0.0023** (13/16), L1m −0.0018, L2 −0.0002; R² rule spreads L4 **0.0116**,
L3 **0.0784**; uncontrolled +0.1740 / +0.0409 / +0.0022; `A′−C` +0.1371 / +0.0283. `costs.md`
regenerated surgically — only the DSR column moved, at `--trials 2536`. The previous pass's
defects 1–3 (`make ladder` missing `--holdout-control`, "392 tests", stale phase-3 §2/§3/§4
numbers) are all fixed. `README.md` says 429 tests; the suite is 429.

Defects:

1. **The withdrawn claim is still standing in `reports/ladder.md`, twice**, because the commit was
   purely additive to that file (214 insertions, 0 deletions):
   - line 16, in the standing *Corrected 2026-08-03* header: "**About half of L4's `A - C` gap is
     that calibration rule, not the split.**"
   - line 1994, in the **Reading notes**, which a reader reaches last: "Roughly half of L4's gap
     and ALL of L1's is the protocol-dependent calibration holdout rather than the split — see the
     control table above. **Quote `A' - C`** …" — i.e. the report still instructs the reader to
     quote the superseded number.
2. **`ladder.md` §"Control: how much of `A - C` is the calibration RULE" is unmarked and still
   false on its face**: "Split, folds, seed, model and metrics are unchanged; only the calibration
   rule moves." The previous pass asked for this sentence specifically; it is unchanged, and the
   `rule share` column (0.611, 0.418, …) is still printed without a superseded marker. The new
   section 100 lines below contradicts it with no back-reference from it.
3. **The same column is given two contradictory labels inside the new section**: `A_rand − C_pos`
   is "the fully uncontrolled comparison reproduced inside this arm" (line 298) and "the split
   effect measured with a rule-independent model, and it is the number to quote" (line 405). On
   raw AUC the identity makes both true; on R² only the first is.
4. **L3's per-fold caveat is attributed to the superseded arm and is not true of the reported
   one.** `RESULTS.md`/`FINDINGS.md`: "on the confounded `A′` arm its gap was positive in only 55
   of 80 protocol-C folds and halved when fold 0 was dropped, where L4's was positive in 79 of
   80". Recomputed on the **clean** arm (`scratch/adv5/analyse.py`, fold-paired, 80 folds):
   **L3 55/80 positive, median +0.0177, worst −0.0829, ex-fold-0 39/64 and median +0.0120** —
   identical weakness. L4 on the clean arm is **80/80, median +0.1986, worst +0.0157, ex-fold-0
   64/64**, i.e. *better* than the arm it replaced. Both facts should be in the report and neither
   is: there is **no per-fold distribution and no worst fold anywhere for the new headline**
   (RULES.md rule 4), which is the third consecutive pass to say so.
5. `INTERVIEW.md` puts "+0.1913 (raw AUC)" as the last column of a table headed "Median R² vs zero
   across 96 cells". It is labelled, but it invites a misread of a rank metric as an R².
6. `FINDINGS.md`'s L3 row reads "+0.0409 | +0.0283 | **+0.0409**", which reads as "the clean
   control restored L3 exactly". It is a coincidence of medians: the two distributions differ
   (mean +0.0457 vs +0.0382, min +0.0217 vs +0.0163).

## 8. Trial count 2,536. Defensible, conservative, and internally inconsistent in the same way as before

384 = 4 arms × 16 cells × 6 rungs, matching trial 20's own convention (96 = 16 × 6, a "fit" being
one (cell, protocol, rung) over five folds). `--holdout-control` runs exactly these four arms and
no fifth. `costs.md` and `METHOD.md` both carry 2,536.

Two frictions, neither of which flatters anything (a larger denominator lowers DSR):

- **Half the 384 fits are duplicates by construction.** `A_pos`'s raw model is byte-identical to
  `A_rand`'s and `C_rand`'s to `C_pos`'s — that is the identity being reported. At most 192
  distinct raw models exist, and only 96 of them enter any quoted number.
- The +384 sits beside trial 21, which re-measured 96 floors with *different* scale rows and
  therefore different predictions, and was counted **+0**. Opposite conventions for the same kind
  of work, as the previous pass already noted for trials 5/9/19.

The uncounted degree of freedom remains the *choice*, not the fits: which model-fit rule to
standardise on, made after seeing that the previous one shrank the gap. My §1 sweep is the check
that would retire that objection and it is not in the repo.

## The four standing items

**Protocols differ in the split and one disclosed rule, and in the decoupled arms not even that.**
The only `protocol ==` branches in `src/` outside `splits.py` are `scale_holdout_mask` line 117
and loop headers. Features, seed, hyperparameters, metric code and the usable-row mask are shared;
`common_usable_mask` is computed once per cell and applied identically.

**Row alignment — CLEAN, verified from the raw CSVs with no `edge.*` import**
(`scratch/adv5/align.py`, 40,000 messages per name). Under "orderbook row *k* is the state
**after** message *k*": INTC 19,677 type-1 adds reproduce the exact `+size` depth delta with **0**
mismatches (1,985 under the off-by-one convention); 18,008 type-3 deletions, 0 mismatches (1,731).
AMZN 19,603 / 0 (334) and 18,598 / 0 (224).

**Lookahead — none on the harness path.** `rolling_zscore`'s window is `[t-window, t)`, strictly
excluding *t*; the constant offset it subtracts is the first finite value. `ofi.py`'s PCA loadings
are fitted per refit block on rows strictly before the block start. No `shift`, no `.iloc[i+`, no
centred window anywhere in `src/`. The `+1`-event lookahead probe fires through the decoupled code
path (INTC event 1s L4: C raw AUC 0.9289 → 0.9426, A 0.9967 → 0.9979).

**Scaler — fitted inside the fold.** `L3Ridge._fit_raw` computes `mu_`/`sigma_` on `X[fit_rows]`
only. The four `.fit(` call sites in `src/` are all on training rows. `frozen_floor` fits on the
whole day deliberately and identically for every protocol.

**Base rate — disclosed, and the headline is not an artifact of it.** Base rate 0.9698 at INTC
calendar 1s; balanced accuracy is exactly 0.3333 (chance on three classes) for every unfitted rung
under every protocol, and `METHOD.md` §6, `ladder.md` and `FINDINGS.md` §1.3 all say to read those
rows as null. AUC is computed on the directional subset only (flats excluded) — I re-implemented
it independently (Mann–Whitney with tie-averaged ranks) and reproduce their values exactly.

**Deadband is a rule, not hindsight.** `ALPHA_TICKS = 0.5` frozen 2026-07-31 in `DECISIONS.md` and
`labels.py`, sourced to `FEATURES.md` §7, a day before any phase-2 number. The phase-3 *trading*
deadband of 0 ticks was set after seeing protocol-C output and is disclosed as such.

**Standard probes, all pass on the decoupled path** (`scratch/adv5/probes.py`, INTC event 1s):
labels shuffled → L4 raw AUC 0.5092 (A) / 0.4825 (C), L1 unchanged as it must be; constant `+1`
predictor → raw AUC exactly 0.5000 in all four arms, gaps vanish; `A−C` at k = 2/3/5/10 →
AUC +0.0650 / +0.0674 / +0.0678 / +0.0676 and R² +0.4604 / +0.4811 / +0.4896 / +0.4949 — smooth
and monotone in k on R², flat on AUC, which is what an *adjacency* mechanism should do (every test
row has a training neighbour at any k, so the channel saturates immediately).

**One number the probes produce that the repo should own.** The per-cell shuffled-label null for
the raw-AUC gap has **sd ≈ 0.05** (both constructions). The 16-cell median of +0.1913 is ~3.5 null
sd, but the **minimum cell (+0.0657) is ~1.3 null sd** — so "clears its worst-case floor in the
worst cell by +0.0657" is a statement about a quantity whose per-cell noise is of the same order.
The 80/80 fold-level sign count is the statement that carries the result, and it is unpublished.

## The narrowest defensible version of the claim

> On one session of two names, with features, rows, folds, seed, hyperparameters, metric code
> **and the model's fit-row rule** held identical, shuffled k-fold buys a GBM about **+0.19 of
> directional AUC on the raw score** over purged walk-forward, and buys a parameter-free rung
> nothing (+0.0023, identical to a frozen predictor's floor in 16 of 16 cells). The figure is
> stable at **+0.191 to +0.201** across every common model-fit rule that does not withhold the
> day's tail from protocol A; the one rule that does withhold it returns +0.137, and that number
> is a measurement of the handicap, not of the calibration. The gap is positive in 16 of 16 cells
> and **80 of 80 protocol-C folds** (worst fold +0.0157) and clears the worst-window
> frozen-predictor floor by a median of **8×** per cell, 3.4× in the worst cell. The calibration
> rule contributes exactly zero to it, by construction. The ridge's gap (+0.0409) is real at the
> cell level but thin at the fold level: 55 of 80 folds, worst −0.0829, and 39 of 64 with fold 0
> dropped. Nothing about the decoupled arm's **R²** columns is quotable: a label-permutation null
> puts that construction's R² gap at +0.062 ± 0.058.

## What has to change (none of it blocks the run)

1. Delete or mark the two surviving statements of the withdrawn attribution in
   `reports/ladder.md` (the line-16 correction header and the line-1994 reading note, which still
   says "quote `A′ − C`"), and the "only the calibration rule moves" sentence plus the unmarked
   `rule share` column in the `A′` section.
2. Withdraw "on R² the rule is worth a median 0.0116 for L4, about 4% of its gap". The numerator
   comes from a table the same paragraph says is not comparable to the ladder, the denominator
   from the ladder, and the statistic sits at its own null (unfitted rungs +0.0022/+0.0046/+0.0066;
   a constant predictor +0.0082). Replace "the overlap is identical, so differences are clean"
   with "identical in fraction, not in effect: measured under a label permutation the decoupled
   R² gap is +0.062 ± 0.058 against the main construction's +0.010 ± 0.009".
3. Publish the per-fold distribution and worst fold for the clean control (L4 80/80, worst
   +0.0157, ex-fold-0 64/64; L3 55/80, worst −0.0829, ex-fold-0 39/64) and move L3's fold-level
   caveat off the superseded arm, where it is currently parked, onto the reported one.
4. Requote the margins as medians of per-cell ratios (8.1× L4, 12.5× L3, min cell 3.4× / 7.3×),
   not ratios of medians labelled "median", and restore a min-cell *ratio*.
5. Say that L1's "below its floor" is an identity — L1's gap and L1's floor are the same 16
   numbers — rather than a control that passes.
6. Add the two wiring tests: `fold.protocol` must not reach the model mask, and `DECOUPLED_ARMS`
   must map `C_*` to protocol C with `decouple_scale=True`. Five reversions of the control,
   including both confounds it exists to remove, currently pass 429 tests.
7. Report the model-fit-rule sweep, or say plainly that the rule was chosen after seeing the
   result. It is the only remaining unlogged degree of freedom in this control, and it happens to
   be harmless — which is worth two lines rather than silence.

## Most likely remaining explanation that is not "the effect is real"

**Intraday burst duplication — adjacency measured on a single session — unchanged from the
previous pass, and the decoupled control does not touch it.** `A − D` = +0.296 is attributed to
adjacency, and the fold-4 size match rules out training-set size, but on one day "training rows
temporally adjacent to test rows" and "training rows that are near-duplicates of test rows because
32% of event stamps sit within 0.5 ms of the previous one" are the same rows. My k-sweep is new
evidence *for* that reading: the AUC gap is flat from k = 2 to k = 10 (+0.0650 → +0.0676) while the
R² gap rises smoothly, which is the signature of a channel that saturates as soon as each test row
has one training neighbour — i.e. duplication, not sample size and not the future. `FINDINGS.md`
§6.1 already names this as the project's own most likely failure; nothing in this commit, in
protocol D, or in any floor here can separate it.

## What I could not check and why

- **Cross-day generality.** One date, two names; day-boundary folds would change what the purge
  removes and cannot be constructed here.
- **Whether the null spread I measured (sd ≈ 0.05 per cell on the AUC gap) is the right null for a
  model with skill.** A label permutation destroys the signal, and the sampling variance of a
  gap between two skilful models need not equal that of a gap between two null ones. A block
  bootstrap over test blocks would answer it; it is not implemented and I may not add it.
- **Why C's raw AUC is so fit-window-sensitive in one cell** (INTC calendar 1s: 0.7994 at all-100%
  against 0.8575 at first-80%, i.e. giving the model the most recent fifth of the past *hurts*).
  It does not generalise — across 16 cells the C level is nearly rule-invariant — but I have no
  mechanism for it and it is the largest single-cell nuisance I found.
- **GBM hyperparameters.** Frozen a priori, never swept, so I cannot bound how much of L4's
  behaviour under C is those settings.
- **Provenance of the trial log.** No `.edge-runs/` artefacts are committed for 2026-08-03, so I
  verified internal consistency and the arm count in `run.py`, not that 2,536 fits were executed.
- **The fee schedule and the 1.234-tick mean spread** behind "~0.92 ticks of cost" — assumptions,
  stated as such, not verifiable from this repo.
