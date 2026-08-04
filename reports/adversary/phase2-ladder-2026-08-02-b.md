VERDICT: KILLED

# Phase 2 red team (second pass) — the three-protocol optimism table

Reviewed: `reports/ladder.md` (mtime 13:20), `RESULTS.md`, `FINDINGS.md` (mtime 13:55),
`src/edge/harness/`. Read-only on `src/`, `tests/`, `RESULTS.md`. All commands below were run
with `/opt/anaconda3/bin/python3` and `PYTHONPATH=src`.

**Note on a moving target.** `src/edge/harness/{__init__,models,run,splits}.py` were rewritten
at 13:24–13:30, i.e. between my first reads and my last. My first four experiments ran against
the pre-correction code and reproduce `reports/ladder.md` bit-exactly; the final experiment ran
against the corrected code and reproduces its own docstring to four decimals. Both states are
labelled below.

---

## The killing check

`reports/ladder.md` is the artifact under review. It was generated with the through-origin
rescale fitted **in-sample** (`Rung.fit` called `_fit_scale(self._raw(X), y)` on the training
rows the model had already fitted). `src/` now implements a causal alternative
(`scale_holdout`), and `RESULTS.md` trial 13 says it "supersedes trial 5 … this is the
headline". I re-ran A and C across all 16 (ticker × clock × horizon) cells with
`score_fold(..., causal_scale=True)` — the project's own corrected code, my own driver:

| rung | med C, ladder.md | med C, causal scale | med A−C, ladder.md | med A−C, causal |
|---|---|---|---|---|
| L1 sign(qi) | +0.0104 | **+0.0091** | +0.0028 | +0.0043 |
| L3 ridge | **−0.0857** | **+0.0266** | +0.2100 | +0.0587 |
| L4 GBM | **−0.2164** | **+0.0273** | +0.5453 | +0.3173 |

Per-cell, under C with the causal scale, **L4 beats L1 in 13 of 16 cells** and L4's worst cell
is −0.0003 against L1's −0.0019.

Claim 3 — "L3 and L4 have NEGATIVE median R2 vs zero, and the best rung is the parameter-free
sign(queue imbalance)" — is false. The sign flips and the ranking reverses. It was an artifact
of calibrating a high-capacity model's amplitude on the rows it had already fitted. The two
other metrics printed in the same report already said so and were not read: under C, median IC
is L3 0.2364 > L4 0.1839 > L1 0.1465, and median balanced accuracy is L4 0.3926 > L3 0.3760 >
L1 0.3333 (= chance exactly).

Validation that I am running the corrected code correctly: `models.py`'s new docstring claims
INTC calendar 10s L4 goes to C = +0.0151 with A−C = 0.4920. I measure +0.0151 and 0.4918.

```
python3 scratch/causalC.py     # 16 cells x {A,C} x 6 rungs, causal_scale=True
```

---

## What I tried and what happened (all ten)

1. **Protocols differ in more than the split.** Seed (0), feature matrix, `common_usable_mask`,
   rung hyperparameters and metric code are computed once per cell and are protocol-independent
   — verified by reading `run.run_cell`. Test universes are identical: A restricts eligibility
   to `ts >= edges[1]`, giving 5 × 39,000 = 195,000 tested rows, exactly B/C's union. **But
   training-set size is not equal**: A trains on 195,000 rows in every fold, C on 38,390 →
   194,390. `reports/ladder.md` line 9 and `DECISIONS.md`'s k-freeze row both assert the
   opposite ("identical count … so the gap is not partly a training-set-size difference";
   "so an A−C gap cannot be partly a difference in the number of training rows"). I measured
   the size effect directly by holding C's test blocks fixed and drawing training rows at the
   same N: **−0.0008 for L4**. The prose is wrong; the confound it denies is real but harmless.
   One new asymmetry has been introduced by the fix: under A the scale holdout is a random 20%
   of training rows, under B/C/D it is the last 20%. A's calibration is therefore still leaky
   by construction, so the corrected A−C still bundles leaky calibration with leaky fitting.
2. **Row alignment.** Verified from the raw LOBSTER CSVs, not from the pipeline. On INTC's
   first 5,000 messages, treating orderbook row *k* as the state **after** message *k* is
   consistent on 1,249 type-1 bid additions with 0 violations; the off-by-one alternative is
   consistent on 222 and violated on 1,025. Independently re-derived `qi_touch` and
   `label_y_mid_1s` on 239 sampled calendar stamps from the raw files: `max|diff| = 0.0` for
   both; against a one-event-ahead book the same column differs by up to 0.474. Clean.
3. **Lookahead in features.** None of `L3_FEATURES` is a `_sadj` (rolling z-score) column, so
   `normalize.py` is not on the path at all. Forward-shift probe through the identical code
   path: features from *t*+500 ms take INTC cal 1s L4 under C from −0.2988 to **+0.2203**. A
   genuine lookahead produces a large jump; the shipped columns do not have it.
4. **Scaler fit on the full sample.** `L3Ridge._fit_raw` fits `mu_`/`sigma_` inside the fold on
   training rows only. No global normalization anywhere on the harness path. Clean — but see
   the killing check: the *other* fitted scalar in the pipeline, `Rung.scale_`, was fitted
   in-sample, and that is what broke the result.
5. **R2 against the test mean.** `metrics.r2_vs_zero` is `1 - SSE/sum(y*y)`. Correct.
   `r2_vs_test_mean` exists but is never the headline. L0 scores exactly 0.0 in all 96 cells,
   which is the arithmetic proof.
6. **The L1 control.** Recomputed. A predictor with **zero** fitted parameters — `k·sign(qi)`
   with `k` fixed globally, an identical vector under all three protocols — still shows
   A−C = **+0.00172** on INTC cal 1s, against L1's published +0.00194. Roughly 90% of L1's gap
   is a fold-aggregation artifact: A's five folds are random subsets of one universe so its
   fold mean approximates the Σy²-weighted pooled metric, while C's contiguous blocks have very
   unequal Σy² and its fold mean is unweighted. The same control at L4's magnitude (one global
   GBM fit, scored unchanged under each partition) gives A−C = **+0.0161**, and under globally
   shuffled labels the harness still reports A−C = **+0.037** for L4. So the artifact floor
   scales with prediction variance, and **L1's small gap does not bound L4's**. `FINDINGS.md`
   §1.1 already says this; `reports/ladder.md`'s reading notes still say the opposite.
7. **Base-rate artifact.** Recomputed from the report. Balanced accuracy is **exactly 0.3333**
   (= 1/3, chance on three classes) for L0, L1, L1m and L2 under every protocol, with MCC NaN,
   because the rescaled sign prediction (`scale_` ≈ 0.02 ticks) never crosses the 0.5-tick
   deadband and the predicted class is always 0. Base rate ≈ 0.75. The classification table
   carries no directional information for four of the six rungs and must not be cited.
8. **Deadband tuning.** α = 0.5 ticks is frozen 2026-07-31 in `DECISIONS.md` line 281 and in
   `RESULTS.md`'s frozen table, sourced to `FEATURES.md`'s own first suggestion, with the sweep
   reported in the phase-1 correlation report and not used to select. The phase-3 0-tick
   *trading* deadband is explicitly disclosed as informed-by-output. No hindsight found in the
   phase-2 numbers.
9. **Trial count.** 288 = 2×2×4×3×6 checks out; 480 = 5×16×6 checks out; 216 = 2×2×2×9×3 checks
   out. The current log adds trials 13–17 (384 + 16 + 192 + 480) for 2,056. Trial 17 correctly
   discloses the 80/20 calibration holdout as a new free parameter chosen after the problem was
   known. Not in the log: the protocol-A k-sweep and the frozen-floor variants I ran here (mine,
   not theirs) — no gap I can attribute to the project.
10. **Pooled-number laundering.** The headline is a mean across folds, and it is fold-0 driven
    for the fitted rungs: INTC cal 1s L4 under C runs −0.6178, −0.2399, −0.4000, −0.2230,
    −0.0130 — monotone in training-set size. `FINDINGS.md` §3 reports that 45 of 80 rows lost
    more than half their gap when fold 0 was dropped; that correction is not in `ladder.md`.

## Additional experiments

- **Labels shuffled.** IC collapses to |0.006| for L1 and L4 under all three protocols. R2 goes
  mildly negative, as it must for any rung with nonzero prediction variance. Passes — but note
  the residual A−C = +0.037 for L4 under the null (item 6).
- **Features shifted +1 / +5 events.** +5 (500 ms) takes L4 under C from −0.2988 to +0.2203.
  Behaves as a leak probe should.
- **Protocol A at k ∈ {2,3,5,10,20}.** L4 gap: +0.4447, +0.4728, +0.4898, +0.5000, +0.5247 —
  smooth and monotone in k. L1 gap: +0.0012, +0.0019, +0.0019, +0.0020, +0.0019. The splitter
  does what it claims.
- **Constant predictor under all three protocols.** Gaps do **not** vanish (item 6). This is the
  single most important negative result here after the killing check.
- **Purge is not a no-op.** B minus C training rows: exactly 610 at h=1s, 620 at 2s, 650 at 5s,
  1,100 at 10s — i.e. (h + E)/100 ms to the row, in every fold. The mask is real. It is also
  ≤1.6% of the training set, which is why B−C ≈ 0 is arithmetic rather than substantive.
- **Where the A−C gap actually comes from.** Holding C's test blocks, fold shapes and
  aggregation fixed and varying only the training set:

  | cell | rung | adjacency (A − all-non-test) | future (all-non-test − C) | total |
  |---|---|---|---|---|
  | INTC cal 1s | L4 | +0.3015 | +0.1882 | +0.4898 |
  | INTC event 1s | L4 | +0.4100 | +0.0764 | +0.4864 |
  | AMZN cal 5s | L4 | +0.4995 | +0.1605 | +0.6601 |

  61–84% of the gap is training rows *temporally adjacent* to test rows — label overlap plus
  serial correlation — which is exactly the mechanism claim 2 assigns a magnitude of ~0.
  Claim 2's arithmetic (B−C ≈ 0) is right; its attribution is wrong. `FINDINGS.md` §1.2 already
  retracts it via protocol D and my numbers agree with theirs; `reports/ladder.md`'s own prose
  and reading notes still carry the retracted reading.

## Most likely remaining explanation that is not "the effect is real"

Even after the causal-scale correction, the surviving A−C gap (L4 +0.3173) is measured on a
**single trading day** with contiguous C blocks that are also *regime* blocks. Protocol A's
training set spans the whole session; C's fold-*k* training set is strictly the morning of the
same day. The clean size-matched control I ran removes training-set *size* as the explanation
(−0.0008) but not training-set *coverage*: A sees the volatility regime of every test block,
C does not. On one day with a strongly time-varying event rate, "A saw this regime, C did not"
and "A saw the future" are not separable, and protocol D does not separate them either — D also
sees every regime. The honest ceiling on the capacity-scaling claim is that it is one day, two
names, and the regime channel is unmeasured.

## What I could not check and why

- **Cross-day and cross-asset generality.** One date, two tickers. `B − C` is near zero *because
  the fold geometry is intraday*; with day-boundary folds the purge removes a different and
  possibly much larger share, and `FINDINGS.md` §5.4 concedes this. Untestable without more data.
- **Whether the corrected `reports/ladder.md` reproduces my medians.** The report has not been
  regenerated since the 13:24 code change. I recomputed A and C but not B or D across all 16
  cells, so I cannot state the corrected B−C or A−B columns.
- **The GBM hyperparameters.** Frozen a priori and never swept, so I cannot tell how much of
  L4's remaining C-protocol behaviour is those specific settings. A single `num_leaves` or
  `n_rounds` sweep would establish whether +0.0273 is a floor or a coincidence, and it would
  cost five more trials.
- **`lightgbm` determinism across machines.** `deterministic=True`, `num_threads=1`, seeded —
  reproduced exactly on this machine, but I did not test a second one.

## What would have to change for this to survive

Nothing in the science. Regenerate `reports/ladder.md` from the causal-scale path, rewrite
`FINDINGS.md` §1's table and §1.3 from it (§1.3's title is currently contradicted by §3 of the
same file), and delete the "L1 is the control" reading note and the "identical training-set
size" sentence from `ladder.md`'s preamble. The corrected result — optimism still scales with
capacity, L4's honest C R2 is small and positive, and the ladder does *not* invert — is a
defensible finding. The published one is not.
