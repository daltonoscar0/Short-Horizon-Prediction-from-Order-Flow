# Edge

**How much of the published short-horizon LOB prediction performance is an artifact of leaky
cross-validation?**

Build the classic microstructure baselines properly, then score the whole ladder under three
evaluation protocols and report the gaps.

| | |
|---|---|
| **A** | shuffled k-fold — what most of the literature does |
| **B** | walk-forward, no purge, no embargo |
| **C** | purged walk-forward with embargo — correct |
| **D** | *control*: C's test blocks, trained on everything outside them, past and future |

`A − C` is total optimism. A fourth split, **D** (C's test blocks, trained on everything
outside them — past *and* future, unpurged), splits that gap into its mechanisms: `A − D` is
the temporal **adjacency** of training rows, `D − C` is seeing the future in disjoint blocks.
That decomposition is the contribution.

Data: LOBSTER level-10, 2012-06-21, INTC (large-tick) and AMZN (small-tick). One trading day
— see [Limitations](#limitations), which you should read before the results.

---

## Headline

**Leaky cross-validation buys optimism in proportion to model capacity, and what it is
actually buying is training rows adjacent to the test rows — not the future.** Median across
all 96 (ticker × clock × horizon) cells, out-of-sample R² against a zero forecast:

| rung | A − C (total) | *floor* | A − D (adjacency) | D − C (future) | B − C (purge) | **C (honest)** |
|---|---|---|---|---|---|---|
| L0 zero | 0.0000 | — | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| L1 `sign(queue imbalance)` | +0.0043 | *+0.0043* | +0.0126 | −0.0085 | +0.0006 | +0.0091 |
| L1m `sign(microprice − mid)` | +0.0058 | *+0.0043* | +0.0030 | +0.0007 | +0.0001 | +0.0042 |
| L2 OLS on OFI | +0.0204 | *+0.0043* | +0.0032 | +0.0030 | +0.0018 | −0.0018 |
| L3 ridge, full set | +0.0587 | *+0.0112* | +0.0296 | +0.0231 | +0.0019 | **+0.0266** |
| L4 GBM, full set | **+0.3173** | *+0.0192* | **+0.2964** | +0.0067 | +0.0025 | **+0.0273** |

Three things fall out of that table.

1. **The gap tracks capacity — and about half of L4's is the calibration rule, not the split.**
   A parameter-free rung gains +0.004; a GBM on the same features, rows, folds and seed gains
   **+0.317**. But `A − C` bundles two mechanisms, and the *floor* column only polices one of
   them. The harness fits each rung's output scale on a held-out slice of training rows, and
   that slice is a **random** 20% under A (exchangeable with A's random test fold) against the
   **last** 20% in time under B/C/D. Re-running A under C's rule — same split, folds, seed and
   model, only the rule moves:

   | rung | `A − C` | **`A′ − C`** (rule held constant) | *floor* | clears floor by |
   |---|---|---|---|---|
   | L1 `sign(queue imbalance)` | +0.0043 | **−0.0021** | *+0.0043* | **not at all** |
   | L3 ridge | +0.0587 | **+0.0542** | *+0.0112* | 4.8× |
   | L4 GBM | **+0.3173** | **+0.1379** | *+0.0192* | 7.2× |

   `A′` is not a perfectly clean control — the holdout rule also moves which rows the model
   trains on — so the honest object is a bracket: standardising on the *other* rule puts L4 at
   +0.5223, and `A − C` sits mid-range in **[+0.138, +0.522]**. On raw AUC the bracket is tight,
   [+0.137, +0.190], which is one more reason to quote AUC.

   The capacity ordering survives; the headline magnitude was overstated roughly 2× for the
   GBM. With the rule held constant the parameter-free rung shows **no optimism at all**, which
   is what a rung with nothing to fit should show and what the first version of this table did
   not verify. The *floor* is a **frozen** model — fitted once on the whole day, scored under
   every protocol with only the scale refitted per fold — so it cannot have learned anything
   from any split.

   The same claim on **directional AUC**, which has no amplitude in it, scored on the raw
   output so the scale's sign cannot touch it:

   | rung | AUC under A | AUC under C | A − C | **A′ − C** | cells positive | *floor* |
   |---|---|---|---|---|---|---|
   | L1 `sign(queue imbalance)` | 0.6545 | 0.6573 | +0.0022 | +0.0023 | 13 / 16 | *+0.0031* |
   | L1m `sign(microprice − mid)` | 0.5880 | 0.5892 | −0.0018 | −0.0018 | 6 / 16 | *+0.0031* |
   | L2 OLS on OFI | 0.6546 | 0.6567 | −0.0002 | −0.0002 | 7 / 16 | *+0.0031* |
   | L3 ridge | 0.7811 | 0.7319 | +0.0409 | **+0.0283** | **16 / 16** | *+0.0025* |
   | L4 GBM | 0.9059 | 0.7122 | **+0.1740** | **+0.1371** | **16 / 16** | *+0.0014* |

   Shuffled k-fold says the GBM ranks up-moves above down-moves 91% of the time. The correct
   protocol says 71%. The three unfitted rungs show no gap in either direction. **That `A′ − C`
   column is the headline** — everything else on this page is a step toward it.

   Margins are quoted against the *worst* artifact floor over four frozen-model fit windows,
   not the whole-day one, which at the GBM's magnitude is measured at an AUC ceiling (~0.996)
   where two protocols cannot differ much. On that basis L4 clears by **~7× (min cell 1.5×)**
   and is positive in **79 of 80 folds**; L3 clears by more in ratio but is a fold-0 story
   (positive in 55 of 80). And the gap is not just training-set size: at the last fold, where
   the two protocols train on within 6% of the same number of rows, L4 is still **+0.094**.
2. **It is adjacency, not the future.** Protocol D hands the model the entire future in blocks
   *disjoint* from the test set and that buys L4 almost nothing (`D − C` = +0.007). Nearly all
   of the gap is `A − D` = +0.296 — training rows that are temporal *neighbours* of test rows.
   That is label overlap, and it is the dominant mechanism. `B − C` is near zero only because
   purge+embargo can touch ~1% of an expanding training set, so it measures label overlap at a
   walk-forward boundary, where it is negligible by construction.
3. **Under the correct protocol the ladder does not invert — but it flattens.** L4 (+0.027)
   and L3 (+0.027) beat L1 (+0.009), and L4 beats L1 in 13 of 16 cells. But an honest R² of
   0.027 is roughly a **twelfth** of what protocol A advertises for the same model, and on the
   two scale-free metrics L3 beats L4 — IC +0.226 vs +0.175, AUC 0.732 vs 0.712, with L4 ahead
   of L3 in only 6 of 16 cells. The capacity is worth something; it is worth far less than the
   literature's protocol implies, and which rung wins depends on whether you are paid for the
   amplitude or for the ordering.

**After costs, nothing pays.** Break-even cost is reported on the gross (mid-to-mid) series,
so it is the total per-side cost a signal can bear:

| INTC calendar 1s, 1ms latency | gross ticks/trade | 95% CI | break-even (ticks) |
|---|---|---|---|
| L1 `sign(queue imbalance)` | 0.0213 | [0.0185, 0.0245] | 0.0106 |
| L4 GBM | 0.0176 | [0.0145, 0.0206] | 0.0088 |
| `CTRL always short` | 0.0027 | [−0.0000, 0.0055] | 0.0014 |
| **`ORACLE` perfect foresight** | 0.9442 | [0.9175, 0.9717] | 0.4721 |

L1's signal is real — its CI excludes the constant control — and it can pay **0.011 ticks**
per side against an actual cost of ~0.92 ticks (half-spread + fee). **Perfect foresight of the
1-second mid move still loses $458** under this fill model. The honest conclusion is that this
ladder is a statement about prediction, not about a tradable strategy.

Full tables: [`reports/ladder.md`](reports/ladder.md),
[`reports/artifact_floor.md`](reports/artifact_floor.md),
[`reports/costs.md`](reports/costs.md). The argument, the failure modes and "what would change
my mind": [`FINDINGS.md`](FINDINGS.md).

## Reproduce

```bash
make data       # raw LOBSTER CSVs -> validated event frame + data-quality report
make features   # -> data/features/*.parquet + the feature x label correlation table
make selfcheck  # Cont-Kukanov-Stoikov reproduction; fails loudly if the relation breaks
make test       # 416 tests

make ladder     # the optimism table + embargo sensitivity -> reports/ladder.md
make floors     # the frozen-predictor artifact floor    -> reports/artifact_floor.md
make costs      # break-even costs, controls, DM tests   -> reports/costs.md
```

Each of those wraps a single module invocation, spelled out here because the report headers
echo the flags rather than the make target:

```bash
python3 -m edge.harness.run --seed 0 --embargo-sweep --in-sample-scale --holdout-control
python3 -m edge.harness.floors --seed 0
python3 -m edge.costs.run --seed 0 --trials 2152 \
    --tickers INTC,AMZN --clocks calendar,event --horizons 1s,5s
```

Every entry point takes `--seed`, and each report echoes the exact arguments that produced it
in its own header. Reproducing a table is one command.

The feature build is **bit-reproducible**: a full rebuild was compared column-by-column across
all four tables (1,028 columns), zero mismatches, NaN placement included.

## What is actually in here

- **A hand-written cross-validation harness.** No `sklearn.model_selection` anywhere —
  `sklearn` is not installed, so the rule is enforced by the environment rather than by
  discipline. The splitter, the metrics, the ridge solve, the Newey–West HAC, the Spearman
  rank transform, the normal quantile function, the block bootstrap and the deflated Sharpe
  are all written out. `lightgbm` is the one library model.
- **Adversarial tests written before the code they test.** `tests/test_splits.py` existed
  before the splitter and contains a deliberately broken splitter that the leak detector is
  asserted to catch — a leak detector that has never caught a leak is a comment. The same
  pattern is used for the fold-boundary position-overlap bug found in phase 3.
- **Nine red-team passes** in [`reports/adversary/`](reports/adversary/), run with fresh
  context and read-only access. They changed the code every time, and three times they killed a
  headline this repo had already written down: the phase-3 pass demoted the break-even column
  after showing a constant predictor scored *inside* the ladder's range on it; the second
  phase-2 pass showed that an in-sample calibration — not leakage — produced every large
  negative protocol-C R² in the first draft; and the third showed that the *artifact floor*
  itself was measured with the largest artifact channel switched off, that about half of the
  GBM's flagship gap is the harness's own calibration rule, and that a calibration-slope result
  published that morning was mechanically forced. Superseded columns stay in
  `reports/ladder.md` so the size of each correction is visible rather than asserted.
- **Controls in the results table, not just in the prose.** always-long, always-short and a
  perfect-foresight oracle run on identical rows, folds and fills, so a reader can see what a
  rung is actually beating.
- **A decision log.** [`DECISIONS.md`](DECISIONS.md) records every choice made without asking,
  dated, including the ones that were wrong and got reverted.

## Repo map

| path | what |
|---|---|
| `src/edge/io/` | LOBSTER reader, alignment invariant, data-quality report |
| `src/edge/features/` | six feature families, two clocks, labels |
| `src/edge/harness/` | the three-protocol splitter, metrics, L0–L4 ladder |
| `src/edge/costs/` | latency, fills, PnL, break-even cost, DM / bootstrap / deflated Sharpe |
| `src/edge/analysis/` | correlation triage, CKS self-check |
| `docs/spec/` | the project specification (`PLAN`, `EVAL_PROTOCOL`, `FEATURES`, …) |
| `docs/INTERFACE.md` | the frozen feature-module contract |
| `reports/ladder.md` | the optimism table, per-fold detail, embargo sweep |
| `reports/artifact_floor.md` | the frozen-predictor floor a gap must clear, per rung |
| `reports/` | every generated table, including the adversary reports |
| `METHOD.md` | how the numbers were produced, in enough detail to attack |
| `FINDINGS.md` | the claim, the evidence, and what would change my mind |
| `FUTURE.md` | what this harness is for next, in priority order |
| `INTERVIEW.md` | the one-page version: problem, method, result, what I'd do with more data |

## Limitations

Stated before anyone else has to point them out.

- **One trading day, two names.** Folds are intraday blocks; `EVAL_PROTOCOL.md`'s "split on
  day boundaries first" is unavailable, and the same-day-vs-next-day leak check **cannot be
  run at all**. It is recorded as unavailable, not as passed. Everything here is a
  single-session result and should be read as one.
- **2012 data vintage.** The free LOBSTER sample. Market structure has moved on; these are not
  present-day numbers.
- **The Nasdaq fee schedule is an assumption**, not a verified citation. The gross-based
  break-even column is the answer to anyone who disagrees with it — it is inclusive of all
  costs, so you can substitute your own.
- **No market impact model.** Stated, not omitted. Ignoring impact biases PnL upward.
- **The passive fill model is dead code in the headline path.** `walk_the_book` and
  `passive_round_trip_pnl` are unit-tested but never called: every fill is priced at the touch
  for 100 shares.
- **The DM tests are indicative, not decisive.** Adjacent rungs are largely nested, where
  standard Diebold–Mariano is undersized; `EVAL_PROTOCOL.md` asks for the Giacomini–White
  variant, which is not implemented.
- **The trading deadband was set after seeing protocol-C output.** Disclosed rather than
  presented as frozen — see the frozen-decisions table in `RESULTS.md`.
- **The microprice recursion does not converge on this data** (|eig(B)| = 0.9906), so
  `mp_micro` is honestly "the six-step microprice" — a fixed truncation, with the iteration
  count part of the feature's definition rather than a cap convergence reaches first.
- **The classification metrics are largely uninformative** at these horizons and are labelled
  as such: predictions are classified with the same 0.5-tick deadband as the labels, so a
  scaled sign predictor lands entirely in the flat class. The regression metrics are the
  informative ones.
- **Not every emitted column has a hand-computed literal test.** Every formula and every
  mutation the audit could construct is covered; roughly 35 OFI level×window cells, 12 label
  columns at the longer horizons and 5 state columns are pinned only structurally.

## Scope that was cut

The original plan had an order-flow language-model arm. `orderflow-lm` stops at tokenization
and no trained checkpoint exists, so there was nothing to probe; the arm was cut rather than
faked, and the phase renumbering is recorded in `DECISIONS.md` (2026-07-31). If a checkpoint
ever exists it becomes a follow-up arm against this harness, which is the point of having
built the harness first — see [`FUTURE.md`](FUTURE.md).
