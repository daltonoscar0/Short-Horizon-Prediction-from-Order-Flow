# METHOD

How the numbers in this repo were produced, in enough detail to attack.

The claim under test: **how much of the published short-horizon LOB prediction performance is
an artifact of leaky cross-validation?** The method is to build the classic microstructure
baselines properly, then score the whole ladder under three evaluation protocols and report
the gaps.

---

## 1. Data

LOBSTER message + orderbook CSV pair, 2012-06-21, level 10, two names:

| ticker | regime | events | calendar stamps (100ms) | event stamps (N=10) |
|---|---|---|---|---|
| INTC | large-tick (spread pinned at 1 tick) | 624,040 | 234,000 | 61,460 |
| AMZN | small-tick (median spread ~12 ticks) | 269,748 | 234,000 | 26,724 |

Both tick regimes are carried deliberately: tick size is the single biggest driver of which
microstructure features work.

**The alignment invariant.** Orderbook row *k* is the book state *after* message *k*. This is
asserted, not assumed — `tests/test_alignment.py` fails if it is off by one, and it was
re-verified independently from the raw CSVs during the phase-0 red team (a type-1 bid add at
message row 102 appears in orderbook row 102 and not 101).

Prices are raw LOBSTER integers (units of 0.01 cents) throughout; every emitted price level
and price difference is converted to **ticks** exactly once, at the point of emission.

## 2. Two clocks

Everything is computed on both, and reported on both, because calendar-time results on a
quiet instrument are partly measuring the clock rather than the market.

- **Calendar** — 100ms grid, feature state as of the last event at or before each stamp.
- **Event** — every 10 book-changing events (types 1–4; types 5–7 leave the displayed book
  untouched and do not tick the clock).

Both resolve a stamp to **the last frame row with `ts_ns <= t`**. Event-clock stamps sharing a
nanosecond collapse to one sample: you cannot act at three distinct moments within the same
nanosecond, and duplicated rows are a leakage hazard the moment folds are cut.

## 3. Features

Six family modules, 144 raw columns plus a seasonality-adjusted twin of each non-label column
(257 total per table). Formulas are in `docs/spec/FEATURES.md`; the module contract is
`docs/INTERFACE.md`.

- **OFI** — Cont–Kukanov–Stoikov per-event contribution, best level and levels 1–10, over
  100ms/1s/5s lookbacks; depth-scaled multi-level; integrated OFI as the first principal
  component (Cont–Cucuringu–Zhang).
- **Queue imbalance** — touch, per level, and depth-weighted under uniform / 1/m / exponential
  weights. `sign(qi_touch)` is baseline L1.
- **Microprice** — Stoikov (2018), the **iterated** absorbing/transient estimate, not the
  first-order term and emphatically not a synonym for weighted mid (see §8).
- **Trade** — signed volume, counts, effective spread, aggressor sign. LOBSTER gives the
  direction explicitly, so no Lee–Ready classification is needed.
- **State** — spread in ticks, depth, book slope, realized volatility, event intensity,
  time-of-day bucket.
- **Labels** — Δmid and Δwmid at 1s/2s/5s/10s plus event-time analogues; classification labels
  with deadband α = **0.5 ticks, frozen by rule**, never tuned.

**Causality.** A feature at stamp `t` may read event rows with `ts_ns <= t` and nothing else.
`label_` is the only forward-looking prefix, so one `grep label_` finds every exception.

**Fitted statistics refit on a schedule.** PCA loadings, depth normalisers and the microprice
transition matrices are refit once per 30-minute block on frame rows **strictly before** the
block start. Block 0 has no prior data and is NaN — 7.69% of stamps, the honest price of a
causal fit, reported rather than papered over with a full-sample fallback.

**Normalisation is rolling only.** `rolling_zscore` baselines on `[t-window, t)` in stamp
positions, strictly excluding `x[t]` — a feature's current draw must not contaminate its own
baseline. The seasonality control z-scores against *earlier stamps in the same time-of-day
bucket*, expanding. There is no full-sample scaler anywhere, and a test greps for the patterns
one leaves behind.

## 4. The three protocols

This is the contribution. The splitter is hand-written; `sklearn` is not installed, so the rule
is enforced by the environment rather than by discipline.

| | training set | purge | embargo |
|---|---|---|---|
| **A** shuffled k-fold | everything outside the test fold, *including the future* | none | none |
| **B** walk-forward | everything strictly before the test block | none | none |
| **C** purged walk-forward | strictly before, minus contaminated rows | h | E |
| **D** *control* | everything outside C's test block, past **and** future | none | none |

- **A − C** is total optimism.
- **A − D** is fold geometry plus the **temporal adjacency** of training rows — this is where
  label overlap actually lives, and it is the large term: median +0.2964 R² for L4, which is
  93% of that rung's whole A−C gap.
- **D − C** is training on the future in blocks *disjoint* from the test set — median +0.0067
  for L4. Handing a model the whole future, but not the neighbours of the test rows, buys it
  very little. The split is capacity-dependent: for L3 the future is worth 44% of the gap, for
  L4 only 2%. A model with enough capacity to memorise needs a near-duplicate, not the future.
- **B − C** is the purge/embargo effect **at a walk-forward boundary**, which is near zero for
  an arithmetic reason: purge+embargo can only remove the last `h + E` of a multi-hour
  training set, 0.6–1.1% of rows.

**This corrects an earlier labelling.** `B − C` was described as "the label-overlap component"
and `A − B` as "training on the future outright". Both were wrong, and protocol D — added
2026-08-02 after the phase-2 red team — is what measures the difference. A near-zero `B − C`
does not license the conclusion that label overlap is unimportant.

**Calibration is causal.** Every rung's raw score is mapped into label units by a
through-origin scale fitted on a **held-out last 20%** of each fold's training rows (a seeded
random 20% under A, where training rows are not time-ordered). Fitting it in-sample — as the
first draft did — returns almost no shrinkage for a high-capacity rung, inflating the
out-of-sample amplitude until R²-vs-zero goes sharply negative. That artifact produced every
large negative protocol-C R² in the earlier ladder, and because it never bites under A, it
inflated the measured optimism itself.

**The artifact floor is measured, and it is measured like-for-like.** A frozen model — fitted
once on the whole day, scored under every protocol with only the per-fold scale refitted, under
that protocol's own holdout rule — still shows a nonzero A−C gap. Two mechanisms produce it.
The first is fold aggregation: A's fold mean approximates the `Σy²`-weighted pooled metric while
C's is an unweighted average over contiguous segments of very unequal `Σy²`. The second is the
calibration rule itself, and it is the larger of the two — freezing the scale as well as the
model gives floors of +0.0020/+0.0020/+0.0153 at L1/L3/L4 magnitudes against the like-for-like
+0.0043/+0.0112/+0.0192. Both columns ship; their difference *is* the calibration channel,
measured on a predictor that cannot leak. Every headline cell reports the like-for-like floor at
its own rung's magnitude, and a rung that does not clear it has measured nothing.

**The calibration rule is a channel in `A − C`, and it is decomposed rather than absorbed.**
Under A the scale's holdout is a random 20% of training rows, which is exchangeable with A's
random test fold; under B/C/D it is the last 20% in time, which is not exchangeable with a later
block. That is a legitimate part of what shuffling costs you, but it is a *different* mechanism
from seeing the future. Re-running A under C's rule (`--holdout-control`) leaves L4's median gap
at **+0.1379** of +0.3173 and takes L1's to **−0.0021**. Found on 2026-08-03 when a
calibration-slope reading was published as a leakage signature and the red team showed a frozen
model reproduces it.

**Folds are cut on wall-clock time, not stamp index.** 32% of INTC event-clock stamps sit
within 0.5 ms of the previous one and carry ~2.3× the correlation of isolated stamps; an index
cut lets a single burst land on both sides of a boundary, so two folds hold the same episode
and duplication reads as optimism. The cost is that per-fold sample counts vary, which is
reported.

**Purge and embargo compose additively**: a training sample survives iff
`t + horizon + embargo < test_block_start`. Purge removes label overlap; embargo removes an
additional serial-correlation window. E = max(60s, 10h), with a sensitivity sweep.

**All three protocols are scored on the same rows.** The walk-forward protocols reserve
segment 0 as initial training data, so A is restricted to testing the same universe — otherwise
A would additionally be tested on the market open, the most volatile part of the session, and
that difference would be charged to leakage. A still *trains* on everything outside its test
fold, because that is what shuffled k-fold does.

**Every rung is scored on a common row mask.** The ladder's premise is that each rung beats the
one below it, which means nothing if the rungs see different samples.

The adversarial tests were written **before** the splitter, and include a deliberately broken
splitter that the leak detector is asserted to catch — a leak detector that has never caught a
leak is a comment.

## 5. The ladder

| rung | what it is | parameters |
|---|---|---|
| **L0** | zero predictor | none |
| **L1** | `sign(queue imbalance)`, `sign(microprice − mid)` | none |
| **L2** | OLS of Δmid on best-level OFI (Cont–Kukanov–Stoikov) | 2 |
| **L3** | ridge on the handcrafted set, standardiser fit per fold on training rows only | 28 |
| **L4** | gradient boosting on the same set, 200 rounds, no early stopping | many |

L1 was intended as the load-bearing control and **is not sufficient for it**. Its
"one-parameter rescale" is fitted, so its A−C gap mixes a partition artifact with a refit
effect; the red team showed those two together account for the whole L1/L1m/L2 gap. The real
control is the frozen predictor described in §4.

Every rung's raw score is mapped into label units by one through-origin scale, fitted on a
held-out slice of the training rows (§4). For least-squares rungs it is close to 1.0 (residuals
are orthogonal to fitted values in-sample), so it does little for L2/L3 and mainly puts L1 and
L4 on a comparable footing. Being rank-preserving, it never changes the IC — which is why the
IC columns are the ones immune to the calibration artifact, and why the surviving result is
stated on IC.

No early stopping on the GBM: stopping on a slice of the test fold is itself a leak, and
carving a validation slice out of training is one more free parameter to log.

## 6. Metrics

- **Out-of-sample R² against a ZERO forecast**, never the test mean. Using the test mean hands
  the model the test set's own average return; on a near-zero-mean series the two are
  numerically close, which is exactly why the mistake survives review. The leaky variant is
  computed alongside purely to show the size of the gap.
- **Information coefficient** — Spearman ρ per fold, hand-written rank transform.
- **Classification** — balanced accuracy and MCC, always printed next to the base rate. At 1s
  ~96% of INTC labels are flat, so plain accuracy is meaningless. Read balanced accuracy and MCC
  on the unfitted rungs as *null*: a rescaled sign predictor never clears the 0.5-tick deadband,
  so its predicted class is always 0, balanced accuracy is exactly 1/3 and MCC is NaN.
- **AUC on the directional subset** — up versus down on the rows the deadband calls
  directional, flats excluded. Pooling the flats into the negative class would make AUC a
  function of tie mass rather than of skill, which is the axis error the phase-1 red team caught
  in the correlation table; under this definition 0.5 is chance in every cell. A one-sided test
  block returns NaN, not 0.5, so an undefined cell cannot enter a mean as a chance-level row.
  AUC ranks rather than thresholds, so it is the only classification metric carrying information
  for the unfitted rungs, and it is the metric the capacity result is restated on
  (`FINDINGS.md` §1.1a). **Quote the raw-score column.** The through-origin rescale is
  rank-preserving only *up to sign*, the fitted scale goes negative in 27 of 400 protocol-C
  folds and 0 of 400 protocol-A folds, and a negative scale sends AUC to 1−AUC — which is what
  produced the unfitted rungs' apparent A−C gap in the first draft of that section.
- **Calibration slope** — the slope of realized on predicted, per fold. `EVAL_PROTOCOL.md` §2
  asks for it because R² and IC cannot separate a model that is directionally right but too
  loud from one that is simply wrong. **Do not read its A-versus-C difference as leakage.**
  Under A the scale's holdout is exchangeable with A's test fold, so the slope is driven to ≈1.0
  mechanically for any predictor; a frozen model that cannot leak reproduces the published split
  to three decimals. It is also unstable at the cell level (its denominator is the prediction's
  variance, and L3 ranges [−4.74, 43.20]), and by median aggregation protocol C's slopes come
  out near 1.0 too. Reported as the spec-mandated diagnostic; no claim rests on it.
- **Never a pooled number alone.** Mean, median and **worst fold** travel together.

## 7. Costs, fills and the break-even number

Protocol C only. A PnL computed under a leaky protocol is not a weaker result, it is a
misleading one, and §4 exists to establish which protocol to believe.

**The decision sequence.** Signal formed at `t` from information ≤ `t`; earliest action at
`t + Δ` for Δ ∈ {1ms, 10ms, 100ms}; position closed at `t + Δ + h`. The entry price is read at
`t + Δ`, which is the whole point of the latency ladder — a signal whose edge is gone by 100ms
is latency arbitrage, not alpha.

**Fills are aggressive**: buy crosses to the ask, sell to the bid, so a round trip pays the
full spread plus two take fees. Fees are the commonly-quoted 2012 Nasdaq standard rates
($0.0030 take / $0.0020 make), entered as an **assumption** and not verified against an
archived price list.

**Position discipline.** Single unit of 100 shares, no inventory, flat at horizon. Decisions
are spaced at least `Δ + h` apart, greedily from the earliest eligible stamp, and **the spacing
watermark carries across fold boundaries** — resetting it per fold let 4 positions on INTC
calendar 1s open 800–1000ms apart against a required 1001ms. Direction is the sign of the
prediction; the deadband is 0 (see §9).

**The break-even column is computed on the GROSS (mid-to-mid) series**, so it reports the total
per-side cost the signal can bear, inclusive of everything. The earlier draft computed it on
the net series, making it *incremental* to a spread already crossed — which made it nearly
model-invariant, since an always-on strategy loses almost exactly the spread whatever it
predicts. The 2026-08-02 red team demonstrated this by scoring a constant always-long predictor
*inside* the ladder's range. The incremental column is retained and demoted rather than
deleted, because removing the number an earlier draft led with would hide the correction.

**Controls ship as rows**, on identical rows, folds, mask and fills: always-long, always-short,
and a perfect-foresight **oracle** that trades the realised sign of the move. The oracle is a
bound, not a rung — it reads the future by construction, and its feature list says so. It is
the most informative row in the table: it still loses money.

**Inferential statistics.** Circular block bootstrap (block length ≥ the label horizon in
stamps) for CIs on the **mean per trade** — an earlier draft printed that interval beside the
*total* PnL, off by a factor of `n_trades`. Diebold–Mariano between adjacent rungs on per-trade
gross PnL, with loss `= −gross ticks` (squaring a PnL would test whether one rung's PnL is
larger in *magnitude*, which is the wrong hypothesis) and a HAC lag equal to the block length.
Deflated Sharpe on the **gross** series at n = 2,152 trials: on the net series every rung has a
per-trade Sharpe near −7.6 and the DSR is 0.000 at every trial count from 1 to 10¹², which is a
tautology rather than a finding.

## 8. What the red team changed

Seven adversary passes ran against artifacts, with fresh context and read-only access. They are
in `reports/adversary/`. The findings that changed code:

- **The in-sample calibration was inflating the phase-2 headline** (§4). Every large negative
  protocol-C R² was an artifact of it, and most reported gaps shrank by 20–95% once the scale
  was fitted causally. The superseded columns are retained in `reports/ladder.md`.
- **`B − C` was labelled with the wrong mechanism** (§4). Protocol D was added to separate
  "saw the future" from "saw the neighbours of the test rows"; the second is much larger.
- **L1 was not a sufficient control** (§5). Replaced by a frozen-predictor floor, which is
  nonzero and which reproduced the old L1 IC gap to five decimals.
- **45 of 80 rows lost more than half their gap when fold 0 was dropped**, so the headline
  table now carries median-based and ex-fold-0 gaps beside the mean.
- **The phase-3 headline column was model-invariant** and was replaced (see §7). The same pass
  found the mislabelled confidence intervals, the vacuous deflated Sharpe, the cross-fold
  position overlap, a reproduction command that rebuilt half its own table, and a stated
  assumption (a 0.5-tick deadband) that contradicted the code.

- **`mp_micro` was Stoikov's first-order term G¹, not the microprice.** The landing matrix
  B = (I−Q)⁻¹R₂ was never built and the outer recursion G⁽ⁱ⁺¹⁾ = G¹ + B·G⁽ⁱ⁾ never ran. Both
  added. The recursion does **not** converge on this data (|eig(B)| = 0.9906, drift −0.018
  ticks/iteration), so `MICRO_ITERS = 6` is part of the feature's *definition*, not a cap —
  `mp_micro` is honestly "the six-step microprice".
- **The event clock violated its own `<= t` contract** on 10.48% of INTC stamps, putting a mid
  move that elapsed in *zero wall-clock time* inside `label_y_mid_*`.
- **A leaking full-sample PCA fit was invisible to every test**, because the only multi-block
  fixture was one level deep, where the first PC of a single column is ±1 regardless of fit
  sample. Six such mutations are now caught.
- **The phase-1 leak triage was read off a truncated console print** (`suspicious[:20]` of a
  488-item list), producing a wrong "AMZN is clean" conclusion.

## 9. Reproduction

```bash
make data       # raw LOBSTER CSVs -> validated event frame + data-quality report
make features   # -> data/features/*.parquet + the correlation table
make selfcheck  # Cont-Kukanov-Stoikov reproduction; fails loudly if the relation breaks
make test       # the full suite
python3 -m edge.harness.run --seed 0 --embargo-sweep --in-sample-scale \
    --holdout-control                                                  # the optimism table
python3 -m edge.harness.floors --seed 0                 # the artifact floor, per rung
python3 -m edge.costs.run   --seed 0 --trials 2152 \
    --tickers INTC,AMZN --clocks calendar,event --horizons 1s,5s   # break-even table
```

The cost command needs its full argument set: `--clocks` defaults to `calendar` alone, and an
earlier draft of this file advertised the short form, which rebuilds half the table. Each
report now echoes the exact arguments that produced it in its own header.

Every entry point takes `--seed`. The feature build is bit-reproducible: a full rebuild was
compared column-by-column across all four tables (1,028 columns) with zero mismatches,
including NaN placement.

## 10. Known limitations

- **One trading day, two names.** Fold boundaries are intraday blocks; the "split on day
  boundaries first" rule in the protocol is unavailable, and the same-day-vs-next-day leak
  check **cannot be run at all** — recorded as unavailable, not as passed.
- **2012 data vintage.** Market structure has moved on; these are not present-day numbers.
- **The Nasdaq fee schedule is an assumption**, not a verified citation (§ `reports/costs.md`).
- **No market impact model.** Stated, not omitted; ignoring impact biases PnL upward.
- **Every fill is priced at the touch for 100 shares.** `walk_the_book` and
  `passive_round_trip_pnl` are unit-tested but never called in the headline path, so the
  "walk the book beyond displayed size" claim is untested against real data.
- **The DM tests are indicative, not decisive.** Adjacent rungs are largely nested, where
  standard Diebold–Mariano is undersized; §2 of the protocol asks for the Giacomini–White
  variant, which is not implemented.
- **The calibration slope is reported but unstable per cell** (2026-08-03). Its denominator is
  the prediction's variance, so a low-variance fold sends the slope to ±40; the medians are
  meaningful, the individual cells often are not. AUC is reported on the directional subset
  only, which is a choice: it makes 0.5 chance in every cell regardless of tie mass, at the
  cost of saying nothing about a rung's ability to identify the flats.
- **The trading deadband was set after seeing protocol-C output** (0 ticks, not the labels'
  0.5). The reasoning is a-priori but the trigger was an empty table; disclosed in the
  frozen-decisions table of `RESULTS.md` rather than presented as clean.
- **Not every emitted column has a hand-computed literal test.** Every formula and every
  mutation the audit could construct is covered, but roughly 35 OFI level×window cells, 12
  label columns at the longer horizons, and 5 state columns are pinned only structurally.
