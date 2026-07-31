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
  first-order term and emphatically not a synonym for weighted mid (see §7).
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

- **A − C** is total optimism.
- **B − C** isolates the label-overlap component (B is already strictly forward in time).
- **A − B** is the residual: training on the future outright.

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

L1 is load-bearing. It has no fitted parameters, so its A−C gap is a **floor on what the
harness itself contributes**: a large L1 gap would mean the decomposition is measuring the
protocols rather than leakage.

Every rung's raw score is mapped into label units by one through-origin scale fitted on
training rows only, applied uniformly. For least-squares rungs it is exactly 1.0 (residuals are
orthogonal to fitted values), so it is a no-op for L2/L3 and only puts L1 and L4 on a
comparable footing. Being rank-preserving, it never changes the IC.

No early stopping on the GBM: stopping on a slice of the test fold is itself a leak, and
carving a validation slice out of training is one more free parameter to log.

## 6. Metrics

- **Out-of-sample R² against a ZERO forecast**, never the test mean. Using the test mean hands
  the model the test set's own average return; on a near-zero-mean series the two are
  numerically close, which is exactly why the mistake survives review. The leaky variant is
  computed alongside purely to show the size of the gap.
- **Information coefficient** — Spearman ρ per fold, hand-written rank transform.
- **Classification** — balanced accuracy and MCC, always printed next to the base rate. At 1s
  ~96% of INTC labels are flat, so plain accuracy is meaningless.
- **Never a pooled number alone.** Mean, median and **worst fold** travel together.

## 7. What the red team changed

Four adversary passes ran against artifacts, with fresh context and read-only access. They are
in `reports/adversary/`. The findings that changed code:

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

## 8. Reproduction

```bash
make data       # raw LOBSTER CSVs -> validated event frame + data-quality report
make features   # -> data/features/*.parquet + the correlation table
make selfcheck  # Cont-Kukanov-Stoikov reproduction; fails loudly if the relation breaks
make test       # the full suite
python3 -m edge.harness.run --seed 0    # the optimism table
python3 -m edge.costs.run   --seed 0    # break-even cost table
```

Every entry point takes `--seed`. The feature build is bit-reproducible: a full rebuild was
compared column-by-column across all four tables (1,028 columns) with zero mismatches,
including NaN placement.

## 9. Known limitations

- **One trading day, two names.** Fold boundaries are intraday blocks; the "split on day
  boundaries first" rule in the protocol is unavailable, and the same-day-vs-next-day leak
  check **cannot be run at all** — recorded as unavailable, not as passed.
- **2012 data vintage.** Market structure has moved on; these are not present-day numbers.
- **The Nasdaq fee schedule is an assumption**, not a verified citation (§ `reports/costs.md`).
- **No market impact model.** Stated, not omitted; ignoring impact biases PnL upward.
- **Not every emitted column has a hand-computed literal test.** Every formula and every
  mutation the audit could construct is covered, but roughly 35 OFI level×window cells, 12
  label columns at the longer horizons, and 5 state columns are pinned only structurally.
