# PLAN — Edge

Short-horizon (1–10s) LOB prediction: do order-flow LM representations beat classic
microstructure baselines under purged walk-forward evaluation?

Target: **3 weeks of evenings**, because the parser and the trained model already exist.
The new work is features + harness + writeup.

---

## Phase 0 — Data contract and repo skeleton (day 1–2)

- [x] Import the LOBSTER parser from orderflow-lm as a package, not a copy-paste. Freeze
      its output schema in `docs/DATA_CONTRACT.md`. (orderflow-lm's parser turned out to be
      C++ with no Python bindings — reads the CSV pair directly instead, orderflow-lm stays
      the convention reference; see `DECISIONS.md` 2026-07-30.)
- [x] Assert the message/orderbook alignment invariant: row *k* of the orderbook file is
      the state **after** message *k*. Write the test that fails if this is off by one.
      (`tests/test_alignment.py::test_orderbook_row_is_state_after_message`, green against
      full raw INTC/AMZN files.)
- [x] Build the event-indexed frame: `(ts_ns, event_type, direction, price, size,
      bid_px[1..L], bid_sz[1..L], ask_px[1..L], ask_sz[1..L])`.
      (`src/edge/io/lobster.py:build_event_frame`.)
- [x] Sanity plots: intraday spread, depth, event-rate profile. Confirm the U-shape.
      If it isn't there, the parse is wrong. (Markdown tables, not plots — `reports/
      data_quality.md`; INTC shows a clean U-shape, AMZN's is weaker at the open —
      logged as an observation, not a bug, since alignment/crossed-book/scaling all pass.)
- [x] Decide and document the universe: start with **one large-tick and one small-tick
      name** (e.g. INTC vs AMZN in the free sample). The tick-size regime is the single
      biggest driver of which features work, and having both is an interview talking point.

**Milestone:** one command turns raw LOBSTER files into a validated event frame, with a
printed data-quality report (event counts by type, crossed-book count = 0, gaps).

## Phase 1 — Feature library (day 3–6)

- [x] Implement everything in `FEATURES.md`, each as a pure function with a unit test on a
      hand-built toy book where the correct value is known by inspection.
      (Six family modules, 144 raw columns; `docs/INTERFACE.md` is the frozen contract.
      Coverage is not yet one hand-computed literal per column — see the gap note below.)
- [x] Two clocks: calendar time (100ms grid) and event time (every N events). Every
      downstream result gets reported on both, because calendar-time results on a quiet
      stock are mostly measuring the clock.
- [x] Rolling normalization only: z-scores from a trailing window, fit on past data only.
      No full-sample scaler anywhere. Write the test that detects one.
      (`normalize.rolling_zscore` baselines `[t-window, t)` in stamp positions, strictly
      excluding `x[t]`; `tests/test_normalize.py::TestNoFullSampleFit`.)
- [x] Label construction: Δmid and Δmicroprice at h ∈ {1s, 2s, 5s, 10s} plus the event-time
      equivalents; classification labels with an explicit deadband α (report the α
      sensitivity, don't tune it). (α = 0.5 ticks frozen by rule; the microprice label is
      substituted by a weighted-mid label — `DECISIONS.md` 2026-07-31.)

**Known gap carried into Phase 2**, recorded rather than papered over: `FEATURES.md`'s
"every feature gets a hand-computed toy-book test" is met for every formula and for every
mutation the 2026-07-31 audit could construct, but not yet as one literal per emitted
column. Untested-by-literal at close: ~35 `ofi_l{m}_{w}` level×window combinations (the
level indexing, depth scaling, PCA orientation and fit-sample causality are each pinned,
but not every cell of the cross product), 12 `label_*` columns at the 2s/5s/10s horizons
(NaN rules asserted, values not), and 5 `st_*` columns that survive replacement by zeros.
None of these blocks the harness; all of them are places a silent regression could still
land.

**Milestone:** `features.parquet` + a correlation table of every feature against every
horizon's label, with intraday-seasonality-adjusted versions alongside raw.

## Phase 2 — Baseline ladder (day 7–10)

Ordered so each rung has to beat the one below it:

- [x] **L0** — constant/zero predictor; class base rate. (Beating this is not news.)
- [x] **L1** — unfitted signs: sign(queue imbalance), sign(microprice − mid). No parameters.
      Surprisingly hard to beat at 1s.
- [x] **L2** — OLS of Δmid on best-level OFI (the Cont–Kukanov–Stoikov specification).
      Recover their result: slope ≈ inversely proportional to depth. If you can't
      reproduce a 20-year-old linear result, stop and fix the features.
- [x] **L3** — ridge / logistic on the full handcrafted set (multi-level OFI, integrated
      OFI, queue imbalance, microprice deviation, spread, depth, trade-sign imbalance,
      realized vol, time-of-day).
- [x] **L4** — gradient boosting on the same handcrafted set. This is the real bar.

**Milestone:** the ladder table, per-fold, on both clocks, both tick regimes.

## Phase 3 — LM representations (day 11–15) — **CUT**

> Cut on 2026-07-31: `orderflow-lm` stops at tokenization and no trained checkpoint exists,
> so there was nothing to probe. The arm was cut rather than faked. It returns as L5/L6
> against this harness the moment a checkpoint exists — `FUTURE.md` sec. 2. The costs phase
> below was renumbered to phase 3 in the working vocabulary; this file keeps the original
> numbering.

- [ ] Extract hidden states from the order-flow LM at each decision timestamp. Fix and
      document the pooling (last token vs mean over the window) and the layer — pick by
      probing on a *held-out-from-everything* dev slice, then freeze.
- [ ] **L5** — handcrafted features + LM states. **L6** — LM states alone.
- [ ] The three controls that decide whether this result means anything:
      - **Untrained control**: identical architecture, random weights, same pooling. If
        L5 ≈ untrained-L5, the gain is a random projection of recent history, not learning.
      - **Lag control**: raw lagged features over the identical lookback window the LM sees.
        The LM has to beat naive memory, not just the contemporaneous snapshot.
      - **Dimension control**: match the parameter count of the augmented model by giving
        the handcrafted model the same number of random-projection features.
- [ ] Alpha decay curve: incremental R² vs horizon. Kolm et al. find the effective horizon
      is roughly two average price changes — check whether yours agrees.

**Milestone:** the incremental-value table with all three controls. This is the paper.

## Phase 4 — Costs, PnL, and statistics (day 16–19)

- [x] Transaction-cost-aware simulated PnL per `EVAL_PROTOCOL.md`: latency delay, explicit
      fill model, half-spread crossing, fees.
- [x] **Break-even cost table** — the cost per trade at which each rung's PnL hits zero.
      This is the headline figure and the most honest number in the project.
- [x] Diebold–Mariano tests between adjacent rungs; block-bootstrap CIs on the metric
      differences; deflated Sharpe ratio using the logged trial count.

**Milestone:** a PnL section that a QR would not immediately dismiss.

## Phase 5 — Writeup (day 20–21)

- [x] `FINDINGS.md`: the claim, the evidence, the failure modes, "what would change my mind".
- [x] `README.md` with the headline table and reproduction instructions.
- [x] A one-page version for interviews: problem, method, result, what you'd do with more
      data — see `INTERVIEW.md`.

**Stretch (only if Phase 3 is a clean result):** cross-asset transfer (train on one name,
test on another), and the regime split (high-vol vs low-vol hours).

---

## Status

**Closed out 2026-08-02.** Phases 0, 1, 2, 4 and 5 are complete; phase 3 (LM representations)
was cut for want of a checkpoint. The study is written up in `FINDINGS.md`, `README.md`,
`METHOD.md`, `INTERVIEW.md` and `FUTURE.md`.

- Headline: optimism scales with model capacity (median A−C: L1 +0.003 with no parameters,
  L4 +0.545 on identical rows), and `B−C` ≈ 0 everywhere — essentially all of it is the
  shuffle, not label overlap. Under protocol C the ladder inverts: L3 and L4 have negative
  median R², and the best rung is the parameter-free `sign(queue imbalance)`.
- Economics: nothing pays. L1 earns 0.0106 ticks per side against ~0.92 ticks of cost, and
  perfect foresight of the 1s mid move still loses $458 under aggressive fills.
- Trials logged: **984** (288 ladder + 480 embargo sweep + 216 cost configurations), which
  deliberately over-counts. See the `RESULTS.md` trial log, including the two discarded
  harness configurations that invalidated their own runs.
- Adversary passes: 7, all in `reports/adversary/`. The phase-3 pass demoted this project's
  own headline column.

Known unmet items, carried in `FUTURE.md` rather than quietly dropped: the calibration slope
and AUC are implemented and tested but not reported; Giacomini–White for nested rungs is not
implemented, so the DM p-values are indicative; the passive fill model is dead code in the
headline path; roughly 52 emitted columns are pinned structurally rather than by a
hand-computed literal.
