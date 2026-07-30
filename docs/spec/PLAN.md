# PLAN — Edge

Short-horizon (1–10s) LOB prediction: do order-flow LM representations beat classic
microstructure baselines under purged walk-forward evaluation?

Target: **3 weeks of evenings**, because the parser and the trained model already exist.
The new work is features + harness + writeup.

---

## Phase 0 — Data contract and repo skeleton (day 1–2)

- [ ] Import the LOBSTER parser from orderflow-lm as a package, not a copy-paste. Freeze
      its output schema in `docs/DATA_CONTRACT.md`.
- [ ] Assert the message/orderbook alignment invariant: row *k* of the orderbook file is
      the state **after** message *k*. Write the test that fails if this is off by one.
- [ ] Build the event-indexed frame: `(ts_ns, event_type, direction, price, size,
      bid_px[1..L], bid_sz[1..L], ask_px[1..L], ask_sz[1..L])`.
- [ ] Sanity plots: intraday spread, depth, event-rate profile. Confirm the U-shape.
      If it isn't there, the parse is wrong.
- [ ] Decide and document the universe: start with **one large-tick and one small-tick
      name** (e.g. INTC vs AMZN in the free sample). The tick-size regime is the single
      biggest driver of which features work, and having both is an interview talking point.

**Milestone:** one command turns raw LOBSTER files into a validated event frame, with a
printed data-quality report (event counts by type, crossed-book count = 0, gaps).

## Phase 1 — Feature library (day 3–6)

- [ ] Implement everything in `FEATURES.md`, each as a pure function with a unit test on a
      hand-built toy book where the correct value is known by inspection.
- [ ] Two clocks: calendar time (100ms grid) and event time (every N events). Every
      downstream result gets reported on both, because calendar-time results on a quiet
      stock are mostly measuring the clock.
- [ ] Rolling normalization only: z-scores from a trailing window, fit on past data only.
      No full-sample scaler anywhere. Write the test that detects one.
- [ ] Label construction: Δmid and Δmicroprice at h ∈ {1s, 2s, 5s, 10s} plus the event-time
      equivalents; classification labels with an explicit deadband α (report the α
      sensitivity, don't tune it).

**Milestone:** `features.parquet` + a correlation table of every feature against every
horizon's label, with intraday-seasonality-adjusted versions alongside raw.

## Phase 2 — Baseline ladder (day 7–10)

Ordered so each rung has to beat the one below it:

- [ ] **L0** — constant/zero predictor; class base rate. (Beating this is not news.)
- [ ] **L1** — unfitted signs: sign(queue imbalance), sign(microprice − mid). No parameters.
      Surprisingly hard to beat at 1s.
- [ ] **L2** — OLS of Δmid on best-level OFI (the Cont–Kukanov–Stoikov specification).
      Recover their result: slope ≈ inversely proportional to depth. If you can't
      reproduce a 20-year-old linear result, stop and fix the features.
- [ ] **L3** — ridge / logistic on the full handcrafted set (multi-level OFI, integrated
      OFI, queue imbalance, microprice deviation, spread, depth, trade-sign imbalance,
      realized vol, time-of-day).
- [ ] **L4** — gradient boosting on the same handcrafted set. This is the real bar.

**Milestone:** the ladder table, per-fold, on both clocks, both tick regimes.

## Phase 3 — LM representations (day 11–15)

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

- [ ] Transaction-cost-aware simulated PnL per `EVAL_PROTOCOL.md`: latency delay, explicit
      fill model, half-spread crossing, fees.
- [ ] **Break-even cost table** — the cost per trade at which each rung's PnL hits zero.
      This is the headline figure and the most honest number in the project.
- [ ] Diebold–Mariano tests between adjacent rungs; block-bootstrap CIs on the metric
      differences; deflated Sharpe ratio using the logged trial count.

**Milestone:** a PnL section that a QR would not immediately dismiss.

## Phase 5 — Writeup (day 20–21)

- [ ] `FINDINGS.md`: the claim, the evidence, the failure modes, "what would change my mind".
- [ ] `README.md` with the headline table and reproduction instructions.
- [ ] A one-page version for interviews: problem, method, result, what you'd do with more
      data — see `INTERVIEW.md`.

**Stretch (only if Phase 3 is a clean result):** cross-asset transfer (train on one name,
test on another), and the regime split (high-vol vs low-vol hours).

---

## Status

- Current phase: 0
- Next 3 tasks: (fill in)
- Trials logged so far: 0
