VERDICT: SURVIVED WITH CAVEATS

Red team on Phase 3 (`reports/costs.md`, `src/edge/costs/*`). Every number below
was recomputed from `data/features/*.parquet` and `data/raw/*.csv` with code that does not
import `edge.costs`, except where noted. Scratch scripts:
`scratch/redteam/`
(`indep.py`, `dsr.py`, `ctrl.py`, `ctrl2.py`, `const.py`, `lat.py`).

The PnL, trade counts, gross ticks/trade and break-even columns are arithmetically right and
reproduce exactly. Two columns are mislabelled by a factor of n_trades, one reading note is
false, one assumption sentence contradicts the code, and the headline column is provably
model-invariant. None of that makes the PnL wrong, so this is not a kill.

## What I tried and what happened

1. **Protocols differ in more than the split** — N/A by construction (Protocol C only), and
   verified: `costs/run.py:178` calls the same `make_folds(..., protocol="C", n_folds=N_FOLDS,
   seed=args.seed)`, same `common_usable_mask`, same `build_ladder(seed)` as `harness/run.py`.
   One code path, one seed. No finding.
2. **Row alignment** — verified independently against the LOBSTER worked example on the raw
   CSVs, not against the phase-0 test. INTC message row 5 is a type-3 deletion of 100 shares at
   275100 (bid side); orderbook row 4 shows `bid_px_1=275100, size=400`, row 5 shows `size=300`.
   Orderbook row k is the state **after** message k. `data/frames/INTC_2012-06-21.parquet` has
   624,040 rows against 624,040 message lines and row 0 matches orderbook row 0. `apply_latency`
   uses `searchsorted(..., side="right") - 1`, i.e. the last book state at or before the action
   time. No lookahead in the fill prices.
3. **Lookahead in features** — forward-shift probe on the PnL itself (`ctrl2.py`), mask
   recomputed on the shifted columns so NaNs do not silently zero the trade count. INTC calendar
   1s gross ticks/trade: L2 0.00707 -> 0.01279 (+1) -> 0.03453 (+10); L3 0.01968 -> 0.02112 ->
   0.03476. Handing the features the future improves them, monotonically, and +10 lands at the
   perfect-foresight bound (0.0342, item 10). The shipped columns sit on the causal side.
   (L1 *falls* under a forward shift, 0.02127 -> 0.01393: sign(queue imbalance) mean-reverts
   after the quote moves. Explained, not alarming, since the fitted rungs respond correctly.)
4. **Scaler fit on the full sample** — `grep -n "\.fit(\|mean(axis=0)"` over `src/edge/harness`
   and `src/edge/costs` returns exactly three sites: `models.py:185` (ridge standardiser, inside
   `_fit_raw`, called only from `Rung.fit`), `harness/run.py:96` and `costs/run.py:111`, both
   `rung.fit(X[tr], y[tr])` on training rows only. The L1 through-origin rescale is likewise
   fitted on `tr`. Nothing global. No finding.
   One nit: `costs/run.py:204` takes the bps denominator as `nanmean(ask_px_1)` over the **whole
   session**, including segment 0 which is never tested. Implied price backed out of the
   ticks/bps ratio is 27.01-27.10 (INTC) and 222.40-223.16 (AMZN); the effect is <0.5%.
5. **R2 against the test mean** — not used in this report. Checked anyway: `r2_vs_zero` is
   `1 - SSE/sum(y^2)`, denominator is the sum of squares about zero, and `r2_test_mean` is
   carried as a separate diagnostic column in the phase-2 table. Correct.
6. **The L1 control** — Phase 3 is Protocol C only so there is no A-C gap to recompute here.
   For the record, `reports/ladder.md` gives L1 A-C gaps of 0.0019 (INTC/cal/1s) to 0.0134
   (INTC/event/1s) in R2 vs zero, against an L1 C-level of 0.0105-0.0719. That is 2-18% relative
   optimism on a rung with no fitted parameters beyond a one-scalar rescale. It is small next to
   L4's 0.49, but it is not zero, and it belongs in the phase-2 red team's scope, not mine.
7. **Base-rate artifact** — the PnL analogue is a directional-bias artifact, and it is real but
   secondary (`const.py`). A constant always-short predictor on the same trade times earns
   +0.00273 gross ticks/trade on INTC calendar 1s and **+0.12645** on AMZN event 5s, against
   L1's 0.30827 on the same rows. So ~41% of L1's AMZN event 5s gross is a one-day short drift,
   not microstructure. Several rungs are heavily one-sided (L1m long fraction 0.110, L2 0.213 on
   INTC calendar 1s). The report has no always-long/always-short baseline row and no de-meaning.
8. **Deadband tuning** — the deadband was moved from 0.5 ticks to 0 **after** looking at the
   first cost run's output. `DECISIONS.md` line 403 and commit `ae67de1` ("found by reading the
   output rather than by a test") both date it 2026-07-31, same day as the table. It is logged
   and the reason given (a scaled sign predictor cannot emit >0.5 ticks at 1s) is sound, but the
   knob that sets turnover, and therefore the entire cost story, was set by looking at protocol-C
   test-fold output. RULES.md rule 5. **Worse: the report still says the old value.**
   `costs/run.py:252` writes "Trades only when |prediction| exceeds the 0.5-tick deadband" while
   `simulate(..., deadband_ticks=0.0)` is never passed anything else. The stated assumption
   contradicts the code that produced the table.
9. **Trial count** — `--trials 288` is the phase-2 ladder count (RESULTS.md trial log row 5,
   "2 x 2 x 4 x 3 x 6 = 288 fits"). Phase 3 ran 2 tickers x 2 clocks x 2 horizons x 6 rungs x
   3 latencies = **144 configurations** (I counted 144 data rows in the table); none of them
   appear in the trial log, which still ends at row 8. The DSR is therefore deflated by a count
   that excludes its own trials. It does not matter, because of item 10's DSR result, but the
   log is out of date and RESULTS.md's "break-even cost (ticks)" column is still empty.
10. **Pooled-number laundering** — yes, everywhere. Every cell is pooled over 5 folds with no
    per-fold distribution and no worst fold (RULES.md rule 4). Recomputed for INTC calendar 1s
    L1 (`dsr.py`): per-fold gross ticks/trade {0.0232, 0.0233, 0.0192, 0.0123, 0.0281} — worst
    fold is 0.0123, 42% below the pooled 0.0213 — over per-fold n {3423, 3375, 3359, 3385, 3501}.
    Per-fold break-even is homogeneous {-0.794, -0.795, -0.799, -0.799, -0.790} only because it
    is dominated by the spread (see below).

## Phase-3-specific column reconciliation

Executable check: parse all 144 rows of `reports/costs.md`, recompute each column from its
neighbours (script inline in the transcript; 120 rows have trades, 24 are L0 with 0 trades).

- **`break-even (ticks)` == `net PnL $` / (2 * 100 * trades) / 0.01 in 120/120 rows.** So the
  column is the **incremental** cost per share per side *on top of* the full spread and the
  $0.0030 take fee already charged — not the total cost at which PnL crosses zero. That is a
  defensible definition and the report's "negative means already underwater" sentence is
  consistent with it. **The reading note is not.** "A break-even cost BELOW the assumed take fee
  of $0.0030/share (0.30 ticks) means the rung does not pay for its own fees" cannot be applied
  to an incremental number: no row in the table could ever be above 0.30 by construction, and
  the comparison is arithmetically meaningless. Delete that note or switch the column to
  `gross/(2*shares*n)`, which for INTC calendar 1s L1 is **+0.0106 ticks** — the number that
  actually answers "what per-side cost can this signal pay", and it is not in the table.
- **`PnL CI lo`/`PnL CI hi` do not bracket `net PnL $` in 0/120 rows. They bracket the mean
  per-trade PnL in 120/120.** INTC calendar 1s L1: net -27105.80, CI [-1.5948, -1.5862],
  mean per trade -1.59044. `block_bootstrap_ci` returns a CI for the mean (`stats.py:114`,
  and `tests/test_stats.py:85` asserts exactly that), but `run.py:271` prints it in a column
  headed "PnL CI" immediately to the right of the total. Off by a factor of n_trades = 17,043.
  The tests are right and the table is wrong.
- **`DSR` is computable, not a default, and completely vacuous.** For INTC calendar 1s L1 the
  per-trade Sharpe is **-7.635** (mean -1.590, sd 0.208 — the net PnL is a near-deterministic
  loss, so its Sharpe is enormous and negative). `deflated_sharpe` at n_trials = 1, 2, 10, 288,
  1e6, 1e12 returns dsr = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0. The threshold moves (0.000, 0.082,
  0.247, 0.453, 0.765, 1.118) and changes nothing. The `--trials 288` in the header is
  decorative: `--trials 1` produces the identical column plus the honesty warning. A DSR of
  0.000 on a strategy that loses the spread on every trade is not a finding, it is a tautology.
  There is no DSR on the gross series, which is the only series where deflation would bite.
- **Gross and net reconcile through the stated fee and spread.** Implied round-trip cost
  (`gross ticks/trade` - `net PnL $`/trades) is 1.6112-1.6178 ticks for INTC across all 60 rows
  and 12.5112-12.8527 for AMZN. Measured mean touch spread from the frames: INTC 1.234 ticks
  (median 1.0), AMZN 13.086 (median 13.0); plus 2 x $0.0030 x 100 shares = 0.60 ticks of fee on
  100 shares. 1.01 + 0.60 = 1.61 and ~12.0 + 0.60 = 12.6. Reconciles.
- **The independent re-derivation matches to the printed digit.** INTC calendar 1s L1 1ms, built
  from parquet with my own splitter call, my own spacing loop and my own fill arithmetic:
  trades 17043, net -27105.7999..., gross 0.021270 ticks/trade, break-even -0.795218 ticks,
  -2.9388 bps. Table: 17043, -27105.80, 0.0213, -0.7952, -2.94.
- **The table reproduces bit-for-bit, but not from the command printed on it.**
  `python3 -m edge.costs.run --seed 0 --trials 288` (the string in the report header, and
  `METHOD.md:176` prints an even shorter one) defaults to `--clocks calendar` and would emit
  half the table. The real command needs `--clocks calendar,event`; with it the body diffs clean
  against the committed file. RULES.md rule 7 ("reproducing a table is one command") is not met
  by the command the table advertises.
- **Trade counts do not match the stated explanation.** The report says counts "differ across
  rungs because the deadband bites differently". The deadband is 0. `_non_overlapping` is
  computed from `ts[te]` alone, so **every rung trades at the identical stamps**; the only
  difference is rows where the prediction is exactly 0. Measured: `qi_touch == 0` on 0.02% of
  INTC calendar stamps (17043 vs 17044) and **10.48%** of AMZN calendar stamps (13986/15458 =
  90.5%, matching to the row). The cause is bid size exactly equal to ask size, not a deadband.
- **Decision spacing mostly matches `latency + horizon`, with 4 violations.** Minimum gap between
  consecutive traded stamps on INTC calendar 1s at 1ms latency is 800 ms against a required
  1001 ms; the violating gaps are {800, 900, 900, 1000} ms — exactly 4, one per fold boundary,
  because `_non_overlapping` resets `next_free` on every fold. 4/17042 = 0.02%, immaterial to
  the number, but `tests/test_pnl.py::TestNonOverlappingDecisions` does not cover the boundary
  and the invariant in the docstring ("at most one is ever open") is false as run.
- **Latency binds, and does what it should.** Holding the decision set fixed and varying only the
  price-read offset (`lat.py`), gross ticks/trade decays monotonically: INTC calendar 1s
  0.02127 / 0.02130 / 0.02010 / 0.01499 at 1/10/100/1000 ms (6.9% of entry rows change at 10 ms,
  33% at 100 ms); AMZN event 1s 0.22255 / 0.19346 / 0.15461 / 0.09021. Caveat: in the shipped
  table the decision *set* also changes with latency on the event clock (7477 -> 7108 trades),
  so the reported `net PnL $` is not comparable across latency rows — the 100 ms row is less
  negative mainly because it trades less. The per-trade columns are the comparable ones.

## The finding that matters most

**The headline column is model-invariant.** Two controls, same trade times, same fills:

- Perfect foresight of the sign of the 1s mid move at the trade instant (`ctrl2.py`) earns
  0.0342 gross ticks/trade, **loses $456 net**, and reports a break-even of **-0.0134 ticks**.
  96.4% of trade stamps have zero 1s mid move; the rest average well under the 1.61-tick round
  trip. An omniscient trader is underwater under this cost and fill model.
- A constant always-long predictor reports break-even **-0.8072** ticks on INTC calendar 1s and
  **-6.4139** on AMZN event 5s. The entire ladder's break-even column spans [-0.8025, -0.7702]
  for INTC and [-6.3457, -5.9923] for AMZN. **The constant predictor sits inside the ladder's
  range.** The gaps do not vanish; they never existed.

So the break-even column is measuring the spread plus the always-on trading rule, to three
significant figures, for every rung and every latency. It cannot answer PLAN.md's actual
question ("does anything beat sign(queue imbalance) after costs?"). The only column that
discriminates is `gross ticks/trade`, and it ships with no CI, no per-fold distribution, no
worst fold and no directional-bias baseline. I computed the CIs: AMZN event 5s L4 GBM
0.5411 [0.3817, 0.7207] vs L1 0.3083 [0.1416, 0.4629] — overlapping, on a series where a
constant short already earns 0.1265.

## Most likely remaining explanation that is not "the effect is real"

The gross-ticks/trade ordering — the one place the ladder appears to separate, e.g. AMZN
event 5s L4 0.5411 > L3 0.5071 > L2 0.3163 > L1 0.3083 — is one day of one name with 2,966
non-overlapping decisions, of which a constant short position accounts for 0.1265 ticks. With
block-bootstrap CIs that overlap between adjacent rungs and a single 2012-06-21 session, the
most likely non-effect explanation is **intraday drift plus fold-level luck on a sample of five
folds**: the pooled ranking would be produced with material probability by five independent
short-biased fits on a day where AMZN fell. The purged split is fine; the sample is one day.

## What I could not check and why

- **The fee schedule.** $0.0030/$0.0020 is flagged as an unverified assumption in the code, the
  report and DECISIONS.md, and I have no archived 2012 Nasdaq price list. It is 0.60 of the
  1.61-tick INTC round trip and 0.60 of AMZN's 12.6, so it is the smaller half of the cost for
  INTC and a rounding error for AMZN. The break-even column does not let a reader substitute
  their own number, because it is incremental to that assumed fee rather than inclusive of it.
- **`walk_the_book` and `passive_round_trip_pnl` are dead code in the headline path.** `simulate`
  prices every fill at `bid_px_1`/`ask_px_1` for 100 shares and never calls either. They are
  unit-tested but they did not produce any number in this table, so the "fill at the displayed
  price up to displayed size, walk the book beyond it" claim in the module docstring is untested
  against the data. I did not check whether 100 shares exceeds displayed size at the touch.
- **AMZN as a "large-tick" comparison.** Median touch spread is 13 ticks. Every AMZN row is a
  statement about a 13-cent spread. Whether the ladder means anything there is a question about
  the universe choice, not about this code, and it is out of my scope.
- **Multi-day and cross-name generalisation.** Single ticker-day per name; nothing here can
  distinguish a signal from a session.
