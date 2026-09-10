# DECISIONS, Edge

Every choice made without stopping to deliberate: what was decided, the one-line rationale, the
date. Frozen choices additionally get a row in the RESULTS.md frozen-decisions table.

- 2026-07-30, Raw LOBSTER CSVs live at `data/raw/lobster/`, fetched from the Hugging Face
  mirror already used by orderflow-lm's fetch script (lobsterdata.com serves an SPA shell on
  the old sample paths). `data/` is gitignored: 250 MB of CSV does not belong in the repo.
- 2026-07-30, Universe is the 10-level full-day (09:30-16:00) free-sample files for INTC
  and AMZN, 2012-06-21, per `DATA_AND_LINKS.md`; the 50-level 1-hour samples in orderflow-lm
  cover different tickers and a shorter window.
- 2026-07-30, Feature storage is `pyarrow` (23.0.1, already in the env); `polars` is not
  installed and adds nothing pyarrow doesn't cover for CSV->parquet.
- 2026-07-30, The orderflow-lm parser is C++ with no Python bindings, so "import as a
  package" cannot be literal. Its SPEC's Decision A applies here too: on level-censored
  LOBSTER samples, message-replay reconstruction diverges from the orderbook file
  (measured today with its `book_source_compare`: 90.3% best-bid disagreement on 10-level
  AMZN, matching its own 95.8% on 10-level AAPL). Edge therefore reads the message +
  orderbook CSV pair directly (numpy/pyarrow), freezes the schema in
  `docs/DATA_CONTRACT.md`, and tests alignment locally via type-3 deletes, replay-based
  validation is structurally impossible on this data. orderflow-lm remains the convention
  reference, not an import.
- 2026-07-30, Alignment convention ("orderbook row k = state after message k") verified
  empirically before any code: 2000/2000 sampled type-3 deletes on each of INTC and AMZN
  satisfy the size-drop identity in the k−1->k direction, 0/2000 the inverted one. The
  lobsterdata.com DataStructure page is unreachable (SPA shell / dead subdomain; archive
  blocked), so the worked example there could not be quoted; orderflow-lm SPEC ("pre-event
  book = orderbook file row i−1") agrees.
- 2026-07-30, Tests use stdlib `unittest`, no pytest: `make test` must run from a fresh
  checkout with only the declared deps.
- 2026-07-30, Phase 0 step 1 (repo skeleton + failing tests) scoped to *not* implement
  `src/edge/io/lobster.py` yet: `RULES.md` rule 2 requires the adversarial test to exist
  before the code it constrains. `make test` is expected to be RED after this step
  (`ImportError: cannot import name 'lobster'`); implementation is a separate step that
  turns it green. Confirmed by running `make test` and pasting the failing output.
- 2026-07-30, `pyproject.toml` declares `numpy`/`pyarrow` deps for documentation and
  optional editable install, but `Makefile` does not require `pip install -e .`, both
  `make data` and `make test` put `src/` on `PYTHONPATH` directly, so a fresh checkout with
  only the two pip deps installed needs no install step.
- 2026-07-30, Output event-frame column order is block-grouped by side/quantity
  (`bid_px[1..L], bid_sz[1..L], ask_px[1..L], ask_sz[1..L]`), per `PLAN.md`'s explicit
  schema, **not** the raw orderbook file's interleaved-per-level order
  (`ask_px,ask_sz,bid_px,bid_sz` repeated per level). The adapter reorders columns on the
  way out; `docs/DATA_CONTRACT.md` documents both orders so this isn't rediscovered later.
- 2026-07-30, `order_id` (raw message col 3) is read by the adapter for row-count/QC
  purposes but dropped from the output parquet schema, not in `PLAN.md`'s frame
  definition and not needed downstream; keeping it out avoids a schema readers have to
  remember to ignore.
- 2026-07-30, Sentinel book levels (`ask 9999999999` / `bid -9999999999`, size 0):
  grepped the full raw INTC and AMZN 10-level files, zero occurrences, the visible book is
  10-levels-deep on both sides all day in this sample. Documented in `DATA_CONTRACT.md` as
  an empirical finding for this sample, not a guaranteed absence; the adapter should still
  treat the sentinel defensively when it's eventually written.
- 2026-07-30, Price-scaling sanity check (median mid, `/10000`) run against the raw
  orderbook files directly (pyarrow, before the adapter exists): INTC median mid $27.02
  (range $26.62-$27.60), AMZN median mid $222.66 (range $220.52-$226.03), both inside the
  spec's [20,35] / [200,250] bounds, confirming `/10000` is the correct dollar conversion.
  Recorded in `docs/DATA_CONTRACT.md`; the adapter-level assertion of this is deferred to
  the data-quality-report step.
- 2026-07-30, `make test`/`make data` both green: `src/edge/io/lobster.py` implements
  `read_message_file`, `read_orderbook_file`, `build_event_frame`, and the CLI exactly per
  `docs/DATA_CONTRACT.md`'s Adapter API. All 12 tests pass, including
  `test_orderbook_row_is_state_after_message` and `test_inverted_direction_does_not_hold`
  run against the full raw INTC (624,040 rows) and AMZN (269,748 rows) files, not just the
  toy fixture. Frame row counts match the raw CSV line counts exactly for both tickers.
- 2026-07-30, `Makefile`'s `PYTHON` is now autodetected (`python3`, falling back to
  `/opt/anaconda3/bin/python3`, then `python3.12`) instead of hardcoded `python3`. Root
  cause: on this machine, bare `python3` resolves to Xcode's `/usr/bin/python3` (3.9, no
  pyarrow) ahead of the conda `base` env that actually has the declared deps (pyarrow
  23.0.1, numpy), macOS's `path_helper` (run from `/etc/zprofile` for login shells)
  re-prepends `/usr/local/bin:/usr/bin:/bin:...` ahead of whatever conda's `.zshrc` block
  added, in every shell mode tested (non-interactive, interactive, login+interactive).
  `/usr/bin/python3` is also below `pyproject.toml`'s declared `requires-python = ">=3.10"`
  regardless of pyarrow. The Makefile fix tries plain `python3` first (a no-op on any
  ordinary single-interpreter checkout) and only falls back if the import check fails, so
  it doesn't change behavior anywhere except this specific quirk. Verified: `make test` and
  `make data` both succeed as bare commands (no manual PATH override) after this change.
- 2026-07-30, Added `src/edge/io/quality.py` (`python3 -m edge.io.quality --frames --out
  --seed`), invoked as the second step of `make data`, to generate `reports/
  data_quality.md` from the built parquet frames (not by re-parsing the raw CSVs a second
  time). It hard-fails (`SystemExit`) if crossed-book count != 0 or median mid falls
  outside the ticker's expected dollar range, per `RULES.md`/`PLAN.md`, those are
  fix-the-parser signals, not report-around-them signals. No new dependency: pyarrow +
  numpy + stdlib only.
- 2026-07-30, Sentinel-safe "valid quote" mask (`ASK_SENTINEL_PRICE`/`BID_SENTINEL_PRICE`,
  now exported from `edge.io.lobster` and reused in `quality.py`): a best-bid/ask pair only
  counts toward the crossed-book check and the spread/depth intraday profile if neither
  side is the LOBSTER sentinel (`ask 9999999999` / `bid -9999999999`) with size 0. Zero
  such rows in this sample (confirmed in the report output), but the check is real code now
  instead of a documented assumption.
- 2026-07-30, Intraday profile bucketing: 30-minute wall-clock buckets (`ts_ns` divided
  into half-hour bins from midnight, labelled by clock time, not by an assumed 09:30
  session-start offset, so it doesn't depend on trusting the filename's start/end fields).
  Event-rate U-shape is checked with a coarse heuristic (first bucket and last bucket both
  above the profile's median bucket count). Result: INTC clean U-shape (**yes**); AMZN
  fails the heuristic (**no**), its close-of-day surge is sharp but its open (09:30-10:00,
  17,520 events) is not the profile's peak (10:00-11:00 buckets run higher). Not treated as
  a parse bug: event-type counts match `DATA_CONTRACT.md` exactly (cross-checked
  independently with `awk` directly on the raw CSV, bypassing the adapter entirely),
  crossed-book count is 0, and median mid is in-range for both tickers, the weaker AMZN
  open-of-day effect is recorded as a genuine finding about this name/day, to revisit if
  AMZN-specific features behave oddly in Phase 1.
- 2026-07-30, `docs/spec/PLAN.md` Phase 0 checkboxes ticked, status moved to Phase 1, next
  three tasks set to `FEATURES.md` implementation / two-clock construction / rolling-only
  normalization (the first three Phase 1 bullets), per the Phase-0 exit criteria in the
  task spec.
- 2026-07-30, U-shape heuristic in `quality.py` replaced: was "first bucket AND last
  bucket both above the profile median", now "mean event rate over the first and last
  quartile of bins exceeds the mean over the interior bins". The old rule turned on two
  single half-hour buckets, so one noisy bucket flipped the verdict, which is exactly what
  happened to AMZN (open 09:30-10:00 is elevated but not the day's peak, so the old rule
  said no while the profile is visibly U-shaped). Under the quartile rule both INTC and
  AMZN report **yes**. This supersedes the "AMZN fails the heuristic" line above; the
  underlying observation (AMZN's open-of-day surge is weaker than INTC's) still stands and
  is still visible in the committed profile table. No parser change, event-type counts,
  crossed-book count and median mid were already all passing.
- 2026-07-30, `quality.py` now also takes `--raw` and re-counts the raw message/orderbook
  CSV lines, making "raw message rows == raw orderbook rows == parquet rows" a hard
  invariant of `make data` rather than something only the adapter checked internally. Cheap
  (streamed line count, no second parse) and it catches a truncated or half-written parquet,
  which the previous report could not.
- 2026-07-30, Added `tests/test_quality.py` (16 hand-computed tests over the pure check
  functions: crossed/locked counting, sentinel rows, ts monotonicity and tie counting, gap
  ranking, price scaling, intraday binning, U-shape rule). `quality.py`'s docstring already
  claimed these were unit tested and they were not; suite is now 28 tests, all green.
- 2026-07-30, Stale test `tests/test_clocks.py::test_frozen_per_ticker_n` asserted the old
  per-ticker `EVENT_CLOCK_N = {"INTC": 3, "AMZN": 1}` against the frozen scalar `10`, so the
  suite was red (67/68). Rewritten as `test_frozen_single_n_for_both_tickers`, asserting the
  scalar and asserting it is *not* a dict, so a silent revert to a per-ticker calibration
  fails a test rather than passing quietly. Suite green at 68.
- 2026-07-30, Phase 2 was invoked with Phase 1 only half-built. The infra layer exists and
  is good (`frame.py`, `book.py`, `clocks.py`, `windows.py`, `normalize.py`) but none of the
  six family modules named in `src/edge/features/__init__.py`'s `FAMILY_PREFIXES` (ofi,
  queue, microprice, trade, state, labels) had been written, there is no `features.parquet`,
  and `docs/INTERFACE.md`, which four module docstrings already cite, does not exist. The
  L0-L4 ladder is defined in terms of those features, so Phase 2's deliverable is
  unreachable without them. Decision: build the remaining Phase 1 families as a prerequisite
  block inside this run, scoped to exactly what `FEATURES.md` secs. 1-5 and 7 specify and no
  further, then run Phase 2 on the result. Not treated as scope creep, the ladder table is
  the Phase 2 milestone and features are its input, but Phase 1's own milestone (the
  feature/label correlation table) is produced too so the phases stay separately auditable.
- 2026-07-30, Fold geometry under a one-day universe. `EVAL_PROTOCOL.md` sec. 1 says split
  on day boundaries first and fall back to intraday blocks "if you need more folds"; the
  free sample is a single session (2012-06-21) per ticker, so intraday blocks are the only
  option, not a preference. Frozen rule, chosen a priori and not by looking at any fold:
  expanding walk-forward, initial train = the first third of the session
  (09:30:00-11:40:00), then **6 test blocks of 2600s** tiling 11:40:00-16:00:00 exactly.
  Consequence recorded up front: `EVAL_PROTOCOL.md` sec. 5 detector 4 (same-day vs next-day)
  **cannot be run** on this universe and will be reported as not-run rather than as passing.
- 2026-07-30, Purge/embargo parameters frozen from the protocol's own defaults, not tuned:
  purge = h at both edges of every test block; embargo E = max(60s, 10*h), giving E = 100s
  at the 10s horizon and 60s at 1s/2s/5s. E-sensitivity gets its own table rather than a
  choice. These go in the RESULTS.md frozen-decisions table.
- 2026-07-30, GBM library: **LightGBM 4.7.0** (the one GBM permitted by RULES.md's
  dependency rule; xgboost/catboost are absent from the environment anyway). `scipy` is
  present in the environment but is **not** on the permitted list, so the harness imports it
  nowhere, Spearman IC, AUC, MCC, ridge, logistic and OLS are all hand-written on numpy.
  `sklearn` is not installed at all, which makes the "no `sklearn.model_selection`" rule
  structurally enforced rather than merely observed.
- 2026-07-31, Phase 0 closed out. All four phase-0 red-team reports returned SURVIVED WITH
  CAVEATS (none KILLED), and the fixes they prompted were verified before committing: 80/80
  tests in the working tree, `make data` exit 0 now reporting `align_hold_rate=1.0000
  (n_checked=4000)` on both tickers, and a fresh clone running 40 tests where it previously ran
  28. Committed as `1022005`, `09e2161`, `3836cdc`. The red team's four "would change my mind"
  conditions are carried forward, in particular that a phase-1 number disagreeing with a
  re-derivation from raw promotes findings 1/2/11 to blockers.
- 2026-07-31, **Project reframed.** The question is now *how much of published short-horizon
  LOB performance is an artifact of leaky cross-validation*, measured by scoring the whole L0-L4
  ladder under three protocols (A shuffled k-fold, B unpurged walk-forward, C purged
  walk-forward with embargo) and reporting the A-C and B-C gaps. The LM arm is cut:
  `orderflow-lm` stops at tokenization and no trained checkpoint exists, so there is nothing to
  probe. Phases renumbered 0-4 (the cost model moves 4->3, the writeup 5->4), and the leakage
  gate written for the absent model is dropped with it. `docs/spec/` is unchanged and still
  authoritative except where it refers to LM representations or phase-3 controls. L1 is now
  load-bearing: an unfitted sign predictor has no parameters, so a near-zero A-C gap on L1 is
  what separates a measurement of optimism from an artifact of the harness.
- 2026-07-31, Phase 1 closed out. The six family modules, both clocks, the labels and the
  correlation table already existed from an earlier session but were recorded as "mid-flight and
  unverified"; this pass verified them rather than rebuilding them. Verification was a
  family-by-family audit in which every candidate finding was then re-checked against a
  deliberate attempt to refute it. 30 candidates, 12 confirmed, 18 refuted, and the refutations
  included several plausible-sounding ones (the `_5s` trade columns are covered by three
  value-asserting tests despite the string never appearing; `st_rv`'s sqrt is realized
  VOLATILITY as FEATURES.md sec. 5 names it, not a spec deviation; the joint per-level validity
  mask in `queue.py` is documented and deliberate), so the confirmed set is what got acted on
  and nothing else.
- 2026-07-31, `mp_micro` was Stoikov's FIRST-ORDER term, not the microprice; fixed. The
  module built only the transient-transient matrix Q (naming it `B`), so
  `x = (I-Q)^-1 R K = G^1`, the expected mid move up to the NEXT price change. Stoikov's
  actual estimator needs the landing matrix `B = (I-Q)^-1 R2` and the outer recursion
  `G^(i+1) = G^1 + B G^i`, which was never built or run. This is exactly the failure
  FEATURES.md sec. 3 warns about ("not as a synonym for weighted mid", G^1 is a subtler
  version of the same mistake, a real quantity that discards every price change after the
  first). `_estimate_block_adjustment` now accumulates R2 counts alongside Q, and
  `_micro_adjustment` runs the recursion. `I - B` is SINGULAR by construction (B is
  row-stochastic on visited states), so the limit cannot be reached by a direct solve; the
  recursion depth is frozen at `MICRO_ITERS = 6`, Stoikov's own repo value, with a TOL early
  exit. Pinned by a new hand-solved test on a convergent fixture where G^1 = 100, the limit
  is 200 and the frozen-depth answer is 100*(2 - 2^-6) = 198.4375, all three distinguishable,
  so neither the old bug nor a changed iteration count can pass silently.
- 2026-07-31, `mp_micro` moved from raw LOBSTER price units to ticks. It is a price LEVEL and
  INTERFACE.md sec. 1 puts every emitted price level and price difference in ticks; sec. 0
  makes INTERFACE.md normative over the module docstring that had declared raw. Internal
  arithmetic still runs in raw integers and converts exactly once at the return.
- 2026-07-31, `tr_last_trade_sign` no longer inherits `tr_eff_spread`'s `isfinite(mid_before)`
  mask. The aggressor sign is `-direction[trade_pos]`, read from the message file with no book
  dependence, so masking it on an invalid preceding touch deleted a perfectly known value. The
  `has_prev` term stays on both, that one is the documented "trade is the very first frame
  row" rule, not a book-validity question.
- 2026-07-31, Four OFI mutations and two state/queue mutations that survived the ENTIRE
  229-test suite are now caught. Root cause in every OFI case was fixture shape, not missing
  assertions: the only multi-block fixture was ONE LEVEL deep, where the first PC of a single
  column is +-1 regardless of what rows it was fitted on, so a full-sample (leaking) PCA fit,
  an `eigvecs[-1, :]` orientation slip and a reversed level index were all arithmetically
  invisible; and the ask-mirror test used a TIED ask price, where swapping `>=` for `<=` is a
  no-op. Added: a two-level multi-block fixture whose covariance structure differs either side
  of the block boundary (so a leaking fit visibly rotates the first PC), a three-level
  hand-derived PCA test (row-vs-column only separate at L >= 3), an ask-side test on a MOVING
  ask, a time-of-day test on stamps deep into the session (every toy stamp sits in bucket 0,
  so `st_tod_bucket` could be hardcoded to zero undetected), and a ten-level queue fixture
  covering `qi_l3`..`qi_l10` and the weight tails. Each fix was verified by re-running the
  mutation and confirming it now fails. The leaking-PCA case is the one that mattered most:
  this project's entire subject is leakage, and its own harness could not detect a leaking fit.
- 2026-07-31, `label_y_wmid_*` / `label_c_wmid_*` stand in for FEATURES.md sec. 7's
  `y'_h = micro(t+h) - micro(t)`, recorded here because the rationale previously lived ONLY in
  the labels.py docstring and RULES.md requires every unasked choice in this file. Two
  reasons, both structural: INTERFACE.md sec. 5 rule 4 forbids labels.py importing
  microprice.py, and the Stoikov microprice is a FITTED quantity on a refit schedule, making
  the label depend on a model's refit cadence (and inheriting its block-0 NaN) is a far bigger
  commitment than "lower-noise alternative" implies. `book.wmid` is exact and unfitted. This
  is a substitution, not a synonym. Not revisited now that the microprice is correct: a label
  that inherits a fitted quantity's warmup is still the wrong shape for Phase 2.
- 2026-07-31, Deadband alpha frozen at 0.5 ticks BY RULE and given its RESULTS.md
  frozen-decisions row (RULES.md requires frozen choices in that table, and the row was still
  blank). Half a tick is FEATURES.md sec. 7's own first suggestion and the smallest move that
  is not a rounding artifact in a whole-cent book. Not fit to data, training or otherwise; the
  sensitivity sweep in reports/feature_correlations.md is reported, never used to pick it.
- 2026-07-31, Red-team pass on phase-1 lookahead/normalization returned
  `VERDICT: SURVIVED WITH CAVEATS` (`reports/adversary/phase1-lookahead-normalization-2026-07-31.md`).
  The flagged feature/label correlations are genuine microstructure, not leakage, established
  six independent executable ways: an independent re-derivation of stamps/OFI/qi/label from the
  raw frame matching the pipeline to 0.0; scrambling every price and size from row 300,000
  onward moving 0 of 229 non-label columns while all 28 label columns moved with row counts
  exactly matching their horizons; forward-shifted (deliberately leaky) features scoring
  strictly HIGHER than shipped (0.374 -> 0.542 at +10) where a pipeline that had already
  leaked could not improve; contemporaneous rho (0.710) exceeding forward rho (0.374), the
  opposite of a boundary leak; the clock gap vanishing once tie mass is conditioned out
  (qi_touch 0.616 calendar vs 0.603 event); and label-shuffle collapsing everything to
  |rho| < 0.01. Phase 1 stands.
- 2026-07-31, F1 fixed: the event clock violated INTERFACE.md sec. 3. `event_stamps` set
  `index[i]` to the TRIGGERING event's own row, which satisfies `ts <= t` only by equality;
  the contract is the LAST row with `ts <= t`. LOBSTER shares nanoseconds across messages, so
  the two differed on 10.48% of INTC event stamps (up to 97 rows skipped). Harmless for
  features (they simply read fewer rows than entitled) but NOT for labels: labels.py anchors
  on this index while resolving its forward endpoint with the correct rule, so
  `label_y_mid_*` on the event clock contained a mid move that elapsed in ZERO wall-clock
  time and was already public at t. Measured inflation +0.021 on `qi_touch`/`qi_l1` (5.5%
  relative), +0.0005 on OFI. It would have appeared identically under protocols A, B and C,
  so it would not have biased the optimism decomposition, but it is a definitional leak into
  a label and it was unasserted, because `tests/test_clocks.py`'s contract test ran on the
  calendar clock only and the toy fixture contains no timestamp tie. Fixed with the same
  `searchsorted(..., side="right") - 1` the calendar clock uses, plus the missing event-clock
  test on a purpose-built tied-timestamp fixture. Event-clock columns change; rebuilt.
- 2026-07-31, F2 recorded, not "fixed": the microprice recursion does NOT converge on this
  data. Measured `|eig(B)| = {1.0, 0.9906, 0.272}` with stationary drift `pi . G^1 = -0.018`
  ticks per iteration, so `G^i` has a persistent linear term and the limit does not exist -
  the increments plateau rather than decay. My own docstring claimed convergence via the
  martingale property; that claim was false and now says so. `MICRO_ITERS = 6` is therefore
  part of the feature's DEFINITION rather than a cap convergence reaches first, and `G^7`
  would be a materially different column. It stays at 6 (Stoikov's repo value, frozen a
  priori, never tuned against a correlation) and `mp_micro` is honestly "the six-step
  microprice". Consequence for Phase 2: on AMZN the truncation puts `mp_micro` outside
  [bid, ask] on 0.32% of finite stamps with `|mp_dev_mid|` reaching 15.8 ticks (p50 0.39,
  p99 0.70); INTC is clean at 0.00%. The tail is meaningless, not leaky, block-refit
  causality is intact, and Phase 2 must winsorise or exclude it with the choice logged.
- 2026-07-31, F3 fixed, and the reasoning error behind it recorded. `correlations.main`
  printed `suspicious[:20]` of a 488-item list, and the phase-1 triage was read off that
  console tail, concluding "19 leak smells, all INTC, AMZN clean". False: 446 leak smells
  (282 INTC, 164 AMZN), and the largest single correlation in all of phase 1 is AMZN/event
  `tr_aggr_imb_100ms` at rho = +0.5307, not anything on INTC. That wrong premise was then
  handed to the red team as a framing. It passes the same causality probes (fwd+1 0.6293,
  bwd-1 0.4538, 81.5% balanced directional accuracy on non-flat labels) so nothing is
  invalidated, but the triage was worthless. The console block now prints a count by scope
  over the WHOLE list and says explicitly that it is not the list. General rule taken from
  this: a summary printed from a truncated slice is worse than no summary.
- 2026-07-31, F4 carried into Phase 2 as a reporting constraint: the 1s label is measuring a
  <=100ms effect. `rho(ofi_best_100ms, mid(t+1s) - mid(t+100ms)) = 0.027` against 0.458 for
  the first 100ms on INTC/event. A `label_y_mid_1s` result must not be presented as one-second
  alpha, and Phase 3's cost model has to price a signal whose content decays inside 100 ms.
- 2026-07-31, F6/F7 carried into Phase 2: `tr_aggr_imb_*` is NaN when its window holds no
  trades (finite on 1.8% of AMZN calendar stamps vs ~100% for `qi_touch`), so per-fold row
  counts will differ by more than an order of magnitude across rungs and must be REPORTED per
  fold, not absorbed. And the trial log is empty against a 7,264-cell univariate surface;
  no model has been fitted yet, but the surface size belongs in the log before the deflated
  Sharpe is computed.
- 2026-07-31, The red team's warning on Phase 2 fold construction, recorded now so it is not
  rediscovered later: 32% of INTC event-clock stamps sit within 0.5ms of the previous stamp
  and carry rho 0.437 against 0.19 for isolated stamps. If Phase 2 cuts folds on STAMP INDEX
  rather than wall clock, a single burst can straddle a boundary and both folds will contain
  the same episode, which would read as optimism and would not be. Folds get cut on time.

## Phase 2, the three-protocol harness

- 2026-07-31, Folds are cut on WALL-CLOCK TIME, not stamp index, and the universe is one
  trading day so EVAL_PROTOCOL.md sec. 1's "split on day boundaries first" is impossible;
  folds are intraday blocks. Time-cutting is the phase-1 red team's explicit warning: 32% of
  INTC event-clock stamps sit within 0.5ms of the previous one and carry ~2.3x the
  correlation of isolated stamps, so an index cut lets one burst land on both sides of a
  boundary and both folds then hold the same episode, which reads as optimism and is really
  duplication. The cost is that per-fold sample counts vary; EVAL_PROTOCOL.md sec. 1 requires
  reporting the per-fold distribution anyway. `same-day vs next-day` (sec. 5 check 4) cannot
  be run at all on one day and is recorded as unavailable, not as passed.
- 2026-07-31, Purge and embargo compose ADDITIVELY at the left edge: a training sample is
  kept iff `t + horizon + embargo < test_block_start`. Purge removes label overlap, embargo
  removes an additional serial-correlation window; they answer two different questions and
  the conservative composition is the defensible one. The inequality is strict so a label
  ending exactly on the boundary is still dropped. A right-edge embargo is implemented but
  cannot bind under expanding walk-forward (all training precedes the test block); it exists
  so the function stays correct if combinatorial purged CV is added later.
- 2026-07-31, All three protocols are SCORED ON THE SAME ROWS. First cut of the harness let
  protocol A k-fold over every sample while B and C reserved segment 0 as initial training
  data, so A was additionally tested on the market open, the most volatile and most
  predictable part of the session, and that difference would have been charged to leakage.
  A now partitions exactly the universe B and C test, while still TRAINING on everything
  outside its test fold including segment 0, because training on all available data is what
  shuffled k-fold actually does. Caught by reading per-fold `n_test` in the first smoke run
  (A 46,799 vs B/C 39,000), not by a test, the test that asserts it now exists.
- 2026-07-31, Every rung is scored on a COMMON usable row mask rather than its own. The
  ladder's premise is that each rung beats the one below it, which means nothing if the rungs
  see different samples. First smoke run had L1/L2 on 39,000 test rows and L3/L4 on 4,576 -
  an 8.5x difference, so the "ladder" was two different experiments. Dominant remaining cost
  is the block-0 warmup on every fitted column (7.69%, exactly the first refit block: the
  honest price of a causal fit), which is reported rather than hidden.
- 2026-07-31, `tr_aggr_imb_1s` dropped from the L3/L4 feature set. It is a RATIO and is NaN
  whenever its window holds no trades: 84.7% of INTC calendar stamps, 86.5% of AMZN's (the
  phase-1 red team's F6, measured). Keeping it would have cost 87% of the sample for one
  column. `tr_signed_vol_1s` and `tr_count_1s` are SUMS, legitimately 0.0 on a quiet window
  (INTERFACE.md sec. 7), and carry the same directional information unnormalised. Chosen on
  the NaN structure of the columns before any fold was scored, not because it improved a
  number.
- 2026-07-31, Every rung's raw score is mapped into label units by ONE through-origin scale
  fitted on TRAINING data only, applied uniformly to all rungs. Without it L1 emits a
  direction in {-1,0,+1} and its R^2 would measure units rather than skill. For any
  least-squares rung the scale is exactly 1.0 (OLS residuals are orthogonal to fitted values),
  so it is a no-op for L2 and near-enough for ridge (measured 1.0031); it only puts L1 and L4
  on a comparable footing, and being rank-preserving it never changes the IC.
- 2026-07-31, GBM hyperparameters frozen a priori (200 rounds, lr 0.05, 31 leaves, min 200
  per leaf, L2 1.0, seeded, single-threaded for determinism) with NO early stopping. Early
  stopping on a slice of the test fold is itself a leak, and carving a validation slice out of
  training is one more free parameter to log; a fixed round count is the honest choice and it
  is the same count under all three protocols.

## Phase 3, costs, fills, PnL

- 2026-07-31, Trading direction is the SIGN of the prediction, not a deadband on its
  magnitude. The first cut reused the labels' 0.5-tick deadband as a trading threshold and
  L0/L1/L1m/L2 then took ZERO trades across the entire session: a scaled sign predictor never
  emits a magnitude above half a tick at a 1s horizon where 96% of realised moves are zero.
  That left the break-even cost table, the phase's headline number, empty for exactly the
  rungs it most needs to price, including L1, the baseline the whole project is trying to
  beat. A deadband is a position-sizing choice, not part of the signal; it remains available
  as a sensitivity and defaults to 0.
- 2026-07-31, Decision points are spaced at least `latency + horizon` apart, greedily from
  the earliest eligible stamp. EVAL_PROTOCOL.md sec. 3 mandates "single unit, no inventory
  accumulation, flat at horizon", and taking a decision at every stamp violates that outright:
  on the 100ms calendar grid with a 1s horizon, ten positions would be open at once and the
  reported PnL would be a leveraged overlapping sum rather than the boring single-unit
  strategy the protocol specifies. This also makes the trade count an honest turnover measure.
- 2026-07-31, PnL is computed under PROTOCOL C ONLY. A PnL number under a leaky protocol is
  not a weaker result, it is a misleading one, and phase 2 exists to establish which protocol
  to believe. The A/B columns stay in the ladder table, where the gap is the subject.
- 2026-07-31, The 2012 Nasdaq take/make rates ($0.0030 / $0.0020 per share) are entered as an
  ASSUMPTION and have NOT been verified against an archived price list; the exact tier depends
  on a member firm's monthly volume, which no public dataset pins down for an anonymous
  participant. Flagged in the module, the report header and METHOD.md rather than presented as
  a citation. The break-even cost table is what makes this survivable: it reports the cost at
  which PnL crosses zero, so a reader who disagrees reads their own answer off it.
- 2026-07-31, The classification columns in the ladder table are largely uninformative and
  are labelled as such rather than quietly reported. Predictions are classified with the same
  0.5-tick deadband as the labels, so a scaled sign predictor lands entirely in the flat class:
  L1 shows balanced accuracy of exactly 1/3 with MCC undefined (a constant predictor has no
  MCC). The regression metrics (R^2 vs zero, and IC) are the informative ones at these
  horizons; fixing the classification treatment properly means a quantile-based threshold and
  is left as future work rather than bolted on late.

## Phase 3 rework after the 2026-08-02 red team

- 2026-08-02, **The headline break-even column is now computed on the GROSS series, not the
  net one.** The old column was incremental to a spread already crossed and a fee already
  paid, which made it very nearly model-invariant: the red team showed a constant always-long
  predictor scores -0.8072 ticks on INTC calendar 1s against a whole-ladder range of
  [-0.8025, -0.7702], i.e. *inside* it, and that perfect foresight of the 1s mid sign still
  loses $456. A column a constant cannot be distinguished from cannot answer PLAN.md's
  question. `break_even_total_ticks` reports the total per-side cost the signal can bear, so a
  reader who rejects the assumed fee schedule can substitute their own, which is what the
  break-even table was always supposed to allow. The incremental column is RETAINED and
  demoted rather than deleted: removing the number an earlier draft led with would hide the
  correction instead of recording it.
- 2026-08-02, **The constant and oracle controls ship as rows in the table.** always-long,
  always-short, and perfect foresight run on the same rows, folds, mask and fills as the
  ladder. The red team had to compute them to find the model-invariance; a reader should not
  have to. The oracle is labelled a BOUND, not a rung, it reads the future by construction
  and its feature list says so.
- 2026-08-02, **The deflated Sharpe is reported on the gross series.** On the net series
  every rung has a per-trade Sharpe near -7.6 (it loses the spread almost deterministically)
  and the DSR is 0.000 at every trial count from 1 to 1e12. That is a tautology, not a
  finding. Deflation only means something where there is something to deflate; the net DSR is
  kept in the row dict but is not the reported column.
- 2026-08-02, Trial count for the DSR set to **984** (SUPERSEDED the same day by the
  phase-2 rework below, which raised it to 2,056): 288 phase-2 ladder fits + 480
  embargo-sweep fits (5 values x 16 cells x 6 rungs) + 216 phase-3 cost configurations
  (2 tickers x 2 clocks x 2 horizons x 9 predictors x 3 latencies). The previous run passed
  `--trials 288`, which is phase 2's count and excludes phase 3's own trials entirely.
  The arguable component is the embargo sweep: it is a pre-registered sensitivity whose value
  was frozen by rule and which never selected anything, so a case exists for excluding it the
  way trials 1-4 are excluded. It is counted anyway. Undercounting is the failure mode that
  makes a deflated Sharpe dishonest; overcounting only makes the threshold harder to clear,
  and the direction of the error should favour the reader, not the result.
- 2026-08-02, `_non_overlapping` carries its watermark ACROSS fold boundaries. Resetting it
  per fold allowed a position opened at the end of one test block to still be open when the
  next block opened one: 4 gaps of 800-1000ms against a required 1001ms on INTC calendar 1s.
  Immaterial to the number (4/17,042) but it falsified the module's own stated invariant, and
  the test that would have caught it did not cover the boundary. It does now, together with
  its own mutation test.
- 2026-08-02, Diebold-Mariano between adjacent rungs is wired to the report. The function was
  implemented, unit-tested, imported into `costs/run.py` and never called, so an
  `EVAL_PROTOCOL.md` sec. 2 deliverable existed as dead code. Loss is `-gross ticks` per trade
  via a new `loss="linear"` mode: squaring a PnL would test whether one rung's PnL is larger
  in MAGNITUDE, which is the wrong hypothesis. Nestedness is flagged in the report, since
  standard DM is undersized there and Giacomini-White is not implemented.
- 2026-08-02, The bps denominator is the mean ask over TRADED rows, not over the whole
  session. Segment 0 is never tested and does not belong in the average price. Effect is
  <0.5%; it is a correctness fix, not a result change.
- 2026-08-02, The report header now echoes the FULL effective argument set. The old header
  printed `--seed 0 --trials 288` while `--clocks` defaults to `calendar`, so the command the
  table advertised rebuilt half the table, a direct violation of RULES.md rule 7.
- 2026-08-02, The 0-tick deadband is disclosed in the report as a choice informed by
  protocol-C output rather than presented as frozen. The reasoning is a-priori (a scaled sign
  predictor cannot emit a magnitude above half a tick at these horizons) but the change was
  triggered by seeing an empty table, which is RULES.md rule 5 territory. Disclosed, not
  reverted: reverting would restore a table with no rows for the baseline the project exists
  to test.

## Phase 2 rework after the 2026-08-02 red team

- 2026-08-02, **The through-origin rescale is now fitted on a held-out slice of the training
  fold, not in-sample.** This is the largest correction in the project. The old scale solved
  `sum(s*y)/sum(s*s)` on the same rows the model was fitted on; for a high-capacity rung whose
  in-sample predictions are near-perfect, that returns essentially no shrinkage, so the
  out-of-sample amplitude is far too large and R2-vs-zero explodes negative. Every large
  negative protocol-C R2 in the earlier ladder was this artifact, NOT a finding: INTC calendar
  1s L4 goes from C = -0.2988 to C = +0.0213, and its A-C gap from 0.4898 to 0.1618. The
  artifact never bit under protocol A, because A's "out-of-sample" rows are the temporal
  neighbours of its training rows, so it inflated the *measured optimism*, not just the
  model. Holdout is the last 20% of the training rows in time; under A, where training rows
  are not time-ordered, a seeded random 20%, which is A's own logic applied to A.
- 2026-08-02, **Protocol D added as a control**: C's test blocks, trained on everything
  outside them, past AND future, no purge. It holds the test universe, fold shapes and
  aggregation fixed and varies only "may the model see the future". It exists because the
  A-B / B-C decomposition attributed the wrong mechanisms, see the next entry. D is NOT a
  fourth protocol under study; `HEADLINE_PROTOCOLS` stays (A, B, C).
- 2026-08-02, **`B - C` is no longer described as "the label-overlap component".** The claim
  was wrong and the earlier README drew a reviewer-facing conclusion from it ("purging is not
  where the money is"). Under expanding walk-forward, purge+embargo can only remove the last
  `h + E` of a multi-hour training set: 578 of 90,333 rows (0.64%) at INTC calendar 1s. A
  column that can touch 1% of the training set is not a measurement of label overlap in
  general, only of label overlap at a walk-forward boundary, where it is negligible by
  construction. Measured with D: training on the future in disjoint blocks is worth ~0.01-0.05
  IC, while temporal adjacency of training rows is worth 0.11-0.36. Label overlap is large and
  it lives inside `A - B`.
- 2026-08-02, **The frozen-predictor artifact floor ships as its own table.** L1 was supposed
  to be the control that made the decomposition readable, and it is not sufficient: its
  "one-parameter rescale" is fitted, so its gap mixes a partition artifact with a refit
  effect. The real control is a predictor fitted ONCE on the whole day and scored unchanged
  under every protocol, where any difference is 100% partition artifact. It is not zero
  (A-C = +0.0014 to +0.0271), because A's fold mean is the sum(y^2)-weighted average of the
  day while C's is unweighted, two different estimands. The frozen gap reproduces the old L1
  IC row to five decimals, which is what proves the L1 row contained no leakage signal at all.
- 2026-08-02, The headline table now carries `A-C med` (gap from per-fold medians) and
  `A-C ex0` (fold 0 dropped) beside the mean. 45 of 80 rows in the earlier draft lost more
  than half their gap when fold 0 was dropped, and the mean is the number that gets quoted.
- 2026-08-02, The superseded in-sample-calibration columns are retained behind
  `--in-sample-scale` and printed in their own section, so the size of the correction is
  visible rather than asserted. Deleting them would hide what changed.
- 2026-08-02, `ALPHA_TICKS` is imported from `edge.features.labels` rather than re-declared
  in `harness/run.py`. The copy could silently diverge from the labels' own deadband, which
  would classify predictions against a different threshold than the labels were built with.
- 2026-08-02, The ladder report header echoes the full effective argument set, and the
  embargo sweep sorts numerically rather than lexically (E was ordering 0, 1, 10, 300, 60).
  The prose claiming an `r2_test_mean` column was removed: no such column was ever emitted.
- 2026-08-02, **The artifact floor is measured at three rung magnitudes, not one.** The second
  red-team pass of the day showed the first version of this fix was itself insufficient: the
  frozen floor measured with `sign(queue imbalance)` is +0.00136, but the identical
  construction at L4's magnitude is +0.01615, an order of magnitude larger, so quoting the
  L1 floor beside an L4 gap understates what that gap has to clear. The floor is an
  interaction between the fold-aggregation asymmetry and the predictor's own variance, so it
  scales with the predictor. `edge.harness.floors` is a separate entry point because a floor
  costs one fit per (cell, rung) while the ladder costs five per protocol.
- 2026-08-02, **The "ladder inverts under protocol C" claim is withdrawn.** It was an artifact
  of the in-sample calibration and did not survive the fix: under the causal scale L4 beats L1
  in 13 of 16 cells, and the median protocol-C R2 goes L1 +0.0091, L3 +0.0266, L4 +0.0273. Two
  metrics already printed in the same report said so and were not read, under C the median IC
  runs L3 0.2364 > L4 0.1839 > L1 0.1465, and the median balanced accuracy runs L4 0.3926 >
  L3 0.3760 > L1 0.3333, which is chance exactly. The corrected claim is narrower: optimism
  still scales with capacity, and the honest protocol-C performance of the fitted rungs is
  small but positive.

## Closing out, the last unmet spec item and the last unactioned red-team instruction

- 2026-08-03, **The calibration slope is reported.** `EVAL_PROTOCOL.md` sec. 2 asks for it by
  name ("slope of realized on predicted, should be ~1; if it's 0.2 your model is confidently
  wrong at scale"); `harness/metrics.py::calibration_slope` was implemented and unit-tested on
  2026-07-31 and then never wired into a report, which `FUTURE.md` sec. 3 recorded as an unmet
  spec item rather than a closed one. It is now a per-fold metric in `score_fold`, summarised
  across folds like every other, and printed with mean/median/worst beside the IC. The cost was
  one full ladder re-run for two columns. It is worth it here specifically: this project spent
  a day on a calibration artifact that a reported slope would have exposed immediately, since
  the in-sample rescale's whole failure mode is an out-of-sample amplitude that is far too
  large, and no metric in the old report measured amplitude.
- 2026-08-03, **AUC is computed on the directional subset, flats excluded.** `EVAL_PROTOCOL.md`
  sec. 2 asks for AUC next to the base rate and does not say against which binary. The choice is
  a-priori and it is the strict one: score up vs down on the rows the 0.5-tick deadband calls
  directional, never up vs {flat, down} pooled. Pooling would make AUC a function of tie mass -
  96.3% of INTC calendar 1s labels are exactly flat, and reporting a number on that axis is the
  exact error the phase-1 red team caught in the correlation table. Under this definition 0.5 is
  chance in every cell regardless of how flat the cell is. A one-sided test block returns NaN
  rather than 0.5, so an undefined cell cannot enter a mean as a chance-level row.
  `tests/test_score_fold.py` constructs a case where the pooled and directional definitions land
  on opposite sides of chance (0.75 vs 0.00), so a regression to the pooled form fails loudly
  rather than shifting a decimal.
- 2026-08-03, AUC earns its place beside balanced accuracy for a reason the report now states:
  balanced accuracy and MCC are NULL for the four unfitted rungs. Their rescaled amplitude never
  crosses the deadband, so the predicted class is always 0, balanced accuracy is exactly 1/3 and
  MCC is NaN. AUC ranks the raw prediction and never thresholds it, so it is the only
  classification column carrying directional information for L0/L1/L1m/L2.
- 2026-08-03, **The "L1 is the control" reading note is deleted from the ladder report.** It
  was the one instruction from the second phase-2 red team (`VERDICT: KILLED`) that was never
  actioned: the science was corrected and `FINDINGS.md` sec. 1.1 was rewritten to "why L1 was
  not it", but `reports/ladder.md`'s own reading notes still told a reader to treat L1's gap as
  the floor. Replaced with a pointer to the `floor` column and the two measured magnitudes
  (+0.0014 at L1's, +0.0161 at L4's). A retracted claim left standing in a generated artifact is
  worse than one never made, because the artifact looks freshly produced.
- 2026-08-03, `make ladder`, `make floors` and `make costs` added. The three phase-2/3 reports
  were the only tables in the project reproduced by a bare module invocation rather than a make
  target, and `TRIALS` is a variable rather than a default buried in `costs/run.py` because
  passing a smaller count silently undeflates the Sharpe.
- 2026-08-03, **The re-run adds 0 to the trial count.** The regenerated ladder is the identical
  configuration at the identical seed, same folds, same rungs, same hyperparameters, with two
  additional diagnostics computed from the same predictions. The R2 and IC columns reproduce the
  2026-08-02 table exactly, which is the check that the two new metrics changed nothing else. No
  new configuration was searched, so counting it as 384 new trials would inflate the denominator
  with fits that explored nothing. Logged explicitly at +0 rather than left silent, because an
  unexplained gap between the run log and the trial count is how a trial count stops being
  trustworthy.

## Phase 2 rework after the 2026-08-03 red team (`VERDICT: KILLED`, third pass)

- 2026-08-03, **The calibration-slope leakage reading is withdrawn, same day it was
  published.** The claim was that protocol A's near-1.0 slopes are a leakage signature ("a
  model is calibrated on rows it has effectively seen"). It is an artifact of the harness's own
  calibration design. `score_fold` picks the holdout BY PROTOCOL: a random 20% of the training
  rows under A, the last 20% in time under B/C/D. Under A that holdout is *exchangeable with A's
  random test fold*, so the reported slope is driven toward 1.0 mechanically, for any predictor,
  with or without skill. Two controls settle it: a frozen whole-day model with only the scale
  refitted reproduces A = 1.0319 / C = 1.3905 at L1's magnitude against the published
  1.0319 / 1.3904, and holding the calibration rule constant REVERSES the claim for L4 (median
  |slope-1| 0.018 -> 1.034 under A, against C's 0.269). The slope is now reported as the
  spec-mandated diagnostic with an explicit do-not-read-this-as-leakage note, and nothing in the
  study rests on it. The previous pass had flagged this asymmetry as newly introduced by its own
  fix; it was promoted to a finding instead of controlled for, which is the specific failure.
- 2026-08-03, **The artifact floor now refits the scale per fold under the protocol's own
  rule.** The old floor froze the scale along with the model, so it could not see the channel
  above, a control that was vacuous by construction, published as a control. This is the same
  class of error the 2026-08-02 pass killed the report over, one level down. Like-for-like
  floors are 2-6x the published ones: R2 +0.0043 (L1 magnitude), +0.0112 (L3), +0.0192 (L4)
  against +0.0020 / +0.0020 / +0.0153. The frozen-scale column is retained beside the corrected
  one, because the pure partition artifact is a real and separate quantity and the difference
  between the two columns IS the calibration channel.
- 2026-08-03, **`--holdout-control` added: protocol A re-run under C's positional holdout.**
  Same split, folds, seed and model; only the calibration rule moves. This is the decomposition
  that has to sit beside the headline, because roughly half of L4's R2 gap lives in this channel
  rather than in the split. It is a legitimate part of what "shuffled k-fold" means, a
  practitioner shuffling really does calibrate on exchangeable rows, but it is a different
  mechanism from seeing the future and the report was not separating them.
- 2026-08-03, **AUC is now reported on the RAW score as well, and the raw column is the one
  quoted.** The through-origin rescale is rank-preserving only UP TO SIGN, and the fitted scale
  goes negative in 27 of 400 protocol-C folds and 0 of 400 protocol-A folds. A negative scale
  sends AUC to 1-AUC. That channel alone manufactured the entire apparent A-C AUC gap on the
  unfitted rungs: on the raw score L1m's and L2's gaps are NEGATIVE (-0.0018, -0.0002) where the
  rescaled column reported +0.0035 and +0.0078. The capacity progression is now quoted on the
  raw score, where L3 (+0.0409) and L4 (+0.1740) are essentially unchanged and the unfitted
  rungs sit at zero, which is what the claim needed them to do and did not verify.
- 2026-08-03, **The claim "a rank statistic's artifact floor cannot scale" is withdrawn.** It
  was the stated *reason* the AUC restatement was worth more than the R2 one, and it is wrong.
  Protocol A's per-fold AUC is a whole-day POOLED AUC while C's is a WITHIN-BLOCK AUC; any
  frozen score with a time-varying offset makes those estimands diverge with nothing refitted,
  and the red team drove the floor to -0.137 that way. The AUC floor is small HERE because these
  frozen predictors happen to be temporally homogeneous. Reported as a measurement on this data,
  never as a property of the metric.
- 2026-08-03, The headline table's `floor` column is now matched to each rung's own magnitude
  (L0/L1/L1m/L2 against L1's, L3 against L3's, L4 against L4's). The 2026-08-02 pass established
  that one magnitude does not bound another and the prose was corrected then, but the GENERATED
  column kept printing the L1 floor beside every rung for another day. Prose is not the artifact.
- 2026-08-03, **`edge.costs.run` now uses the causal calibration too.** It had kept the
  in-sample rescale that was killed for the ladder on 2026-08-02, found by the third red team,
  outside its own scope. My first estimate of the impact was that the economics would be
  invariant, on the grounds that with a 0-tick trading deadband the traded direction is
  `sign(scale) * sign(raw)` and only the scale's sign can change a trade. **That was wrong, and
  re-running it rather than reasoning about it is what showed so**: `scale_holdout` also changes
  what the RAW MODEL is fitted on (80% of the training rows instead of 100%), so the fitted
  rungs' predictions move. L4's INTC calendar 1s break-even goes 0.0080 -> 0.0088 and L3's
  0.0098 -> 0.0100. A first version of this entry then claimed the unfitted rungs and controls
  were unchanged to four decimals and offered that as the check that nothing else moved; the
  fourth red team showed **18 of 72 unfitted rows moved**, because the causal scale can come
  out negative and with a 0-tick deadband that inverts an unfitted rung's direction outright.
  Every quoted cost number was updated from the regenerated report.
- 2026-08-03, The cost controls' `fit` now accepts and IGNORES `scale_holdout`. They override
  `fit` deliberately, a control whose rescale can flip its sign is not a control, and that
  override silently fell out of step with `Rung.fit` the moment the cost path started passing
  the keyword, killing a full cost run with a `TypeError` that no test covered.
  `tests/test_pnl.py::test_every_cost_predictor_accepts_the_harness_calling_convention` now
  walks every predictor the cost runner builds and asserts both halves: the call must not raise,
  and the controls' scale must still be exactly 1.0.
- 2026-08-03, **`.venv-edge` added and put first in the Makefile's interpreter probe.** The
  `/opt/anaconda3` install this project was developed against disappeared from the machine
  partway through the session, `make` would have found no candidate with `pyarrow` and every
  target would have failed. A repo whose reproduction story is "one command" cannot depend on an
  interpreter it does not control, so the venv is created from `/opt/homebrew/bin/python3` and
  probed first; the older candidates stay in the list and simply fall through when absent. The
  full suite passes on it unchanged (400 tests, 594 subtests) under Python 3.14 and numpy 2.5.1
  against the 3.12/numpy-1.x it was written on, which is a free clean-room check of the
  zero-dependency claim. `.venv-edge/` and the red team's `scratch/` are gitignored.

## Phase 2 after the 2026-08-03 fourth red team (`SURVIVED WITH CAVEATS`)

- 2026-08-03, **The artifact floor is now measured over four frozen-model fit windows and the
  WORST one is what margins are quoted against.** Fitting the frozen model on the whole day
  minimises the floor, and at L4's magnitude it does so mechanically: a whole-day GBM scores AUC
  ~0.996, a ceiling where two protocols have almost no room to differ, while the rung it is meant
  to bound scores ~0.72 under C. Refit on half the day and the raw-AUC floor goes from +0.0014 to
  +0.0198 median and +0.0383 worst cell. **The "98x its floor" margin published earlier the same
  day was that ceiling.** The real figure is ~7x median and 1.5x in the worst cell, still the
  claim, an order of magnitude less impressive than stated. This is the third distinct way this
  project's floor has been wrong, and the pattern each time is the same: a control that is not
  exposed to everything the measured thing is exposed to.
- 2026-08-03, **`A' - C` is presented as a bracket, not a share.** The control changes the
  calibration rule, but `Rung.fit` fits the model on the complement of the holdout, so it also
  changes which rows the model trains on, `A'` is a temporally handicapped model, not the same
  model. Standardising on the other rule instead (`C'`, give C protocol A's random holdout) puts
  L4's gap at +0.5223 against `A'`'s +0.1379: each protocol's default rule is the one that
  flatters it, and the published `A - C` sits mid-bracket rather than at the top. A third rule
  (holdout = first 20% in time) returns essentially the whole uncontrolled gap on raw AUC
  (+0.1727 of +0.1740), which says the AUC shrink under `A'` is a lost training window rather
  than the rule. "48% of the gap is the rule" is withdrawn as a point estimate. The clean control
, hold the model's fit rows fixed, move only the scale rows, accept the overlap, is not
  implemented and is logged in `FUTURE.md`.
- 2026-08-03, **`tests/test_floors.py` added: the floor had zero test coverage.** Six mutations
  reverting the day's corrections, flipping `refit_scale`'s default, fitting the floor's scale
  in-sample, giving every protocol A's holdout rule, collapsing `_floor_key` to L1, measuring one
  magnitude, and `score_fold` ignoring `holdout_rule`, all passed a 400-test suite, including
  two verbatim reversions of corrections earlier passes killed the report over. `frozen_floor`
  now returns its per-fold `scales` and `frozen_raw` so the tests can re-derive them by exact
  equality; an inequality-based test passes against the in-sample mutation because the two
  numbers are numerically close. A seventh mutation (phase 3 reverted to the in-sample rescale)
  is covered by a spy rung in `tests/test_pnl.py`.
- 2026-08-03, `make ladder`, `README.md` and `METHOD.md` now pass `--holdout-control`. Without
  it the published command regenerated `reports/ladder.md` **without the control table the
  headline numbers come from**, a documented reproduction that did not reproduce the artifact
  (RULES.md rule 7). The report header always echoed the flag correctly; the three places a
  reader would copy from did not.
- 2026-08-03, The `AUC raw` column now carries its worst fold, like every other metric in the
  project. It shipped for a day without one, which is a RULES.md rule 4 miss on the single
  column the report tells the reader to quote. It matters: L4's gap is positive in 79 of 80
  protocol-C folds while L3's is positive in 55 of 80 and halves when fold 0 is dropped, and the
  "16 / 16 cells" column both rungs share hides that difference entirely.
- 2026-08-03, Three stale phase-3 numbers in `FINDINGS.md` were replaced from the regenerated
  `reports/costs.md`, and one conclusion weakened with them: the Diebold-Mariano comparison of L4
  against L3 on INTC calendar 1s is **p = 0.087**, not the p = 0.016 quoted from a superseded
  run, and the AMZN cell cited as a null now leans the other way at p = 0.088. The honest reading
  is no significant difference in either direction. The break-even ordering the section actually
  rests on is unaffected.

## The clean calibration control (2026-08-03)

- 2026-08-03, **`Rung.fit` gains `model_mask`, and the decoupled control it enables reverses an
  attribution published the same day.** `A' - C` was built to isolate the calibration rule's
  contribution to the A-versus-C gap and could not: `Rung.fit` fits the raw model on the
  complement of the calibration holdout, so switching A to the positional rule also took away
  the model's training tail. `model_mask` decouples the two row sets, the raw model is fitted on
  a seeded random 80% that is IDENTICAL in every arm, and only the scale rows move, accepting
  that they now overlap the model's.
- 2026-08-03, **The calibration rule is worth exactly zero on a rank metric, by construction,
  and that is the finding.** Raw AUC is a rank statistic on the raw model, so it never sees the
  scale; once the model's fit rows stop moving with the rule, the rule cannot move raw AUC. It
  measures 0.000000 across all 96 rows. Therefore the whole `A' - C` shrink (+0.1740 -> +0.1371)
  was the model's lost training window, not the calibration. The split-only gap is **+0.1913**
  for L4, LARGER than the uncontrolled figure, positive in 16 of 16 cells, +0.0657 to +0.3132.
  "About half of L4's gap is the calibration rule" and the bracket [+0.138, +0.522] are both
  withdrawn. On R², where the metric does see amplitude, the rule is worth a median 0.0116 for
  L4, about 4% of its gap.
- 2026-08-03, **The decoupled arm's absolute R² levels are not quoted anywhere, and the report
  says so on its face.** The accepted overlap makes each arm's scale partly in-sample, which is
  precisely the calibration killed on 2026-08-02, and it bites the same way: under the
  walk-forward arms a high-capacity rung's amplitude is far too large and R2-vs-zero goes sharply
  negative. The overlap is identical in every arm so the DIFFERENCES are clean, but the levels
  are not comparable to the headline ladder. This is a real cost of the control and it is the
  reason the result is stated on AUC.
- 2026-08-03, `tests/test_decoupled_control.py` was written before the branch and was RED on the
  import. It pins the two properties that make the thing a control at all: the model's fit rows
  do not move when the rule moves, and the scale's do. It also contains a guard test asserting
  that WITHOUT the control the model's raw AUC does move with the rule, without that, the
  headline test could pass vacuously.
- 2026-08-03, Adding `model_mask` broke every `fit` override in the repo (`_ConstantDirection`,
  `_Oracle`, `L0Zero`, and the `StubRung` in two test files) with a `TypeError`. That is the
  second signature drift in two days, and the first one had a test written for it the day before
, which did not catch this because it only passed `scale_holdout`. The test now passes every
  keyword the harness can pass, not just the one that broke it first.
- 2026-08-03, Trial count 2,152 -> **2,536**: +384 for the four decoupled arms. Trial 20, the
  confounded control it supersedes, stays counted on the same principle as trials 5 and 9, it
  was fitted, it produced a number that was reported, and removing it would flatter the
  deflation.

## After the 2026-08-03 fifth red team (`SURVIVED WITH CAVEATS`)

- 2026-08-03, **"The overlap is identical in every arm, so the differences are clean" is
  withdrawn.** It is true of the overlap FRACTION and false of its effect. The in-sample scale
  bias grows with capacity per training row, and C trains on 1/1.8 of A's rows at the median and
  1/8.7 at fold 0, so the accepted cost lands almost entirely on one side of every difference:
  at INTC calendar 1s the A arm is unharmed (R2 0.18378 vs the main ladder's 0.18316) while the C
  arm is destroyed (-0.25953 vs +0.02134). Under permuted labels the decoupled construction
  manufactures an R2 gap of +0.062 +/- 0.058 out of noise against the main ladder's
  +0.010 +/- 0.009. **The decoupled arm's R2 columns are now marked unusable INCLUDING the
  differences**, and the "rule is worth 0.0116 on R2, about 4%" figure built on them is
  withdrawn, it sat inside its own null (a constant predictor scores +0.0082 on the same
  statistic). Raw AUC carries the same null as the main ladder and is unaffected.
- 2026-08-03, **The model-fit-rule sweep is reported, because without it the control looks like
  a free parameter chosen after the fact.** `A' - C` was already model-fit-rule-matched; the
  headline `A - C` is the rule-INCONSISTENT one, so the commit relocated a choice rather than
  removing one. Across six common rules the L4 gap is +0.1740 / +0.1371 / +0.1913 / +0.1927 /
  +0.1977 / +0.2010, and the only low value is the rule that strips protocol A of the day's tail.
  The defensible claim is a range, +0.19 to +0.20 under every rule that does not remove A's own
  adjacency, not the point estimate the first draft gave.
- 2026-08-03, Margins are quoted as the **median of per-cell ratios**, not the ratio of medians:
  L4 8.1x (worst cell 3.4x), L3 12.5x (worst cell 7.3x). The previous pass flagged exactly this
  confusion in a different row and it recurred here; the min-cell *ratio* had also been silently
  replaced by a min-cell *absolute margin*, which is the weaker statement dressed as the stronger.
- 2026-08-03, **L1 is no longer described as "clearing" or "failing" its floor.** L1's raw-AUC
  gap and L1's raw-AUC floor are the same 16 numbers, ratio 1.000 in 16 of 16 cells, because L1
  has no model to refit. Presenting that as a control that passes is circular. The underlying
  fact is the important one and it is now stated directly: a rung with nothing to fit is
  numerically indistinguishable from a frozen predictor.
- 2026-08-03, `per_fold_auc_raw` added; the headline metric had shipped without a fold
  distribution for two revisions (RULES.md rule 4, third occurrence). It matters: on the clean
  arm L4 is positive in **80 of 80** folds (worst +0.0157) while L3 is positive in **55 of 80**
  (worst -0.0829), and the "16/16 cells" column both share hides that completely.
- 2026-08-03, **Five test mutations that survived a 429-test suite are now caught.** All five
  were in the WIRING rather than in `model_fit_mask` itself: a mask varying by protocol at the
  call site, a positional mask at the call site, `DECOUPLED_ARMS["C_pos"]` pointing at protocol A,
  `A_rand` silently positional, and the four arms run with `decouple_scale=False`. Two of the
  original property tests were tautological, they called `model_fit_mask` twice with identical
  arguments and compared the result to itself, so they could not fail for any deterministic
  implementation. The properties are structural, so they are now asserted structurally: the
  protocol must not be reachable from the mask's signature or from its call site, and the arms
  must name the protocols their keys claim. The end-to-end test runs under both protocols; it ran
  under A only, which is how the protocol-dependence mutation escaped.
- 2026-08-03, **The strongest surviving alternative explanation is recorded as item 0 of "what
  would change my mind", not buried.** Across protocol-A fold counts k the raw-AUC gap is flat
  (+0.0650 -> +0.0676) while the R2 gap rises smoothly, the signature of a channel that saturates
  once each test row has one training neighbour, which is what intraday burst duplication would
  produce and what "more training data leaks more" would not. Nothing in this repo separates it
  from the §1 reading on a single session.
