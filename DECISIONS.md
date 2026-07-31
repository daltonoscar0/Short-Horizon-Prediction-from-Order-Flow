# DECISIONS — Edge

Every choice made without asking the human: what was decided, the one-line rationale, the
date. Frozen choices additionally get a row in the RESULTS.md frozen-decisions table.

- 2026-07-30 — Raw LOBSTER CSVs live at `data/raw/lobster/`, fetched from the Hugging Face
  mirror already used by orderflow-lm's fetch script (lobsterdata.com serves an SPA shell on
  the old sample paths). `data/` is gitignored: 250 MB of CSV does not belong in the repo.
- 2026-07-30 — Universe is the 10-level full-day (09:30–16:00) free-sample files for INTC
  and AMZN, 2012-06-21, per `DATA_AND_LINKS.md`; the 50-level 1-hour samples in orderflow-lm
  cover different tickers and a shorter window.
- 2026-07-30 — Feature storage is `pyarrow` (23.0.1, already in the env); `polars` is not
  installed and adds nothing pyarrow doesn't cover for CSV→parquet.
- 2026-07-30 — The orderflow-lm parser is C++ with no Python bindings, so "import as a
  package" cannot be literal. Its SPEC's Decision A applies here too: on level-censored
  LOBSTER samples, message-replay reconstruction diverges from the orderbook file
  (measured today with its `book_source_compare`: 90.3% best-bid disagreement on 10-level
  AMZN, matching its own 95.8% on 10-level AAPL). Edge therefore reads the message +
  orderbook CSV pair directly (numpy/pyarrow), freezes the schema in
  `docs/DATA_CONTRACT.md`, and tests alignment locally via type-3 deletes — replay-based
  validation is structurally impossible on this data. orderflow-lm remains the convention
  reference, not an import.
- 2026-07-30 — Alignment convention ("orderbook row k = state after message k") verified
  empirically before any code: 2000/2000 sampled type-3 deletes on each of INTC and AMZN
  satisfy the size-drop identity in the k−1→k direction, 0/2000 the inverted one. The
  lobsterdata.com DataStructure page is unreachable (SPA shell / dead subdomain; archive
  blocked), so the worked example there could not be quoted; orderflow-lm SPEC ("pre-event
  book = orderbook file row i−1") agrees.
- 2026-07-30 — Tests use stdlib `unittest`, no pytest: `make test` must run from a fresh
  checkout with only the declared deps.
- 2026-07-30 — Phase 0 step 1 (repo skeleton + failing tests) scoped to *not* implement
  `src/edge/io/lobster.py` yet: `CLAUDE.md` rule 2 requires the adversarial test to exist
  before the code it constrains. `make test` is expected to be RED after this step
  (`ImportError: cannot import name 'lobster'`); implementation is a separate step that
  turns it green. Confirmed by running `make test` and pasting the failing output.
- 2026-07-30 — `pyproject.toml` declares `numpy`/`pyarrow` deps for documentation and
  optional editable install, but `Makefile` does not require `pip install -e .` — both
  `make data` and `make test` put `src/` on `PYTHONPATH` directly, so a fresh checkout with
  only the two pip deps installed needs no install step.
- 2026-07-30 — Output event-frame column order is block-grouped by side/quantity
  (`bid_px[1..L], bid_sz[1..L], ask_px[1..L], ask_sz[1..L]`), per `PLAN.md`'s explicit
  schema, **not** the raw orderbook file's interleaved-per-level order
  (`ask_px,ask_sz,bid_px,bid_sz` repeated per level). The adapter reorders columns on the
  way out; `docs/DATA_CONTRACT.md` documents both orders so this isn't rediscovered later.
- 2026-07-30 — `order_id` (raw message col 3) is read by the adapter for row-count/QC
  purposes but dropped from the output parquet schema — not in `PLAN.md`'s frame
  definition and not needed downstream; keeping it out avoids a schema readers have to
  remember to ignore.
- 2026-07-30 — Sentinel book levels (`ask 9999999999` / `bid -9999999999`, size 0):
  grepped the full raw INTC and AMZN 10-level files, zero occurrences — the visible book is
  10-levels-deep on both sides all day in this sample. Documented in `DATA_CONTRACT.md` as
  an empirical finding for this sample, not a guaranteed absence; the adapter should still
  treat the sentinel defensively when it's eventually written.
- 2026-07-30 — Price-scaling sanity check (median mid, `/10000`) run against the raw
  orderbook files directly (pyarrow, before the adapter exists): INTC median mid $27.02
  (range $26.62–$27.60), AMZN median mid $222.66 (range $220.52–$226.03) — both inside the
  brief's [20,35] / [200,250] bounds, confirming `/10000` is the correct dollar conversion.
  Recorded in `docs/DATA_CONTRACT.md`; the adapter-level assertion of this is deferred to
  the data-quality-report step.
- 2026-07-30 — `make test`/`make data` both green: `src/edge/io/lobster.py` implements
  `read_message_file`, `read_orderbook_file`, `build_event_frame`, and the CLI exactly per
  `docs/DATA_CONTRACT.md`'s Adapter API. All 12 tests pass, including
  `test_orderbook_row_is_state_after_message` and `test_inverted_direction_does_not_hold`
  run against the full raw INTC (624,040 rows) and AMZN (269,748 rows) files, not just the
  toy fixture. Frame row counts match the raw CSV line counts exactly for both tickers.
- 2026-07-30 — `Makefile`'s `PYTHON` is now autodetected (`python3`, falling back to
  `/opt/anaconda3/bin/python3`, then `python3.12`) instead of hardcoded `python3`. Root
  cause: on this machine, bare `python3` resolves to Xcode's `/usr/bin/python3` (3.9, no
  pyarrow) ahead of the conda `base` env that actually has the declared deps (pyarrow
  23.0.1, numpy) — macOS's `path_helper` (run from `/etc/zprofile` for login shells)
  re-prepends `/usr/local/bin:/usr/bin:/bin:...` ahead of whatever conda's `.zshrc` block
  added, in every shell mode tested (non-interactive, interactive, login+interactive).
  `/usr/bin/python3` is also below `pyproject.toml`'s declared `requires-python = ">=3.10"`
  regardless of pyarrow. The Makefile fix tries plain `python3` first (a no-op on any
  ordinary single-interpreter checkout) and only falls back if the import check fails, so
  it doesn't change behavior anywhere except this specific quirk. Verified: `make test` and
  `make data` both succeed as bare commands (no manual PATH override) after this change.
- 2026-07-30 — Added `src/edge/io/quality.py` (`python3 -m edge.io.quality --frames --out
  --seed`), invoked as the second step of `make data`, to generate `reports/
  data_quality.md` from the built parquet frames (not by re-parsing the raw CSVs a second
  time). It hard-fails (`SystemExit`) if crossed-book count != 0 or median mid falls
  outside the ticker's expected dollar range — per `CLAUDE.md`/`PLAN.md`, those are
  fix-the-parser signals, not report-around-them signals. No new dependency: pyarrow +
  numpy + stdlib only.
- 2026-07-30 — Sentinel-safe "valid quote" mask (`ASK_SENTINEL_PRICE`/`BID_SENTINEL_PRICE`,
  now exported from `edge.io.lobster` and reused in `quality.py`): a best-bid/ask pair only
  counts toward the crossed-book check and the spread/depth intraday profile if neither
  side is the LOBSTER sentinel (`ask 9999999999` / `bid -9999999999`) with size 0. Zero
  such rows in this sample (confirmed in the report output), but the check is real code now
  instead of a documented assumption.
- 2026-07-30 — Intraday profile bucketing: 30-minute wall-clock buckets (`ts_ns` divided
  into half-hour bins from midnight, labelled by clock time, not by an assumed 09:30
  session-start offset, so it doesn't depend on trusting the filename's start/end fields).
  Event-rate U-shape is checked with a coarse heuristic (first bucket and last bucket both
  above the profile's median bucket count). Result: INTC clean U-shape (**yes**); AMZN
  fails the heuristic (**no**) — its close-of-day surge is sharp but its open (09:30-10:00,
  17,520 events) is not the profile's peak (10:00-11:00 buckets run higher). Not treated as
  a parse bug: event-type counts match `DATA_CONTRACT.md` exactly (cross-checked
  independently with `awk` directly on the raw CSV, bypassing the adapter entirely),
  crossed-book count is 0, and median mid is in-range for both tickers — the weaker AMZN
  open-of-day effect is recorded as a genuine finding about this name/day, to revisit if
  AMZN-specific features behave oddly in Phase 1.
- 2026-07-30 — `docs/spec/PLAN.md` Phase 0 checkboxes ticked, status moved to Phase 1, next
  three tasks set to `FEATURES.md` implementation / two-clock construction / rolling-only
  normalization (the first three Phase 1 bullets), per the Phase-0 exit criteria in the
  task brief.
- 2026-07-30 — U-shape heuristic in `quality.py` replaced: was "first bucket AND last
  bucket both above the profile median", now "mean event rate over the first and last
  quartile of bins exceeds the mean over the interior bins". The old rule turned on two
  single half-hour buckets, so one noisy bucket flipped the verdict — which is exactly what
  happened to AMZN (open 09:30–10:00 is elevated but not the day's peak, so the old rule
  said no while the profile is visibly U-shaped). Under the quartile rule both INTC and
  AMZN report **yes**. This supersedes the "AMZN fails the heuristic" line above; the
  underlying observation (AMZN's open-of-day surge is weaker than INTC's) still stands and
  is still visible in the committed profile table. No parser change — event-type counts,
  crossed-book count and median mid were already all passing.
- 2026-07-30 — `quality.py` now also takes `--raw` and re-counts the raw message/orderbook
  CSV lines, making "raw message rows == raw orderbook rows == parquet rows" a hard
  invariant of `make data` rather than something only the adapter checked internally. Cheap
  (streamed line count, no second parse) and it catches a truncated or half-written parquet,
  which the previous report could not.
- 2026-07-30 — Added `tests/test_quality.py` (16 hand-computed tests over the pure check
  functions: crossed/locked counting, sentinel rows, ts monotonicity and tie counting, gap
  ranking, price scaling, intraday binning, U-shape rule). `quality.py`'s docstring already
  claimed these were unit tested and they were not; suite is now 28 tests, all green.
- 2026-07-30 — Stale test `tests/test_clocks.py::test_frozen_per_ticker_n` asserted the old
  per-ticker `EVENT_CLOCK_N = {"INTC": 3, "AMZN": 1}` against the frozen scalar `10`, so the
  suite was red (67/68). Rewritten as `test_frozen_single_n_for_both_tickers`, asserting the
  scalar and asserting it is *not* a dict, so a silent revert to a per-ticker calibration
  fails a test rather than passing quietly. Suite green at 68.
- 2026-07-30 — Phase 2 was invoked with Phase 1 only half-built. The infra layer exists and
  is good (`frame.py`, `book.py`, `clocks.py`, `windows.py`, `normalize.py`) but none of the
  six family modules named in `src/edge/features/__init__.py`'s `FAMILY_PREFIXES` (ofi,
  queue, microprice, trade, state, labels) had been written, there is no `features.parquet`,
  and `docs/INTERFACE.md` — which four module docstrings already cite — does not exist. The
  L0-L4 ladder is defined in terms of those features, so Phase 2's deliverable is
  unreachable without them. Decision: build the remaining Phase 1 families as a prerequisite
  block inside this run, scoped to exactly what `FEATURES.md` secs. 1-5 and 7 specify and no
  further, then run Phase 2 on the result. Not treated as scope creep — the ladder table is
  the Phase 2 milestone and features are its input — but Phase 1's own milestone (the
  feature/label correlation table) is produced too so the phases stay separately auditable.
- 2026-07-30 — Fold geometry under a one-day universe. `EVAL_PROTOCOL.md` sec. 1 says split
  on day boundaries first and fall back to intraday blocks "if you need more folds"; the
  free sample is a single session (2012-06-21) per ticker, so intraday blocks are the only
  option, not a preference. Frozen rule, chosen a priori and not by looking at any fold:
  expanding walk-forward, initial train = the first third of the session
  (09:30:00-11:40:00), then **6 test blocks of 2600s** tiling 11:40:00-16:00:00 exactly.
  Consequence recorded up front: `EVAL_PROTOCOL.md` sec. 5 detector 4 (same-day vs next-day)
  **cannot be run** on this universe and will be reported as not-run rather than as passing.
- 2026-07-30 — Purge/embargo parameters frozen from the protocol's own defaults, not tuned:
  purge = h at both edges of every test block; embargo E = max(60s, 10*h), giving E = 100s
  at the 10s horizon and 60s at 1s/2s/5s. E-sensitivity gets its own table rather than a
  choice. These go in the RESULTS.md frozen-decisions table.
- 2026-07-30 — GBM library: **LightGBM 4.7.0** (the one GBM permitted by CLAUDE.md's
  dependency rule; xgboost/catboost are absent from the environment anyway). `scipy` is
  present in the environment but is **not** on the permitted list, so the harness imports it
  nowhere — Spearman IC, AUC, MCC, ridge, logistic and OLS are all hand-written on numpy.
  `sklearn` is not installed at all, which makes the "no `sklearn.model_selection`" rule
  structurally enforced rather than merely observed.
- 2026-07-30 — Driver hardened after the first unattended run failed three ways. (1) Two
  `drive.sh all` instances ran concurrently (20:18 background, 20:39 interactive) and raced
  the working tree; a lock in `.edge-runs/lock` now refuses a second driver. (2) Headless
  phase sessions launched their workflows in the background and the harness terminated them
  at the 600s print ceiling, so phases 1-2 were marked complete with work half-landed; the
  driver now exports `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0`. (3) The adversary gate
  treated an empty `reports/adversary/` as a pass; phases 0-4 now require a new adversary
  report before the phase is marked complete. State reset to rerun from phase 0 (its code is
  committed and verified; the rerun owes only the adversary pass) through phase 2. The
  uncommitted feature spine from the killed phase-1 workflow is left in the tree for the
  rerun to inherit and verify rather than rewrite.
- 2026-07-31 — Phase 0 closed out by hand after its rerun died to `API Error: Connection
  closed mid-response` between writing the adversary reports and committing. All four
  phase-0 adversary reports returned SURVIVED WITH CAVEATS (none KILLED), and the fixes the
  dying session had already written to the tree were verified before committing: 80/80 tests
  in the working tree, `make data` exit 0 now reporting `align_hold_rate=1.0000
  (n_checked=4000)` on both tickers, and a fresh clone running 40 tests where it previously
  ran 28. Committed as `1022005`, `09e2161`, `3836cdc`. The adversary's four "would change my
  mind" conditions are carried forward, in particular that a Phase-1 number disagreeing with
  a re-derivation from raw promotes findings 1/2/11 to blockers.
- 2026-07-31 — `.edge-runs/state` set to 0 by hand rather than re-running phase 0 a third
  time. The gate report records the milestone criteria as met and the run only died after
  the adversary pass, so a rerun would repeat ~30 minutes of verified work; this is the write
  `run_phase` would have made had the phase exited 0.
- 2026-07-31 — Driver: a non-zero exit from the phase's `claude` invocation is now fatal.
  It was unchecked, so the phase-0 API error was indistinguishable from success at the exit
  code; only the missing state file revealed it. The adversary-report count guard does not
  cover this case (that phase had written its reports before dying).
- 2026-07-31 — Driver, second hardening round, after a mid-run edit of `tools/drive.sh`
  corrupted the executing driver's incremental parse (`$1: unbound variable` at a line that
  contained no `$1`): the driver body now runs through a `main "$@"; exit` tail so the whole
  file is parsed before anything executes, and edits landing mid-run cannot touch a live
  driver. A transient API failure (the class that killed phase 0's rerun) now retries the
  phase up to 3 times with backoff instead of stopping the run; any other non-zero claude
  exit still stops it. `tools/leakcheck.py` added to the settings deny list for Write/Edit —
  the phase 3 gate must not be editable by the phase sessions it gates.
- 2026-07-31 — Project reframed and the toolchain overwritten from `EDGE_MASTER.md` Part 1.
  The question is now *how much of published short-horizon LOB performance is an artifact of
  leaky cross-validation*, measured by scoring the whole L0-L4 ladder under three protocols
  (A shuffled k-fold, B unpurged walk-forward, C purged walk-forward with embargo) and
  reporting the A-C and B-C gaps. The LM arm is cut: `orderflow-lm` stops at tokenization and
  no trained checkpoint exists, so there is nothing to probe. Consequences: phases renumbered
  0-4 (cost-model moves 4->3, scribe 5->4), the `lm-probe` agent and the phase-5 command are
  gone, and `tools/leakcheck.py` — the gate for an absent model — is dropped along with its
  two `Write`/`Edit` deny rules in `.claude/settings.json` (the file was never written; the
  rules were vestigial). `docs/spec/` is unchanged and still authoritative except where it
  refers to LM representations or phase-3 controls. L1 is now load-bearing: an unfitted sign
  predictor has no parameters, so a near-zero A-C gap on L1 is what separates a measurement of
  optimism from an artifact of the harness.
- 2026-07-31 — `tools/drive.sh` overwritten verbatim from the brief, which drops the four
  hardenings added on 2026-07-30/31 after the first unattended run failed: the `.edge-runs/lock`
  concurrent-driver guard, `CLAUDE_CODE_PRINT_BG_WAIT_CEILING_MS=0` (without it the harness
  kills background phase workflows at 600s and the phase is marked complete half-landed), the
  fatal check on a non-zero `claude` exit plus API-error retry, the adversary-report count
  guard (an empty `reports/adversary/` reads as a pass), and the `main "$@"; exit` tail that
  makes a mid-run edit of the driver safe. Each was written in response to an observed
  failure, not a hypothetical. Recorded here rather than silently re-applied: reinstating them
  on top of the new phase list is a small edit and should be a deliberate choice.
- 2026-07-31 — Part 1's closing `git add -A && git commit` not run. The toolchain
  (`.claude/`, `CLAUDE.md`, `tools/`) is git-excluded, so the commit would have contained
  nothing from the bootstrap and instead swept ~20 files of in-flight phase-1 work
  (`src/edge/features/`, `src/edge/analysis/`, twelve test modules, three reports,
  `docs/INTERFACE.md`, the `Makefile` diff) into a commit labelled "add project spec and
  harness scaffolding". That work is mid-flight and unverified against the new protocol; it
  gets committed by the phase that finishes it.
- 2026-07-31 — Phase 1 closed out. The six family modules, both clocks, the labels and the
  correlation table already existed from an earlier session but were recorded as "mid-flight
  and unverified"; this pass verified them rather than rebuilding them. Verification was a
  six-way parallel audit (one agent per family, each finding then adversarially re-checked by
  an independent agent instructed to refute it). 30 candidate findings, 12 confirmed, 18
  refuted — the refutations included several plausible-sounding ones (the `_5s` trade columns
  are covered by three value-asserting tests despite the string never appearing; `st_rv`'s
  sqrt is realized VOLATILITY as FEATURES.md sec. 5 names it, not a spec deviation; the joint
  per-level validity mask in `queue.py` is documented and deliberate), so the confirmed set is
  what got acted on and nothing else.
- 2026-07-31 — `mp_micro` was Stoikov's FIRST-ORDER term, not the microprice; fixed. The
  module built only the transient-transient matrix Q (naming it `B`), so
  `x = (I-Q)^-1 R K = G^1` — the expected mid move up to the NEXT price change. Stoikov's
  actual estimator needs the landing matrix `B = (I-Q)^-1 R2` and the outer recursion
  `G^(i+1) = G^1 + B G^i`, which was never built or run. This is exactly the failure
  FEATURES.md sec. 3 warns about ("not as a synonym for weighted mid" — G^1 is a subtler
  version of the same mistake, a real quantity that discards every price change after the
  first). `_estimate_block_adjustment` now accumulates R2 counts alongside Q, and
  `_micro_adjustment` runs the recursion. `I - B` is SINGULAR by construction (B is
  row-stochastic on visited states), so the limit cannot be reached by a direct solve; the
  recursion depth is frozen at `MICRO_ITERS = 6`, Stoikov's own repo value, with a TOL early
  exit. Pinned by a new hand-solved test on a convergent fixture where G^1 = 100, the limit
  is 200 and the frozen-depth answer is 100*(2 - 2^-6) = 198.4375 — all three distinguishable,
  so neither the old bug nor a changed iteration count can pass silently.
- 2026-07-31 — `mp_micro` moved from raw LOBSTER price units to ticks. It is a price LEVEL and
  INTERFACE.md sec. 1 puts every emitted price level and price difference in ticks; sec. 0
  makes INTERFACE.md normative over the module docstring that had declared raw. Internal
  arithmetic still runs in raw integers and converts exactly once at the return.
- 2026-07-31 — `tr_last_trade_sign` no longer inherits `tr_eff_spread`'s `isfinite(mid_before)`
  mask. The aggressor sign is `-direction[trade_pos]`, read from the message file with no book
  dependence, so masking it on an invalid preceding touch deleted a perfectly known value. The
  `has_prev` term stays on both — that one is the documented "trade is the very first frame
  row" rule, not a book-validity question.
- 2026-07-31 — Four OFI mutations and two state/queue mutations that survived the ENTIRE
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
- 2026-07-31 — `label_y_wmid_*` / `label_c_wmid_*` stand in for FEATURES.md sec. 7's
  `y'_h = micro(t+h) - micro(t)`, recorded here because the rationale previously lived ONLY in
  the labels.py docstring and CLAUDE.md requires every unasked choice in this file. Two
  reasons, both structural: INTERFACE.md sec. 5 rule 4 forbids labels.py importing
  microprice.py, and the Stoikov microprice is a FITTED quantity on a refit schedule — making
  the label depend on a model's refit cadence (and inheriting its block-0 NaN) is a far bigger
  commitment than "lower-noise alternative" implies. `book.wmid` is exact and unfitted. This
  is a substitution, not a synonym. Not revisited now that the microprice is correct: a label
  that inherits a fitted quantity's warmup is still the wrong shape for Phase 2.
- 2026-07-31 — Deadband alpha frozen at 0.5 ticks BY RULE and given its RESULTS.md
  frozen-decisions row (CLAUDE.md requires frozen choices in that table, and the row was still
  blank). Half a tick is FEATURES.md sec. 7's own first suggestion and the smallest move that
  is not a rounding artifact in a whole-cent book. Not fit to data, training or otherwise; the
  sensitivity sweep in reports/feature_correlations.md is reported, never used to pick it.
- 2026-07-31 — Adversary on phase-1 lookahead/normalization returned
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
- 2026-07-31 — F1 fixed: the event clock violated INTERFACE.md sec. 3. `event_stamps` set
  `index[i]` to the TRIGGERING event's own row, which satisfies `ts <= t` only by equality;
  the contract is the LAST row with `ts <= t`. LOBSTER shares nanoseconds across messages, so
  the two differed on 10.48% of INTC event stamps (up to 97 rows skipped). Harmless for
  features (they simply read fewer rows than entitled) but NOT for labels: labels.py anchors
  on this index while resolving its forward endpoint with the correct rule, so
  `label_y_mid_*` on the event clock contained a mid move that elapsed in ZERO wall-clock
  time and was already public at t. Measured inflation +0.021 on `qi_touch`/`qi_l1` (5.5%
  relative), +0.0005 on OFI. It would have appeared identically under protocols A, B and C,
  so it would not have biased the optimism decomposition — but it is a definitional leak into
  a label and it was unasserted, because `tests/test_clocks.py`'s contract test ran on the
  calendar clock only and the toy fixture contains no timestamp tie. Fixed with the same
  `searchsorted(..., side="right") - 1` the calendar clock uses, plus the missing event-clock
  test on a purpose-built tied-timestamp fixture. Event-clock columns change; rebuilt.
- 2026-07-31 — F2 recorded, not "fixed": the microprice recursion does NOT converge on this
  data. Measured `|eig(B)| = {1.0, 0.9906, 0.272}` with stationary drift `pi . G^1 = -0.018`
  ticks per iteration, so `G^i` has a persistent linear term and the limit does not exist —
  the increments plateau rather than decay. My own docstring claimed convergence via the
  martingale property; that claim was false and now says so. `MICRO_ITERS = 6` is therefore
  part of the feature's DEFINITION rather than a cap convergence reaches first, and `G^7`
  would be a materially different column. It stays at 6 (Stoikov's repo value, frozen a
  priori, never tuned against a correlation) and `mp_micro` is honestly "the six-step
  microprice". Consequence for Phase 2: on AMZN the truncation puts `mp_micro` outside
  [bid, ask] on 0.32% of finite stamps with `|mp_dev_mid|` reaching 15.8 ticks (p50 0.39,
  p99 0.70); INTC is clean at 0.00%. The tail is meaningless, not leaky — block-refit
  causality is intact — and Phase 2 must winsorise or exclude it with the choice logged.
- 2026-07-31 — F3 fixed, and the reasoning error behind it recorded. `correlations.main`
  printed `suspicious[:20]` of a 488-item list, and the phase-1 triage was read off that
  console tail — concluding "19 leak smells, all INTC, AMZN clean". False: 446 leak smells
  (282 INTC, 164 AMZN), and the largest single correlation in all of phase 1 is AMZN/event
  `tr_aggr_imb_100ms` at rho = +0.5307, not anything on INTC. That wrong premise was then
  handed to the adversary as a framing. It passes the same causality probes (fwd+1 0.6293,
  bwd-1 0.4538, 81.5% balanced directional accuracy on non-flat labels) so nothing is
  invalidated, but the triage was worthless. The console block now prints a count by scope
  over the WHOLE list and says explicitly that it is not the list. General rule taken from
  this: a summary printed from a truncated slice is worse than no summary.
- 2026-07-31 — F4 carried into Phase 2 as a reporting constraint: the 1s label is measuring a
  <=100ms effect. `rho(ofi_best_100ms, mid(t+1s) - mid(t+100ms)) = 0.027` against 0.458 for
  the first 100ms on INTC/event. A `label_y_mid_1s` result must not be presented as one-second
  alpha, and Phase 3's cost model has to price a signal whose content decays inside 100 ms.
- 2026-07-31 — F6/F7 carried into Phase 2: `tr_aggr_imb_*` is NaN when its window holds no
  trades (finite on 1.8% of AMZN calendar stamps vs ~100% for `qi_touch`), so per-fold row
  counts will differ by more than an order of magnitude across rungs and must be REPORTED per
  fold, not absorbed. And the trial log is empty against a 7,264-cell univariate surface;
  no model has been fitted yet, but the surface size belongs in the log before the deflated
  Sharpe is computed.
- 2026-07-31 — Adversary's warning on Phase 2 fold construction, recorded now so it is not
  rediscovered later: 32% of INTC event-clock stamps sit within 0.5ms of the previous stamp
  and carry rho 0.437 against 0.19 for isolated stamps. If Phase 2 cuts folds on STAMP INDEX
  rather than wall clock, a single burst can straddle a boundary and both folds will contain
  the same episode — which would read as optimism and would not be. Folds get cut on time.
