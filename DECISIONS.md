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
