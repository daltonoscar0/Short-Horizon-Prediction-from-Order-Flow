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
