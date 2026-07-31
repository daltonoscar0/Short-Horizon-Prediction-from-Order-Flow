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
