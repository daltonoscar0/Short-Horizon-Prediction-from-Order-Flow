# DECISIONS — Edge

Every choice made without asking the human: what was decided, the one-line rationale, the
date. Frozen choices additionally get a row in the RESULTS.md frozen-decisions table.

- 2026-07-30 — Raw LOBSTER CSVs live at `data/raw/lobster/`, fetched from the Hugging Face
  mirror already used by orderflow-lm's fetch script (lobsterdata.com serves an SPA shell on
  the old sample paths). `data/` is gitignored: 250 MB of CSV does not belong in the repo.
- 2026-07-30 — Universe is the 10-level full-day (09:30–16:00) free-sample files for INTC
  and AMZN, 2012-06-21, per `DATA_AND_LINKS.md`; the 50-level 1-hour samples in orderflow-lm
  cover different tickers and a shorter window.
