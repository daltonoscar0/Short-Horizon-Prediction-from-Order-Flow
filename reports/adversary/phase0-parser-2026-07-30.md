VERDICT: SURVIVED WITH CAVEATS

Target: `src/edge/io/lobster.py`, `tests/test_lobster.py`, `tests/test_alignment.py`,
`docs/DATA_CONTRACT.md`, `src/edge/io/quality.py`, `reports/data_quality.md`.
Red team, fresh context, 2026-07-30. All scripts under `/tmp/redteam/`.

## What I tried and what happened

1. **Row alignment.** Not trusted, re-derived. I ran the size-drop identity over the
   **entire population** (not the repo's 500-sample) for four independent event families and
   five candidate offsets, on both tickers, and separately on the built parquet. Offset −1
   (`row k` = state **after** message `k`) is the unique winner at exactly 100%: INTC type-1
   adds 304,790/304,790, type-2 8,028/8,028, type-3 278,739/278,739, type-4 28,923/28,923;
   AMZN 131,954 / 2,917 / 123,458 / 8,974 — all 100%. Nearest rival offsets score 1.5–21.6%.
   Independent fifth check the repo does not do: type-5 hidden executions must leave the book
   **byte-identical** between rows `k−1` and `k` — 3,559/3,559 (INTC) and 2,444/2,444 (AMZN)
   at offset −1, 23%/25% at any other offset. The convention in `DATA_CONTRACT.md` is right,
   and the same test also independently pins the `size`/`price` column order and the
   `direction` (+1 bid / −1 ask) semantics, since swapping either destroys the identity.
   Caveats below on the *test* (weak inverted assertion, skip-on-missing-data) and on the
   *evidence claim* (`0/2000` is false).
2. **Lookahead in features.** Not applicable at phase 0 — no feature code is committed. The
   uncommitted phase-1 `src/edge/features/` is out of scope for this report.
3. **Scaler fit on the full sample.** Not applicable at phase 0 — `grep -rn "\.fit(" src/`
   returns nothing under `src/edge/io/`; no normalization exists yet.
4. **Label overlap across boundaries.** Not applicable at phase 0 — no labels, no splitter.
5. **R² against the test mean.** Not applicable at phase 0 — no metric code exists.
6. **Base-rate artifact.** Not applicable at phase 0 — no classifier, no accuracy number.
7. **Deadband tuning.** Not applicable at phase 0 — no α. *But* the phase-0 analogue exists
   and is real: the U-shape criterion was rewritten after AMZN failed the first one (finding
   F8).
8. **Dev-slice contamination.** Not applicable at phase 0 — no folds, no dev slice.
9. **Trial count.** Not applicable at phase 0 — `RESULTS.md` trial log is empty and should
   be. I scanned `.edge-runs/phase0-*.log` and `git log` for undisclosed configurations and
   found none.
10. **Pooled-number laundering.** Not applicable at phase 0 — the report is per-ticker
    throughout; there is no pooled number.

## Target-specific attacks

- **`_parse_ts_ns` fuzz.** Every one of 894k time strings in both raw files reparsed against
  `Decimal(s) * 10**9` floored — **0 mismatches**. No decimal point → correct. Leading
  whitespace → correct (`int()` strips). Trailing whitespace/newline, `.5`, `""`, `1e4`,
  `nan`, 14-digit seconds → all **raise**, none corrupt silently. Truncation is exact floor
  for positive input, so it never rounds the wrong way. **It does break on negatives**
  (F4). Fractional-digit histogram shows 7 rows with 12 digits (5 INTC, 2 AMZN), floored to
  ns — matches exact-decimal floor. The string-parse design decision is weakly load-bearing:
  a naive `float64*1e9` would differ on exactly 3 INTC rows, 0 AMZN.
- **`discover_pairs`.** Level-count mismatch in the filename → `FileNotFoundError`. Level
  count mismatch in the *body* → `ArrowInvalid` from pyarrow. Date mismatch →
  `FileNotFoundError`. Truncated orderbook → `ValueError` row-count mismatch. All good.
  Breaks: duplicate ticker+date with different windows silently collide on one output path
  (F5); lowercase tickers and tickers containing digits are silently skipped and
  `edge.io.lobster.main` exits 0 having written nothing; `2012-13-45` passes the regex (F10).
- **Sentinels.** I counted them myself over **all 40 columns** of both raw files, not just
  the best level: **0** ask sentinels, **0** bid sentinels, **0** zero-size cells, **0**
  non-positive prices, **0** crossed rows, ask levels strictly increasing and bid strictly
  decreasing on 100% of 893,788 rows. The `DATA_CONTRACT.md` "zero occurrences" claim is
  **true and stronger than stated**. What breaks the first time it is not zero: nothing in
  `io/` excludes them (F3) — `crossed_locked_counts`, `median_mid_dollars` and
  `build_intraday_profile` all consume raw `bid_px_1`/`ask_px_1` unmasked, so a single
  sentinel row puts `9999999999` into the spread mean with no hard failure. (Phase-1's
  uncommitted `features/frame.py` and `features/book.py` *do* mask properly.)
- **Position join with no key.** Confirmed: row count is the only check. A one-row-rotated
  AMZN orderbook (identical row count) built cleanly and then passed the **entire** quality
  gate — `Row-count equality: PASS`, `crossed-book count: 0 PASS`, `monotonic: True`,
  `median mid $222.6550 PASS`, `U-shape: True`, exit 0. Nothing validates alignment on the
  built artifact (F6).
- **int8 casts.** Not silent. `event_type=300` → `ArrowInvalid: Integer value 300 not in
  range: -128 to 127`; `direction=128` likewise. pyarrow `cast` defaults to `safe=True`.
  Real max magnitudes: price 2,298,000, size 175,583 — nowhere near int64 limits, and
  correctly int64 rather than int32 (sentinel would overflow int32).
- **Round-trip.** Verified, not assumed. For both tickers at full size,
  `build_event_frame(...).equals(pq.read_table(...))` is **True**, schema and column order
  included. I also re-read both raw files with a plain `csv.reader` + `Decimal` and confirmed
  every parquet column is element-wise identical, which rules out pyarrow block-parallel row
  reordering and confirms the interleaved ask-first column mapping in file order.
- **Determinism.** `make data` run twice into `/tmp` gives **byte-identical** parquet
  (md5 `b20b8a40…` / `439fa6b8…`), identical to the committed `data/frames/`, and the
  regenerated `data_quality.md` is byte-identical to the committed one.
- **Every number in `reports/data_quality.md` recomputed from the raw CSVs by an independent
  path** (awk for event counts, `csv.reader`+`Decimal` for timestamps): row counts, all 7
  event-type counts per ticker, 65,886/8,483 ties, 3,600/5,202 gaps>1s, the top-5 gap table,
  all 26 intraday bin counts, both median mids. **All match exactly.** Zero events fall
  outside 09:30–16:00, so the `np.clip` in `build_intraday_profile` is not inflating the
  edge bins.

## Findings

- **F1 (major) — a committed evidentiary number is false.** `DECISIONS.md` and
  `docs/DATA_CONTRACT.md` both state the inverted (`k → k+1`) comparison "held on **0/2000**"
  samples. Driving the repo's own `_run(ticker, "before")` at `MIN_SAMPLES=2000`, seed 0:
  **110/2000 (INTC, 5.50%)** and **130/2000 (AMZN, 6.50%)**. Full population: 5.15% / 6.72%.
  Zero is not attainable — `0.95^2000 ≈ 10^-45`. The companion claim, "2000/2000 in the
  k−1→k direction", reproduces exactly. So the log overstates the discrimination of its own
  evidence in the one direction that flatters it. The conclusion is unaffected (5.5% vs 100%
  still separates decisively, and I verified −1 at 100% on 750k+ events), which is why this
  is not a kill, but a number in the auditable decision log that does not reproduce is
  exactly what the log exists to prevent.
- **F2 (major) — a fresh clone's green `make test` never runs the alignment check.**
  `git clone` of the repo into `/tmp` and `make test`: `Ran 28 tests … OK (skipped=2)`. The
  two skipped tests are the entire phase-0 invariant. `data/` is gitignored, so
  `_raw_files_present()` is False on every fresh checkout and CI. Worse,
  `_raw_files_present()` is all-or-nothing across four hardcoded filenames: I pointed
  `EDGE_RAW_DIR` at a directory containing only a **deliberately corrupted** AMZN pair and
  got `OK (skipped=2)`. With INTC restored the same corrupted data correctly fails
  (`33 != 500`). The guard works; it is just opt-out-by-default in the environment where a
  reviewer would rely on it.
- **F3 (major) — `DECISIONS.md` describes sentinel-masking code that no longer exists.** The
  entry claims "a best-bid/ask pair only counts toward the crossed-book check and the
  spread/depth intraday profile if neither side is the LOBSTER sentinel … the check is real
  code now instead of a documented assumption." It was real code at `6692c9e`
  (`_valid_quote_mask`, used in the crossed count, the mid and the profile); `7ad99b4`
  ("tighten data-quality checks") removed it, and no decision entry records the removal.
  Current `quality.py` only *counts* sentinel rows at the best level and prints the count as
  "informational". Its docstring — "checked here so a future universe with real sentinel rows
  doesn't silently corrupt the crossed-book count" — is not true of counting. Zero impact on
  today's numbers (0 sentinels, verified independently).
- **F4 (minor, latent) — `_parse_ts_ns` gets the sign wrong on negative decimal seconds.**
  `"-1.5"` → `-500000000` (true: `-1500000000`); `"-0.5"` → `+500000000`, a **sign flip**.
  The fraction is added rather than applied with the sign of the seconds field. Unreachable
  in LOBSTER (0 negative times in either file, verified), but this function is the only
  timestamp path in the project and phase 4 may well introduce signed time deltas.
- **F5 (minor) — silent output collision on duplicate ticker-date.** `discover_pairs` keys on
  `(ticker, date)` only; `main` writes `{TICKER}_{DATE}.parquet`. Two message files for the
  same ticker-day with different `start_ns`/`end_ns` are both discovered and the second
  **overwrites** the first with no warning. Demonstrated. Harmless for a one-window-per-day
  universe; a live data-loss bug the moment a second window or a re-download appears.
- **F6 (minor) — position join, row count is the only key.** Demonstrated end-to-end above.
  The cheap fix that does not exist: nothing in `make data` re-runs the type-3/type-5
  identity on the built frame, so the artifact's alignment is guaranteed only transitively,
  by a test that reads different files and can skip itself (F2).
- **F7 (minor) — `test_inverted_direction_does_not_hold` does not test what it claims.**
  `assertLess(n_holds, n_checked)` passes if **499/500** samples support the inverted
  convention. The docstring says the inverted direction is "expected to fail". Observed
  values are 34/500 and 33/500; a threshold like `n_holds < 0.25 * n_checked` would encode
  the actual finding. As written it is nearly vacuous.
- **F8 (minor) — the U-shape verdict is a rule changed after seeing the answer.** The
  original rule (first and last bin both above the profile median) gives INTC **True**, AMZN
  **False** — I reproduced both. It was replaced by a quartile-mean rule under which AMZN is
  **True**. `DECISIONS.md` discloses this honestly and dates it, and `PLAN.md` does not make
  the U-shape a hard gate, which is why this is minor. But `reports/data_quality.md` prints a
  bare "U-shape (open/close activity > midday) confirmed: True" for a ticker whose opening
  bin (17,520) is **below** its own interior mean (18,279). The report should carry the
  qualification the decision log carries.
- **F9 (minor) — undocumented sub-ns truncation.** `DATA_CONTRACT.md` calls the time field an
  "ns-resolution fraction"; 7 rows in the sample carry 12 fractional digits and are floored,
  not rounded. Floor is monotone so it cannot create a strict decrease, and it agrees with
  exact-decimal floor, so the numbers are unaffected. It can in principle manufacture a
  timestamp tie, which matters once event/calendar clocks are built on `ts_ns`.
- **F10 (minor) — regex silently drops valid filenames.** `[A-Z]+` skips lowercase tickers
  and any ticker containing a digit; `\d{4}-\d{2}-\d{2}` accepts `2012-13-45`. On an empty
  discovery `edge.io.lobster.main` prints nothing and exits **0** (`quality.py` correctly
  exits 1). A mis-cased download directory produces a silent no-op build.

## Most likely remaining explanation that is not "the signal is real"

Phase 0 has no signal, so the analogous question is: what is the most likely way these
artifacts are still wrong? **The raw CSVs themselves.** Every check I ran — mine and the
repo's — is *internal* consistency between two files that were downloaded together from a
Hugging Face mirror (`DECISIONS.md`, 2026-07-30), not from lobsterdata.com, and neither the
repo nor I verified them against a published checksum or against LOBSTER's own worked
example (the spec page does not resolve from this machine). If the mirror shipped a
message/orderbook pair that was internally consistent but not the true LOBSTER output — a
re-derived orderbook file, say, or a re-serialisation — every test in this repo and every
test in this report would pass, because they all only assert that the orderbook file is a
faithful replay of the message file. The 100% type-1/2/3/4 identity is in fact *suspiciously*
clean: it implies the message file contains no event at a price outside the visible ten
levels, on either side, all day, for both names. That is consistent with a level-censored
LOBSTER export, and it is also exactly what you would see if the orderbook file had been
generated *from* the message file. Distinguishing the two needs an external reference the
environment cannot reach. Second-most-likely: a reviewer takes a green `make test` on a
fresh clone (F2) as evidence the alignment invariant holds, when it was never executed.

## What I could not check and why

- **LOBSTER's authoritative data-structure page.** `curl https://data.lobsterdata.com/info/DataStructure.php`
  → `Could not resolve host` (the apex `lobsterdata.com` resolves, returns a 457-byte SPA
  shell). The claim of unreachability in `DATA_CONTRACT.md` is verified for this environment.
  I substituted five independent empirical identities instead.
- **The orderflow-lm corroboration.** I did verify it verbatim: `/Users/oscardalton/orderflow-lm/SPEC.md:114`
  reads "pre-event book (orderbook file row i-1, per Decision A)", which is equivalent to the
  Edge convention. Note it is same-author, same-family evidence, not independent.
- **`DECISIONS.md`'s "90.3% best-bid disagreement on 10-level AMZN"** from orderflow-lm's
  `book_source_compare`. I did not rebuild the C++ tool; the analogous AAPL figures (95.8% /
  79.9%) are in that SPEC at lines 72–73, so the claim is plausible but unverified here.
- **Provenance of the raw CSVs.** No published per-file checksum reachable offline; both
  copies in `data/raw/` and `data/raw/lobster/` are md5-identical to each other, which
  proves only that the duplicate is a duplicate.
- **Row 0's book state has no `k−1` predecessor** and so cannot be checked by the size-drop
  identity directly. It is checked indirectly: the type-1 identity at `k = 1` (row 0 → row 1)
  holds, which would fail if the orderbook file carried a leading pre-open state row.
- **Items 2–10** of the standard order of suspicion have no phase-0 surface, as stated above.
  I did not pad them; I confirmed by grep and by reading `RESULTS.md` that the corresponding
  code and numbers do not exist yet.

## Reproduction

```
PYTHONPATH=src python3 /tmp/redteam/align.py          # full-population, 5 offsets, 4 event families
PYTHONPATH=src python3 /tmp/redteam/align_parquet.py  # same, against data/frames/*.parquet
PYTHONPATH=src python3 /tmp/redteam/ts_fuzz.py        # adversarial time strings vs exact Decimal
PYTHONPATH=src python3 /tmp/redteam/ts_full.py        # all 894k timestamps vs exact Decimal
PYTHONPATH=src python3 /tmp/redteam/pairs.py          # 14 discover_pairs / cast / sentinel cases
PYTHONPATH=src python3 /tmp/redteam/sentinel_book.py  # sentinels + book ordering, all 40 columns
PYTHONPATH=src python3 /tmp/redteam/roundtrip.py      # parquet-on-disk == in-memory frame
PYTHONPATH=src python3 /tmp/redteam/order.py          # parquet == plain csv.reader, element-wise
python3 /tmp/redteam/qc_indep.py                      # independent recompute of the QC report
PYTHONPATH=src python3 /tmp/redteam/inv.py            # reproduces (and falsifies) the 2000-sample claims
```
