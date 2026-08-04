VERDICT: SURVIVED WITH CAVEATS

Target: `reports/data_quality.md`, `src/edge/io/quality.py`, `tests/test_quality.py`.
Claim under attack: **"the parse is validated."**

Every number printed in `reports/data_quality.md` is arithmetically correct — I recomputed
all of them from the raw CSVs in pure Python without importing a single line of `edge` code.
The parse itself is correct, and the row-alignment invariant is correct at a confidence far
above what the repo's own test establishes. But the *claim* is overstated in three specific
ways, and one committed artifact (`DECISIONS.md`) describes code that no longer exists.

---

## What I tried and what happened

1. **Row alignment.** Killed my own attempt. Independent full-sample check (`/tmp/redteam/align.py`,
   parses the raw CSVs with pyarrow directly, no `edge` import) over **all** messages at three
   offsets. At offset 0 ("orderbook row *k* = state after message *k*") the size-delta identity
   holds on **131954/131954 AMZN type-1 adds, 126375/126375 type-2/3 reductions, 8974/8974 type-4
   executions** and **304790/304790, 286767/286767, 28923/28923** for INTC — 100.00% on 1.05M
   messages. At offsets ±1 it holds on 4–22%. Fourth independent signature: type-5 hidden
   executions must leave the *visible* book unchanged — 2444/2444 (AMZN) and 3559/3559 (INTC)
   at offset 0, vs 24.6%/23.2% at offset +1. Parquet row *i* equals raw row *i* (message and
   book fields) on 2006/2006 sampled rows including the first and last three. **Alignment is
   right.** Verdict on the existing test: `test_orderbook_row_is_state_after_message` does test
   what it claims — I re-ran it against a deliberately 1-row-shifted orderbook and it failed
   (34/500). Two weaknesses: (a) it samples only 500 type-3 deletes, and (b) its sibling
   `test_inverted_direction_does_not_hold` asserts `assertLess(n_holds, n_checked)`, which passes
   if 499/500 samples hold — near-vacuous; the true inverted hold-rate is ~7%, so it can never
   fail in practice. The forward test carries all the load. Separately, `data/` is gitignored, so
   on a fresh clone both alignment tests `skipUnless` and `make test` prints OK with the phase's
   highest-risk invariant untested.
2. **Lookahead in features.** Not applicable at phase 0. `grep` for `.iloc[i+`, `shift(`,
   `rolling(`, `min_periods` over `src/edge/io/` returns nothing; there is no feature, label or
   window code in the phase-0 surface. (Phase-1 modules exist uncommitted in the tree but are
   out of scope here.)
3. **Scaler fit on the full sample.** Not applicable at phase 0. No `.fit(` anywhere in
   `src/edge/io/`; no normalization site exists yet.
4. **Label overlap across boundaries.** Not applicable at phase 0. No labels, no folds, no
   splitter exists.
5. **R2 against the test mean.** Not applicable at phase 0. No metric code exists; `RESULTS.md`
   is an empty template with zero rows filled.
6. **Base-rate artifact.** Not applicable at phase 0. No accuracy number exists.
7. **Deadband tuning.** Not applicable as such (no α exists) — **but the phase-0 analogue fired.**
   See the U-shape finding below: the parse-validity predicate was redefined after it failed on
   AMZN. `DECISIONS.md` 2026-07-30 logs both the original rule, its AMZN failure, and the
   replacement rule under which AMZN passes, same day, in that order.
8. **Dev-slice contamination.** Not applicable at phase 0. No fold was scored.
9. **Trial count.** Not applicable at phase 0. `PLAN.md` says "Trials logged so far: 0" and
   `RESULTS.md`'s trial log is empty; no model configuration has been run, so nothing is
   under-logged. (`git log` shows 8 commits, all infrastructure.)
10. **Pooled-number laundering.** Not applicable at phase 0. Nothing is pooled; both tickers are
    reported separately with full per-bin tables.

### Phase-0 substitutes for the standard experiments

- *Labels shuffled* → no labels. Substituted: **event-type relabelling** (1000 type-1 rows
  rewritten as type-6). The report prints `**MISMATCH**` on two rows and **`make data` exits 0**.
- *Features shifted +1 event* → substituted: **orderbook shifted +1 row** and **+10 rows**
  against the message file. Both produce `crossed=0 monotonic=True u_shape=True median_mid=$222.6550
  (bounds ok=True)` and **`make data` exits 0, PASS**. This is the headline caveat.
- *Purging disabled* → no splitter exists.
- *Phase-3 Gaussian-noise control* → no LM states exist.

---

## Invariant-by-invariant: HARD vs PRINTED

Method: copied the AMZN raw pair to `/tmp/redteam/work/`, mutated the copy, ran the real
(unmodified) `edge.io.lobster` then `edge.io.quality` against it, and recorded both exit codes.
`make data` fails iff either step is non-zero.

| # | invariant | mutation I injected | exit | verdict |
|---|---|---|---|---|
| 1 | crossed book (`bid_px_1 >= ask_px_1`) | one row's bid set above its ask | **1** | **HARD** |
| 1b | same, column-order check | bid/ask columns swapped (contract's ask-first claim inverted) | **1** | **HARD** — 269748 crossed. This is a genuine validation of the ask-first column order. |
| 2 | raw msg / raw book / parquet row equality | truncate raw book by 10 rows | **1** (adapter raises) | **HARD** |
| 2b | same, stale-parquet path | good parquet, then truncate both raw files by 5 | **1** (quality) | **HARD** |
| 3 | `ts_ns` non-decreasing | swap two message rows | **1** | **HARD** |
| 4 | event-type counts vs `DATA_CONTRACT.md` | 1000 type-1 → type-6 | **0** | PRINTED ONLY |
| 5 | median mid inside sanity bounds | all prices ÷100 → median mid $2.2266 | **0** | PRINTED ONLY (`bounds ok=False`) |
| 6 | U-shape | timestamps re-spread uniformly across the session | **0** | PRINTED ONLY (`u_shape=False`) |
| 7 | sentinel best-level rows | 1000 rows given `ask=9999999999, sz=0` | **0** | PRINTED ONLY — and the intraday spread column is poisoned to **5,706,503 ticks** in the open bin while the build still says PASS |
| 8 | locked book | — | n/a | printed; subsumed by #1 because crossed uses `>=` |
| 9 | gap counts, top-5 gaps, tie counts | — | n/a | informational, no threshold at all |
| **10** | **message↔orderbook row alignment** | **book shifted +1 row; book shifted +10 rows** | **0** | **NOT CHECKED. `make data` passes clean.** |
| 11 | per-row level monotonicity (`ask_px_1<ask_px_2<…`) | levels 2–10 reversed | **0** | NOT CHECKED (real data: 0 violations, I verified) |
| 12 | zero-size best level (`bid_sz_1==0`) | — | n/a | NOT CHECKED (real data: 0, I verified) |
| 13 | timestamps inside 09:30–16:00 | — | n/a | NOT CHECKED — `build_intraday_profile` **`np.clip`s** out-of-session stamps into the first/last bins |

**Item 10 is the finding.** `docs/DATA_CONTRACT.md` calls the alignment off-by-one "the single
highest-risk bug in the phase" — and the artifact whose job is to say "the parse is validated"
is completely blind to it. A shifted book preserves row counts, timestamps, event-type counts,
median mid, non-crossedness and the intraday profile, so every hard gate passes. Only
`make test` catches it, `make test` is not a prerequisite of `make data`, and `data/` is
gitignored so those tests skip on a fresh checkout and still report OK.

**Item 13 is subtler and worth a line.** The U-shape check is `PLAN.md`'s designated
parse-validity detector, and out-of-session timestamps — a canonical parse error — get silently
folded *into the two edge bins*, which is exactly the direction that makes `u_shape=True` more
likely. The detector is strengthened by the bug it is supposed to detect.
`tests/test_quality.py::test_out_of_range_stamps_clip_into_the_end_bins` enshrines this clipping
as intended behaviour rather than flagging it. No live impact here: I counted 0 clipped rows on
both tickers.

---

## crossed=0: recounted raw, no exclusions

The open question was whether the crossed-book test is computed over all rows or only some, and whether
sentinels/halts/pre-open rows are excluded in a way that could hide crossings.

**Current code applies no mask at all.** `crossed_locked_counts(bid_px_1, ask_px_1)` is
`np.sum(bid >= ask)` over the whole column. My own recount, iterating every line of both raw
orderbook CSVs with no filtering of any kind:

| ticker | rows scanned | crossed (`bid1>=ask1`) | locked (`bid1==ask1`) | sentinel rows | zero-size L1 rows | type-7 halts |
|---|---|---|---|---|---|---|
| AMZN | 269,748 | **0** | 0 | 0 | 0 | 0 |
| INTC | 624,040 | **0** | 0 | 0 | 0 | 0 |

Raw number: **zero crossings in 893,788 rows, nothing excluded.** The report's headline claim is
true and I could not dent it. I also checked deep-level monotonicity (ask strictly increasing,
bid strictly decreasing across all 10 levels): **0 violating rows on either ticker** — an
invariant the report does not check but which the data satisfies anyway.

**However — `git show 6692c9e:src/edge/io/quality.py` shows the *previous* version did exactly
what I suspected:**

```python
valid = _valid_quote_mask(bid_px_1, ask_px_1, bid_sz_1, ask_sz_1)
crossed_count = int(np.sum(valid & (bid_px_1 >= ask_px_1)))
```

That mask was removed in `7ad99b4`. So the shipped number is now unfiltered (good), but
`DECISIONS.md` still carries the entry asserting the mask exists — see below.

---

## The U-shape predicate: satisfiable by almost any right-skewed profile

The predicate is `mean(first quartile of bins + last quartile) > mean(interior)`. For 13 bins
that is `mean(bins 1-3, 11-13) > mean(bins 4-10)`. It is a one-sided edge-heaviness test, not a
U-shape test. Profiles I fed it (`/tmp/redteam/ushape.py`, calling the real `confirm_u_shape`):

| profile | `confirm_u_shape` |
|---|---|
| flat except a single spike at 10:30-11:00 — `[10,10,1000,10,10,…]` | **True** |
| **inverted U** (midday peak) with a large close bin | **True** |
| monotone all-day decline (open peak only, no close leg) | **True** |
| monotone all-day ramp `[1,2,…,12,20]` (no open leg at all) | **True** |
| flat | False |

So: yes, a profile that is obviously not U-shaped passes. Three separate ones do. Worse, this
predicate is *printed only* — my `flat_profile` mutation produced `u_shape=False` and `make data`
still exited 0. `PLAN.md` says "Confirm the U-shape. If it isn't there, the parse is wrong." The
implemented check cannot discharge that instruction and would not stop the build if it tried.

### Resolving the AMZN contradiction

`reports/data_quality.md` line 85: `U-shape (open/close activity > midday) confirmed: True`.
`docs/spec/PLAN.md` line 26-27: "AMZN's is weaker at the open — logged as an observation."

**PLAN.md is the accurate document. The data-quality report is the one that is wrong**, not
arithmetically but semantically: its parenthetical "(open/close activity > midday)" is a claim
about AMZN's open that the data contradicts.

AMZN, my recount:
- open bin 09:30–10:00 = **17,520** events. Interior-bin mean = **18,279**. Day mean per bin = **20,750**.
  The open half-hour is *below* both.
- First **5 minutes** = 1,806 events = **0.52× the day mean** — the quietest part of AMZN's morning.
  Last 5 minutes = 2.13×. (INTC for contrast: first 5 min = 3.46×, last = 2.33× — a real U.)
- `confirm_u_shape` returns True only because the "edge" quartile reaches to 11:00, sweeping in
  the 10:00–10:30 (24,960) and 10:30–11:00 (24,502) bins. Edges mean 23,632.8 vs interior 18,278.7.
  Strip the mid-morning bins and AMZN has no open leg. It is a J, not a U.

`DECISIONS.md` is worse than the report on this point. It states AMZN's open "is elevated but not
the day's peak" (it is *below* the day mean) and that the old rule "said no while the profile is
**visibly U-shaped**" (it is not). And the rule was changed the same day, immediately after
observing that AMZN failed it, to a rule under which AMZN passes. The change is honestly logged —
credit for that — but the recorded justification is contradicted by the numbers, and a
parse-validity detector that is redefined until it returns True on the data under test has no
remaining power to detect anything.

---

## `DECISIONS.md` describes code that is not there

`RULES.md` makes `DECISIONS.md` the auditable record of the study. Two of its phase-0 entries are
false of the shipped `quality.py`:

1. *"It hard-fails (`SystemExit`) if crossed-book count != 0 **or median mid falls outside the
   ticker's expected dollar range**."* — **False now.** In `6692c9e` median-mid *was* a
   `raise SystemExit`. In `7ad99b4` it became a printed `**FAIL**` string with exit 0. Proved:
   dividing all prices by 100 yields `median_mid=$2.2266 (bounds ok=False)` and `make data` **passes**.
2. *"Sentinel-safe 'valid quote' mask … a best-bid/ask pair only counts toward the crossed-book
   check and the spread/depth intraday profile if neither side is the LOBSTER sentinel … the check
   is real code now instead of a documented assumption."* — **False now.** `_valid_quote_mask` was
   deleted in `7ad99b4`. `sentinel_row_count` counts and prints; nothing is excluded. Proved:
   injecting 1000 sentinel ask rows leaves `crossed-book count: 0 -- PASS` and drives the open bin's
   mean spread to 5,706,503 ticks, exit 0.

The commit that did this is titled *"tighten data-quality checks"*. It genuinely tightened one
thing (row counts, previously adapter-internal, became a quality-level hard gate) — and that entry
is logged — but it also **demoted one hard gate to a warning and deleted a defensive mask**, and
neither loosening is recorded. The U-shape entry explicitly says "This supersedes the … line
above"; these two do not, so a reader auditing the protocol from `DECISIONS.md` will believe two
gates exist that do not.

Neither affects a printed number on this sample (median mid is in range; sentinel count is 0). This
is an audit-trail defect, not a data defect — hence CAVEATS, not KILLED.

---

## `tests/test_quality.py`: covers the helpers, not the invariants

`DECISIONS.md` records that the docstring "claimed these were unit tested and they were not" and
that 16 tests were added. The 16 tests are real and hand-computed, and they do cover every *pure*
primitive: `crossed_locked_counts`, `sentinel_row_count`, `check_monotonic_ts`, `top_k_gaps`,
`gaps_over_threshold`, `median_mid_dollars`, `ns_to_hhmmss`, `build_intraday_profile`,
`confirm_u_shape`.

**Untested — no test anywhere in the repo:**

- **`main()`'s exit code.** Nothing asserts that a crossed book, a row-count mismatch, or a
  non-monotone timestamp returns non-zero. The file's docstring headline — *"Hard invariants
  (non-zero exit, fails `make data`)"* — is the single load-bearing claim of the module and it has
  zero coverage. This is not hypothetical: the median-mid gate was demoted from `SystemExit` to a
  printed string between `6692c9e` and `7ad99b4` and **the suite stayed green**, because no test
  existed to notice. The same demotion could happen to `crossed` tomorrow, silently.
- **`build_ticker_report`'s `hard_failures` assembly** — the escalation policy itself.
- **`count_lines`** — the row-count invariant's own primitive.
- **`_event_count_table` mismatch detection** — the `**MISMATCH**` path is never exercised.
- **`ascii_bar`, `_gap_table`, `_intraday_table`, `write_report`** — no coverage.

**Weak among those that do exist:**

- `TestUShape.test_open_close_heavy_profile_passes` recomputes the implementation's own arithmetic
  in the test body (`assertGreater(sum(edges)/len(edges), sum(middle)/len(middle))`) and then
  asserts `confirm_u_shape` agrees. That is a tautology, not an independent expectation. It also
  uses a 12-bin list while the real profile is 13 bins.
- `test_flat_profile_fails` only covers the exactly-flat case, which fails on the strict `>`. There
  is no negative case pinning down a non-U profile that *should* fail — which is why the three
  counterexamples above went unnoticed.
- `test_out_of_range_stamps_clip_into_the_end_bins` locks in silent clipping of out-of-session
  timestamps as desired behaviour.
- `test_alignment.py::test_inverted_direction_does_not_hold` — `assertLess(n_holds, n_checked)`
  passes on 499/500. Cannot fail in practice.

---

## Arithmetic audit of `reports/data_quality.md`

Recomputed from raw CSVs in pure Python, no `edge` import (`/tmp/redteam/recount.py`,
`recount2.py`). **Every figure reproduces exactly.**

| check | AMZN | INTC | agrees with report |
|---|---|---|---|
| raw message lines | 269,748 | 624,040 | yes |
| raw orderbook lines | 269,748 | 624,040 | yes |
| parquet rows | 269,748 | 624,040 | yes |
| Σ event types 1–7 | 269,748 | 624,040 | **sums to row count exactly** |
| type 1 / 2 / 3 / 4 / 5 / 6 / 7 | 131954 / 2917 / 123458 / 8974 / 2445 / 0 / 0 | 304790 / 8028 / 278739 / 28924 / 3559 / 0 / 0 | yes, all 14 |
| crossed / locked / sentinel | 0 / 0 / 0 | 0 / 0 / 0 | yes |
| ts ties / strict decreases | 8,483 / 0 | 65,886 / 0 | yes |
| gaps > 1s | 5,202 | 3,600 | yes |
| top-5 gaps (s) | 12.281996, 10.598153, 10.576862, 10.429486, 9.839008 | 6.972331, 5.595666, 5.311282, 5.058452, 4.579354 | yes, to 6 dp, indices too |
| median mid | $222.6550 | $27.0250 | yes |
| Σ intraday bin counts | 269,748 | 624,040 | **sums to row count exactly**; 0 rows clipped |
| all 13 mean spreads (ticks) | match to 3 dp | match to 3 dp | yes |
| all 13 mean L1 depths | match to 0 dp | match to 0 dp | yes |

Session bounds: AMZN `ts` spans 09:30:00.017 → 15:59:59.959; INTC 09:30:00.006 → 15:59:59.948.
Both strictly inside 09:30:00–16:00:00. No pre-open or post-close rows exist, so the "excluded rows
could hide crossings" hypothesis has nothing to hide behind.

Cross-checks: `docs/DATA_CONTRACT.md`'s price ranges (INTC 26.62–27.60, AMZN 220.52–226.03) match my
recount (26.615–27.595, 220.52–226.025) to the stated rounding. The committed
`reports/data_quality.md` is **byte-identical** to what the current code regenerates — the report is
not stale relative to the code, despite `quality.py` being rewritten in the same commit. The
apparent inconsistency in the gap table (a 12.28s gap between wall-clocks 13s apart) is
`ns_to_hhmmss` truncating sub-seconds: 13:10:06.905 → 13:10:19.187. Not a defect.

Housekeeping: `data/raw/*.csv` and `data/raw/lobster/*.csv` are byte-identical duplicates
(md5-verified, ~258 MB wasted). `discover_pairs` does not recurse, so only `data/raw/lobster` is
read and there is no double-counting risk — but a future `RAW_DIR=data/raw` would silently process
the stray copies.

---

## Most likely remaining explanation that is not "the signal is real"

There is no signal at phase 0, so the phase-0 form of this question is: *what is the most likely
way "the parse is validated" is a claim the artifacts do not support?*

**Answer: `make data` validates the parse against corruptions that are easy to see and blind to the
one corruption that matters.** All three hard gates — crossed, row counts, monotonicity — are
properties of the message file and the orderbook file *considered separately*. Not one of them
constrains the *join*. I shifted the orderbook by one row and by ten rows against the message file
and `make data` printed `crossed=0 monotonic=True u_shape=True median_mid=$222.6550 (bounds ok=True)`
and exited 0 both times, and wrote a report identical in every PASS field to the committed one. The
committed `reports/data_quality.md` would look exactly the same if the alignment were wrong. The
alignment is in fact right — I verified it four independent ways at 100% over 1.05M messages — but
that fact is established by `tests/test_alignment.py`, not by the quality report, and the quality
report is what the milestone points at.

Second-most-likely: the U-shape line. It is the one check `PLAN.md` designates as the
parse-is-wrong detector, it was redefined after failing on AMZN, its replacement passes on
profiles that are monotone ramps and inverted-Us, and even the replacement is print-only. For AMZN
specifically, `confirmed: True` is carried entirely by the 10:00–11:00 bins; the actual open half-hour
is below the day mean.

## What I could not check and why

- **The LOBSTER convention against its own authority.** `data.lobsterdata.com/info/DataStructure.php`
  is unreachable from this sandbox, as `DATA_CONTRACT.md` and `test_alignment.py` both already
  record. I could not quote the vendor's worked example. What I did instead is stronger than a
  citation for this dataset: four independent structural signatures (type-1 adds, type-2/3
  reductions, type-4 executions, type-5 non-effects) all hold at exactly 100.00% at offset 0 and at
  4–25% at offsets ±1, on both tickers, over every message. If the convention were the other way,
  offset 0 could not be 100%. I regard the convention as established empirically regardless of the
  docs.
- **`orderflow-lm`'s `SPEC.md`**, cited in `DATA_CONTRACT.md` as independent confirmation, is
  outside this repo; I did not read it and the alignment conclusion does not depend on it.
- **Whether the 2012-06-21 free-sample files are unmodified vendor files.** No vendor checksum is
  available, so a corruption present in both the message and orderbook file consistently would be
  invisible to every check here, mine included. The event-type counts in `DATA_CONTRACT.md` were
  derived from these same files, so that cross-check is circular by construction — it verifies the
  adapter against the raw file, not the raw file against LOBSTER.
- **Generalisation beyond this sample.** Every "0" in the report (sentinels, halts, type-6, locked,
  crossed, zero-size levels, out-of-session rows) is a property of one ticker-day pair. The removal
  of `_valid_quote_mask` means the crossed check and the spread profile have no defence the first
  time a day with real sentinel rows arrives — I demonstrated the failure mode by injecting them
  (spread column reads 5.7M ticks, build still PASSes) but cannot say when it will bite.
- **Items 2–10 of the standard order of suspicion** have no phase-0 surface: no features, no
  scaler, no labels, no folds, no metrics, no trials. I state that as not-applicable, not as passing.
  Nothing in `src/edge/io/` reaches past a timestamp, calls `.fit(`, or computes a metric, and
  `RESULTS.md` contains no filled row.

### What would falsify my SURVIVED WITH CAVEATS if I am wrong

If the alignment convention were inverted, my offset-0 hold rate on type-1 adds would not be
131954/131954 — it would be near the 5% background rate, and offset −1 or +1 would be the one at
100%. It is not. If `crossed=0` were an artifact of exclusion, my unfiltered raw recount over
893,788 rows would have found crossings. It found none. If the report's numbers were fabricated or
stale, my independent recount would disagree somewhere in the ~90 figures I recomputed. It does not
disagree anywhere.
