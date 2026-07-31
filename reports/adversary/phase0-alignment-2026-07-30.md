VERDICT: SURVIVED WITH CAVEATS

Target: the phase-0 alignment invariant ("orderbook row k is the book state AFTER message k")
and `tests/test_alignment.py`. Adversary run 2026-07-30, fresh context, read-only on `src/`,
`tests/`, `RESULTS.md`. All numbers below were produced by throwaway scripts in `/tmp/edge_adv/`
that share no code with the test under attack.

## What I tried and what happened

1. **Row alignment (the only live phase-0 surface).** Attacked from five independent angles;
   the convention is CORRECT and I could not break it. Detail in the sections below.
2. **Lookahead in features.** Not applicable at phase 0 — no feature code is committed
   (`src/edge/features/` exists only as uncommitted phase-1 work and is out of scope for this
   report). Nothing in `src/edge/io/` indexes forward: `build_event_frame` is a positional
   column copy, `quality.py` reads only backward differences. Stated as N/A, not as a pass.
3. **Scaler fit on the full sample.** N/A at phase 0 — no scaler, no `.fit(` anywhere under
   `src/edge/io/`. There is no model and no normalization in the committed phase-0 code.
4. **Label overlap across boundaries.** N/A — no labels, no horizon, no splitter exists yet.
5. **R2 against the test mean.** N/A — no metric code, `RESULTS.md` headline table is empty.
6. **Base-rate artifact.** N/A — no accuracy number is claimed anywhere in phase 0.
7. **Deadband tuning.** N/A — no deadband, `RESULTS.md` frozen-decisions row for alpha is blank.
8. **Dev-slice contamination.** N/A — no folds, no dev slice. Both ticker-days were used for
   parser verification, which is the correct place to spend them; nothing was scored.
9. **Trial count.** N/A — the trial log is empty and no configuration has been run.
10. **Pooled-number laundering.** N/A — no pooled number. The only per-ticker numbers
    (quality report) are reported per ticker already, not pooled.

## Item 1 in full: what I did to the alignment invariant

### 1a. I reached the authoritative spec the phase said was unreachable — and it agrees

`DECISIONS.md` (2026-07-30, alignment entry) says the lobsterdata.com DataStructure page is
unreachable and "archive blocked"; `docs/DATA_CONTRACT.md` line 68 says "no cached mirror
reachable either". Both statements are false from this machine. One command:

    curl -sSL "https://web.archive.org/web/2023/https://lobsterdata.com/info/DataStructure.php"

returns HTTP 200, 30,948 bytes, snapshot 2024-04-27, the full page. It states verbatim:

> "The k-th row in the 'message' file describes the limit order event causing the change in
> the limit order book from line k-1 to line k in the 'orderbook' file."

plus the worked example (a type-3 delete on message line 2 drops best-ask volume from 9484 to
9384 between orderbook lines 1 and 2). That is exactly the convention the phase adopted. The
same page independently confirms every other frozen claim in `DATA_CONTRACT.md`: the
interleaved `Ask Price 1, Ask Size 1, Bid Price 1, Bid Size 1, Ask Price 2, ...` layout; price
= dollars x 10000; direction -1 = sell limit / +1 = buy limit (side of the standing order);
sentinels ask 9999999999 / bid -9999999999 with volume 0; event types 1-7; and "N is the number
of events in the requested price range", which is why every type-3 delete in a 10-level file
lands inside the displayed window.

So the convention is right and the provenance chain is now first-party rather than
"orderflow-lm said so". But the audit trail contains a false statement about a verification
step that takes one curl. That is the single most damaging thing I found, and it is a
documentation defect, not a wrong number.

### 1b. Five independent invariants, full population, both tickers — no exceptions

My probe (`/tmp/edge_adv/align.py`) reuses nothing from `tests/test_alignment.py`. For every
message k it checks the message's own effect at its own price on its own side, comparing
row pair (k-1, k) for the "after" hypothesis and (k, k+1) for the "before" hypothesis:

| type | invariant | INTC "after" | AMZN "after" | INTC "before" | AMZN "before" |
|---|---|---|---|---|---|
| 1 new limit | size(k) - size(k-1) = +msg_size | 304790/304790 (100.0000%) | 131954/131954 (100.0000%) | 4.33% | 4.97% |
| 2 partial cancel | ... = -msg_size | 8028/8028 (100.0000%) | 2917/2917 (100.0000%) | 8.04% | 31.31% |
| 3 full delete | ... = -msg_size | 278739/278739 (100.0000%) | 123458/123458 (100.0000%) | 5.15% | 6.95% |
| 4 visible execution | ... = -msg_size | 28923/28923 (100.0000%) | 8974/8974 (100.0000%) | 22.31% | 8.22% |
| 5 hidden execution | entire 40-column row unchanged | 3559/3559 (100.0000%) | 2444/2444 (100.0000%) | 23.15% | 24.54% |

(28923 not 28924 for INTC type 4 because message 0 is a type-4 and has no row k-1.)

Sixth invariant, `/tmp/edge_adv/others.py`: for every type-1..4 message, **every displayed price
other than the message price** must carry identical size in rows k-1 and k. Result: **0
mismatches out of 11,789,120 level comparisons (INTC) and 5,078,757 (AMZN)**. Under a one-row
shift the same check produces 3.04% / 2.63% mismatches. This is the check that would catch a
level permutation or a column mis-parse, and the type-3-only test cannot see it.

### 1c. Negative controls, so 100% is not trivially achievable

- Consecutive book rows differ on **100.00%** of type-1..4 events (620,480 / 267,303). The
  identity is not being satisfied by a book that never moves.
- Random-pairing control (message k against a randomly chosen row pair (r-1, r), seed 0): the
  identity holds on **0.0475%** (INTC) / **0.0363%** (AMZN). The observed 100% sits ~2000x
  above chance. This is the phase-0 analogue of the label-shuffle control.

### 1d. Deliberate off-by-one injection — my probes catch it

`/tmp/edge_adv/shift.py` pairs message k with book row k+s and re-runs everything:

| shift | INTC type-3 "after" hold rate | AMZN type-3 "after" hold rate |
|---|---|---|
| 0 (clean) | 100.0000% | 100.0000% |
| +1 | 5.1463% | 6.9535% |
| -1 | 5.2265% | 7.1867% |
| +2 | 11.7831% | 2.5480% |

Consistency check that the harness is not fooling me: at shift -1 the *inverted* ("before")
direction returns 100.0000% on all five types, exactly as it must if shifting the book back by
one row makes k -> k+1 the true k-1 -> k comparison. The probes detect the bug they exist to
catch, in both directions.

### 1e. Is the committed test's construction sound? Yes — and narrower than it looks

I answered the specific traps rather than assuming them:

- **Duplicate price levels.** `_level_size` returns the first matching level. I checked: **0
  rows** out of 624,040 (INTC) and 269,748 (AMZN) contain a duplicated price on either side,
  and ask prices are strictly ascending / bid prices strictly descending across levels 1..10 in
  every row. The first-match scan is unambiguous on this data.
- **Levels pushed out of the 10-level window.** Handled correctly: prev_size = S, next_size = 0,
  identity holds iff the delete emptied the level, which is the definition of type 3. My
  unrestricted (no in-window filter) counts are also 100%, so no delete is silently dropped.
- **"Can a level absent from BOTH rows silently pass?"** No — and this is the opposite of a
  weakness. `0 - 0 == msg_size` requires `msg_size == 0`, and the minimum message size in both
  files is **1**. Absent-from-both therefore *fails*, making the 0-default *stricter*. I proved
  this by swapping the bid/ask column blocks: all 500 samples became 0-0 lookups and the primary
  test went to **0/500 FAIL**. The latent trap is only if the type filter is ever widened to
  include type 7 (halt), whose spec-defined size is 0 — then `0 - 0 == 0` would vacuously hold.
  No type-7 rows exist in this universe.
- **Can it pass while the parse is off by one?** Not for a *uniform* shift: measured hold rate
  under any shift is 4-8% for type 3, so P(500/500) is astronomically small. The primary
  `assertEqual(n_holds, n_checked)` does all the real work and it does it correctly.

### 1f. The inverted-direction test is very close to a tautology

`assertLess(n_holds, n_checked)` with `n_checked == 500` on both tickers **passes for any
n_holds from 0 to 499**. In words: the inverted-hold rate could be **99.8%** and the test would
still be green. It fails only on a perfect 500/500. Realized values: **34/500 (INTC), 33/500
(AMZN)** — nowhere near the threshold, so the number is fine, but the assertion carries almost
no information. I demonstrated its emptiness directly: under a bid/ask column swap the primary
test fails 0/500 while the inverted test still **passes** (0/500 < 500). A threshold such as
`assertLess(n_holds, 0.25 * n_checked)` would be the assertion the docstring implies.

### 1g. What the committed test cannot see (measured, not asserted)

- **Depth-order permutation.** I re-ran the test's exact construction against a book with levels
  1..10 reversed, and against a book with levels 1 and 2 swapped: **500/500 PASS in both cases,
  both tickers.** `_level_size` scans all ten levels for a price match, so level ordering is
  invisible to it. A level-1/2 swap is separately caught by the 2-level toy fixture in
  `tests/test_lobster.py`; a permutation among levels 3..10 of the real files is caught by
  nothing in the suite. (The real parse is correct — I verified strict monotonicity of level
  prices independently.)
- **Localized misalignment.** 500 samples from 278,739 type-3 events gives 95% power only
  against corrupted fractions >= **0.597%**. A contiguous corrupted block of 1,000 type-3 events
  slips through with probability **0.166** on INTC (0.017 on AMZN, which has fewer events).
- **Event types other than 3.** Types 1, 2, 4, 5 are 100% untested by the alignment file. My
  probe covers them; the committed test does not.
- **The artifact downstream code actually reads.** `test_alignment.py` exercises
  `read_message_file` / `read_orderbook_file`, never `build_event_frame`, and
  `TestBuildEventFrame.test_schema_and_join` asserts column *names* and row count only — zero
  value assertions. I closed this myself (`/tmp/edge_adv/frame_check.py`): the committed
  `data/frames/{INTC,AMZN}_2012-06-21.parquet` match the raw CSVs with **0 mismatches across all
  40 book columns, event_type, direction, price, size and ts_ns**, ts_ns non-decreasing, row
  counts equal. The frames are correct; the test suite is not what establishes that.
- **A green build that proves nothing.** `data/` is gitignored and the class is wrapped in
  `skipUnless(_raw_files_present())`. I ran `EDGE_RAW_DIR=/nonexistent make test`-equivalent:
  **"Ran 68 tests ... OK (skipped=2)"**. On a fresh clone the phase-0 headline invariant is
  silently not enforced and the build is still green.

### 1h. A claim in the decision log that I falsified

`DECISIONS.md` (2026-07-30): "replay-based validation is structurally impossible on this data."
Multi-step cumulative replay is indeed impossible on level-censored samples — I do not dispute
the 90.3% best-bid disagreement figure quoted there. But **single-step** replay validation is
entirely possible and is strictly stronger than the type-3 test: it is what sections 1b/1c above
are, run on 887,783 events across both tickers with zero exceptions. The blanket wording in the
decision log is what justified a test covering one event type instead of five.

## Cross-checks outside the stated target

Because "a quality number that is false" is a kill criterion, I recomputed
`reports/data_quality.md` from the raw CSVs independently. Every number reproduces exactly:
row counts (624040 / 269748), all seven event-type counts per ticker, crossed = 0, locked = 0,
sentinels = 0, ts ties (65886 / 8483), strict decreases = 0, median mid ($27.0250 / $222.6550),
gaps > 1s (3600 / 5202), and the top-5 gap table to the microsecond including indices and
wall-clock stamps. The one discrepancy was mine: "mean L1 depth (shares)" is bid+ask summed,
not averaged — documented as such in `quality.py` line 174. No false quality number.

Process checks: `tests/test_alignment.py` was committed in c8315a5 (20:31) as a failing test,
`src/edge/io/lobster.py` in c73710b (20:50), and the test file is byte-identical to its first
commit with no working-tree modifications — it was not weakened after seeing the implementation.
`tests/test_quality.py` (7ad99b4) landed *after* `quality.py` (6692c9e), a minor deviation from
CLAUDE.md rule 2 if that rule is read as covering more than the CV harness.

## Most likely remaining explanation that is not "the signal is real"

There is no signal at phase 0, so the honest form of this question is: what would make the
alignment invariant wrong anyway? The one surviving story is **a defect that is uniform across
the whole file and therefore invisible to any per-event differencing test, mine included**.
Every check I ran — and every check in the repo — is a *relative* check: it confirms that
row k minus row k-1 equals the effect of message k. If the vendor's export applied a constant
transformation to both files together (for instance, both files truncated by the same k rows at
the start, or the message and orderbook exports being consistent with each other but both
shifted relative to true exchange time), all my invariants hold at 100% and the study still
inherits a systematic offset. The only defence against that is the vendor spec text plus an
external cross-reference (e.g. an independent NASDAQ ITCH reconstruction of the same
ticker-day), and only the first half of that is now in place. Given the spec text I retrieved,
plus the worked example matching the observed file structure exactly, I rate this unlikely — but
it is the class of error that survives everything done so far.

## What I could not check and why

- **The live lobsterdata.com page.** `https://lobsterdata.com/info/DataStructure.php` returns
  HTTP 200 with a 457-byte SPA shell; `data.lobsterdata.com` and `www.lobsterdata.com` do not
  resolve. The phase's description of the live site is accurate. My quotation is from the
  Wayback snapshot of 2024-04-27, which is a mirror, not the live page — I cannot rule out that
  the live page has since changed, though a change to the k-th-row sentence would be a change to
  the format itself.
- **Whether the implementing agent's environment could actually reach web.archive.org.** I only
  know that plain `curl` reaches it from this machine today. If the agent's fetch tool was
  sandboxed differently, "archive blocked" may have been true of its tooling and false of the
  machine. The defect is that the claim was recorded as a property of the environment without
  the command that produced it.
- **orderflow-lm's `SPEC.md`.** The repository is not present under this working directory, so I
  could not verify the quoted sentence "pre-event book = orderbook file row i-1". This no longer
  matters: the primary source now says the same thing. Note also that the quoted phrasing
  ("pre-event book = row i-1") and the Edge phrasing ("row k = state after message k") are the
  same statement, so there is no contradiction to resolve.
- **Cross-vendor validation.** No independent reconstruction of INTC/AMZN 2012-06-21 from raw
  ITCH is available here, so the uniform-offset story in the section above stays open.
- **Ticker-days beyond 2012-06-21 INTC/AMZN.** Everything above is verified on exactly two
  ticker-days. `discover_pairs` will happily ingest a 50-level file whose `_level_size`
  first-match assumption I have not tested against sentinel-padded rows, because zero sentinel
  rows exist in this universe.

## Falsifiers, if this were wrong

If the alignment convention were inverted, the type-1/2/3/4/5 hold rates in section 1b would sit
at 4-31% in the "after" column and 100% in the "before" column. They do the opposite, on
887,783 events, on two tickers, on five event types, with 16.87M supporting level comparisons
and a 0.04% random-pairing floor. If the parquet frames were mis-joined, `frame_check.py` would
report non-zero mismatches; it reports zero on all 45 columns.
