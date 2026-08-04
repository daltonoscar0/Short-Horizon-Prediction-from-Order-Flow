# DATA CONTRACT — Edge

Frozen schemas for the LOBSTER adapter (`src/edge/io/lobster.py`). This is Phase 0 step 1:
the schemas below are the target the adapter must satisfy (`tests/test_lobster.py`,
`tests/test_alignment.py`); the adapter itself is written in a later step (tests-first, per
`RULES.md`).

## Raw input: LOBSTER message file

`{TICKER}_{YYYY-MM-DD}_{start_ns}_{end_ns}_message_{levels}.csv`, no header row, one row per
event, 6 comma-separated columns:

| col | name | type | notes |
|---|---|---|---|
| 1 | `time` | decimal string | seconds after midnight, ns-resolution fraction (e.g. `34200.005742728`). Parsed by splitting on `.` and combining as integers — **never** via `float64 * 1e9`, which can perturb ns ties. |
| 2 | `type` | int | event type, see table below |
| 3 | `order_id` | int | exchange order id; read for QC only, dropped from the output frame |
| 4 | `size` | int | shares; for reductions, the amount removed |
| 5 | `price` | int | units of 0.01 cents (`price / 10000` = dollars) |
| 6 | `direction` | int | `+1` = bid side, `-1` = ask side (side of the *standing* order, not the aggressor) |

Event types (col 2), counts observed in the 2012-06-21 free sample (10-level, full day):

| type | meaning | INTC count | AMZN count |
|---|---|---|---|
| 1 | new limit order (visible) | 304,790 | 131,954 |
| 2 | partial cancel (size reduction of a standing order) | 8,028 | 2,917 |
| 3 | full delete (total removal of a standing order) | 278,739 | 123,458 |
| 4 | execution of a visible limit order | 28,924 | 8,974 |
| 5 | execution of a hidden order (book unaffected) | 3,559 | 2,445 |
| 6 | cross/auction trade (book unaffected) | 0 | 0 |
| 7 | trading halt indicator (book unaffected) | 0 | 0 |

Neither sample file contains a type-6 or type-7 row; the data-quality report still counts
them (should print 0/0 for this universe, not "absent").

## Raw input: LOBSTER orderbook file

`{TICKER}_{YYYY-MM-DD}_{start_ns}_{end_ns}_orderbook_{levels}.csv`, no header row, exactly
as many rows as the paired message file, `4 * levels` columns, **interleaved per level, ask
first**:

```
ask_px_1, ask_sz_1, bid_px_1, bid_sz_1, ask_px_2, ask_sz_2, bid_px_2, bid_sz_2, ...
```

Verified against row 0 of the INTC file: col0 (`275200`) > col2 (`275100`) — ask before bid,
consistent with the column order above. All price/size fields are integers; prices are the
same 0.01-cent units as the message file.

**Sentinel levels.** LOBSTER's documented convention for an occupied-level shortfall is
ask `9999999999` / bid `-9999999999` with size `0`. Checked (`grep -c` over the full raw
files): **zero occurrences of either sentinel in the INTC or AMZN 10-level full-day files**
— the visible book is 10-levels-deep on both sides for the entire trading day in this
sample. The adapter must still handle the sentinel defensively (treat `size == 0` or the
sentinel price as "no level"), since this is a property of this specific sample, not a
guarantee of the format.

## Alignment convention (the phase-0 invariant)

**Row *k* of the orderbook file is the book state immediately AFTER message *k* of the
paired message file** (1-indexed both; 0-indexed in code, so array index `i` of the parsed
table pairs message `i` with post-state row `i`).

Convention authority: sourced to the primary spec directly, not inferred from an empirical
check alone. Two hosts are involved and both are dead *live*: `data.lobsterdata.com` does
not resolve (dead subdomain), and the sibling live page
`lobsterdata.com/info/DataStructure.php` serves only a 457-byte SPA shell with no content.
A cached mirror of the latter *is* reachable, though — the Wayback Machine snapshot
(https://web.archive.org/web/20240427184519/https://lobsterdata.com/info/DataStructure.php,
HTTP 200, 30,950 bytes, captured 2024-04-27) has the full page, and states verbatim:

> "The k-th row in the 'message' file describes the limit order event causing the change in
> the limit order book from line k-1 to line k in the 'orderbook' file."

— i.e. message `k` causes the change from orderbook row `k-1` to orderbook row `k`, so row
`k` is the state *after* message `k`: exactly the convention above. The same page gives a
worked example, reproduced here for the record: the type-3 delete on line 2 of its sample
message file removes 100 shares from the ask side at price 118600; the paired orderbook
file's line 1 -> 2 shows the level-1 ask volume dropping from 9484 to 9384 by exactly that
amount.

1. **Primary source**: the quoted sentence and worked example above (Wayback snapshot,
   2024-04-27, as cited).
2. **Empirical verification**, done independently before any adapter code was written, as
   the executable check on this repo's actual data rather than the spec's toy rows: for a
   sampled type-3 (full delete) message `k` with `size = S` at `(side, price)`, the
   displayed size at that `(side, price)` in orderbook row `k-1` exceeds row `k` by exactly
   `S` (or the level disappears entirely from row `k`, when the delete empties it). At
   seed 0: the spec-direction comparison (row `k-1` vs row `k`) held on 500/500 sampled
   type-3 deletes on each of INTC and AMZN (100%); the inverted comparison (row `k` vs row
   `k+1`) held on only 110/2000 (5.5%) for INTC and 130/2000 (6.5%) for AMZN. A nonzero
   inverted rate is expected — a level can coincidentally shrink by the same size across the
   *next*, unrelated event too — which is why the inverted check uses a threshold rather
   than asserting exactly 0; the ~100%-vs-~6% separation is what pins the convention down,
   not the exact inverted rate. `tests/test_alignment.py` re-runs the spec-direction check
   at n=500+ per ticker against the live raw files as the executable version of this check
   (`test_orderbook_row_is_state_after_message`), plus an explicit assertion that the
   inverted direction does *not* hold anywhere near as often on the same samples.
3. **Independent confirmation**: `orderflow-lm`'s `SPEC.md` (Phase 2 tokenizer section)
   states the identical convention for the same data — "pre-event book = orderbook file row
   `i-1`" — derived independently for that project's tokenizer design.

An off-by-one here silently inverts every book-state feature (spread, imbalance, microprice
computed one event too early or too late), so this is treated as the single highest-risk bug
in the phase and is pinned down before any feature code is written.

## Price scaling

Raw prices are integers in units of **0.01 cents**; dollars = `price / 10000`. Verified
during this pass (median mid = `(ask_px_1 + bid_px_1) / 2 / 10000`, full trading day):

| ticker | median mid ($) | range ($) | expected range |
|---|---|---|---|
| INTC | 27.02 | 26.62 – 27.60 | [20, 35] |
| AMZN | 222.66 | 220.52 – 226.03 | [200, 250] |

Both inside the sanity bounds the phase-0 spec sets — confirms `/10000` is the
correct scaling, not `/100` or `/1000`. The output frame keeps prices in raw integer units
(int64); the `/10000` conversion to dollars is a display/feature-time operation, done
downstream, never baked into the stored frame.

## Output: event-indexed frame

One row per raw message/orderbook-row pair, one Parquet file per ticker-day at
`data/frames/{TICKER}_{YYYY-MM-DD}.parquet`:

| column | type | source |
|---|---|---|
| `ts_ns` | int64 | message `time`, parsed to nanoseconds-after-midnight (string split, no float64) |
| `event_type` | int8 | message `type`, passthrough |
| `direction` | int8 | message `direction`, passthrough (`+1` bid / `-1` ask) |
| `price` | int64 | message `price`, passthrough, raw 0.01-cent units |
| `size` | int64 | message `size`, passthrough |
| `bid_px_1` … `bid_px_L` | int64 | orderbook row, same index as the message, bid side, best to worst |
| `bid_sz_1` … `bid_sz_L` | int64 | orderbook row, bid sizes |
| `ask_px_1` … `ask_px_L` | int64 | orderbook row, ask side, best to worst |
| `ask_sz_1` … `ask_sz_L` | int64 | orderbook row, ask sizes |

Column order is exactly `ts_ns, event_type, direction, price, size, bid_px[1..L],
bid_sz[1..L], ask_px[1..L], ask_sz[1..L]` (block-grouped by side/quantity, *not* the raw
file's interleaved-per-level order — the adapter reorders on the way out). `order_id` is
read from the raw message file for adapter-internal use only and is **not** part of the
output schema. `L = 10` for this universe (free-sample depth).

## Adapter API (target for the next implementation step)

`src/edge/io/lobster.py`:

- `read_message_file(path: str) -> pyarrow.Table` — columns `ts_ns, event_type, order_id,
  size, price, direction`.
- `read_orderbook_file(path: str, num_levels: int) -> pyarrow.Table` — columns
  `ask_px_i, ask_sz_i, bid_px_i, bid_sz_i` for `i` in `1..num_levels` (raw interleaved
  names; column *order* in the returned table is not load-bearing, only the names are).
- `build_event_frame(message_path: str, orderbook_path: str, num_levels: int) ->
  pyarrow.Table` — joins the two by row position (validates equal row counts) and reorders
  into the frozen output schema above.
- CLI: `python3 -m edge.io.lobster --raw RAW_DIR --out OUT_DIR --seed SEED` — discovers
  `{message,orderbook}` file pairs under `RAW_DIR` by filename pattern, writes one parquet
  per ticker-day to `OUT_DIR`. `--seed` is logged even though this step is deterministic
  (CSV → parquet has no randomness) — every entry point takes `--seed` per `RULES.md`.

## What this contract does not cover yet

- Data-quality report (`reports/data_quality.md`): crossed-book count, timestamp
  monotonicity/ties, event-rate and spread intraday profiles, gap detection — produced by
  `make data` once the adapter exists, not part of this schema-freeze step.
- `PLAN.md` status update — deferred to the step that lands a green `make test` and a
  committed `reports/data_quality.md`.
