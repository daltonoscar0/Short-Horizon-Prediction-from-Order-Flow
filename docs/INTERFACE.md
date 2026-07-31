# INTERFACE — the frozen family-module contract

The contract between the `edge.features` infrastructure (`frame.py`, `book.py`, `clocks.py`,
`windows.py`, `normalize.py`) and the six family modules named in `FAMILY_PREFIXES`
(`ofi`, `queue`, `microprice`, `trade`, `state`, `labels`).

Those six are written **in parallel by agents that never talk to each other**, and their
outputs are merged into one table. Anything left ambiguous here becomes six incompatible
guesses and a silent column collision. So this file is normative: where it disagrees with a
module docstring, this file wins, and the module docstring is the bug.

Formulas live in `docs/spec/FEATURES.md`. The frame schema and the alignment invariant live
in `docs/DATA_CONTRACT.md`. Evaluation rules live in `docs/spec/EVAL_PROTOCOL.md`. This file
says only how the pieces plug together.

Cited by the module docstrings of `edge/features/__init__.py`, `book.py`, `clocks.py` and
`windows.py`. Structural invariants asserted mechanically in `tests/test_interface.py`.

---

## 1. Units

| quantity | representation |
|---|---|
| price | raw LOBSTER integer, units of 0.01 cents (`price / PRICE_SCALE` = dollars) |
| tick | `TICK_SIZE_RAW = 100` raw units = $0.01, both tickers, whole session |
| size | shares, integer |
| time | `ts_ns`, int64 nanoseconds after midnight |

**Compute in raw units, report price quantities in ticks.** Raw integers keep every
intermediate exact (no float drift on a 2.7e5 price); ticks make the number comparable across
INTC and AMZN, which is the whole point of carrying a large-tick and a small-tick name.

The only sanctioned conversion is `edge.features.book.ticks(x)` (`x / TICK_SIZE_RAW`), never a
local `/ 100`. Any emitted column whose value is a price *difference* or a price *level* is in
ticks and its docstring says so. Sizes stay in shares; do not normalise them to lots.

Session bounds (`SESSION_START_NS`, `SESSION_END_NS`), the lookback ladder (`WINDOWS_NS`,
`FEATURE_WINDOWS`), the label horizons (`LABEL_HORIZONS`, `LABEL_HORIZONS_EVENTS`), the
time-of-day bucket (`TOD_BUCKET_NS`) and the refit block (`REFIT_BLOCK_NS`) all live in
`edge/features/__init__.py`. Import them; never re-declare a magic number.

## 2. Causality — the `<= t` convention

**A feature at stamp time `t` may read event rows with `ts_ns <= t` and nothing else.**

It is `<=`, not `<`, because orderbook row `k` is the book state *after* message `k`
(`docs/DATA_CONTRACT.md`, alignment invariant): an event stamped exactly at `t` has already
printed by `t`, so its post-state is public at `t`. Ties are included, and a stamp landing
strictly between two events resolves to the **earlier** one.

Consequences, none of them optional:

- `stamps.index[i]` is that last row. A family may slice `frame[start : stamps.index[i] + 1]`
  and no further. Reading `stamps.index[i] + 1` is a one-row lookahead and **will not raise** —
  it will just quietly inflate your IC.
- A trailing window of lookback `w` is the half-open interval `(t - w, t]`. An event exactly
  `w` old is **excluded**. This is what `windows.trailing_start_index` implements
  (`searchsorted(..., side="right")`); do not roll your own boundary.
- The interval between consecutive stamps is likewise `(previous stamp, this stamp]`
  (`windows.between_stamps_bounds`); the first stamp anchors at frame row 0.
- **`label_` is the only forward-looking prefix.** Columns under it read `ts_ns > t` by
  construction; that is their job. Every other column in the merged table is causal, and a
  single `grep label_` finds every exception with no judgement call.

Two places are deliberately **stricter** than `<= t`:

- `normalize.rolling_zscore` baselines on `[t - window, t)` in *stamp positions*, excluding
  `x[t]` itself. A feature's current draw must not contaminate its own baseline, or a single
  outlier both is and defines "extreme". `x[t]` is legal information; it is just not allowed
  to be its own reference point.
- Any **fitted** statistic is estimated on frame rows *strictly* before the block it is applied
  to — sec. 8.

## 3. Stamps and clocks

```python
Stamps = namedtuple("Stamps", ["index", "ts_ns"])   # edge.features.clocks
```

- `ts_ns[i]` — int64, the decision time of sample `i`. Strictly increasing.
- `index[i]` — int64, position in `frame` of the **last** event with `frame.ts_ns <= ts_ns[i]`.
  Non-decreasing, always `>= 0`.
- Same length. `len(stamps.index)` is the row count of everything a family returns.

```python
stamps = stamps_for_clock(frame, "calendar")          # 100ms grid, CALENDAR_GRID_NS
stamps = stamps_for_clock(frame, "event")             # every EVENT_CLOCK_N = 10 book-changing events
```

`calendar_stamps` / `event_stamps` are the direct builders; `stamps_for_clock` is the frozen
dispatcher and is what the assembly stage calls. Book-changing types are `(1, 2, 3, 4)`; types
5/6/7 leave the displayed book untouched and never tick the event clock.

Sizes on 2012-06-21, for your performance budget:

| ticker | events | calendar stamps | event stamps (N=10) |
|---|---|---|---|
| INTC | 624,040 | 234,000 | 62,048 |
| AMZN | 269,748 | 234,000 | 26,730 |

**Family code must not branch on which clock it was handed.** If a feature only makes sense on
one clock, it is the wrong feature. `stamps` is the only thing that differs.

**Grid points before the first event are dropped, not sentinel-indexed** — 234,000 stamps from
234,001 grid points on both tickers (09:30:00.000 is dropped). `searchsorted(...) - 1` leaves
`-1`, which numpy reads as the *last* row of the frame: a lookahead bug that does not raise. So
the array is shorter instead.

Therefore: **stamp arrays align by position only within one clock and one ticker-day.** Never
assume the calendar clock has `calendar_grid_stamp_count()` entries, never zip a calendar array
against an event-time array, never join two feature tables on row number. Join on `ts_ns`.

An empty `stamps` is legal and must not raise: return the same keys with length-0 float64
arrays.

## 4. The book spine — `edge.features.book`

Best-quote extraction, the weighted-mid crossing and the validity rule are defined **once**,
here. No family re-derives them.

```python
book = book_at_stamps(frame, stamps)     # one row per stamp   <- normally handed to you
book = book_at_events(frame)             # one row per frame event, identical fields
```

`BookAtStamps` fields, all length `n` (`len(stamps.index)`, or `frame.n_events`):

| field | dtype | meaning |
|---|---|---|
| `bid_px`, `bid_sz`, `ask_px`, `ask_sz` | int64 `(n,)` | touch, raw units, verbatim from the frame |
| `bid_px_l`, `bid_sz_l`, `ask_px_l`, `ask_sz_l` | int64 `(n, n_levels)` | levels 1..L, **column 0 is the touch**, best to worst |
| `mid` | float64 `(n,)` | `(bid_px + ask_px) / 2`, raw units |
| `wmid` | float64 `(n,)` | `(bid_px*ask_sz + ask_px*bid_sz) / (ask_sz + bid_sz)` — **crossed weights** |
| `spread` | float64 `(n,)` | `ask_px - bid_px`, raw units |
| `spread_ticks` | float64 `(n,)` | `spread / TICK_SIZE_RAW` |
| `valid` | bool `(n,)` | touch usable on both sides |
| `valid_l` | bool `(n, n_levels)` | same test per level; `valid == valid_l[:, 0]` |
| `n_levels` | int | `min(NUM_LEVELS, levels in the frame)` — **10 on the real frames, 2 on the toy fixture** |

Iterate levels over `book.n_levels`. A hardcoded `10` turns every toy-book test into an
`IndexError`, which is exactly the bug the toy fixture is there to catch.

**The crossed weights.** The bid price carries the **ask** size (`FEATURES.md` sec. 3). A heavy
ask queue pulls `wmid` *toward the bid*, because that is where the resting sell pressure is.
Uncrossing them produces a number of the right magnitude in the wrong direction, which survives
every sanity check and inverts the sign of every microprice deviation feature. Pinned by
`tests/test_book.py::TestWeightedMidCrossedWeights`.

**Validity.** A level is valid iff both sides show a real price and a positive size:
`bid_px != BID_SENTINEL_PRICE and ask_px != ASK_SENTINEL_PRICE and bid_sz > 0 and ask_sz > 0`.
`mid`, `wmid`, `spread`, `spread_ticks` are **NaN wherever `valid` is False**; the raw integer
columns pass through unmasked so a family can see what LOBSTER actually printed. A sentinel
price is finite arithmetic — `(100000 + 9999999999)/2` is a mid of ~$500,000 that no magnitude
check catches once it has been averaged into something. **Mask with `valid`/`valid_l` before
aggregating any raw integer column.** There are zero invalid rows in this sample; that is a
property of this sample, not of the format.

Crossed books (`bid_px >= ask_px`) are a hard `make data` failure and cannot reach here.

Other exports:

- `ticks(x)` — the only sanctioned raw-to-ticks conversion.
- `event_frame_arrays(frame) -> EventArrays(ts_ns, event_type, direction, price, size)` — message
  columns as **int64 copies**. int64 because under NEP 50 an int8 array times a Python int stays
  int8 and wraps silently (`direction * 100` overflows), corrupting any signed-volume aggregate.
  Copies, so a family cannot mutate the shared frame through them.
- `frame_num_levels(frame)`, capped at `NUM_LEVELS`.

**Per-event work without a second full book.** `book_at_events(frame)` materialises `4 * L`
int64 columns (~200 MB on INTC at L=10); six families each building one is not free. For
touch-level per-event work use the `EventFrame`'s own cached, read-only properties instead —
they are built once and shared across every family that touches the same frame object:
`frame.best_bid_px`, `best_bid_sz`, `best_ask_px`, `best_ask_sz`, `frame.mid`, `frame.wmid`,
`frame.spread` (all float64 `(n_events,)`, NaN at absent levels) and `frame.bid_px_f`,
`bid_sz_f`, `ask_px_f`, `ask_sz_f` (float64 `(n_events, L)`). One difference to be deliberate
about: the frame's `*_f` masks are **per side**, `book.valid_l` is **joint** (both sides at that
level). Pick the one your formula means.

## 5. The `compute()` contract

Every family module exposes exactly one entry point, with exactly this signature:

```python
def compute(frame, stamps, book=None) -> dict[str, np.ndarray]:
    """One line, plus the FEATURES.md section / paper equation the formulas come from."""
    if book is None:                                  # standalone/test convenience only
        book = book_at_stamps(frame, stamps)
    ...
```

No extra parameters, no keyword-only extras, no config object. Everything tunable is a frozen
constant in `edge/features/__init__.py`. The assembly stage is then a three-line loop over the
six modules, which is the point.

`book` **is** passed in by the assembly stage: `book_at_stamps` is built once per (ticker,
clock) and shared, because six rebuilds of a `(234000, 10)` int64 gather is six times the work
and six times the peak memory for a bit-identical result. `book=None` exists so a unit test can
call `compute(frame, stamps)` without ceremony; when it is `None` you build it yourself with
`book_at_stamps(frame, stamps)` and the result must be **bit-identical** to the passed-in case.

Rules, all load-bearing:

1. **Every returned array is float64, 1-D, of length `len(stamps.index)`, positionally aligned
   to `stamps`.** Booleans and counts are float64 too — one dtype, one storage path, one NaN
   convention. Never return an int array, a 2-D array, or a shorter array.
2. **Pure.** No I/O, no globals, no module-level mutable state, no caching. Two calls with equal
   arguments return equal arrays. Import must have no side effects.
3. **Never mutates `frame`, `stamps` or `book`.** They are shared across six families. Copy
   before you write.
4. **No cross-family imports.** `microprice` may not import `queue` to get an imbalance bucket;
   it recomputes it from `book` in three lines. Six agents writing in parallel means any
   cross-import is a circular-dependency and an ordering assumption at the same time. The
   infra modules (`frame`, `book`, `clocks`, `windows`, `normalize`) are the only shared code.
5. **Only `ts_ns <= stamps.ts_ns[i]` information** (sec. 2). `labels.py` is the sole exception.
6. **Vectorised over stamps.** 234,000 stamps against 624,040 events is not a Python loop; use
   `windows.prefix_sums` + `window_sum` / `window_count`, an O(1) prefix difference per stamp.
   Loops over levels (<=10), windows (3), horizons (4) and refit blocks (14) are fine — they are
   loops over *constants*, not over data.
7. **Deterministic.** No RNG. Nothing that depends on dict ordering, thread count, or wall clock.
8. **Dependencies:** stdlib + numpy only inside a family module. No `scipy` (installed, still
   forbidden), no `sklearn`, no `pandas`, no `polars`, no `statsmodels`. Hand-write the
   statistics — the hand-written harness is the contribution.
9. Do not put the literal strings `.fit(`, `StandardScaler` or `sklearn` anywhere under
   `src/edge/features/`, comments included:
   `tests/test_normalize.py::TestNoFullSampleFit` greps for them and will fail the suite. Name
   your fitting helper `_estimate_*`, not `.fit(`.
10. Budget roughly **<= 64 columns per family**. More than that and you are emitting a cross
    product `FEATURES.md` never asked for, which inflates the trial count for free.

## 6. Column naming

The six dicts are merged into one table. A collision silently overwrites a column, so each
module owns a prefix and **may not emit a single key outside it**. No prefix is a prefix of
another (asserted in `tests/test_interface.py`).

| module | prefix | examples |
|---|---|---|
| `ofi.py` | `ofi_` | `ofi_best_1s`, `ofi_l3_1s`, `ofi_integrated_1s` |
| `queue.py` | `qi_` | `qi_touch`, `qi_sign`, `qi_depth_l10` |
| `microprice.py` | `mp_` | `mp_micro`, `mp_dev_mid`, `mp_dev_wmid` |
| `trade.py` | `tr_` | `tr_signed_vol_1s`, `tr_eff_spread`, `tr_count_5s` |
| `state.py` | `st_` | `st_spread_ticks`, `st_rv_5s`, `st_tod_bucket` |
| `labels.py` | `label_` | `label_y_mid_1s`, `label_c_mid_1s`, `label_y_micro_e10` |

Labels split *inside* their prefix: `label_y_*` regression (`LABEL_REGRESSION_PREFIX`),
`label_c_*` classification (`LABEL_CLASSIFICATION_PREFIX`). One grep for `label_` therefore
finds every forward-looking column with no exceptions — which is why the split is a sub-prefix
and not two top-level prefixes.

**Suffix grammar.** A column name is:

```
<prefix><base>[_l{m}][_{window}][_sadj]
```

in exactly that order, lower-case ASCII, digits and underscores only:

- **lookback / horizon** — `_100ms`, `_1s`, `_2s`, `_5s`, `_10s`, spelled as the `WINDOWS_NS`
  keys, never as `_1000ms` or `_1sec`. Features use `FEATURE_WINDOWS = ('100ms','1s','5s')`;
  labels use `LABEL_HORIZONS = ('1s','2s','5s','10s')`.
- **event-time horizon** — `_e{N}` from `LABEL_HORIZONS_EVENTS`: `label_y_mid_e10`.
- **book level** — `_l{m}`, 1-indexed from the touch, so `_l1` is the touch and `_l10` the tenth
  level. Level comes *before* window: `ofi_l3_1s`, never `ofi_1s_l3`.
- **seasonality-adjusted twin** — `SEASONAL_ADJ_SUFFIX = '_sadj'`, always last:
  `ofi_l3_1s_sadj`. **A family never emits a `_sadj` column.** The assembly stage produces the
  twins by running `normalize.expanding_bucket_zscore(col, tod_bucket_id)` over the raw
  columns, so the adjustment is defined in one place and every raw feature gets one
  (`FEATURES.md` sec. 5: "produce every result twice"). Emitting your own would produce two
  different definitions of the same word.

A time-invariant feature carries no window suffix (`qi_touch`, `mp_dev_mid`). Do not pad it with
`_0s`.

## 7. Missing data

**NaN, never 0, never a forward-fill, never a magic sentinel.** Zero is a legitimate value of
every signed feature in this project — OFI, imbalance, microprice deviation, signed volume — so
zero must not double as "missing". Emit NaN when:

- the book is invalid at that stamp (`~book.valid`, or `~book.valid_l[:, m]` for a level feature);
- the trailing window is empty (`window_count(...) == 0`) and the statistic needs at least one
  observation. A *sum* over an empty window is legitimately 0.0; a *mean*, *ratio* or
  *z-score* over one is NaN. Decide per feature and say which in the docstring;
- warmup is insufficient (a rolling z-score before `min_periods`, a refit block with no prior
  data — sec. 8);
- a denominator is zero (`q^b + q^a == 0` in queue imbalance). Use `np.divide(..., where=...)`
  into a NaN-filled output, not `0/0` with an `errstate` suppression;
- a label horizon runs past the end of the session (`label_y_mid_10s` in the last 10 seconds).

**Who drops NaN rows: the harness, at fit time, per fold. Not the family module, not the
assembly stage.** A family never drops a row, never shortens its output, never imputes, never
interpolates. The reason is that the drop is a *modelling* decision that depends on which
columns a rung actually uses — L2 uses one OFI column and should not lose rows to a NaN in an
L4-only column — and it has to be counted and reported per fold (`EVAL_PROTOCOL.md`: report the
per-fold distribution, and a fold that silently lost 40% of its rows is a result, not a
detail). Length is fixed at `len(stamps.index)` from the first column to the last.

## 8. Refit on a schedule

Any statistic that must be **fitted** — the integrated-OFI PCA loadings (`FEATURES.md` sec. 1),
the Stoikov microprice transition matrices (sec. 3), depth normalisers used to make multi-level
OFI comparable across levels (sec. 1) — is refit on a fixed wall-clock schedule:

**One fit per `REFIT_BLOCK_NS` (30 min) block, estimated on frame rows STRICTLY BEFORE the block
start, held constant for every stamp inside the block.**

Strictly before, because an event stamped exactly at the boundary belongs to the block being
predicted, not to the sample fitting it (`windows.refit_fit_end_index` uses `side="left"`). The
schedule is fixed a priori rather than per-feature so the refit cadence is not a free parameter
anyone can tune, and 30 min matches `TOD_BUCKET_NS` so the seasonality control and the refit
grid line up.

`windows.refit_blocks(frame_ts_ns, stamp_ts_ns)` returns
`RefitBlocks(stamp_block_id, block_id, block_start_ns, fit_end_index)`, where
`fit_end_index[j]` is the **exclusive** end of block `block_id[j]`'s fitting slice, i.e. the fit
sample is `frame[0 : fit_end_index[j]]`. On INTC there are 14 blocks (0..13; block 13 holds the
single 16:00:00.000 stamp) and `fit_end_index[0] == 0`: **block 0 has no prior data and its
stamps are NaN.** That is correct, it is the honest cost of a causal fit, and it is 18,000
calendar stamps — do not paper over it by falling back to a full-sample fit.

Copy this shape:

```python
import numpy as np
from edge.features.book import book_at_stamps
from edge.features.windows import refit_blocks

MIN_FIT_ROWS = 1000   # per-family, frozen a priori, documented in your docstring


def compute(frame, stamps, book=None):
    """Integrated OFI: first PC of the L multi-level OFIs (Cont-Cucuringu-Zhang)."""
    if book is None:
        book = book_at_stamps(frame, stamps)

    per_event = _level_contributions(frame)              # (n_events, L) float64, causal per row
    window = _windowed(per_event, frame, stamps)         # (n_stamps, L) float64, (t-w, t]

    blocks = refit_blocks(frame.ts_ns, stamps.ts_ns)
    out = np.full(stamps.index.shape[0], np.nan, dtype=np.float64)

    # Loop over 14 blocks, never over 234,000 stamps.
    for block, fit_end in zip(blocks.block_id, blocks.fit_end_index):
        if fit_end < MIN_FIT_ROWS:                       # block 0: nothing strictly earlier -> NaN
            continue
        loadings = _estimate_first_pc(per_event[:fit_end])   # fitted on rows < block start ONLY
        sel = blocks.stamp_block_id == block
        out[sel] = window[sel] @ loadings

    return {"ofi_integrated_1s": out}
```

Three ways this goes wrong, all of which a reviewer checks first:

- fitting on `per_event[:blocks.fit_end_index[-1]]` once and reusing it — that is a full-sample
  fit wearing a loop;
- fitting on the *stamp* array instead of the frame rows — the stamp array inside the block is
  the thing you are predicting;
- sign indeterminacy: a PCA eigenvector's sign is arbitrary, so it flips between blocks and your
  feature changes sign at 10:00. Fix it deterministically inside `_estimate_first_pc` (e.g.
  force the loading on level 1 to be non-negative) and unit-test that.

The same three apply to the microprice transition matrices and to depth normalisers.

## 9. Normalisation

`edge.features.normalize` only. `rolling_zscore(values, window, min_periods=2)` where `window`
is a **count of preceding stamps**, not ns — so it is clock-agnostic; a 5s baseline on the 100ms
calendar clock is `window=50`. Baseline is `[t-window, t)`, strictly excluding `x[t]`.

`expanding_bucket_zscore(values, bucket_id)` is the seasonality control. This universe is one
trading day, so `FEATURES.md`'s "past days only" baseline does not exist; the frozen substitute
is *earlier stamps in the same time-of-day bucket on the same day*, expanding. Strictly causal,
and honest about what one session can support. The assembly stage applies it to build the
`_sadj` twins; a family does not call it.

A family generally emits the **raw** feature and lets the harness standardise per fold. Emit a
z-scored column only where the z-score *is* the feature (a normalised realized-vol surprise,
say), and say so in the name.

## 10. The toy-book test requirement

Every feature gets a unit test in `tests/test_<family>.py` (unittest, not pytest) built on
`tests/toybook.py`'s 8-event, 2-level fixture, and:

**the expected value is a hand-computed literal, written into the test, with the arithmetic
shown in a comment. Not a re-implementation of the function under test.** A test that recomputes
the formula a second way passes whenever both copies share the same misreading of the spec,
which is precisely the failure mode here (the crossed weights, the `<= t` tie, the off-by-one).
`tests/test_book.py` and `tests/test_normalize.py` are the house style; copy their comment
format, which derives the number line by line before asserting it.

`tests/toybook.py` gives you two doors, and you use the existing fixture rather than inventing a
second format:

- `toy_event_frame()` — the shared 8-event, 2-level `EventFrame`, read through the frozen
  adapter, so your test sees the real output schema.
- `toy_arrays()` + `make_event_frame(**d)` — a mutable copy of the fixture's columns to tweak
  for a case the 8 events do not contain. `FEATURES.md` sec. 1's third OFI case (an aggressive
  sell that clears the bid) is one of these: every bid-side price change in the fixture comes
  from a cancel/delete, not a type-4 execution. Build the variant, do not build a new fixture.

Required per family, over and above the per-feature literals:

- **a lookahead test**: truncate the frame, recompute, assert the surviving stamps are
  **bit-identical** (`np.testing.assert_array_equal`, not `allclose`). `tests/test_interface.py`
  runs a generic version of this for you; a family whose formula has a subtle boundary
  (`trade`'s "last trade before t", `microprice`'s block fit) writes a targeted one too.
- **an edge case per NaN rule you claim**: invalid book, empty window, warmup, zero denominator.
  If your docstring says a case is NaN, a test asserts it is NaN.
- **a runs-on-both-clocks test**: `stamps_for_clock(frame, "calendar")` and
  `stamps_for_clock(frame, "event")` both return the right shapes and dtypes.

`labels.py` is the one module whose lookahead test is inverted: it asserts the label at `t`
**does** change when future events change, and that a label whose horizon runs past the last
event is NaN rather than clipped to the last available price.

## 11. Done checklist

- [ ] `compute(frame, stamps, book=None)`, exactly that signature, returns
      `dict[str, np.ndarray]`, every array float64, 1-D, `len(stamps.index)` long.
- [ ] `book=None` builds `book_at_stamps(frame, stamps)` and gives a bit-identical result.
- [ ] Every key starts with the module's `FAMILY_PREFIXES` prefix; suffix grammar per sec. 6;
      no `_sadj`.
- [ ] Module docstring is one line plus the `FEATURES.md` section / paper equation, and names
      any convention stricter than `<= t` that it uses.
- [ ] NaN, not 0, for every undefined case; nothing dropped, imputed or forward-filled.
- [ ] Fitted statistics go through `windows.refit_blocks`; block 0 is NaN.
- [ ] Hand-computed literal tests on `tests/toybook.py`; lookahead test; NaN-rule tests.
- [ ] No Python loop over stamps. Runs on INTC (624,040 events, 234,000 calendar stamps).
- [ ] No `scipy`/`sklearn`/`pandas`/`polars`; no `.fit(`, `StandardScaler`, `sklearn` as literal
      strings under `src/edge/features/`.
- [ ] `make test` green.
