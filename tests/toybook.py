"""Shared toy event frame every Phase-1 family agent imports to hand-verify features against.

Two ways in:

* `build_toy_frame()` / `toy_event_frame()` -- the same 8-message, 2-level fixture
  `tests/test_lobster.py` and `tests/test_alignment.py` already exercise
  (`tests/fixtures/toy_message.csv` + `tests/fixtures/toy_orderbook.csv`), read through
  the frozen adapter `edge.io.lobster.build_event_frame`. The CSVs are themselves the
  hand-built artifact (see `test_lobster.py` for the per-row derivation) and going
  through the real adapter guarantees every family test sees the exact frozen output
  schema (`docs/DATA_CONTRACT.md`), not a hand-rolled table that could drift from it.

* `make_frame(...)` / `make_event_frame(...)` -- build a VARIANT book inline from plain
  Python lists, in the identical schema. `toy_arrays()` hands back a mutable copy of the
  fixture's columns to tweak: `d = toy_arrays(); d["event_type"][4] = 1;
  frame = make_event_frame(**d)`. Use this for scenarios the 8-event fixture does not
  contain -- e.g. FEATURES.md sec. 1's third OFI case, an aggressive sell that clears the
  bid, is NOT one of these 8 events (every bid-side price change here comes from a
  cancel/delete, not a type-4 execution). `tests/test_frame.py` asserts
  `make_frame(**toy_arrays())` equals `build_toy_frame()` column for column, so the
  inline builder cannot drift from the adapter.

Event-by-event summary (0-indexed; full derivation in test_lobster.py / test_alignment.py):
  0: type=1 (new bid limit),   dir=+1, price=100500, size=10  -> bid_sz_1 10 (from 0)
  1: type=1 (new bid limit),   dir=+1, price=100500, size=20  -> bid_sz_1 10->30 (+20 add)
  2: type=2 (partial cancel),  dir=+1, price=100500, size=5   -> bid_sz_1 30->25 (-5 cancel)
  3: type=3 (full delete),     dir=+1, price=100500, size=25  -> bid_px_1 100500->100000 (level vanishes)
  4: type=4 (visible execution), dir=-1, price=101000, size=10 -> ask_sz_1 50->40 (ask depletion)
  5: type=5 (hidden execution), dir=-1, price=101000, size=5  -> book unchanged (row4 == row5)
  6: type=1 (new ask limit),   dir=-1, price=103000, size=15  -> invisible at L=2 (beyond ask_px_2)
  7: type=3 (full delete),     dir=-1, price=102000, size=30  -> ask_px_2 102000->103000 (level vanishes)

Derived reference values (raw price units, TICK_SIZE_RAW = 100), hand-computed:
  mid  = [100750, 100750, 100750, 100500, 100500, 100500, 100500, 100500]
  wmid = [100583.33.., 100687.5, 100666.66.., 100444.44.., 100500, 100500, 100500, 100500]
  spread = [500, 500, 500, 1000, 1000, 1000, 1000, 1000]  (5, 5, 5, 10, 10, 10, 10, 10 ticks)
"""
import os

import pyarrow as pa

from edge.features.frame import event_frame_from_table
from edge.io.lobster import build_event_frame

FIXTURES_DIR = os.path.join(os.path.dirname(__file__), "fixtures")
TOY_MESSAGE_PATH = os.path.join(FIXTURES_DIR, "toy_message.csv")
TOY_ORDERBOOK_PATH = os.path.join(FIXTURES_DIR, "toy_orderbook.csv")
TOY_NUM_LEVELS = 2

# Verbatim from the fixture CSVs, so a family test can assert against these without
# re-parsing the CSV itself.
TOY_TS_NS = [
    34200000000001,
    34200000000002,
    34200100000000,
    34200100000500,
    34200200000000,
    34200200000001,
    34200300000000,
    34200400000000,
]
TOY_EVENT_TYPE = [1, 1, 2, 3, 4, 5, 1, 3]
TOY_DIRECTION = [1, 1, 1, 1, -1, -1, -1, -1]
TOY_PRICE = [100500, 100500, 100500, 100500, 101000, 101000, 103000, 102000]
TOY_SIZE = [10, 20, 5, 25, 10, 5, 15, 30]

# Post-event book state, one row per event, level 1 first (transcribed from
# tests/fixtures/toy_orderbook.csv, which stores them ask-first interleaved per level).
TOY_BID_PX = [
    [100500, 100000],
    [100500, 100000],
    [100500, 100000],
    [100000, 98000],
    [100000, 98000],
    [100000, 98000],
    [100000, 98000],
    [100000, 98000],
]
TOY_BID_SZ = [
    [10, 40],
    [30, 40],
    [25, 40],
    [40, 15],
    [40, 15],
    [40, 15],
    [40, 15],
    [40, 15],
]
TOY_ASK_PX = [
    [101000, 102000],
    [101000, 102000],
    [101000, 102000],
    [101000, 102000],
    [101000, 102000],
    [101000, 102000],
    [101000, 102000],
    [101000, 103000],
]
TOY_ASK_SZ = [
    [50, 30],
    [50, 30],
    [50, 30],
    [50, 30],
    [40, 30],
    [40, 30],
    [40, 30],
    [40, 15],
]


def build_toy_frame():
    """The shared 8-event, 2-level toy book as a pyarrow Table, via the frozen adapter."""
    return build_event_frame(TOY_MESSAGE_PATH, TOY_ORDERBOOK_PATH, TOY_NUM_LEVELS)


def toy_event_frame():
    """The shared 8-event toy book as an `edge.features.frame.EventFrame`."""
    return event_frame_from_table(build_toy_frame(), num_levels=TOY_NUM_LEVELS)


def toy_arrays():
    """Mutable per-column copy of the toy book, ready to tweak and splat into make_frame."""
    return {
        "ts_ns": list(TOY_TS_NS),
        "event_type": list(TOY_EVENT_TYPE),
        "direction": list(TOY_DIRECTION),
        "price": list(TOY_PRICE),
        "size": list(TOY_SIZE),
        "bid_px": [list(row) for row in TOY_BID_PX],
        "bid_sz": [list(row) for row in TOY_BID_SZ],
        "ask_px": [list(row) for row in TOY_ASK_PX],
        "ask_sz": [list(row) for row in TOY_ASK_SZ],
    }


def make_frame(ts_ns, event_type, direction, price, size, bid_px, bid_sz, ask_px, ask_sz):
    """Build a variant event frame (pyarrow Table) in the frozen schema from plain lists."""
    n = len(ts_ns)
    for name, col in (
        ("event_type", event_type),
        ("direction", direction),
        ("price", price),
        ("size", size),
        ("bid_px", bid_px),
        ("bid_sz", bid_sz),
        ("ask_px", ask_px),
        ("ask_sz", ask_sz),
    ):
        if len(col) != n:
            raise ValueError(f"{name} has {len(col)} rows, expected {n}")
    num_levels = len(bid_px[0])
    for name, col in (("bid_px", bid_px), ("bid_sz", bid_sz), ("ask_px", ask_px), ("ask_sz", ask_sz)):
        if any(len(row) != num_levels for row in col):
            raise ValueError(f"{name} is ragged; every row needs {num_levels} levels")

    columns = {
        "ts_ns": pa.array(ts_ns, type=pa.int64()),
        "event_type": pa.array(event_type, type=pa.int8()),
        "direction": pa.array(direction, type=pa.int8()),
        "price": pa.array(price, type=pa.int64()),
        "size": pa.array(size, type=pa.int64()),
    }
    # Column ORDER is the frozen one from docs/DATA_CONTRACT.md: block-grouped by
    # side/quantity, not the raw file's interleaved-per-level order.
    for side, kind, values in (
        ("bid", "px", bid_px),
        ("bid", "sz", bid_sz),
        ("ask", "px", ask_px),
        ("ask", "sz", ask_sz),
    ):
        for m in range(num_levels):
            columns[f"{side}_{kind}_{m + 1}"] = pa.array(
                [row[m] for row in values], type=pa.int64()
            )
    return pa.table(columns)


def make_event_frame(*args, **kwargs):
    """`make_frame` result wrapped as an EventFrame, the type family modules consume."""
    return event_frame_from_table(make_frame(*args, **kwargs))
