"""Loader turning a frozen parquet event frame (docs/DATA_CONTRACT.md) into numpy arrays.

Every family module consumes an `EventFrame`, never a pyarrow Table, so the 2-D level
arrays (n_events, L) are column-stacked exactly once instead of once per family. Level
column m of `bid_px` / `bid_sz` / `ask_px` / `ask_sz` is level m+1 of the book (index 0
is the touch), best to worst, matching the frozen schema's `bid_px_1 .. bid_px_L`.

Row i is the book state immediately AFTER message i (docs/DATA_CONTRACT.md "Alignment
convention"). That is the whole reason the `<= t` convention is safe: an event stamped
exactly at t has already printed by t, so its post-state is public information at t.

Prices and sizes stay in raw LOBSTER integer units (0.01 cents; TICK_SIZE_RAW = 100).
The `*_f` float64 views carry NaN at any level LOBSTER marks as absent -- the documented
sentinel (ask 9999999999 / bid -9999999999) or a zero displayed size. There are zero
such levels in the 2012-06-21 INTC/AMZN sample, but the mask is computed rather than
assumed, because that is a property of this sample and not of the format.
"""
import os
from collections import namedtuple

import numpy as np
import pyarrow.parquet as pq

from edge.features import NUM_LEVELS
from edge.io.lobster import ASK_SENTINEL_PRICE, BID_SENTINEL_PRICE

SCALAR_COLUMNS = ("ts_ns", "event_type", "direction", "price", "size")
LEVEL_COLUMN_KINDS = (("bid", "px"), ("bid", "sz"), ("ask", "px"), ("ask", "sz"))

_Schema = namedtuple("_Schema", ["names"])


def _level_valid(px, sz, sentinel):
    """Displayed-level mask: not the LOBSTER sentinel price, positive price, positive size."""
    return (px != sentinel) & (px > 0) & (sz > 0)


class _ColumnView:
    """pyarrow-ChunkedArray stand-in so table-shaped consumers also accept an EventFrame."""

    def __init__(self, values):
        self._values = values

    def to_numpy(self, zero_copy_only=True):
        """Return the backing numpy array; the kwarg exists only for call-site compatibility."""
        return self._values

    def to_pylist(self):
        """Python-list view, matching pyarrow's accessor of the same name."""
        return self._values.tolist()

    def __len__(self):
        return self._values.shape[0]

    def __array__(self, dtype=None):
        return np.asarray(self._values, dtype=dtype)


class EventFrame:
    """Numpy view of one ticker-day event frame; see module docstring for the row convention."""

    def __init__(self, ts_ns, event_type, direction, price, size, bid_px, bid_sz, ask_px, ask_sz):
        self.ts_ns = np.ascontiguousarray(ts_ns, dtype=np.int64)
        self.event_type = np.ascontiguousarray(event_type, dtype=np.int8)
        self.direction = np.ascontiguousarray(direction, dtype=np.int8)
        self.price = np.ascontiguousarray(price, dtype=np.int64)
        self.size = np.ascontiguousarray(size, dtype=np.int64)
        self.bid_px = np.ascontiguousarray(bid_px, dtype=np.int64)
        self.bid_sz = np.ascontiguousarray(bid_sz, dtype=np.int64)
        self.ask_px = np.ascontiguousarray(ask_px, dtype=np.int64)
        self.ask_sz = np.ascontiguousarray(ask_sz, dtype=np.int64)
        self._cache = {}
        self._validate()

    def _validate(self):
        """Fail loudly on the two errors that would silently corrupt every window feature."""
        n = self.ts_ns.shape[0]
        for name in ("event_type", "direction", "price", "size"):
            got = getattr(self, name).shape
            if got != (n,):
                raise ValueError(f"{name} has shape {got}, expected ({n},)")
        shape = self.bid_px.shape
        if len(shape) != 2 or shape[0] != n:
            raise ValueError(f"bid_px has shape {shape}, expected ({n}, L)")
        for name in ("bid_sz", "ask_px", "ask_sz"):
            got = getattr(self, name).shape
            if got != shape:
                raise ValueError(f"{name} has shape {got}, expected {shape}")
        # searchsorted-based stamping and every prefix-sum window are only correct on a
        # non-decreasing time axis; LOBSTER files are sorted, so a violation means a
        # corrupt or concatenated frame, not a market event.
        if n > 1 and not bool(np.all(np.diff(self.ts_ns) >= 0)):
            raise ValueError("ts_ns is not non-decreasing; frame is not a single sorted ticker-day")

    @property
    def n_events(self):
        """Number of rows (one per LOBSTER message)."""
        return self.ts_ns.shape[0]

    @property
    def num_levels(self):
        """Book depth L carried by the level arrays."""
        return self.bid_px.shape[1]

    def _cached(self, key, builder):
        """Memoise a derived array so five family modules share one allocation."""
        if key not in self._cache:
            value = builder()
            value.flags.writeable = False
            self._cache[key] = value
        return self._cache[key]

    @property
    def bid_valid(self):
        """(n, L) bool: bid level m is a real displayed level, not a sentinel/empty slot."""
        return self._cached(
            "bid_valid", lambda: _level_valid(self.bid_px, self.bid_sz, BID_SENTINEL_PRICE)
        )

    @property
    def ask_valid(self):
        """(n, L) bool: ask level m is a real displayed level, not a sentinel/empty slot."""
        return self._cached(
            "ask_valid", lambda: _level_valid(self.ask_px, self.ask_sz, ASK_SENTINEL_PRICE)
        )

    def _masked(self, key, values, mask):
        """float64 copy of an integer level array with NaN at masked-out levels."""
        return self._cached(
            key, lambda: np.where(mask, values.astype(np.float64), np.nan)
        )

    @property
    def bid_px_f(self):
        """(n, L) float64 bid prices, raw units, NaN at absent levels."""
        return self._masked("bid_px_f", self.bid_px, self.bid_valid)

    @property
    def bid_sz_f(self):
        """(n, L) float64 bid sizes, NaN at absent levels."""
        return self._masked("bid_sz_f", self.bid_sz, self.bid_valid)

    @property
    def ask_px_f(self):
        """(n, L) float64 ask prices, raw units, NaN at absent levels."""
        return self._masked("ask_px_f", self.ask_px, self.ask_valid)

    @property
    def ask_sz_f(self):
        """(n, L) float64 ask sizes, NaN at absent levels."""
        return self._masked("ask_sz_f", self.ask_sz, self.ask_valid)

    @property
    def best_bid_px(self):
        """(n,) float64 best bid price, raw units, NaN when the touch is absent."""
        return self._cached("best_bid_px", lambda: np.array(self.bid_px_f[:, 0]))

    @property
    def best_bid_sz(self):
        """(n,) float64 best bid size, NaN when the touch is absent."""
        return self._cached("best_bid_sz", lambda: np.array(self.bid_sz_f[:, 0]))

    @property
    def best_ask_px(self):
        """(n,) float64 best ask price, raw units, NaN when the touch is absent."""
        return self._cached("best_ask_px", lambda: np.array(self.ask_px_f[:, 0]))

    @property
    def best_ask_sz(self):
        """(n,) float64 best ask size, NaN when the touch is absent."""
        return self._cached("best_ask_sz", lambda: np.array(self.ask_sz_f[:, 0]))

    @property
    def mid(self):
        """(n,) float64 mid price in raw units, FEATURES.md sec. 3: mid = (P^b + P^a)/2."""
        return self._cached("mid", lambda: 0.5 * (self.best_bid_px + self.best_ask_px))

    @property
    def wmid(self):
        """(n,) float64 weighted mid, FEATURES.md sec. 3: (P^b q^a + P^a q^b)/(q^a + q^b)."""

        def build():
            qb, qa = self.best_bid_sz, self.best_ask_sz
            denom = qa + qb
            with np.errstate(invalid="ignore", divide="ignore"):
                out = (self.best_bid_px * qa + self.best_ask_px * qb) / denom
            return np.where(denom > 0, out, np.nan)

        return self._cached("wmid", build)

    @property
    def spread(self):
        """(n,) float64 quoted spread in raw price units (divide by TICK_SIZE_RAW for ticks)."""
        return self._cached("spread", lambda: self.best_ask_px - self.best_bid_px)

    # -- pyarrow-Table compatibility shim ---------------------------------------------
    # Six family modules are written in parallel; some will reach for `.column(name)`
    # out of habit. Answering that call correctly is cheaper than debugging six
    # AttributeErrors, and it lets any table-shaped helper accept an EventFrame.

    @property
    def num_rows(self):
        """Row count, spelled the way pyarrow spells it."""
        return self.n_events

    def column_names(self):
        """Frozen-schema column names in the docs/DATA_CONTRACT.md order."""
        names = list(SCALAR_COLUMNS)
        for side, kind in LEVEL_COLUMN_KINDS:
            names += [f"{side}_{kind}_{m + 1}" for m in range(self.num_levels)]
        return names

    @property
    def schema(self):
        """Schema shim exposing `.names`, so `frame.schema.names` works on either type."""
        return _Schema(names=self.column_names())

    def column(self, name):
        """Fetch one frozen-schema column by name, pyarrow-Table style."""
        if name in SCALAR_COLUMNS:
            return _ColumnView(getattr(self, name))
        head, _, level = name.rpartition("_")
        if head in {f"{side}_{kind}" for side, kind in LEVEL_COLUMN_KINDS} and level.isdigit():
            m = int(level)
            if 1 <= m <= self.num_levels:
                return _ColumnView(getattr(self, head)[:, m - 1])
        raise KeyError(f"{name!r} is not a column of this event frame")


def event_frame_from_table(table, num_levels=None):
    """Build an EventFrame from a pyarrow Table in the frozen schema; L inferred if omitted."""
    names = set(table.schema.names)
    missing = [c for c in SCALAR_COLUMNS if c not in names]
    if missing:
        raise ValueError(f"event frame is missing required column(s): {missing}")
    if num_levels is None:
        num_levels = 0
        while f"bid_px_{num_levels + 1}" in names:
            num_levels += 1
    if num_levels < 1:
        raise ValueError("event frame carries no bid_px_1 column; not a frozen-schema frame")

    def col(name):
        return table.column(name).to_numpy(zero_copy_only=False)

    n = table.num_rows
    levels = {}
    for side, kind in (("bid", "px"), ("bid", "sz"), ("ask", "px"), ("ask", "sz")):
        out = np.empty((n, num_levels), dtype=np.int64)
        for m in range(num_levels):
            out[:, m] = col(f"{side}_{kind}_{m + 1}")
        levels[f"{side}_{kind}"] = out

    return EventFrame(
        ts_ns=col("ts_ns"),
        event_type=col("event_type"),
        direction=col("direction"),
        price=col("price"),
        size=col("size"),
        bid_px=levels["bid_px"],
        bid_sz=levels["bid_sz"],
        ask_px=levels["ask_px"],
        ask_sz=levels["ask_sz"],
    )


def load_event_frame(path, num_levels=None):
    """Read data/frames/{TICKER}_{DATE}.parquet into an EventFrame."""
    return event_frame_from_table(pq.read_table(path), num_levels=num_levels)


def frame_path(ticker, date="2012-06-21", frames_dir="data/frames"):
    """Canonical location of a built event frame for a ticker-day."""
    return os.path.join(frames_dir, f"{ticker}_{date}.parquet")


def column(frame, name):
    """Fetch a scalar column as numpy from either an EventFrame or a raw pyarrow Table."""
    if isinstance(frame, EventFrame):
        return getattr(frame, name)
    return frame.column(name).to_numpy(zero_copy_only=False)


__all__ = [
    "ASK_SENTINEL_PRICE",
    "BID_SENTINEL_PRICE",
    "NUM_LEVELS",
    "SCALAR_COLUMNS",
    "EventFrame",
    "column",
    "event_frame_from_table",
    "frame_path",
    "load_event_frame",
]
