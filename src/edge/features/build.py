"""Phase-1 assembly stage: merges the six family modules into one feature table per
ticker-day-clock (docs/INTERFACE.md sec. 5's "three-line loop over the six modules").

    python3 -m edge.features.build --frames data/frames --out data/features --seed 0

For each (ticker, clock) pair: load the frame, build `stamps` once via `stamps_for_clock`,
build `book_at_stamps` once and hand the SAME object to all six families (INTERFACE.md
sec. 5: six independent rebuilds of a `(234000, 10)` int64 gather is six times the work and
six times the peak memory for a bit-identical result), merge the six `dict[str, ndarray]`s
-- raising on any duplicate key rather than silently overwriting one -- prepend the three
identity columns (`ts_ns`, `stamp_index`, `tod_bucket`, all computed HERE from `stamps`,
never taken from a family), and append the seasonality-adjusted `_sadj` twin of every
non-identity, non-`label_` column (FEATURES.md sec. 5's "produce every result twice";
INTERFACE.md sec. 6: a family may never emit its own `_sadj` column, so the definition
lives in exactly one place -- this one -- via `normalize.expanding_bucket_zscore`). Labels
are exempt from the twin: a seasonality-adjusted label is a different label and would
silently become the thing Phase 2 predicts.

Column order is deterministic: identity columns first, then `FAMILY_PREFIXES`'s own
definition order (ofi, queue, microprice, trade, state, labels -- a plain `dict` preserves
insertion order), then each family's own dict-insertion order, with a raw column's `_sadj`
twin placed immediately after it. Two runs on the same inputs therefore produce
byte-comparable schemas.

`--seed` is accepted because every entry point takes one (CLAUDE.md rule 7), not because
anything here is stochastic: `compute()` is pure by contract (INTERFACE.md sec. 5 rule 2)
and this module adds no randomness of its own, so the flag is parsed and logged, never
wired into a computation.
"""
import argparse
import os
import time

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from edge.features import (
    FAMILY_PREFIXES,
    LABEL_PREFIX,
    SEASONAL_ADJ_SUFFIX,
    SESSION_START_NS,
    TOD_BUCKET_NS,
)
from edge.features import labels, microprice, ofi, queue, state, trade
from edge.features.book import book_at_stamps
from edge.features.clocks import stamps_for_clock
from edge.features.frame import frame_path, load_event_frame
from edge.features.normalize import expanding_bucket_zscore

TICKERS = ("INTC", "AMZN")
CLOCKS = ("calendar", "event")
DATE = "2012-06-21"  # the only ticker-day in this sample (docs/DATA_AND_LINKS.md)

# Family name -> module, keyed the same as FAMILY_PREFIXES so the loop below can walk
# FAMILY_PREFIXES's own definition order and use this only as a lookup, never a second
# source of ordering truth.
_FAMILY_MODULES = {
    "ofi": ofi,
    "queue": queue,
    "microprice": microprice,
    "trade": trade,
    "state": state,
    "labels": labels,
}
assert set(_FAMILY_MODULES) == set(FAMILY_PREFIXES)


def _identity_columns(stamps):
    """`ts_ns` / `stamp_index` / `tod_bucket`, built HERE from `stamps` alone -- never
    taken from any family's own (float64) `st_tod_bucket`, which is a different column."""
    ts_ns = np.asarray(stamps.ts_ns, dtype=np.int64)
    stamp_index = np.asarray(stamps.index, dtype=np.int64)
    tod_bucket = np.floor_divide(ts_ns - SESSION_START_NS, TOD_BUCKET_NS).astype(np.int64)
    return {"ts_ns": ts_ns, "stamp_index": stamp_index, "tod_bucket": tod_bucket}


def build_table(frame, clock):
    """Run all six families for one (ticker, clock) pair and merge into one column set.

    Returns `(columns, order)`: `columns` is the full `{name: ndarray}` mapping (raw,
    identity and `_sadj`), `order` is the deterministic emission order described in the
    module docstring.
    """
    stamps = stamps_for_clock(frame, clock)
    book = book_at_stamps(frame, stamps)  # built ONCE, shared by all six families

    identity = _identity_columns(stamps)
    columns = dict(identity)
    order = list(identity)
    tod_bucket = identity["tod_bucket"]

    for name in FAMILY_PREFIXES:  # frozen order: ofi, queue, microprice, trade, state, labels
        out = _FAMILY_MODULES[name].compute(frame, stamps, book)
        for key, values in out.items():
            if key in columns:
                raise ValueError(f"duplicate column {key!r} emitted by family {name!r}")
            columns[key] = values
            order.append(key)

            if not key.startswith(LABEL_PREFIX):
                sadj_key = f"{key}{SEASONAL_ADJ_SUFFIX}"
                if sadj_key in columns:
                    raise ValueError(f"duplicate column {sadj_key!r} produced for family {name!r}")
                columns[sadj_key] = expanding_bucket_zscore(values, tod_bucket)
                order.append(sadj_key)

    return columns, order


def _nan_fraction_report(columns, order, top_n=10):
    """[(column, nan_fraction)] for the `top_n` widest-NaN columns, worst first."""
    n = columns["ts_ns"].shape[0]
    fractions = [
        (key, float(np.isnan(columns[key]).sum()) / n if n > 0 else 0.0) for key in order
    ]
    fractions.sort(key=lambda kv: kv[1], reverse=True)
    return fractions[:top_n]


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Assemble the six feature families into one table per ticker-day-clock."
    )
    parser.add_argument("--frames", required=True, help="directory of {TICKER}_{DATE}.parquet event frames")
    parser.add_argument("--out", required=True, help="output directory for {TICKER}_{DATE}_{clock}.parquet")
    parser.add_argument("--date", default=DATE, help="ticker-day date stamp (default: the only date in this sample)")
    parser.add_argument("--seed", type=int, default=0)  # see module docstring: unused, logged only
    args = parser.parse_args(argv)

    print(f"seed={args.seed} (deterministic step, no randomness consumed)")
    os.makedirs(args.out, exist_ok=True)

    for ticker in TICKERS:
        path = frame_path(ticker, date=args.date, frames_dir=args.frames)
        frame = load_event_frame(path)
        for clock in CLOCKS:
            t0 = time.time()
            columns, order = build_table(frame, clock)
            dt = time.time() - t0

            table = pa.table({key: columns[key] for key in order})
            out_path = os.path.join(args.out, f"{ticker}_{args.date}_{clock}.parquet")
            pq.write_table(table, out_path)

            n_rows = table.num_rows
            n_cols = len(order)
            print(f"{ticker} {clock}: {n_rows} rows x {n_cols} cols -> {out_path} ({dt:.1f}s)")
            print(f"  widest NaN offenders (fraction of {n_rows} rows):")
            for key, frac in _nan_fraction_report(columns, order):
                print(f"    {key:40s} {frac:6.2%}")


if __name__ == "__main__":
    main()
