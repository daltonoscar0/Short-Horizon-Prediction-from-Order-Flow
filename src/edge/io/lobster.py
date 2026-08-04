"""Adapter around LOBSTER message/orderbook CSVs; frozen schema in docs/DATA_CONTRACT.md."""
import argparse
import os
import re

import numpy as np
import pyarrow as pa
import pyarrow.csv as pa_csv
import pyarrow.parquet as pq

NS_PER_SEC = 1_000_000_000

# LOBSTER's documented "no displayed level" sentinel: ask 9999999999 / bid -9999999999,
# size 0. Zero occurrences in the free 10-level full-day sample (grepped, see
# DECISIONS.md), but downstream level lookups (quality report, features) must not treat
# these as real prices -- exported here so callers don't hardcode the magic numbers.
ASK_SENTINEL_PRICE = 9999999999
BID_SENTINEL_PRICE = -9999999999

_MESSAGE_COLUMNS = ["time", "event_type", "order_id", "size", "price", "direction"]

_FILE_RE = re.compile(
    r"^(?P<ticker>[A-Z]+)_(?P<date>\d{4}-\d{2}-\d{2})_(?P<start>\d+)_(?P<end>\d+)"
    r"_message_(?P<levels>\d+)\.csv$"
)


def _parse_ts_ns(time_strings):
    """Decimal seconds -> ns-after-midnight via string split, never float64 * 1e9 (loses ns ties)."""
    out = np.empty(len(time_strings), dtype=np.int64)
    for i, s in enumerate(time_strings):
        sec_str, _, frac_str = s.partition(".")
        frac_str = (frac_str + "000000000")[:9]  # right-pad/truncate to ns resolution
        out[i] = int(sec_str) * NS_PER_SEC + int(frac_str)
    return out


def read_message_file(path):
    """Parse a LOBSTER message CSV: ts_ns, event_type, order_id, size, price, direction."""
    read_options = pa_csv.ReadOptions(column_names=_MESSAGE_COLUMNS)
    convert_options = pa_csv.ConvertOptions(
        column_types={
            "time": pa.string(),  # keep as string: forcing double here is exactly the float64*1e9 bug
            "event_type": pa.int64(),
            "order_id": pa.int64(),
            "size": pa.int64(),
            "price": pa.int64(),
            "direction": pa.int64(),
        }
    )
    raw = pa_csv.read_csv(path, read_options=read_options, convert_options=convert_options)

    ts_ns = pa.array(_parse_ts_ns(raw.column("time").to_pylist()), type=pa.int64())
    return pa.table(
        {
            "ts_ns": ts_ns,
            "event_type": raw.column("event_type").cast(pa.int8()),
            "order_id": raw.column("order_id"),
            "size": raw.column("size"),
            "price": raw.column("price"),
            "direction": raw.column("direction").cast(pa.int8()),
        }
    )


def _orderbook_columns(num_levels):
    """Raw interleaved column names, ask before bid per level, per docs/DATA_CONTRACT.md."""
    cols = []
    for lvl in range(1, num_levels + 1):
        cols += [f"ask_px_{lvl}", f"ask_sz_{lvl}", f"bid_px_{lvl}", f"bid_sz_{lvl}"]
    return cols


def read_orderbook_file(path, num_levels):
    """Parse a LOBSTER orderbook CSV into ask/bid px/sz columns for 1..num_levels."""
    columns = _orderbook_columns(num_levels)
    read_options = pa_csv.ReadOptions(column_names=columns)
    # int64: sentinel levels (ask 9999999999 / bid -9999999999) overflow int32, passthrough as-is.
    convert_options = pa_csv.ConvertOptions(column_types={c: pa.int64() for c in columns})
    return pa_csv.read_csv(path, read_options=read_options, convert_options=convert_options)


def build_event_frame(message_path, orderbook_path, num_levels):
    """Join message + orderbook rows by position into the frozen event-indexed schema."""
    messages = read_message_file(message_path)
    book = read_orderbook_file(orderbook_path, num_levels)
    if messages.num_rows != book.num_rows:
        raise ValueError(
            f"row count mismatch: {message_path} has {messages.num_rows} rows, "
            f"{orderbook_path} has {book.num_rows} rows -- alignment invariant requires equality"
        )

    columns = {
        "ts_ns": messages.column("ts_ns"),
        "event_type": messages.column("event_type"),
        "direction": messages.column("direction"),
        "price": messages.column("price"),
        "size": messages.column("size"),
    }
    for side, kind in (("bid", "px"), ("bid", "sz"), ("ask", "px"), ("ask", "sz")):
        for lvl in range(1, num_levels + 1):
            name = f"{side}_{kind}_{lvl}"
            columns[name] = book.column(name)
    return pa.table(columns)


def discover_pairs(raw_dir):
    """Find {message,orderbook} file pairs under raw_dir by the LOBSTER filename pattern."""
    pairs = []
    for fname in sorted(os.listdir(raw_dir)):
        m = _FILE_RE.match(fname)
        if not m:
            continue
        levels = m.group("levels")
        ob_name = (
            f"{m.group('ticker')}_{m.group('date')}_{m.group('start')}_{m.group('end')}"
            f"_orderbook_{levels}.csv"
        )
        ob_path = os.path.join(raw_dir, ob_name)
        if not os.path.exists(ob_path):
            raise FileNotFoundError(f"no matching orderbook file for {fname}: expected {ob_name}")
        pairs.append(
            {
                "ticker": m.group("ticker"),
                "date": m.group("date"),
                "message_path": os.path.join(raw_dir, fname),
                "orderbook_path": ob_path,
                "num_levels": int(levels),
            }
        )
    return pairs


def main(argv=None):
    parser = argparse.ArgumentParser(description="Build validated event frames from raw LOBSTER CSVs.")
    parser.add_argument("--raw", required=True, help="directory of raw LOBSTER message/orderbook CSVs")
    parser.add_argument("--out", required=True, help="output directory for {TICKER}_{DATE}.parquet")
    parser.add_argument("--seed", type=int, default=0)  # every entry point takes --seed, per RULES.md
    args = parser.parse_args(argv)

    print(f"seed={args.seed} (deterministic step, no randomness consumed)")
    os.makedirs(args.out, exist_ok=True)

    for pair in discover_pairs(args.raw):
        frame = build_event_frame(pair["message_path"], pair["orderbook_path"], pair["num_levels"])
        out_path = os.path.join(args.out, f"{pair['ticker']}_{pair['date']}.parquet")
        pq.write_table(frame, out_path)
        print(f"{pair['ticker']}_{pair['date']}: {frame.num_rows} rows -> {out_path}")


if __name__ == "__main__":
    main()
