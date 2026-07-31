"""Data-quality report for built LOBSTER event frames.

Reads the parquet frames `edge.io.lobster` produces (not the raw CSVs again) and writes
`reports/data_quality.md`. Invoked as the last step of `make data`, or standalone:
`python3 -m edge.io.quality --frames data/frames --out reports/data_quality.md --seed 0`.
"""
import argparse
import glob
import os
import random

import numpy as np
import pyarrow.parquet as pq

from edge.io.lobster import ASK_SENTINEL_PRICE, BID_SENTINEL_PRICE

DOLLARS_PER_RAW_UNIT = 1.0 / 10000.0  # price is in 0.01-cent units; see DATA_CONTRACT.md
TICK_SIZE_RAW = 100  # $0.01 == 100 raw units

# Sanity bounds from docs/spec/DATA_AND_LINKS.md / the phase-0 brief: median mid outside
# this range means the price scaling (or the parse generally) is broken -- fix it, don't
# report around it. Only asserted for tickers this universe actually covers.
MEDIAN_MID_BOUNDS_DOLLARS = {"INTC": (20.0, 35.0), "AMZN": (200.0, 250.0)}

BUCKET_MINUTES = 30
NS_PER_SEC = 1_000_000_000
BUCKET_NS = BUCKET_MINUTES * 60 * NS_PER_SEC
GAP_THRESHOLD_NS = NS_PER_SEC  # "gaps: intervals > 1s with no events" per PLAN.md

EVENT_TYPES = [1, 2, 3, 4, 5, 6, 7]
EVENT_TYPE_LABELS = {
    1: "new limit order",
    2: "partial cancel",
    3: "full delete",
    4: "visible execution",
    5: "hidden execution",
    6: "cross/auction trade",
    7: "trading halt",
}


def _valid_quote_mask(bid_px, ask_px, bid_sz, ask_sz):
    """True where both best bid and best ask are real displayed levels.

    Defensive against LOBSTER's documented sentinel (ask 9999999999 / bid -9999999999,
    size 0) for an occupied-level shortfall -- zero occurrences in this sample (see
    DECISIONS.md) but a naive `bid_px >= ask_px` crossed check would misfire on it.
    """
    bid_ok = (bid_px != BID_SENTINEL_PRICE) & (bid_sz > 0)
    ask_ok = (ask_px != ASK_SENTINEL_PRICE) & (ask_sz > 0)
    return bid_ok & ask_ok


def _bucket_label(bucket_index):
    start_s = bucket_index * BUCKET_MINUTES * 60
    end_s = start_s + BUCKET_MINUTES * 60
    fmt = lambda s: f"{int(s // 3600):02d}:{int((s % 3600) // 60):02d}"
    return f"{fmt(start_s)}-{fmt(end_s)}"


def _ns_to_clock(ts_ns):
    total_s = ts_ns // NS_PER_SEC
    return f"{int(total_s // 3600):02d}:{int((total_s % 3600) // 60):02d}:{int(total_s % 60):02d}"


def analyze_frame(path, ticker):
    table = pq.read_table(path)
    n = table.num_rows

    ts_ns = table.column("ts_ns").to_numpy()
    event_type = table.column("event_type").to_numpy()
    bid_px_1 = table.column("bid_px_1").to_numpy()
    bid_sz_1 = table.column("bid_sz_1").to_numpy()
    ask_px_1 = table.column("ask_px_1").to_numpy()
    ask_sz_1 = table.column("ask_sz_1").to_numpy()

    type_counts = {t: int(np.sum(event_type == t)) for t in EVENT_TYPES}

    valid = _valid_quote_mask(bid_px_1, ask_px_1, bid_sz_1, ask_sz_1)
    n_invalid_quote_rows = int(n - np.sum(valid))
    crossed_count = int(np.sum(valid & (bid_px_1 >= ask_px_1)))

    diffs = np.diff(ts_ns)
    n_ties = int(np.sum(diffs == 0))
    n_non_monotonic = int(np.sum(diffs < 0))
    max_gap_ns = int(diffs.max()) if diffs.size else 0
    gap_idx = np.where(diffs > GAP_THRESHOLD_NS)[0]
    n_gaps_over_1s = int(gap_idx.size)
    top_gaps = sorted(
        (
            (int(diffs[i]), _ns_to_clock(int(ts_ns[i])), _ns_to_clock(int(ts_ns[i + 1])))
            for i in gap_idx
        ),
        reverse=True,
    )[:5]

    mid_raw = (bid_px_1[valid].astype(np.float64) + ask_px_1[valid].astype(np.float64)) / 2.0
    median_mid_dollars = float(np.median(mid_raw)) * DOLLARS_PER_RAW_UNIT
    min_mid_dollars = float(mid_raw.min()) * DOLLARS_PER_RAW_UNIT
    max_mid_dollars = float(mid_raw.max()) * DOLLARS_PER_RAW_UNIT

    bucket_idx = (ts_ns // BUCKET_NS).astype(np.int64)
    bucket_rows = []
    for b in np.unique(bucket_idx):
        mask = bucket_idx == b
        n_events = int(np.sum(mask))
        v = mask & valid
        if np.any(v):
            spread_ticks = (
                ask_px_1[v].astype(np.float64) - bid_px_1[v].astype(np.float64)
            ) / TICK_SIZE_RAW
            depth = (bid_sz_1[v].astype(np.float64) + ask_sz_1[v].astype(np.float64)) / 2.0
            mean_spread = float(spread_ticks.mean())
            mean_depth = float(depth.mean())
        else:
            mean_spread = float("nan")
            mean_depth = float("nan")
        bucket_rows.append((_bucket_label(int(b)), n_events, mean_spread, mean_depth))

    return {
        "ticker": ticker,
        "n_rows": n,
        "type_counts": type_counts,
        "crossed_count": crossed_count,
        "n_invalid_quote_rows": n_invalid_quote_rows,
        "n_ties": n_ties,
        "n_non_monotonic": n_non_monotonic,
        "max_gap_ns": max_gap_ns,
        "n_gaps_over_1s": n_gaps_over_1s,
        "top_gaps": top_gaps,
        "median_mid_dollars": median_mid_dollars,
        "min_mid_dollars": min_mid_dollars,
        "max_mid_dollars": max_mid_dollars,
        "bucket_rows": bucket_rows,
    }


def _is_u_shaped(bucket_rows):
    """Coarse check: first and last bucket's event rate both above the profile's median."""
    counts = np.array([r[1] for r in bucket_rows], dtype=np.float64)
    if counts.size < 3:
        return False
    median_count = float(np.median(counts))
    return counts[0] > median_count and counts[-1] > median_count


def render_ticker_section(stats):
    lines = []
    ticker = stats["ticker"]
    lines.append(f"## {ticker}")
    lines.append("")
    lines.append(f"Rows (== raw message/orderbook row count): **{stats['n_rows']}**")
    lines.append("")

    lines.append("### Event counts by type")
    lines.append("")
    lines.append("| type | meaning | count |")
    lines.append("|---|---|---|")
    for t in EVENT_TYPES:
        lines.append(f"| {t} | {EVENT_TYPE_LABELS[t]} | {stats['type_counts'][t]} |")
    lines.append("")

    lines.append("### Book integrity")
    lines.append("")
    lines.append(f"- Crossed-book count (best bid >= best ask, valid quotes only): "
                  f"**{stats['crossed_count']}**")
    lines.append(f"- Rows with an invalid/sentinel best-level quote (excluded from the "
                  f"crossed check and spread/depth stats): {stats['n_invalid_quote_rows']}")
    lines.append("")

    lines.append("### Timestamp monotonicity")
    lines.append("")
    lines.append(f"- Non-monotonic (ts_ns decreasing) adjacent pairs: "
                  f"**{stats['n_non_monotonic']}**")
    lines.append(f"- Exact ts_ns ties (adjacent pairs, diff == 0): {stats['n_ties']}")
    lines.append(f"- Largest inter-event gap: {stats['max_gap_ns'] / NS_PER_SEC:.3f} s")
    lines.append(f"- Gaps > 1s: {stats['n_gaps_over_1s']}")
    if stats["top_gaps"]:
        lines.append("")
        lines.append("| gap (s) | from | to |")
        lines.append("|---|---|---|")
        for gap_ns, t_from, t_to in stats["top_gaps"]:
            lines.append(f"| {gap_ns / NS_PER_SEC:.3f} | {t_from} | {t_to} |")
    lines.append("")

    lines.append("### Price scaling sanity")
    lines.append("")
    lines.append(f"- Median mid: ${stats['median_mid_dollars']:.2f} "
                  f"(range ${stats['min_mid_dollars']:.2f}-${stats['max_mid_dollars']:.2f})")
    bounds = MEDIAN_MID_BOUNDS_DOLLARS.get(stats["ticker"])
    if bounds is not None:
        lines.append(f"- Expected range: [{bounds[0]}, {bounds[1]}] -- "
                      f"{'PASS' if bounds[0] <= stats['median_mid_dollars'] <= bounds[1] else 'FAIL'}")
    lines.append("")

    lines.append(f"### Intraday profile ({BUCKET_MINUTES}-min buckets)")
    lines.append("")
    lines.append("| bucket | events | mean spread (ticks) | mean top-of-book depth (shares) |")
    lines.append("|---|---|---|---|")
    for label, n_events, mean_spread, mean_depth in stats["bucket_rows"]:
        lines.append(f"| {label} | {n_events} | {mean_spread:.3f} | {mean_depth:.1f} |")
    lines.append("")
    lines.append(f"U-shape in event rate (first and last bucket above median): "
                  f"**{'yes' if _is_u_shaped(stats['bucket_rows']) else 'no'}**")
    lines.append("")

    return "\n".join(lines)


def render_report(all_stats):
    lines = [
        "# Data quality report",
        "",
        "Generated by `src/edge/io/quality.py` from the parquet frames in `data/frames/` "
        "(reproduce with `make data`). Not hand-edited.",
        "",
    ]
    for stats in all_stats:
        lines.append(render_ticker_section(stats))
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description="Data-quality report over built event frames.")
    parser.add_argument("--frames", required=True, help="directory of {TICKER}_{DATE}.parquet frames")
    parser.add_argument("--out", required=True, help="path to write the markdown report to")
    parser.add_argument(
        "--seed", type=int, required=True, help="logged for consistency; this step is deterministic"
    )
    args = parser.parse_args(argv)

    random.seed(args.seed)
    print(f"[edge.io.quality] seed={args.seed} (no randomness in this step)")

    paths = sorted(glob.glob(os.path.join(args.frames, "*.parquet")))
    if not paths:
        raise SystemExit(f"[edge.io.quality] no parquet frames found under {args.frames!r}")

    all_stats = []
    for path in paths:
        ticker = os.path.basename(path).split("_", 1)[0]
        stats = analyze_frame(path, ticker)
        all_stats.append(stats)

        bounds = MEDIAN_MID_BOUNDS_DOLLARS.get(ticker)
        if bounds is not None and not (bounds[0] <= stats["median_mid_dollars"] <= bounds[1]):
            raise SystemExit(
                f"[edge.io.quality] {ticker}: median mid ${stats['median_mid_dollars']:.2f} "
                f"outside expected [{bounds[0]}, {bounds[1]}] -- price scaling is broken, "
                f"fix the adapter, do not report around it"
            )
        if stats["crossed_count"] != 0:
            raise SystemExit(
                f"[edge.io.quality] {ticker}: crossed-book count is "
                f"{stats['crossed_count']} (expected 0) -- the parse is broken, fix it "
                f"before continuing (see DECISIONS.md)"
            )

    report = render_report(all_stats)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write(report)
    print(f"[edge.io.quality] wrote {args.out}")


if __name__ == "__main__":
    main()
