"""Data-quality checks + report for the event-indexed LOBSTER frame (Phase 0 milestone).

Sanity plots are rendered as markdown tables + ASCII bars, not matplotlib, to keep the
zero-dependency ethos (stdlib + numpy + pyarrow only) -- see DECISIONS.md, 2026-07-30.

Hard invariants (non-zero exit, fails `make data`):
  - crossed-book count (bid_px_1 >= ask_px_1) > 0
  - row-count mismatch across raw message / raw orderbook / parquet frame
  - non-monotone ts_ns (any strict decrease)
Everything else (event-count-vs-contract mismatch, median-mid out of the sanity bounds,
absence of the U-shape) is logged loudly in the report but does not block the build -- these
are "look at this" signals, not proof by themselves that the parser is broken.
"""
import argparse
import os
import sys

import numpy as np
import pyarrow.parquet as pq

from edge.io.lobster import ASK_SENTINEL_PRICE, BID_SENTINEL_PRICE, discover_pairs

TICK_UNITS_PER_DOLLAR = 10_000  # raw price units (0.01-cent) per dollar
TICK_SIZE_UNITS = 100  # $0.01 NMS tick == 100 raw price units

EVENT_TYPE_NAMES = {
    1: "new limit order",
    2: "partial cancel",
    3: "full delete",
    4: "visible execution",
    5: "hidden execution",
    6: "cross/auction trade",
    7: "trading halt",
}

# docs/DATA_CONTRACT.md counts table, 2012-06-21 free sample, 10-level, full day.
EXPECTED_EVENT_COUNTS = {
    "INTC": {1: 304790, 2: 8028, 3: 278739, 4: 28924, 5: 3559, 6: 0, 7: 0},
    "AMZN": {1: 131954, 2: 2917, 3: 123458, 4: 8974, 5: 2445, 6: 0, 7: 0},
}

SANITY_MID_BOUNDS = {
    "INTC": (20.0, 35.0),
    "AMZN": (200.0, 250.0),
}

DAY_START_NS = 34_200 * 1_000_000_000  # 09:30:00 after midnight
DAY_END_NS = 57_600 * 1_000_000_000  # 16:00:00 after midnight
BIN_NS = 30 * 60 * 1_000_000_000  # 30-minute intraday bins
GAP_THRESHOLD_NS = 1_000_000_000  # 1s, per PLAN.md's gap-detection ask


# --------------------------------------------------------------------------------------
# Pure checks (numpy in, plain python out) -- unit tested in tests/test_quality.py.
# --------------------------------------------------------------------------------------


def count_lines(path):
    """Row count of a headerless CSV via a streamed line count (no full parse needed)."""
    n = 0
    with open(path, "rb") as f:
        for _ in f:
            n += 1
    return n


def crossed_locked_counts(bid_px_1, ask_px_1):
    """(crossed count: bid>=ask, locked count: bid==ask). Crossed includes locked, per spec."""
    bid = np.asarray(bid_px_1)
    ask = np.asarray(ask_px_1)
    crossed = int(np.sum(bid >= ask))
    locked = int(np.sum(bid == ask))
    return crossed, locked


def sentinel_row_count(bid_px_1, ask_px_1):
    """Rows where the best level is LOBSTER's no-displayed-level sentinel (informational only).

    Zero occurrences confirmed for this sample in `docs/DATA_CONTRACT.md`; checked here so a
    future universe with real sentinel rows doesn't silently corrupt the crossed-book count.
    """
    bid = np.asarray(bid_px_1)
    ask = np.asarray(ask_px_1)
    return int(np.sum((bid == BID_SENTINEL_PRICE) | (ask == ASK_SENTINEL_PRICE)))


def check_monotonic_ts(ts_ns):
    """ts_ns non-decreasing check: is_monotonic, exact-tie count, violation count/index."""
    ts = np.asarray(ts_ns)
    if len(ts) < 2:
        return {
            "is_monotonic": True,
            "num_ties": 0,
            "num_violations": 0,
            "first_violation_index": None,
        }
    diffs = np.diff(ts)
    num_ties = int(np.sum(diffs == 0))
    violation_idxs = np.nonzero(diffs < 0)[0]
    num_violations = int(len(violation_idxs))
    first_violation_index = int(violation_idxs[0] + 1) if num_violations else None
    return {
        "is_monotonic": num_violations == 0,
        "num_ties": num_ties,
        "num_violations": num_violations,
        "first_violation_index": first_violation_index,
    }


def top_k_gaps(ts_ns, k=5):
    """Largest inter-event gaps, descending: list of dicts with gap_ns and both endpoints."""
    ts = np.asarray(ts_ns)
    if len(ts) < 2:
        return []
    diffs = np.diff(ts)
    k = min(k, len(diffs))
    top_idx = np.argsort(diffs)[::-1][:k]
    out = [
        {
            "gap_ns": int(diffs[i]),
            "before_idx": int(i),
            "after_idx": int(i + 1),
            "before_ts_ns": int(ts[i]),
            "after_ts_ns": int(ts[i + 1]),
        }
        for i in top_idx
    ]
    out.sort(key=lambda d: d["gap_ns"], reverse=True)
    return out


def gaps_over_threshold(ts_ns, threshold_ns):
    """Count of inter-event intervals strictly greater than threshold_ns."""
    ts = np.asarray(ts_ns)
    if len(ts) < 2:
        return 0
    diffs = np.diff(ts)
    return int(np.sum(diffs > threshold_ns))


def median_mid_dollars(bid_px_1, ask_px_1):
    bid = np.asarray(bid_px_1, dtype=np.float64)
    ask = np.asarray(ask_px_1, dtype=np.float64)
    mid_units = (bid + ask) / 2.0
    return float(np.median(mid_units)) / TICK_UNITS_PER_DOLLAR


def ns_to_hhmmss(ts_ns):
    total_seconds = int(ts_ns) // 1_000_000_000
    hh = total_seconds // 3600
    mm = (total_seconds % 3600) // 60
    ss = total_seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def build_intraday_profile(
    ts_ns,
    bid_px_1,
    bid_sz_1,
    ask_px_1,
    ask_sz_1,
    day_start_ns=DAY_START_NS,
    day_end_ns=DAY_END_NS,
    bin_ns=BIN_NS,
):
    """Per-bin event count, mean L1 spread (ticks & dollars), mean L1 depth (bid+ask shares)."""
    ts = np.asarray(ts_ns)
    bid_px = np.asarray(bid_px_1, dtype=np.float64)
    ask_px = np.asarray(ask_px_1, dtype=np.float64)
    bid_sz = np.asarray(bid_sz_1, dtype=np.float64)
    ask_sz = np.asarray(ask_sz_1, dtype=np.float64)

    spread_units = ask_px - bid_px
    depth = bid_sz + ask_sz  # shares at best bid + best ask, per spec (sum, not average)

    n_bins = int(np.ceil((day_end_ns - day_start_ns) / bin_ns))
    bin_idx = np.clip(((ts - day_start_ns) // bin_ns).astype(np.int64), 0, n_bins - 1)

    rows = []
    for b in range(n_bins):
        mask = bin_idx == b
        count = int(np.sum(mask))
        bin_start = day_start_ns + b * bin_ns
        bin_end = bin_start + bin_ns
        if count > 0:
            mean_spread_units = float(np.mean(spread_units[mask]))
            mean_depth = float(np.mean(depth[mask]))
        else:
            mean_spread_units = float("nan")
            mean_depth = float("nan")
        rows.append(
            {
                "bin_start": ns_to_hhmmss(bin_start),
                "bin_end": ns_to_hhmmss(bin_end),
                "event_count": count,
                "mean_spread_ticks": mean_spread_units / TICK_SIZE_UNITS if count else float("nan"),
                "mean_spread_dollars": mean_spread_units / TICK_UNITS_PER_DOLLAR if count else float("nan"),
                "mean_l1_depth": mean_depth,
            }
        )
    return rows


def confirm_u_shape(event_counts):
    """True if the mean rate in the first+last quarter of bins exceeds the interior mean.

    Deliberately simple, defensible operationalisation of PLAN.md's "confirm the U-shape":
    open/close activity should dominate the midday lull. Quartile edges (not just the first
    and last single bin) so one noisy half-hour doesn't flip the verdict.
    """
    counts = list(event_counts)
    n = len(counts)
    if n < 4:
        return False
    edge_n = max(1, n // 4)
    edges = counts[:edge_n] + counts[-edge_n:]
    middle = counts[edge_n:-edge_n]
    if not middle:
        return False
    return (sum(edges) / len(edges)) > (sum(middle) / len(middle))


def ascii_bar(value, max_value, width=30, char="#"):
    if not max_value or max_value <= 0 or value != value:  # value != value catches NaN
        return ""
    n = int(round(width * value / max_value))
    n = max(0, min(width, n))
    return char * n


# --------------------------------------------------------------------------------------
# Report assembly
# --------------------------------------------------------------------------------------


def _event_count_table(ticker, event_type_counts):
    expected = EXPECTED_EVENT_COUNTS.get(ticker, {})
    lines = [
        "| type | meaning | count | contract expected | match |",
        "|---|---|---|---|---|",
    ]
    mismatches = []
    for t in range(1, 8):
        count = event_type_counts.get(t, 0)
        exp = expected.get(t)
        exp_str = str(exp) if exp is not None else "n/a"
        if exp is None:
            match = "n/a"
        elif count == exp:
            match = "OK"
        else:
            match = "**MISMATCH**"
            mismatches.append((t, count, exp))
        lines.append(f"| {t} | {EVENT_TYPE_NAMES[t]} | {count} | {exp_str} | {match} |")
    return "\n".join(lines), mismatches


def _gap_table(gaps):
    if not gaps:
        return "(fewer than 2 events; no gaps to report)"
    lines = [
        "| rank | gap (s) | before idx | before wall-clock | after wall-clock |",
        "|---|---|---|---|---|",
    ]
    for rank, g in enumerate(gaps, start=1):
        gap_s = g["gap_ns"] / 1e9
        lines.append(
            f"| {rank} | {gap_s:.6f} | {g['before_idx']} | "
            f"{ns_to_hhmmss(g['before_ts_ns'])} | {ns_to_hhmmss(g['after_ts_ns'])} |"
        )
    return "\n".join(lines)


def _intraday_table(profile):
    max_count = max((r["event_count"] for r in profile), default=0)
    finite_spreads = [r["mean_spread_ticks"] for r in profile if r["event_count"] > 0]
    finite_depths = [r["mean_l1_depth"] for r in profile if r["event_count"] > 0]
    max_spread = max(finite_spreads, default=0)
    max_depth = max(finite_depths, default=0)

    lines = [
        "| bin | events | rate bar | mean spread (ticks) | spread bar | "
        "mean spread ($) | mean L1 depth (shares) | depth bar |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in profile:
        spread_ticks = r["mean_spread_ticks"]
        spread_dollars = r["mean_spread_dollars"]
        depth = r["mean_l1_depth"]
        spread_ticks_str = f"{spread_ticks:.3f}" if spread_ticks == spread_ticks else "n/a"
        spread_dollars_str = f"{spread_dollars:.4f}" if spread_dollars == spread_dollars else "n/a"
        depth_str = f"{depth:.0f}" if depth == depth else "n/a"
        lines.append(
            f"| {r['bin_start']}-{r['bin_end']} | {r['event_count']} | "
            f"`{ascii_bar(r['event_count'], max_count)}` | {spread_ticks_str} | "
            f"`{ascii_bar(spread_ticks, max_spread)}` | {spread_dollars_str} | "
            f"{depth_str} | `{ascii_bar(depth, max_depth)}` |"
        )
    return "\n".join(lines)


def build_ticker_report(ticker, date, raw_msg_rows, raw_book_rows, parquet_rows, table):
    ts_ns = table.column("ts_ns").to_numpy(zero_copy_only=False)
    event_type = table.column("event_type").to_numpy(zero_copy_only=False)
    bid_px_1 = table.column("bid_px_1").to_numpy(zero_copy_only=False)
    bid_sz_1 = table.column("bid_sz_1").to_numpy(zero_copy_only=False)
    ask_px_1 = table.column("ask_px_1").to_numpy(zero_copy_only=False)
    ask_sz_1 = table.column("ask_sz_1").to_numpy(zero_copy_only=False)

    row_counts_match = raw_msg_rows == raw_book_rows == parquet_rows

    event_type_counts = {t: int(np.sum(event_type == t)) for t in range(1, 8)}
    event_table, mismatches = _event_count_table(ticker, event_type_counts)

    crossed, locked = crossed_locked_counts(bid_px_1, ask_px_1)
    n_sentinel = sentinel_row_count(bid_px_1, ask_px_1)

    mono = check_monotonic_ts(ts_ns)
    gaps = top_k_gaps(ts_ns, k=5)
    n_gaps_over_1s = gaps_over_threshold(ts_ns, GAP_THRESHOLD_NS)

    median_mid = median_mid_dollars(bid_px_1, ask_px_1)
    lo, hi = SANITY_MID_BOUNDS[ticker]
    mid_ok = lo <= median_mid <= hi

    profile = build_intraday_profile(ts_ns, bid_px_1, bid_sz_1, ask_px_1, ask_sz_1)
    u_shape = confirm_u_shape([r["event_count"] for r in profile])

    hard_failures = []
    if crossed > 0:
        hard_failures.append(f"crossed-book count = {crossed} (expected 0)")
    if not row_counts_match:
        hard_failures.append(
            f"row-count mismatch: raw message={raw_msg_rows}, "
            f"raw orderbook={raw_book_rows}, parquet={parquet_rows}"
        )
    if not mono["is_monotonic"]:
        hard_failures.append(
            f"ts_ns not monotone: {mono['num_violations']} violation(s), "
            f"first at index {mono['first_violation_index']}"
        )

    lines = [
        f"## {ticker} -- {date}",
        "",
        "### Row counts",
        "",
        "| source | rows |",
        "|---|---|",
        f"| raw message file | {raw_msg_rows} |",
        f"| raw orderbook file | {raw_book_rows} |",
        f"| parquet frame | {parquet_rows} |",
        "",
        f"Row-count equality: {'PASS' if row_counts_match else '**FAIL**'}",
        "",
        "### Event counts by type",
        "",
        event_table,
        "",
    ]
    if mismatches:
        lines.append(
            f"**WARNING**: {len(mismatches)} event type(s) mismatch the "
            f"`docs/DATA_CONTRACT.md` counts table: "
            + ", ".join(f"type {t} (got {c}, expected {e})" for t, c, e in mismatches)
        )
    else:
        lines.append("All event-type counts match `docs/DATA_CONTRACT.md` exactly.")
    lines += [
        "",
        "### Book integrity",
        "",
        f"- crossed-book count (`bid_px_1 >= ask_px_1`): {crossed} (expected 0) -- "
        f"{'PASS' if crossed == 0 else '**FAIL**'}",
        f"- locked-book count (`bid_px_1 == ask_px_1`): {locked}",
        f"- sentinel best-level rows (informational, defensive check): {n_sentinel}",
        "",
        "### Timestamp monotonicity",
        "",
        f"- monotonic non-decreasing: {mono['is_monotonic']}",
        f"- exact-tie events (consecutive equal ts_ns): {mono['num_ties']}",
        f"- violations (strict decrease): {mono['num_violations']}"
        + (f" (first at index {mono['first_violation_index']})" if mono["num_violations"] else ""),
        "",
        "### Top-5 largest inter-event gaps",
        "",
        _gap_table(gaps),
        "",
        f"Gaps > 1s with no events: {n_gaps_over_1s}",
        "",
        "### Price scaling sanity",
        "",
        f"- median mid: ${median_mid:.4f} (contract bounds [{lo}, {hi}]) -- "
        f"{'PASS' if mid_ok else '**FAIL**'}",
        "",
        "### Intraday profile (30-minute bins)",
        "",
        _intraday_table(profile),
        "",
        f"U-shape (open/close activity > midday) confirmed: {u_shape}",
        "",
    ]
    stats = {
        "crossed_book": crossed,
        "locked_book": locked,
        "monotonic": mono["is_monotonic"],
        "num_ties": mono["num_ties"],
        "u_shape": u_shape,
        "median_mid": median_mid,
        "mid_ok": mid_ok,
        "row_counts_match": row_counts_match,
        "hard_failures": hard_failures,
    }
    return "\n".join(lines), stats


def write_report(out_path, header, ticker_sections):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(header)
        f.write("\n\n")
        f.write("\n\n".join(ticker_sections))
        f.write("\n")


REPORT_HEADER = """# Data Quality Report

Generated by `python3 -m edge.io.quality`, part of `make data`. Universe: LOBSTER free
sample, 2012-06-21, 10 levels, INTC (large-tick) and AMZN (small-tick). See
`docs/DATA_CONTRACT.md` for the frozen schema and `docs/spec/PLAN.md` for the phase-0
milestone this report satisfies.

Hard invariants (fail `make data` if violated): crossed-book count = 0, raw
message/orderbook/parquet row counts equal, ts_ns non-decreasing. Everything else is a
loud warning, not a build failure. No matplotlib: all profiles below are markdown tables
with ASCII bar columns (see DECISIONS.md, 2026-07-30)."""


def main(argv=None):
    parser = argparse.ArgumentParser(description="Data-quality report for the LOBSTER event frame.")
    parser.add_argument("--raw", required=True, help="directory of raw LOBSTER message/orderbook CSVs")
    parser.add_argument("--frames", required=True, help="directory of {TICKER}_{DATE}.parquet frames")
    parser.add_argument("--out", required=True, help="output path for the markdown report")
    parser.add_argument("--seed", type=int, default=0)  # every entry point takes --seed, per CLAUDE.md
    args = parser.parse_args(argv)

    print(f"seed={args.seed} (deterministic step, no randomness consumed)")

    pairs = discover_pairs(args.raw)
    if not pairs:
        print(f"no message/orderbook pairs found under {args.raw}", file=sys.stderr)
        return 1

    sections = []
    all_hard_failures = []
    summary = {}
    for pair in pairs:
        ticker, date = pair["ticker"], pair["date"]
        parquet_path = os.path.join(args.frames, f"{ticker}_{date}.parquet")
        if not os.path.exists(parquet_path):
            print(f"missing parquet frame for {ticker}_{date}: {parquet_path}", file=sys.stderr)
            return 1

        raw_msg_rows = count_lines(pair["message_path"])
        raw_book_rows = count_lines(pair["orderbook_path"])
        table = pq.read_table(
            parquet_path,
            columns=["ts_ns", "event_type", "bid_px_1", "bid_sz_1", "ask_px_1", "ask_sz_1"],
        )
        parquet_rows = table.num_rows

        section, stats = build_ticker_report(
            ticker, date, raw_msg_rows, raw_book_rows, parquet_rows, table
        )
        sections.append(section)
        summary[ticker] = stats
        for failure in stats["hard_failures"]:
            all_hard_failures.append(f"{ticker}_{date}: {failure}")
            print(f"HARD FAILURE [{ticker}_{date}]: {failure}", file=sys.stderr)

    write_report(args.out, REPORT_HEADER, sections)
    print(f"wrote {args.out}")
    for ticker, stats in summary.items():
        print(
            f"{ticker}: crossed={stats['crossed_book']} monotonic={stats['monotonic']} "
            f"u_shape={stats['u_shape']} median_mid=${stats['median_mid']:.4f} "
            f"(bounds ok={stats['mid_ok']})"
        )

    if all_hard_failures:
        print(f"{len(all_hard_failures)} hard invariant failure(s), failing the build:", file=sys.stderr)
        for f in all_hard_failures:
            print(f"  - {f}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
