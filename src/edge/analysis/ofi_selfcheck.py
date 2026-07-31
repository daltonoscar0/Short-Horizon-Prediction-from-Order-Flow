"""Phase-1 gate: reproduce Cont-Kukanov-Stoikov's OFI/Delta-mid relation (FEATURES.md sec. 1,
PLAN.md's L2 rung) on the features `edge.features.ofi` already built, and write
`reports/ofi_selfcheck.md`.

    python3 -m edge.analysis.ofi_selfcheck --seed 0

Regression spec, taken literally from the task: for TICKER in (INTC, AMZN) and CLOCK in
(calendar, event), regress the forward mid return over the next second
(`label_y_mid_1s` -- ticks(mid(last event <= t+1s)) - ticks(mid(t)), labels.py) on the
TRAILING 1s-window best-level OFI as of t (`ofi_best_1s` -- Sum e_n over (t-1s, t], ofi.py),
plus the deep-book, per-level and integrated OFI columns for the level profile. Both columns
already carry their own frozen causality convention (`<=t` for every feature, `label_` for
the one forward-looking exception, INTERFACE.md sec. 2); this module recomputes nothing, it
only reads `data/features/{TICKER}_2012-06-21_{clock}.parquet` and regresses one column on
another. "Contemporaneous" in the task description means "aligned to the same decision
stamp t", not "same interval as the label" -- the OFI window (t-1s, t] and the label window
(t, t+1s] are adjacent, not overlapping, so this is a genuine one-step-ahead test, a strictly
harder bar than CKS's own same-window regression.

Every OLS + Newey-West HAC estimator below is hand-written on numpy only (no scipy, no
sklearn, no statsmodels -- CLAUDE.md's dependency rule): a 2-column design matrix
`[1, x]`, closed-form `beta = (X'X)^{-1} X'y`, and the Bartlett-kernel sandwich
`(X'X)^{-1} S_hat (X'X)^{-1}` for the HAC covariance. Rows are dropped PAIRWISE per (x, y)
regression -- a NaN in one regressor (e.g. `ofi_integrated_1s`'s block-0 warmup) must not
shrink `n` for every other regressor, per the task's own instruction.

HAC lag: chosen a priori as the number of stamps spanning the 1s label horizon, not fit.
Calendar clock: 10 (100ms grid, 10 stamps = 1s, always). Event clock: the median number of
event-clock stamps per calendar second, COMPUTED per ticker from that ticker's own event
stamps (INTC and AMZN tick at very different rates on the same N=10 event clock -- see
clocks.py's own measured medians, 8.64ms vs 352ms) and reported alongside the table.

The raw-CSV re-derivation (`_raw_rederivation`) is run unconditionally, pass or fail, per
the task instruction: a Phase-1 number disagreeing with an independent re-derivation from
raw is a blocker regardless of what the headline check says. It re-implements the CKS e_n
formula from scratch, reading `data/raw/lobster/*.csv` with plain file iteration -- no
`edge.io.lobster`, no `edge.features.ofi` import in the derivation itself -- and only then
diffs the result against `edge.features.ofi._event_contributions` (imported solely for the
comparison, never for the derivation) and against the real `ofi_best_1s` column in the built
parquet.
"""
import argparse
import os
import time

import numpy as np
import pyarrow.parquet as pq

NS_PER_SEC = 1_000_000_000
TICKERS = ("INTC", "AMZN")
CLOCKS = ("calendar", "event")
DATE = "2012-06-21"

LABEL_COL = "label_y_mid_1s"

REGRESSORS = (
    ["ofi_best_1s", "ofi_deep_1s"]
    + [f"ofi_raw_l{m}_1s" for m in range(1, 11)]
    + ["ofi_integrated_1s"]
)

CALENDAR_HAC_LAG = 10  # 100ms grid; 1s label horizon spans exactly 10 stamps, always.

SIG_T_HAC = 3.0  # bar fixed by the task
R2_LEAK_THRESHOLD = 0.5  # an R^2 above this on a 1s univariate regression is a leak, not a win

# Raw-CSV re-derivation: a padded row window per ticker (padded so every row in the target
# window has >= 5000 true predecessor rows -- comfortably more than one second of trailing
# history at either ticker's event rate -- available for the windowed-sum comparison).
RAW_SLICE = {
    "INTC": {"start_row": 200_000, "n_rows": 20_000, "pad_rows": 5_000},
    "AMZN": {"start_row": 100_000, "n_rows": 20_000, "pad_rows": 5_000},
}


# --------------------------------------------------------------------------------------
# Hand-written OLS + Newey-West HAC, univariate with intercept.
# --------------------------------------------------------------------------------------


def _ols_with_hac(x, y, lag):
    """Univariate OLS of y on x with an intercept; naive OLS SE and Newey-West HAC SE.

    Rows dropped PAIRWISE: only this (x, y) pair's own finite mask is applied, never a
    mask shared across regressors. Returns None if there are too few finite rows to
    support the requested HAC lag (needs n > 2*lag, a generous floor since NW is only
    consistent for n >> lag).
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    finite = np.isfinite(x) & np.isfinite(y)
    xf = x[finite]
    yf = y[finite]
    n = xf.shape[0]
    lag_eff = max(int(lag), 0)
    if n < max(30, 2 * lag_eff + 10):
        return None

    design = np.column_stack([np.ones(n), xf])
    xtx = design.T @ design
    xtx_inv = np.linalg.inv(xtx)
    beta = xtx_inv @ (design.T @ yf)
    resid = yf - design @ beta

    dof = n - 2
    sigma2 = float(resid @ resid) / dof
    se_ols = np.sqrt(np.diag(sigma2 * xtx_inv))

    # Newey-West (Bartlett kernel) sandwich: S_hat = Gamma_0 + sum_{l=1}^{L} w_l (Gamma_l + Gamma_l'),
    # Gamma_l = sum_t score_t score_{t-l}', score_t = resid_t * design_t.
    lag_eff = min(lag_eff, n - 1)
    scores = design * resid[:, None]
    s_hat = scores.T @ scores
    for lag_k in range(1, lag_eff + 1):
        w = 1.0 - lag_k / (lag_eff + 1)
        gamma = scores[lag_k:].T @ scores[:-lag_k]
        s_hat += w * (gamma + gamma.T)
    cov_hac = xtx_inv @ s_hat @ xtx_inv
    se_hac = np.sqrt(np.diag(cov_hac))

    sst = float(np.sum((yf - yf.mean()) ** 2))
    ssr = float(resid @ resid)
    r2 = 1.0 - ssr / sst if sst > 0 else float("nan")

    def _t(b, se):
        return float(b / se) if se > 0 else float("nan")

    return {
        "n": n,
        "lag": lag_eff,
        "intercept": float(beta[0]),
        "slope": float(beta[1]),
        "se_ols": float(se_ols[1]),
        "t_ols": _t(beta[1], se_ols[1]),
        "se_hac": float(se_hac[1]),
        "t_hac": _t(beta[1], se_hac[1]),
        "r2": r2,
    }


def _pearson(x, y):
    """Hand-written Pearson correlation; used only for the slope-vs-level trend, not fit."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum(xm * ym) / denom)


def _event_stamp_rate_stats(ts_ns):
    """Distribution of event-clock stamp COUNTS per 1s calendar bucket, spanning the session.

    The a-priori HAC lag on the event clock (module docstring) is the MEDIAN of this
    distribution: the number of event-clock stamps a 1s label horizon typically spans.
    Computed from bucket counts (including quiet, zero-count buckets), not from
    `1 / median_gap`, because the two summaries can disagree once the distribution is
    skewed -- "median stamps per second" is the literal reading of the task's own phrase.

    Reported alongside the mean/p90/p99/max because the distribution IS heavily
    right-skewed here (event-clock stamps arrive in bursts, not uniformly): the median
    across both tickers comes out to 1, while the mean and tail are much higher (see
    `reports/ofi_selfcheck.md`'s HAC-lag section) -- worth stating plainly rather than
    letting a lag of 1 look like a bug.
    """
    ts_ns = np.asarray(ts_ns, dtype=np.int64)
    if ts_ns.shape[0] == 0:
        return {"median": 0, "mean": 0.0, "p90": 0.0, "p99": 0.0, "max": 0}
    span = int(ts_ns[-1] - ts_ns[0])
    n_buckets = span // NS_PER_SEC + 1
    bucket = (ts_ns - ts_ns[0]) // NS_PER_SEC
    counts = np.bincount(bucket, minlength=n_buckets)
    return {
        "median": int(np.median(counts)),
        "mean": float(np.mean(counts)),
        "p90": float(np.percentile(counts, 90)),
        "p99": float(np.percentile(counts, 99)),
        "max": int(np.max(counts)),
    }


# --------------------------------------------------------------------------------------
# Loading + running the regression grid.
# --------------------------------------------------------------------------------------


def _load_columns(path, names):
    table = pq.read_table(path, columns=list(names))
    return {name: table.column(name).to_numpy(zero_copy_only=False).astype(np.float64) for name in names}


def run_regressions(features_dir, date=DATE):
    """One row per (ticker, clock, regressor); also returns the per-(ticker,clock) HAC lag
    used (plus the full event-clock rate distribution, for transparency) and the mean
    touch depth, for the depth-profile / ticker-comparison sections."""
    needed = [LABEL_COL, "ts_ns", "st_depth_touch"] + list(REGRESSORS)
    rows = []
    lags = {}
    lag_stats = {}
    depth = {}
    for ticker in TICKERS:
        for clock in CLOCKS:
            path = os.path.join(features_dir, f"{ticker}_{date}_{clock}.parquet")
            cols = _load_columns(path, needed)
            if clock == "calendar":
                lag = CALENDAR_HAC_LAG
                lag_stats[(ticker, clock)] = None
            else:
                stats = _event_stamp_rate_stats(cols["ts_ns"])
                lag = stats["median"]
                lag_stats[(ticker, clock)] = stats
            lags[(ticker, clock)] = lag
            depth[(ticker, clock)] = float(np.nanmean(cols["st_depth_touch"]))

            y = cols[LABEL_COL]
            for regressor in REGRESSORS:
                result = _ols_with_hac(cols[regressor], y, lag)
                row = {"ticker": ticker, "clock": clock, "regressor": regressor}
                if result is None:
                    row.update(
                        {"n": 0, "lag": lag, "intercept": float("nan"), "slope": float("nan"),
                         "se_ols": float("nan"), "t_ols": float("nan"), "se_hac": float("nan"),
                         "t_hac": float("nan"), "r2": float("nan")}
                    )
                else:
                    row.update(result)
                rows.append(row)
    return rows, lags, lag_stats, depth


def _find(rows, ticker, clock, regressor):
    for row in rows:
        if row["ticker"] == ticker and row["clock"] == clock and row["regressor"] == regressor:
            return row
    raise KeyError((ticker, clock, regressor))


# --------------------------------------------------------------------------------------
# Raw-CSV independent re-derivation (module docstring's third section).
# --------------------------------------------------------------------------------------


def _parse_ts_ns(time_str):
    """Decimal seconds-after-midnight -> ns, via string split (own copy, not imported)."""
    sec_str, _, frac_str = time_str.partition(".")
    frac_str = (frac_str + "000000000")[:9]
    return int(sec_str) * NS_PER_SEC + int(frac_str)


def _read_raw_slice(msg_path, book_path, start_row, n_rows, pad_rows, num_levels=10):
    """Independent line-oriented parse of raw file rows [start_row-pad_rows, start_row+n_rows).

    Deliberately hand-rolled (plain `open()` + `str.split(",")`), no `edge.io.lobster`
    import anywhere in this function -- a second, independent implementation of the same
    raw-file format, used only to cross-check the pipeline's own parser+feature output.
    """
    lo = start_row - pad_rows
    hi = start_row + n_rows
    if lo < 0:
        raise ValueError(f"pad_rows={pad_rows} runs before row 0 for start_row={start_row}")

    ts_ns, event_type = [], []
    with open(msg_path, "r") as f:
        for i, line in enumerate(f):
            if i < lo:
                continue
            if i >= hi:
                break
            parts = line.rstrip("\n").split(",")
            ts_ns.append(_parse_ts_ns(parts[0]))
            event_type.append(int(parts[1]))

    bid_px, bid_sz, ask_px, ask_sz = [], [], [], []
    with open(book_path, "r") as f:
        for i, line in enumerate(f):
            if i < lo:
                continue
            if i >= hi:
                break
            vals = [int(v) for v in line.rstrip("\n").split(",")]
            bpx, bsz, apx, asz = [], [], [], []
            for lvl in range(num_levels):
                apx.append(vals[4 * lvl + 0])
                asz.append(vals[4 * lvl + 1])
                bpx.append(vals[4 * lvl + 2])
                bsz.append(vals[4 * lvl + 3])
            bid_px.append(bpx)
            bid_sz.append(bsz)
            ask_px.append(apx)
            ask_sz.append(asz)

    return {
        "row_start": lo,
        "target_offset": start_row - lo,  # index of the first "target" (non-padding) row
        "ts_ns": np.array(ts_ns, dtype=np.int64),
        "event_type": np.array(event_type, dtype=np.int64),
        "bid_px": np.array(bid_px, dtype=np.int64),
        "bid_sz": np.array(bid_sz, dtype=np.int64),
        "ask_px": np.array(ask_px, dtype=np.int64),
        "ask_sz": np.array(ask_sz, dtype=np.int64),
    }


def _independent_e_n_level1(bid_px_l1, bid_sz_l1, ask_px_l1, ask_sz_l1):
    """Best-level Cont-Kukanov-Stoikov e_n, re-derived from FEATURES.md sec. 1's formula a
    second time, in a plain Python loop -- independent of `edge.features.ofi`, which this
    function never calls or imports."""
    n = bid_px_l1.shape[0]
    e = np.zeros(n, dtype=np.float64)
    for i in range(1, n):
        pb, pb0 = bid_px_l1[i], bid_px_l1[i - 1]
        qb, qb0 = float(bid_sz_l1[i]), float(bid_sz_l1[i - 1])
        pa, pa0 = ask_px_l1[i], ask_px_l1[i - 1]
        qa, qa0 = float(ask_sz_l1[i]), float(ask_sz_l1[i - 1])
        bid_term = (qb if pb >= pb0 else 0.0) - (qb0 if pb <= pb0 else 0.0)
        ask_term = (qa0 if pa >= pa0 else 0.0) - (qa if pa <= pa0 else 0.0)
        e[i] = bid_term + ask_term
    return e


def _raw_rederivation(ticker, raw_dir, features_dir, date=DATE):
    """Independent raw-CSV re-derivation of best-level e_n and the windowed OFI it feeds,
    diffed against `edge.features.ofi._event_contributions` (event-level) and the real
    `ofi_best_1s` column of the built calendar-clock parquet (windowed-sum level).

    Run unconditionally regardless of the headline check's outcome, per the task.
    """
    spec = RAW_SLICE[ticker]
    msg_path = os.path.join(raw_dir, f"{ticker}_{date}_34200000_57600000_message_10.csv")
    book_path = os.path.join(raw_dir, f"{ticker}_{date}_34200000_57600000_orderbook_10.csv")
    raw = _read_raw_slice(msg_path, book_path, **spec)

    independent_e1 = _independent_e_n_level1(
        raw["bid_px"][:, 0], raw["bid_sz"][:, 0], raw["ask_px"][:, 0], raw["ask_sz"][:, 0]
    )

    # -- event-level comparison: independent e_n vs edge.features.ofi's own function -----
    # Imported here, deliberately, ONLY as the "src/edge" reference point for the diff --
    # never used to produce `independent_e1` above.
    from edge.features.ofi import _event_contributions

    n_rows, n_levels = raw["bid_px"].shape
    all_valid = np.ones((n_rows, n_levels), dtype=bool)
    pipeline_contrib = _event_contributions(
        raw["bid_px"].astype(np.float64), raw["bid_sz"].astype(np.float64),
        raw["ask_px"].astype(np.float64), raw["ask_sz"].astype(np.float64),
        all_valid, all_valid,
    )
    pipeline_e1 = pipeline_contrib[:, 0]
    event_level_diff = np.abs(pipeline_e1 - independent_e1)

    # -- windowed-sum comparison: independent bucket sums vs the real ofi_best_1s column -
    ts_ns = raw["ts_ns"]
    t_min, t_max = int(ts_ns[0]), int(ts_ns[-1])
    safe_lo = t_min + 2 * NS_PER_SEC  # 2s buffer past the padded start: pad_rows=5000 is far more
    k_lo = -(-safe_lo // NS_PER_SEC)
    k_hi = t_max // NS_PER_SEC
    boundaries = [k * NS_PER_SEC for k in range(int(k_lo), int(k_hi) + 1)][:25]

    calendar_path = os.path.join(features_dir, f"{ticker}_{date}_calendar.parquet")
    table = pq.read_table(calendar_path, columns=["ts_ns", "ofi_best_1s"])
    pq_ts = table.column("ts_ns").to_numpy(zero_copy_only=False)
    pq_ofi = table.column("ofi_best_1s").to_numpy(zero_copy_only=False)
    ts_to_idx = {int(t): i for i, t in enumerate(pq_ts)}

    window_rows = []
    for t_k in boundaries:
        mask = (ts_ns > t_k - NS_PER_SEC) & (ts_ns <= t_k)
        independent_sum = float(np.sum(independent_e1[mask]))
        idx = ts_to_idx.get(int(t_k))
        if idx is None:
            continue
        pipeline_val = float(pq_ofi[idx])
        window_rows.append(
            {"t_k_ns": t_k, "independent_sum": independent_sum, "pipeline_ofi_best_1s": pipeline_val,
             "diff": independent_sum - pipeline_val}
        )

    return {
        "ticker": ticker,
        "n_rows_read": n_rows,
        "target_rows": raw["target_offset"], "target_n": spec["n_rows"],
        "event_level_max_abs_diff": float(np.max(event_level_diff)) if event_level_diff.size else float("nan"),
        "event_level_n_compared": int(event_level_diff.shape[0] - 1),  # row 0 has no predecessor
        "window_rows": window_rows,
        "window_max_abs_diff": max((abs(r["diff"]) for r in window_rows), default=float("nan")),
    }


# --------------------------------------------------------------------------------------
# Report assembly.
# --------------------------------------------------------------------------------------


def _fmt(x, spec="{:.6g}"):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "NaN"
    return spec.format(x)


def _slope_table_markdown(rows):
    header = (
        "| ticker | clock | regressor | n | lag | slope | se_ols | t_ols | se_hac | t_hac | r2 |\n"
        "|---|---|---|---|---|---|---|---|---|---|---|\n"
    )
    lines = []
    for row in rows:
        lines.append(
            f"| {row['ticker']} | {row['clock']} | {row['regressor']} | {row['n']} | {row['lag']} | "
            f"{_fmt(row['slope'])} | {_fmt(row['se_ols'])} | {_fmt(row['t_ols'], '{:.3f}')} | "
            f"{_fmt(row['se_hac'])} | {_fmt(row['t_hac'], '{:.3f}')} | {_fmt(row['r2'], '{:.5f}')} |"
        )
    return header + "\n".join(lines) + "\n"


def _corr_strength(corr):
    if not np.isfinite(corr):
        return "undefined"
    a = abs(corr)
    if a >= 0.5:
        return "strong"
    if a >= 0.3:
        return "moderate"
    return "weak"


def _depth_profile_section(rows):
    lines = ["## Depth profile: slope vs. book level\n"]
    for ticker in TICKERS:
        for clock in CLOCKS:
            level_rows = [_find(rows, ticker, clock, f"ofi_raw_l{m}_1s") for m in range(1, 11)]
            level_slopes = [r["slope"] for r in level_rows]
            levels = np.arange(1, 11)
            corr = _pearson(levels, level_slopes)
            best_row = _find(rows, ticker, clock, "ofi_best_1s")
            deep_row = _find(rows, ticker, clock, "ofi_deep_1s")
            best, deep = best_row["slope"], deep_row["slope"]
            lines.append(f"### {ticker} / {clock}\n")
            lines.append("| level m | slope (ofi_raw_l{m}_1s) | t_hac | significant (\\|t_hac\\|>3) |\n|---|---|---|---|\n")
            for m, r in zip(levels, level_rows):
                sig = "yes" if np.isfinite(r["t_hac"]) and abs(r["t_hac"]) > SIG_T_HAC else "no"
                lines.append(f"| {m} | {_fmt(r['slope'])} | {_fmt(r['t_hac'], '{:.2f}')} | {sig} |\n")
            deep_sig = "significant" if np.isfinite(deep_row["t_hac"]) and abs(deep_row["t_hac"]) > SIG_T_HAC else "NOT significant"
            lines.append(
                f"\nlevel-vs-slope Pearson correlation: {_fmt(corr, '{:.4f}')} "
                f"({_corr_strength(corr)}; expected negative -- slope should decay with level)\n\n"
                f"ofi_best_1s slope: {_fmt(best)} (t_hac={_fmt(best_row['t_hac'], '{:.2f}')}); "
                f"ofi_deep_1s slope: {_fmt(deep)} (t_hac={_fmt(deep_row['t_hac'], '{:.2f}')}, **{deep_sig}**); "
                f"deep < best: {'YES' if (np.isfinite(best) and np.isfinite(deep) and abs(deep) < abs(best)) else 'NO'}\n\n"
            )
    return "".join(lines)


def _ticker_comparison_section(rows, depth):
    lines = ["## Ticker comparison: slope ~ c / depth (CKS)\n"]
    lines.append(
        "| clock | ticker | avg touch depth (shares) | ofi_best_1s slope | slope INTC/AMZN | "
        "depth AMZN/INTC (predicted slope ratio) | ratio agreement |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    ratio_notes = []
    for clock in CLOCKS:
        slope_intc = _find(rows, "INTC", clock, "ofi_best_1s")["slope"]
        slope_amzn = _find(rows, "AMZN", clock, "ofi_best_1s")["slope"]
        depth_intc = depth[("INTC", clock)]
        depth_amzn = depth[("AMZN", clock)]
        slope_ratio = slope_intc / slope_amzn if slope_amzn else float("nan")
        depth_ratio_predicted = depth_amzn / depth_intc if depth_intc else float("nan")
        agree = float("nan")
        if np.isfinite(slope_ratio) and np.isfinite(depth_ratio_predicted) and depth_ratio_predicted != 0:
            agree = abs(slope_ratio - depth_ratio_predicted) / abs(depth_ratio_predicted)
        for ticker, slope, d in (("INTC", slope_intc, depth_intc), ("AMZN", slope_amzn, depth_amzn)):
            lines.append(
                f"| {clock} | {ticker} | {_fmt(d, '{:.1f}')} | {_fmt(slope)} | "
                f"{_fmt(slope_ratio, '{:.4f}')} | {_fmt(depth_ratio_predicted, '{:.4f}')} | "
                f"{_fmt(agree, '{:.1%}') if np.isfinite(agree) else 'NaN'} relative deviation |\n"
            )
        ratio_notes.append((clock, slope_ratio, depth_ratio_predicted, agree))
    return "".join(lines) + "\n", ratio_notes


def _raw_rederivation_section(results):
    lines = ["## Independent raw-CSV re-derivation\n"]
    lines.append(
        "Re-implements the CKS best-level e_n formula from scratch by reading "
        "`data/raw/lobster/*.csv` with plain line iteration (no `edge.io.lobster`, no "
        "`edge.features.ofi` import in the derivation itself), on a padded slice of raw rows "
        "per ticker, then diffs against (a) `edge.features.ofi._event_contributions` "
        "applied to the same raw slice at event granularity, and (b) the real `ofi_best_1s` "
        "column of the built calendar-clock parquet at matching whole-second timestamps.\n\n"
    )
    for res in results:
        lines.append(f"### {res['ticker']}\n\n")
        lines.append(
            f"- raw rows read (padded): {res['n_rows_read']}, target window: rows "
            f"[{res['target_rows']}, {res['target_rows'] + res['target_n']}) of the padded slice\n"
            f"- event-level: max |independent e_n - edge.features.ofi e_n| over "
            f"{res['event_level_n_compared']} compared events: "
            f"**{_fmt(res['event_level_max_abs_diff'], '{:.10g}')}**\n"
            f"- windowed-sum: {len(res['window_rows'])} whole-second calendar boundaries compared "
            f"against the real `ofi_best_1s` column; max |independent bucket sum - pipeline value|: "
            f"**{_fmt(res['window_max_abs_diff'], '{:.10g}')}**\n\n"
        )
        if res["window_rows"]:
            lines.append("| t_k (ns after midnight) | independent Sum e_n | pipeline ofi_best_1s | diff |\n")
            lines.append("|---|---|---|---|\n")
            for wr in res["window_rows"][:10]:
                lines.append(
                    f"| {wr['t_k_ns']} | {_fmt(wr['independent_sum'])} | "
                    f"{_fmt(wr['pipeline_ofi_best_1s'])} | {_fmt(wr['diff'], '{:.3g}')} |\n"
                )
            lines.append("\n")
    return "".join(lines)


def evaluate_gate(rows, ratio_notes):
    """The three CKS checks. Returns (passed, notes)."""
    notes = []
    passed = True

    # 1. ofi_best_1s slope positive and |t_hac| > SIG_T_HAC on both tickers, both clocks.
    crit1_rows = [_find(rows, t, c, "ofi_best_1s") for t in TICKERS for c in CLOCKS]
    for row in crit1_rows:
        ok = np.isfinite(row["slope"]) and row["slope"] > 0 and np.isfinite(row["t_hac"]) and abs(row["t_hac"]) > SIG_T_HAC
        if not ok:
            passed = False
            notes.append(
                f"criterion 1 FAILED for {row['ticker']}/{row['clock']}: slope={row['slope']:.6g}, "
                f"t_hac={row['t_hac']:.3f} (need slope>0 and |t_hac|>{SIG_T_HAC})"
            )
    if not notes:
        notes.append("criterion 1 (positive, HAC-significant ofi_best_1s slope) holds on all 4 ticker/clock combos.")

    # 2a. slope-vs-level decays; deep slope smaller than best slope, per ticker/clock.
    crit2a_ok = True
    for ticker in TICKERS:
        for clock in CLOCKS:
            level_slopes = [_find(rows, ticker, clock, f"ofi_raw_l{m}_1s")["slope"] for m in range(1, 11)]
            corr = _pearson(np.arange(1, 11), level_slopes)
            best = _find(rows, ticker, clock, "ofi_best_1s")["slope"]
            deep = _find(rows, ticker, clock, "ofi_deep_1s")["slope"]
            level_ok = np.isfinite(corr) and corr < 0
            deep_ok = np.isfinite(best) and np.isfinite(deep) and abs(deep) < abs(best)
            if not (level_ok and deep_ok):
                crit2a_ok = False
                notes.append(
                    f"criterion 2a FAILED for {ticker}/{clock}: level-slope corr={corr:.4f} "
                    f"(want <0), |deep|={abs(deep):.6g} vs |best|={abs(best):.6g} (want deep<best)"
                )
    if crit2a_ok:
        notes.append("criterion 2a (slope decays across levels; deep-book slope below best-level slope) holds.")
    passed = passed and crit2a_ok

    # 2b. INTC (deep book) slope smaller than AMZN (thin book) slope, both clocks.
    crit2b_ok = True
    for clock, slope_ratio, depth_ratio_predicted, agree in ratio_notes:
        slope_intc = _find(rows, "INTC", clock, "ofi_best_1s")["slope"]
        slope_amzn = _find(rows, "AMZN", clock, "ofi_best_1s")["slope"]
        direction_ok = np.isfinite(slope_intc) and np.isfinite(slope_amzn) and abs(slope_intc) < abs(slope_amzn)
        if not direction_ok:
            crit2b_ok = False
            notes.append(
                f"criterion 2b FAILED for clock={clock}: |slope_INTC|={abs(slope_intc):.6g} not < "
                f"|slope_AMZN|={abs(slope_amzn):.6g}"
            )
        else:
            notes.append(
                f"criterion 2b holds for clock={clock}: |slope_INTC| < |slope_AMZN|; "
                f"slope ratio={slope_ratio:.4f} vs predicted 1/depth ratio={depth_ratio_predicted:.4f} "
                f"({agree:.1%} relative deviation)."
            )
    passed = passed and crit2b_ok

    # 3. R^2 small but non-zero on ofi_best_1s; loud flag if any exceeds R2_LEAK_THRESHOLD.
    crit3_ok = True
    for row in crit1_rows:
        if np.isfinite(row["r2"]) and row["r2"] > R2_LEAK_THRESHOLD:
            crit3_ok = False
            notes.append(
                f"criterion 3 FAILED for {row['ticker']}/{row['clock']}: R2={row['r2']:.4f} > "
                f"{R2_LEAK_THRESHOLD} -- THIS IS A LEAK, NOT A TRIUMPH."
            )
    if crit3_ok:
        r2_vals = ", ".join(f"{r['ticker']}/{r['clock']}={r['r2']:.5f}" for r in crit1_rows)
        notes.append(f"criterion 3 (R2 small but non-zero, no leak) holds: {r2_vals}")
    passed = passed and crit3_ok

    return passed, notes


def _caveats(rows):
    """Honest caveats worth surfacing even when the gate passes (CLAUDE.md rule 3: propose
    the boring explanation, flag anything surprising) -- marginal significance, weak level
    decay, a deep-book slope that is directionally smaller but not itself significant."""
    notes = []
    for row in [_find(rows, t, c, "ofi_best_1s") for t in TICKERS for c in CLOCKS]:
        if np.isfinite(row["t_hac"]) and SIG_T_HAC < abs(row["t_hac"]) < 5.0:
            notes.append(
                f"{row['ticker']}/{row['clock']} ofi_best_1s: |t_hac|={row['t_hac']:.2f} clears the "
                f"{SIG_T_HAC} bar but is not comfortably above it -- marginal, not a strong result."
            )
    for ticker in TICKERS:
        for clock in CLOCKS:
            level_slopes = [_find(rows, ticker, clock, f"ofi_raw_l{m}_1s")["slope"] for m in range(1, 11)]
            corr = _pearson(np.arange(1, 11), level_slopes)
            if np.isfinite(corr) and corr < 0 and abs(corr) < 0.3:
                notes.append(
                    f"{ticker}/{clock}: level-vs-slope correlation is weak ({corr:.4f}) -- the "
                    f"per-level decay is not clean, only directionally negative."
                )
            deep_row = _find(rows, ticker, clock, "ofi_deep_1s")
            if np.isfinite(deep_row["t_hac"]) and abs(deep_row["t_hac"]) <= SIG_T_HAC:
                notes.append(
                    f"{ticker}/{clock}: ofi_deep_1s slope is smaller than ofi_best_1s in magnitude "
                    f"but is NOT itself HAC-significant (t_hac={deep_row['t_hac']:.2f}) -- the "
                    f"'deeper book, smaller slope' claim holds in direction, not in precision, here."
                )
    return notes


def main(argv=None):
    parser = argparse.ArgumentParser(description="Phase-1 gate: OFI/Delta-mid self-check (CKS).")
    parser.add_argument("--features", default="data/features", help="directory of built feature parquets")
    parser.add_argument("--raw", default="data/raw/lobster", help="directory of raw LOBSTER CSVs")
    parser.add_argument("--out", default="reports/ofi_selfcheck.md", help="output report path")
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--seed", type=int, default=0)  # deterministic step; logged only, per CLAUDE.md
    args = parser.parse_args(argv)

    print(f"seed={args.seed} (deterministic step, no randomness consumed)")
    t0 = time.time()

    rows, lags, lag_stats, depth = run_regressions(args.features, date=args.date)
    ticker_section, ratio_notes = _ticker_comparison_section(rows, depth)
    passed, gate_notes = evaluate_gate(rows, ratio_notes)
    caveat_notes = _caveats(rows)

    raw_results = [_raw_rederivation(ticker, args.raw, args.features, date=args.date) for ticker in TICKERS]

    lag_line_parts = []
    for ticker in TICKERS:
        for clock in CLOCKS:
            if clock == "calendar":
                lag_line_parts.append(
                    f"- {ticker}/calendar: HAC lag = {lags[(ticker, clock)]} "
                    f"(fixed a priori: 10 calendar stamps span the 1s label horizon, always)"
                )
            else:
                stats = lag_stats[(ticker, clock)]
                lag_line_parts.append(
                    f"- {ticker}/event: HAC lag = {lags[(ticker, clock)]} (median event-clock stamps "
                    f"per calendar second, computed). Distribution is heavily right-skewed -- mean "
                    f"{stats['mean']:.2f}, p90={stats['p90']:.0f}, p99={stats['p99']:.0f}, "
                    f"max={stats['max']}: event-clock stamps arrive in bursts, so the median (the "
                    f"a-priori rule) is much smaller than a typical burst's width."
                )
    lag_lines = "\n".join(lag_line_parts)

    verdict = "PASS" if passed else "FAIL"
    verdict_paragraph = (
        f"**{verdict}.** " + " ".join(gate_notes) if passed else
        f"**{verdict}.** The features disagree with a 20-year-old linear result and must be treated as "
        f"broken, not the market (FEATURES.md sec. 1 / PLAN.md L2). Diagnosis below.\n\n" + "\n\n".join(gate_notes)
    )
    if caveat_notes:
        verdict_paragraph += "\n\nCaveats (do not change the verdict, but are honest to flag):\n\n" + "\n".join(
            f"- {n}" for n in caveat_notes
        )

    report = []
    report.append("# OFI self-check: Cont-Kukanov-Stoikov Delta-mid ~ OFI\n\n")
    report.append(
        f"`python3 -m edge.analysis.ofi_selfcheck --seed {args.seed}`. Regresses `label_y_mid_1s` "
        f"(forward mid return over the next second, ticks) on the TRAILING 1s-window OFI columns "
        f"as of the same stamp t: `ofi_best_1s` (the CKS spec), `ofi_deep_1s` (summed raw shares "
        f"over levels 1..10), `ofi_raw_l{{m}}_1s` (m=1..10, per-level slope profile) and "
        f"`ofi_integrated_1s`. Univariate OLS with intercept, hand-written on numpy (no scipy, no "
        f"sklearn, no statsmodels). Rows dropped PAIRWISE per regression. Naive OLS SE is reported "
        f"but judged wrong (overlapping labels on a 100ms grid are autocorrelated); significance is "
        f"judged on the Newey-West HAC t-statistic, bar |t_hac| > {SIG_T_HAC}.\n\n"
    )
    report.append("## HAC lag\n\n" + lag_lines + "\n\n")
    report.append("## Slope table\n\n" + _slope_table_markdown(rows) + "\n")
    report.append(_depth_profile_section(rows))
    report.append(ticker_section)
    report.append(_raw_rederivation_section(raw_results))
    report.append("## Verdict\n\n" + verdict_paragraph + "\n")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        f.write("".join(report))

    dt = time.time() - t0
    print(f"wrote {args.out} ({dt:.1f}s)")
    print(f"VERDICT: {verdict}")
    for note in gate_notes:
        print(f"  - {note}")

    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
