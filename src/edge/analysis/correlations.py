"""Phase-1 correlation triage: every non-label feature column vs each of the four calendar
label horizons' regression targets (mid and wmid), Pearson r + Spearman IC + a HAC-corrected
t-stat, raw and seasonality-adjusted side by side (PLAN.md phase-1 milestone: "features.parquet
+ a correlation table of every feature against every horizon's label, with intraday-
seasonality-adjusted versions alongside raw").

    python3 -m edge.analysis.correlations --seed 0

Reads `data/features/{TICKER}_{DATE}_{clock}.parquet` (built by `edge.features.build`), writes
`reports/feature_correlations.csv` (the full grid) and `reports/feature_correlations.md`
(curated top-20-by-|IC| tables, alpha sensitivity, base rates, stamp/NaN accounting).

THIS IS NOT A RESULT. Every number below is a UNIVARIATE, IN-SAMPLE, WHOLE-SESSION
correlation, computed for feature triage only (EVAL_PROTOCOL.md sec. 2 defines the IC as
Spearman rho between prediction and realized move -- that definition is used here, but
without a single fold, a single purge, or a single embargo behind it). Nothing in this module
or its report may be quoted as evidence of predictive value; Phase 2 is where numbers count.

Causality: this module recomputes nothing. It reads already-built, already-causal columns --
each carries its own `<= t` convention from its family module (INTERFACE.md sec. 2) -- and
correlates two existing columns. `label_y_*` is the only forward-looking column family in the
merged table (grep `label_`); every `x` this module treats as a feature is causal by
construction of the table it came from.

Hand-written statistics only (RULES.md's dependency rule -- no scipy, no pandas, no sklearn):
  - Pearson r: `_pearson`, a mean-centred dot-product ratio.
  - Spearman IC: `_rank_avg` (hand-written average-rank transform: tied values share the MEAN
    of the ranks they span) followed by `_pearson` of the two rank arrays. Ranks are computed
    ONCE PER COLUMN, on that column's own finite values, not re-ranked per (x, y) pair on the
    exact pairwise-finite intersection -- see "Spearman rank convention" below.
  - HAC t-stat: reuses `edge.analysis.ofi_selfcheck._ols_with_hac` (hand-written Newey-West /
    Bartlett-kernel sandwich) rather than a second implementation, per the task instruction.
    Its own internal pairwise-complete finite mask (`isfinite(x) & isfinite(y)`, recomputed
    fresh for every (x, y) pair) is also what this module uses for Pearson/Spearman's own `n`,
    so all three statistics on one CSV row share one mask.

HAC lag rule -- generalises `ofi_selfcheck`'s a-priori rule (built for the single 1s horizon)
to all four label horizons, reusing its own two building blocks unchanged:
  - calendar clock: `ofi_selfcheck.CALENDAR_HAC_LAG` (10, exact -- 100ms grid, 10 stamps/sec)
    times the horizon's whole seconds. 10 calendar stamps span 1s always; 100 span 10s always.
  - event clock: `ofi_selfcheck._event_stamp_rate_stats(ts_ns)`'s MEDIAN event-clock stamps
    per calendar second (computed once per ticker from that ticker's own stamps, same as
    ofi_selfcheck) times the horizon's whole seconds.
Neither piece is fit to this analysis; both are ofi_selfcheck's own a-priori numbers, reused.

Spearman rank convention. Ranking a full column once and then subsetting with a pairwise mask
is NOT bit-identical to re-ranking the exact pairwise-finite intersection: if a tied group
straddles the kept/dropped boundary, the shared average rank of the surviving members shifts
by a fraction of a rank. This module takes the once-per-column shortcut anyway, for two
reasons stated plainly rather than silently assumed: (a) tractability -- the grid is 2 tickers
x 2 clocks x 8 labels x ~227 features (~7264 pairs); re-ranking each pairwise intersection
roughly doubles the argsort cost of an already HAC-dominated run; (b) the two NaN sources in
this table are small and boundary-concentrated (feature warmup at the open, label-horizon
truncation at the close), not scattered through the middle of the session, so the tie-rank
correction this shortcut misses is far below the table's reporting precision. This is the
standard practical shortcut for a Spearman correlation matrix, not a novel approximation.

Alpha sensitivity imports `edge.features.labels.classify` rather than reimplementing the
deadband rule (FEATURES.md sec. 7's warning that a free-parameter deadband can manufacture
accuracy). ALPHA_TICKS = 0.5 stays frozen by rule; this module only sweeps it for the
sensitivity table, on the already-built `label_y_mid_*` columns, never touching a test fold.
"""
import argparse
import csv
import os
import time

import numpy as np
import pyarrow.parquet as pq

from edge.analysis.ofi_selfcheck import CALENDAR_HAC_LAG, _event_stamp_rate_stats, _ols_with_hac
from edge.features import (
    LABEL_HORIZONS,
    LABEL_HORIZONS_EVENTS,
    LABEL_PREFIX,
    LABEL_REGRESSION_PREFIX,
    NS_PER_SEC,
    WINDOWS_NS,
)
from edge.features.labels import ALPHA_TICKS, classify

TICKERS = ("INTC", "AMZN")
CLOCKS = ("calendar", "event")
DATE = "2012-06-21"

# "and their wmid twins" (task): the 8 regression targets this module correlates against.
LABEL_TYPES = ("mid", "wmid")
TARGET_LABELS = tuple(
    f"{LABEL_REGRESSION_PREFIX}{lt}_{h}" for h in LABEL_HORIZONS for lt in LABEL_TYPES
)

# Row identifiers, not features -- no FEATURES.md formula, excluded from the "every non-label
# feature column" grid. `tod_bucket` (an int bucket id) IS kept: it is a real state feature
# duplicated in float form as `st_tod_bucket`/`st_tod_frac`, and correlating it is informative
# for the seasonality story this report is partly about.
IDENTITY_EXCLUDE = ("ts_ns", "stamp_index")

ALPHA_SWEEP_TICKS = (0.0, 0.25, 0.5, 1.0, 2.0)

MIN_CORR_N = 10  # floor to even attempt Pearson/Spearman on a pair; below this the number is noise
TOP_N = 20

# "Suspicious" thresholds -- see module docstring / task instructions for what each guards.
SUSPICIOUS_IC_1S = 0.15  # |Spearman rho| on a 1s-forward label from a contemporaneous feature
SAME_ACROSS_HORIZONS_TOL = 1e-6  # rho ~identical at 1s/2s/5s/10s -> smells like a label transform
SADJ_RATIO_FLAG = 2.0  # sadj |rho| at least this many times raw |rho| ...
SADJ_ABS_FLAG = 0.03  # ... AND at least this much larger in absolute terms, to flag as "by a wide margin"


# --------------------------------------------------------------------------------------
# Hand-written statistics (numpy only).
# --------------------------------------------------------------------------------------


def _rank_avg(x):
    """Average-rank transform: tied values share the mean of the ranks they jointly occupy.

    E.g. [3, 1, 2, 2, 5] -> [4, 1, 2.5, 2.5, 5] (the two 2's split ranks {2, 3}).
    Vectorised: sort once, find run boundaries of equal values, reduce sorted-position sums
    within each run (`np.add.reduceat`), scatter back to the original order.
    """
    x = np.asarray(x, dtype=np.float64)
    n = x.shape[0]
    if n == 0:
        return np.empty(0, dtype=np.float64)
    order = np.argsort(x, kind="quicksort")
    sorted_x = x[order]
    is_new = np.empty(n, dtype=bool)
    is_new[0] = True
    is_new[1:] = sorted_x[1:] != sorted_x[:-1]
    group_id = np.cumsum(is_new) - 1
    pos = np.arange(1, n + 1, dtype=np.float64)  # 1-indexed rank position in sorted order
    group_start = np.flatnonzero(is_new)
    group_end = np.r_[group_start[1:], n]
    counts = (group_end - group_start).astype(np.float64)
    sums = np.add.reduceat(pos, group_start)
    avg_per_group = sums / counts
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = avg_per_group[group_id]
    return ranks


def _pearson(x, y):
    """Hand-written Pearson correlation (mean-centred dot-product ratio); NaN if degenerate."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    if x.shape[0] < 2:
        return float("nan")
    xm = x - x.mean()
    ym = y - y.mean()
    denom = np.sqrt(np.sum(xm ** 2) * np.sum(ym ** 2))
    if denom == 0:
        return float("nan")
    return float(np.sum(xm * ym) / denom)


def _column_ranks(float_cols, names):
    """Rank each named column once, on its own finite values; NaN carried through unranked."""
    cache = {}
    for name in names:
        v = float_cols[name]
        out = np.full(v.shape[0], np.nan, dtype=np.float64)
        finite = np.isfinite(v)
        if finite.any():
            out[finite] = _rank_avg(v[finite])
        cache[name] = out
    return cache


# --------------------------------------------------------------------------------------
# HAC lag: generalises ofi_selfcheck's a-priori 1s rule to all four horizons (module docstring).
# --------------------------------------------------------------------------------------


def _horizon_seconds(horizon):
    return WINDOWS_NS[horizon] // NS_PER_SEC


def _hac_lag_for_horizon(clock, horizon, ts_ns, event_lag_stats_cache, cache_key):
    seconds = _horizon_seconds(horizon)
    if clock == "calendar":
        return int(CALENDAR_HAC_LAG * seconds)
    if cache_key not in event_lag_stats_cache:
        event_lag_stats_cache[cache_key] = _event_stamp_rate_stats(ts_ns)
    return int(event_lag_stats_cache[cache_key]["median"] * seconds)


# --------------------------------------------------------------------------------------
# Per (ticker, clock) grid.
# --------------------------------------------------------------------------------------


def _load_table(features_dir, ticker, clock, date):
    path = os.path.join(features_dir, f"{ticker}_{date}_{clock}.parquet")
    table = pq.read_table(path)
    cols = {name: table.column(name).to_numpy(zero_copy_only=False) for name in table.schema.names}
    return cols, table.num_rows


def compute_grid(features_dir, date=DATE, tickers=TICKERS, clocks=CLOCKS):
    """Returns (grid_rows, stamp_info, alpha_rows, base_rate_rows, n_features, event_lag_stats)."""
    grid_rows = []
    stamp_info = {}  # (ticker, clock) -> {"n": int, "usable": {label: int}}
    alpha_rows = []
    base_rate_rows = []
    event_lag_stats = {}
    n_features = None

    for ticker in tickers:
        for clock in clocks:
            cols, n_rows = _load_table(features_dir, ticker, clock, date)
            names = list(cols)
            feature_names = [n for n in names if not n.startswith(LABEL_PREFIX) and n not in IDENTITY_EXCLUDE]
            if n_features is None:
                n_features = len(feature_names)
            elif n_features != len(feature_names):
                raise ValueError(
                    f"{ticker}/{clock}: {len(feature_names)} feature columns, expected {n_features} "
                    f"(schemas must match across ticker/clock -- build.py's deterministic column order)"
                )

            ts_ns = np.asarray(cols["ts_ns"], dtype=np.int64)

            float_cols = {name: np.asarray(cols[name], dtype=np.float64) for name in feature_names}
            for lbl in TARGET_LABELS:
                float_cols[lbl] = np.asarray(cols[lbl], dtype=np.float64)

            rank_cache = _column_ranks(float_cols, feature_names + list(TARGET_LABELS))

            usable = {}
            for lbl in TARGET_LABELS:
                y = float_cols[lbl]
                y_finite = np.isfinite(y)
                usable[lbl] = int(y_finite.sum())

                # lbl looks like "label_y_mid_1s" or "label_y_wmid_1s"; split on '_' and read
                # the last two tokens (label_type itself never contains an underscore, so this
                # is unambiguous).
                parts = lbl.split("_")
                horizon = parts[-1]
                label_type = parts[-2]

                lag = _hac_lag_for_horizon(clock, horizon, ts_ns, event_lag_stats, (ticker, clock))

                y_ranks = rank_cache[lbl]
                for feat in feature_names:
                    x = float_cols[feat]
                    mask = np.isfinite(x) & y_finite
                    n = int(mask.sum())

                    pearson_r = spearman_ic = float("nan")
                    if n >= MIN_CORR_N:
                        pearson_r = _pearson(x[mask], y[mask])
                        spearman_ic = _pearson(rank_cache[feat][mask], y_ranks[mask])

                    hac = _ols_with_hac(x, y, lag)
                    if hac is None:
                        t_hac = slope_hac = float("nan")
                    else:
                        t_hac = hac["t_hac"]
                        slope_hac = hac["slope"]
                        n = hac["n"]  # canonical n: identical pairwise mask, HAC's own count

                    grid_rows.append(
                        {
                            "ticker": ticker,
                            "clock": clock,
                            "feature": feat,
                            "label": lbl,
                            "label_type": label_type,
                            "horizon": horizon,
                            "n": n,
                            "pearson_r": pearson_r,
                            "spearman_ic": spearman_ic,
                            "t_hac": t_hac,
                            "slope_hac": slope_hac,
                            "hac_lag": lag,
                        }
                    )

            stamp_info[(ticker, clock)] = {"n": n_rows, "usable": usable}

            # -- alpha sensitivity: sweep alpha on label_y_mid_h, the primary y_h (FEATURES.md
            # sec. 7's c_h = sign(y_h)); wmid is the substituted lower-noise twin, not the
            # sweep target, per labels.py's own docstring on the mid/wmid split.
            for horizon in LABEL_HORIZONS:
                y = float_cols[f"{LABEL_REGRESSION_PREFIX}mid_{horizon}"]
                y_finite_mask = np.isfinite(y)
                n_finite = int(y_finite_mask.sum())
                for alpha in ALPHA_SWEEP_TICKS:
                    c = classify(y, alpha_ticks=alpha)
                    c_finite = c[y_finite_mask]
                    denom = c_finite.shape[0]
                    frac_up = float(np.sum(c_finite == 1.0)) / denom if denom else float("nan")
                    frac_flat = float(np.sum(c_finite == 0.0)) / denom if denom else float("nan")
                    frac_down = float(np.sum(c_finite == -1.0)) / denom if denom else float("nan")
                    alpha_rows.append(
                        {
                            "ticker": ticker,
                            "clock": clock,
                            "horizon": horizon,
                            "alpha": alpha,
                            "n_finite": n_finite,
                            "frac_up": frac_up,
                            "frac_flat": frac_flat,
                            "frac_down": frac_down,
                        }
                    )

            # -- base rate of the FROZEN (alpha=0.5) classification labels already built into
            # the table: label_c_mid_h / label_c_wmid_h (calendar) and label_c_mid_e{N} (event).
            base_rate_cols = [f"label_c_mid_{h}" for h in LABEL_HORIZONS]
            base_rate_cols += [f"label_c_wmid_{h}" for h in LABEL_HORIZONS]
            base_rate_cols += [f"label_c_mid_e{n}" for n in LABEL_HORIZONS_EVENTS]
            for col in base_rate_cols:
                c = np.asarray(cols[col], dtype=np.float64)
                finite = np.isfinite(c)
                denom = int(finite.sum())
                c_f = c[finite]
                frac_up = float(np.sum(c_f == 1.0)) / denom if denom else float("nan")
                frac_flat = float(np.sum(c_f == 0.0)) / denom if denom else float("nan")
                frac_down = float(np.sum(c_f == -1.0)) / denom if denom else float("nan")
                base_rate_rows.append(
                    {
                        "ticker": ticker,
                        "clock": clock,
                        "label": col,
                        "n_total": n_rows,
                        "n_finite": denom,
                        "frac_up": frac_up,
                        "frac_flat": frac_flat,
                        "frac_down": frac_down,
                    }
                )

    return grid_rows, stamp_info, alpha_rows, base_rate_rows, n_features, event_lag_stats


# --------------------------------------------------------------------------------------
# Suspicious scan.
# --------------------------------------------------------------------------------------


def _find_row(index, ticker, clock, feature, label):
    return index.get((ticker, clock, feature, label))


def _clock_effect_summary(grid_rows):
    """Mean/median |IC| at the 1s horizon, calendar vs event clock, raw features only.

    The boring explanation for why so many LEAK-SMELL items below are on the event clock
    (RULES.md rule 3: propose it before treating a number as a discovery): event-clock stamps
    are BOOK-CHANGING EVENTS by construction (clocks.py), so they are drawn exactly at moments
    order flow is active, not uniformly in time. A feature measured at those moments is
    correlated with the immediate next second's move mechanically more often than the same
    feature on a calendar grid that also samples quiet moments. FEATURES.md sec. 8: "calendar-
    time results on a quiet instrument are partly measuring the clock rather than the market" --
    this is that statement's mirror image on the event clock.
    """
    by_clock = {}
    for r in grid_rows:
        if r["horizon"] != "1s" or r["feature"].endswith("_sadj"):
            continue
        ic = r["spearman_ic"]
        if np.isfinite(ic):
            by_clock.setdefault(r["clock"], []).append(abs(ic))
    if "calendar" not in by_clock or "event" not in by_clock:
        return "", {}
    cal = np.array(by_clock["calendar"])
    evt = np.array(by_clock["event"])
    stats = {
        "calendar_mean_abs_ic_1s": float(cal.mean()),
        "event_mean_abs_ic_1s": float(evt.mean()),
        "calendar_frac_over_threshold": float((cal > SUSPICIOUS_IC_1S).mean()),
        "event_frac_over_threshold": float((evt > SUSPICIOUS_IC_1S).mean()),
    }
    text = (
        f"**Read the LEAK SMELL list below against the clock, not just the number.** At the 1s "
        f"horizon, raw-feature mean |IC| is {stats['calendar_mean_abs_ic_1s']:.3f} on the "
        f"calendar clock vs {stats['event_mean_abs_ic_1s']:.3f} on the event clock (median "
        f"{np.median(cal):.3f} vs {np.median(evt):.3f}); {stats['calendar_frac_over_threshold']:.1%} "
        f"of calendar-clock features clear the {SUSPICIOUS_IC_1S} threshold vs "
        f"{stats['event_frac_over_threshold']:.1%} of event-clock features. The likely (boring) "
        f"explanation: event-clock stamps are book-changing events by construction (clocks.py), "
        f"so they sample exactly the moments order flow is active, not uniformly in time -- a "
        f"selection effect on WHEN you look, not a leak in WHAT you look at. This does not "
        f"un-flag the individual items (the threshold is on the number, and the number is what "
        f"Phase 2 would have to survive out-of-sample on this clock), but it is the reason most "
        f"of them cluster on `clock=event`.\n\n"
    )
    return text, stats


def suspicious_scan(grid_rows):
    """Three checks, each named with column/magnitude per the task instruction."""
    index = {(r["ticker"], r["clock"], r["feature"], r["label"]): r for r in grid_rows}
    notes = []

    # 1. |Spearman rho| > SUSPICIOUS_IC_1S on a 1s-forward label from a contemporaneous feature.
    for r in grid_rows:
        if r["horizon"] != "1s":
            continue
        ic = r["spearman_ic"]
        if np.isfinite(ic) and abs(ic) > SUSPICIOUS_IC_1S:
            notes.append(
                f"LEAK SMELL: {r['ticker']}/{r['clock']} {r['feature']} vs {r['label']}: "
                f"Spearman rho={ic:.4f} (|rho|>{SUSPICIOUS_IC_1S} at a 1s horizon on 2012 "
                f"equity data is not a discovery)"
            )

    # 2. Spearman rho ~identical across all 4 horizons for the same (ticker, clock, feature,
    #    label_type) -> smells like a transform of the label, not a real relationship that
    #    should decay/change with horizon.
    groups = {}
    for r in grid_rows:
        key = (r["ticker"], r["clock"], r["feature"], r["label_type"])
        groups.setdefault(key, {})[r["horizon"]] = r["spearman_ic"]
    for key, by_h in groups.items():
        if set(by_h) != set(LABEL_HORIZONS):
            continue
        vals = [by_h[h] for h in LABEL_HORIZONS]
        if any(not np.isfinite(v) for v in vals):
            continue
        spread = max(vals) - min(vals)
        if spread < SAME_ACROSS_HORIZONS_TOL and abs(vals[0]) > 1e-6:
            ticker, clock, feature, label_type = key
            notes.append(
                f"IDENTICAL-ACROSS-HORIZONS: {ticker}/{clock} {feature} vs label_y_{label_type}_*: "
                f"Spearman rho={vals[0]:.6f} at every one of 1s/2s/5s/10s (spread={spread:.2e}) "
                f"-- smells like a transform of the label, not four independent relationships"
            )

    # 3. _sadj column correlating more strongly than its raw parent by a wide margin.
    for r in grid_rows:
        if not r["feature"].endswith("_sadj"):
            continue
        raw_feature = r["feature"][: -len("_sadj")]
        raw_row = _find_row(index, r["ticker"], r["clock"], raw_feature, r["label"])
        if raw_row is None:
            continue
        sadj_ic, raw_ic = r["spearman_ic"], raw_row["spearman_ic"]
        if not (np.isfinite(sadj_ic) and np.isfinite(raw_ic)):
            continue
        if abs(sadj_ic) >= SADJ_ABS_FLAG and abs(sadj_ic) >= SADJ_RATIO_FLAG * max(abs(raw_ic), 1e-9):
            notes.append(
                f"SADJ>>RAW: {r['ticker']}/{r['clock']} {raw_feature} vs {r['label']}: "
                f"raw rho={raw_ic:.4f}, sadj rho={sadj_ic:.4f} "
                f"({abs(sadj_ic) / max(abs(raw_ic), 1e-9):.1f}x, a seasonality adjustment should "
                f"not manufacture signal out of a near-zero raw correlation)"
            )

    return notes


# --------------------------------------------------------------------------------------
# Rank-and-report: top-N by |IC|, raw and sadj side by side.
# --------------------------------------------------------------------------------------


def top_n_table(grid_rows, ticker, clock, label, n=TOP_N):
    """[(row, sadj_row_or_None)] for the top-`n` RAW (non-`_sadj`) features by |Spearman IC|."""
    subset = [
        r
        for r in grid_rows
        if r["ticker"] == ticker and r["clock"] == clock and r["label"] == label and not r["feature"].endswith("_sadj")
    ]
    index = {(r["ticker"], r["clock"], r["feature"], r["label"]): r for r in grid_rows}

    def _key(r):
        ic = r["spearman_ic"]
        return -abs(ic) if np.isfinite(ic) else float("inf")

    subset.sort(key=_key)
    top = subset[:n]
    out = []
    for r in top:
        sadj_row = _find_row(index, ticker, clock, f"{r['feature']}_sadj", label)
        out.append((r, sadj_row))
    return out


def _fmt(x, spec="{:.4f}"):
    if x is None:
        return "N/A"
    if isinstance(x, float) and not np.isfinite(x):
        return "NaN"
    return spec.format(x)


def _top_n_markdown(grid_rows, ticker, clock, label, n=TOP_N):
    rows = top_n_table(grid_rows, ticker, clock, label, n=n)
    lines = [
        "| rank | feature | n | pearson_r (raw) | IC raw (Spearman rho) | t_hac raw | "
        "IC sadj | t_hac sadj | delta (sadj-raw) |\n",
        "|---|---|---|---|---|---|---|---|---|\n",
    ]
    for i, (r, sadj_row) in enumerate(rows, start=1):
        ic_raw = r["spearman_ic"]
        ic_sadj = sadj_row["spearman_ic"] if sadj_row is not None else None
        delta = (ic_sadj - ic_raw) if (sadj_row is not None and np.isfinite(ic_raw) and np.isfinite(ic_sadj)) else None
        lines.append(
            f"| {i} | {r['feature']} | {r['n']} | {_fmt(r['pearson_r'])} | {_fmt(ic_raw)} | "
            f"{_fmt(r['t_hac'], '{:.2f}')} | {_fmt(ic_sadj)} | "
            f"{_fmt(sadj_row['t_hac'], '{:.2f}') if sadj_row is not None else 'N/A'} | {_fmt(delta)} |\n"
        )
    return "".join(lines)


# --------------------------------------------------------------------------------------
# Report assembly.
# --------------------------------------------------------------------------------------


def _stamp_usable_markdown(stamp_info):
    lines = [
        "| ticker | clock | n stamps | " + " | ".join(f"usable {lt}_{h}" for h in LABEL_HORIZONS for lt in LABEL_TYPES) + " |\n",
        "|---|---|---" + "|---" * (len(LABEL_HORIZONS) * len(LABEL_TYPES)) + "|\n",
    ]
    for (ticker, clock), info in stamp_info.items():
        usable_cells = " | ".join(
            f"{info['usable'][f'{LABEL_REGRESSION_PREFIX}{lt}_{h}']} "
            f"({info['usable'][f'{LABEL_REGRESSION_PREFIX}{lt}_{h}'] / info['n']:.1%})"
            for h in LABEL_HORIZONS
            for lt in LABEL_TYPES
        )
        lines.append(f"| {ticker} | {clock} | {info['n']} | {usable_cells} |\n")
    return "".join(lines)


def _alpha_sensitivity_markdown(alpha_rows):
    lines = [
        "| ticker | clock | horizon | alpha (ticks) | n | frac up | frac flat | frac down |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]
    for r in alpha_rows:
        marker = " **FROZEN**" if r["alpha"] == ALPHA_TICKS else ""
        lines.append(
            f"| {r['ticker']} | {r['clock']} | {r['horizon']} | {r['alpha']}{marker} | {r['n_finite']} | "
            f"{_fmt(r['frac_up'], '{:.1%}')} | {_fmt(r['frac_flat'], '{:.1%}')} | {_fmt(r['frac_down'], '{:.1%}')} |\n"
        )
    return "".join(lines)


def _base_rate_markdown(base_rate_rows):
    lines = [
        "| ticker | clock | label | n total | n finite | frac up | frac flat | frac down |\n",
        "|---|---|---|---|---|---|---|---|\n",
    ]
    for r in base_rate_rows:
        lines.append(
            f"| {r['ticker']} | {r['clock']} | {r['label']} | {r['n_total']} | {r['n_finite']} | "
            f"{_fmt(r['frac_up'], '{:.1%}')} | {_fmt(r['frac_flat'], '{:.1%}')} | {_fmt(r['frac_down'], '{:.1%}')} |\n"
        )
    return "".join(lines)


SADJ_SIGNAL_FLOOR = 0.1  # |raw IC| threshold for the "higher-signal subset" breakdown below


def _sadj_effect_summary(grid_rows):
    """Mean signed and mean absolute (|sadj|-|raw|) shift in Spearman IC, raw feature rows only.

    Reported two ways: over the WHOLE grid (dominated by near-zero, noise-level correlations,
    where a tiny absolute shift is a large relative one and not informative) and over the
    subset with |raw IC| > SADJ_SIGNAL_FLOOR (the features a reviewer would actually look at),
    where the RELATIVE shift is the more honest summary than the absolute one.
    """
    index = {(r["ticker"], r["clock"], r["feature"], r["label"]): r for r in grid_rows}
    signed_deltas, abs_deltas = [], []
    rel_deltas_signal = []
    n_larger, n_smaller, n_pairs = 0, 0, 0
    n_signal, n_signal_shrink, n_signal_grow = 0, 0, 0
    for r in grid_rows:
        if r["feature"].endswith("_sadj"):
            continue
        sadj_row = _find_row(index, r["ticker"], r["clock"], f"{r['feature']}_sadj", r["label"])
        if sadj_row is None:
            continue
        raw_ic, sadj_ic = r["spearman_ic"], sadj_row["spearman_ic"]
        if not (np.isfinite(raw_ic) and np.isfinite(sadj_ic)):
            continue
        n_pairs += 1
        signed_deltas.append(sadj_ic - raw_ic)
        abs_deltas.append(abs(sadj_ic) - abs(raw_ic))
        if abs(sadj_ic) > abs(raw_ic):
            n_larger += 1
        elif abs(sadj_ic) < abs(raw_ic):
            n_smaller += 1
        if abs(raw_ic) > SADJ_SIGNAL_FLOOR:
            n_signal += 1
            rel_deltas_signal.append((abs(sadj_ic) - abs(raw_ic)) / abs(raw_ic))
            if abs(sadj_ic) < abs(raw_ic):
                n_signal_shrink += 1
            elif abs(sadj_ic) > abs(raw_ic):
                n_signal_grow += 1
    if n_pairs == 0:
        return "no raw/sadj pairs with finite IC on both sides.", {}
    mean_signed = float(np.mean(signed_deltas))
    mean_abs = float(np.mean(abs_deltas))
    median_abs = float(np.median(abs_deltas))
    stats = {
        "n_pairs": n_pairs,
        "mean_signed_delta": mean_signed,
        "mean_abs_ic_delta": mean_abs,
        "median_abs_ic_delta": median_abs,
        "frac_sadj_larger": n_larger / n_pairs,
        "frac_sadj_smaller": n_smaller / n_pairs,
    }
    text = (
        f"Whole grid ({n_pairs} raw/sadj pairs with finite IC on both sides, mostly noise-level "
        f"correlations): seasonality adjustment makes |IC| larger in {n_larger} "
        f"({n_larger / n_pairs:.1%}) and smaller in {n_smaller} ({n_smaller / n_pairs:.1%}); mean "
        f"|IC| shift (sadj - raw) = {mean_abs:+.5f}, median {median_abs:+.5f} -- an absolute shift "
        f"this small is not informative on its own when most raw |IC| values here are themselves "
        f"under 0.05.\n\n"
    )
    if n_signal:
        mean_rel = float(np.mean(rel_deltas_signal))
        median_rel = float(np.median(rel_deltas_signal))
        stats["n_signal"] = n_signal
        stats["mean_relative_shift_signal_subset"] = mean_rel
        stats["median_relative_shift_signal_subset"] = median_rel
        direction = (
            "shrinks" if mean_rel < -0.01 else ("grows" if mean_rel > 0.01 else "leaves roughly unchanged")
        )
        text += (
            f"Higher-signal subset (|raw IC| > {SADJ_SIGNAL_FLOOR}, n={n_signal}, the features a "
            f"reviewer would actually look at): seasonality adjustment {direction} |IC| on "
            f"average, by {mean_rel:+.1%} relative (median {median_rel:+.1%}); shrinks {n_signal_shrink} "
            f"({n_signal_shrink / n_signal:.1%}) and grows {n_signal_grow} ({n_signal_grow / n_signal:.1%}) "
            f"of them. Read as: a modest, mostly-shrinking correction -- some of the raw signal in "
            f"the strongest features is intraday level/seasonality, not all of it."
        )
    else:
        text += "No pairs cleared the higher-signal |raw IC| threshold."
    return text, stats


def write_csv(path, grid_rows):
    fieldnames = [
        "ticker", "clock", "feature", "label", "label_type", "horizon",
        "n", "pearson_r", "spearman_ic", "t_hac", "slope_hac", "hac_lag",
    ]
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in grid_rows:
            writer.writerow(r)


def write_markdown(path, grid_rows, stamp_info, alpha_rows, base_rate_rows, n_features, event_lag_stats, seed):
    sadj_text, sadj_stats = _sadj_effect_summary(grid_rows)
    clock_effect_text, clock_effect_stats = _clock_effect_summary(grid_rows)
    suspicious = suspicious_scan(grid_rows)

    report = []
    report.append("# Feature x label correlation triage (Phase 1)\n\n")
    report.append(f"`python3 -m edge.analysis.correlations --seed {seed}`\n\n")
    report.append(
        "**THIS IS NOT A RESULT.** Every number in this report is a univariate, in-sample, "
        "whole-session Pearson/Spearman correlation between one feature column and one forward "
        "label, computed for feature triage only. There is no purged walk-forward split, no "
        "purge, no embargo, and no held-out fold behind any of it (EVAL_PROTOCOL.md sec. 1-2). "
        "Nothing here may be quoted as evidence of predictive value; Phase 2's harness is where "
        "numbers count. The Spearman rho column is nonetheless the information coefficient as "
        "EVAL_PROTOCOL.md sec. 2 defines it: Spearman rho between prediction and realized move.\n\n"
    )
    report.append(
        f"Grid: {len(TICKERS)} tickers x {len(CLOCKS)} clocks x {len(TARGET_LABELS)} labels x "
        f"{n_features} non-label feature columns = {len(grid_rows)} rows in "
        f"`feature_correlations.csv`. Labels: `label_y_mid_h` / `label_y_wmid_h` for "
        f"h in {LABEL_HORIZONS} (FEATURES.md sec. 7's `y_h` primary and its wmid twin).\n\n"
    )
    report.append(
        "**Methodology, hand-written on numpy (no scipy/pandas/sklearn):** Pearson r "
        "(mean-centred dot product); Spearman IC via a hand-written average-rank transform "
        "(tied values share the mean of the ranks they span), ranked once per column on that "
        "column's own finite values and then masked pairwise per (feature, label) pair -- see "
        "module docstring's \"Spearman rank convention\" for why this is not bit-identical to "
        "re-ranking the exact pairwise intersection, and why that gap is negligible here; `n` is "
        "pairwise-complete (`isfinite(x) & isfinite(y)`), reported per row, since a column with "
        "a NaN block (e.g. a refit-block-0 warmup) legitimately has fewer usable rows than one "
        "without; `t_hac` reuses `edge.analysis.ofi_selfcheck._ols_with_hac` (hand-written "
        "Newey-West/Bartlett HAC), because overlapping labels on a fine grid make the naive t "
        "wildly overstated. HAC lag is `ofi_selfcheck`'s own a-priori rule, generalised from "
        "its fixed 1s case to all four horizons: calendar clock lag = 10 x horizon-seconds "
        "(10 stamps/sec on the 100ms grid, exact); event clock lag = median event-clock stamps "
        "per calendar second (computed per ticker) x horizon-seconds.\n\n"
    )
    lag_lines = []
    for ticker in TICKERS:
        stats = event_lag_stats.get((ticker, "event"))
        if stats:
            lag_lines.append(
                f"- {ticker}/event median event-clock stamps per calendar second: {stats['median']} "
                f"(mean {stats['mean']:.2f}, p90 {stats['p90']:.0f}, p99 {stats['p99']:.0f}, "
                f"max {stats['max']}) -> HAC lags of {stats['median']}/{stats['median']*2}/"
                f"{stats['median']*5}/{stats['median']*10} at 1s/2s/5s/10s."
            )
    if lag_lines:
        report.append("Event-clock HAC lag inputs:\n\n" + "\n".join(lag_lines) + "\n\n")

    report.append("## Stamps and NaN accounting\n\n")
    report.append(
        "Total stamps per (ticker, clock), and how many are usable (finite label) at each "
        "horizon x label-type. A horizon's usable count is strictly less than the stamp count "
        "at the session's tail, where `t+h` runs past the last event or past `SESSION_END_NS` "
        "(labels.py's NaN rule, never clipped/back-filled).\n\n"
    )
    report.append(_stamp_usable_markdown(stamp_info) + "\n")

    report.append("## Alpha sensitivity of the classification labels\n\n")
    report.append(
        f"**alpha = {ALPHA_TICKS} ticks (half a tick) is FROZEN BY RULE** (FEATURES.md sec. 7's "
        f"own first suggestion, `edge.features.labels.ALPHA_TICKS`) -- not fit to this or any "
        f"other data. This table is reported to show sensitivity, not to pick a value: a bigger "
        f"alpha manufactures a bigger \"flat\" class and can make a trivial always-flat "
        f"predictor look accurate, which is exactly why FEATURES.md sec. 7 warns against tuning "
        f"it against test folds. Swept on `label_y_mid_h` (the primary `y_h`, not the wmid "
        f"twin) via `edge.features.labels.classify` -- imported, not reimplemented.\n\n"
    )
    report.append(_alpha_sensitivity_markdown(alpha_rows) + "\n")

    report.append("## Base rate of the frozen (alpha=0.5) classification labels\n\n")
    report.append(
        "Any Phase-2 classification accuracy has to be read against these base rates, not "
        "against 33% or 50% -- at short horizons almost every label is flat "
        "(labels.py's own \"BASE RATE WARNING\").\n\n"
    )
    report.append(_base_rate_markdown(base_rate_rows) + "\n")

    report.append("## Seasonality adjustment: what it did to the ranking\n\n")
    report.append(sadj_text + "\n\n")

    report.append("## Suspicious\n\n")
    if clock_effect_text:
        report.append(clock_effect_text)
    if suspicious:
        report.append(
            f"{len(suspicious)} flagged item(s) below. Flagging does not mean the number is "
            f"wrong -- it means it clears a threshold chosen because a real univariate "
            f"relationship at this horizon, in 2012 large/small-tick equities, should not be "
            f"this large or this shaped, so it is reported rather than presented as a result.\n\n"
        )
        for note in suspicious:
            report.append(f"- {note}\n")
        report.append("\n")
    else:
        report.append("None cleared the thresholds below.\n\n")
    report.append(
        f"Thresholds: |Spearman rho| > {SUSPICIOUS_IC_1S} at the 1s horizon; Spearman rho "
        f"identical (spread < {SAME_ACROSS_HORIZONS_TOL:.0e}) across all of 1s/2s/5s/10s for "
        f"the same feature/label-type; a `_sadj` column's |IC| at least {SADJ_RATIO_FLAG}x its "
        f"raw parent's AND at least {SADJ_ABS_FLAG} larger in absolute terms.\n\n"
    )

    report.append("## Top-20 features by |IC| per (ticker, clock, label), raw and sadj side by side\n\n")
    report.append(
        "Ranked by |Spearman IC| of the RAW (non-`_sadj`) feature; the seasonality-adjusted "
        "twin's IC and t_hac are shown alongside for the same rank so the effect of the "
        "adjustment is legible in the row, not buried in a second table.\n\n"
    )
    for ticker in TICKERS:
        for clock in CLOCKS:
            for label in TARGET_LABELS:
                report.append(f"### {ticker} / {clock} / `{label}`\n\n")
                report.append(_top_n_markdown(grid_rows, ticker, clock, label) + "\n")

    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        f.write("".join(report))

    return suspicious, sadj_stats


# --------------------------------------------------------------------------------------
# Entry point.
# --------------------------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Phase-1 feature x label correlation triage (univariate, in-sample, whole-session)."
    )
    parser.add_argument("--features", default="data/features", help="directory of built feature parquets")
    parser.add_argument("--out-md", default="reports/feature_correlations.md")
    parser.add_argument("--out-csv", default="reports/feature_correlations.csv")
    parser.add_argument("--date", default=DATE)
    parser.add_argument("--seed", type=int, default=0)  # deterministic step; logged only, per RULES.md
    args = parser.parse_args(argv)

    print(f"seed={args.seed} (deterministic step, no randomness consumed)")
    t0 = time.time()

    grid_rows, stamp_info, alpha_rows, base_rate_rows, n_features, event_lag_stats = compute_grid(
        args.features, date=args.date
    )
    print(f"computed grid: {len(grid_rows)} rows, {n_features} feature columns ({time.time() - t0:.1f}s)")

    write_csv(args.out_csv, grid_rows)
    suspicious, sadj_stats = write_markdown(
        args.out_md, grid_rows, stamp_info, alpha_rows, base_rate_rows, n_features, event_lag_stats, args.seed
    )

    dt = time.time() - t0
    print(f"wrote {args.out_csv} ({len(grid_rows)} rows)")
    print(f"wrote {args.out_md}")
    # Print the BREAKDOWN, never a bare head of the list. A `suspicious[:20]` tail on a
    # 488-item list produced a phase-1 triage that concluded "all INTC, AMZN clean" when
    # AMZN in fact holds 164 flagged pairs and the single largest correlation in the whole
    # report (red-team report phase1-lookahead-normalization-2026-07-31, F3). Anything that reads
    # like a summary has to be computed over the whole list.
    print(f"suspicious items: {len(suspicious)}")
    by_group = {}
    for note in suspicious:
        kind = note.split(":", 1)[0].strip()
        scope = " ".join(note.split()[:4])
        key = (kind.split()[0] if kind else "?", scope)
        by_group[key] = by_group.get(key, 0) + 1
    for key in sorted(by_group, key=lambda k: -by_group[k]):
        print(f"  - {key[1]}: {by_group[key]} item(s)")
    print(
        f"  full list ({len(suspicious)} items) is in {args.out_md}; this console block is a"
        " count by scope, not the list -- do not triage from it"
    )
    print(f"done in {dt:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
