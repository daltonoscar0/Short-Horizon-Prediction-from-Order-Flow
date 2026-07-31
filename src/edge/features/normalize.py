"""Rolling (never full-sample) normalisation for feature arrays already aligned to stamps.

`rolling_zscore` convention: `z[t] = (x[t] - mean(x[t-window:t])) / std(x[t-window:t],
ddof=1)` -- the baseline window is the half-open `[t-window, t)` slice of STAMP-INDEX
positions (not ns), so it ends strictly *before* t and always excludes `x[t]` itself,
even though `x[t]` is legal "at or before t" information under the clocks' `<= t`
convention. This is deliberately stricter than that convention: a feature's current
draw must never contaminate its own baseline mean/std, or a single outlier both is and
defines "extreme."

Window length is a count of preceding stamps, not ns, so this module stays clock-
agnostic; a caller normalising a feature on the 100ms calendar clock over a trailing 5s
passes `window=50`.

`expanding_bucket_zscore` is the seasonality control FEATURES.md sec. 5 asks for. The
spec says per-bucket z-score "using past days only". THIS UNIVERSE IS ONE TRADING DAY,
so there are no past days and a cross-day baseline does not exist. Rather than silently
substitute a full-sample per-bucket mean (which would leak the whole session into every
stamp), the baseline here is the EARLIER STAMPS IN THE SAME TIME-OF-DAY BUCKET ON THE
SAME DAY, expanding. That is strictly causal and it is what the single-day sample can
honestly support; its limitation is that within one day it cannot separate intraday
seasonality from intraday drift, and the first `min_periods` stamps of every bucket are
NaN warmup. With two or more days the correct change is to add a `day_id` argument and
restrict the baseline to strictly earlier days.

Every statistic here is a difference of two prefix cumulative sums taken over a trailing
slice, so by construction it can never see `x[t]` or anything after it. See
tests/test_normalize.py::test_no_full_sample_fit, which also greps this package for the
patterns a full-sample scaler leaves behind.
"""
import numpy as np


def _causal_offset(values, finite):
    """First finite value, used as a constant shift so the prefix sums stay well-conditioned."""
    idx = np.flatnonzero(finite)
    return float(values[idx[0]]) if idx.size else 0.0


def rolling_zscore(values, window, min_periods=2):
    """z-score of `values[t]` against the preceding `window` stamps; NaN under `min_periods`."""
    if window < 1:
        raise ValueError("window must be >= 1")
    if min_periods < 2:
        raise ValueError("min_periods must be >= 2 (std is undefined below 2 points)")

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError("rolling_zscore expects a 1-D array aligned to stamps")
    n = values.shape[0]
    finite = np.isfinite(values)

    # A raw mid price is ~2.7e5 with a per-stamp variance of ~1e0; the textbook
    # sum-of-squares variance would then cancel 11 significant digits away. Shifting by
    # a constant fixes the conditioning and cannot leak: the shift is the FIRST finite
    # value, so any stamp with a usable baseline is strictly after it, and a constant
    # shift cancels exactly in (x - mean)/std.
    offset = _causal_offset(values, finite)
    centred = np.where(finite, values - offset, 0.0)

    # cum*[i] = statistic over the first i entries (cum*[0] = 0, the empty prefix).
    count_cum = np.concatenate(([0], np.cumsum(finite.astype(np.int64))))
    sum_cum = np.concatenate(([0.0], np.cumsum(centred)))
    sumsq_cum = np.concatenate(([0.0], np.cumsum(centred * centred)))

    idx = np.arange(n)
    lo = np.maximum(0, idx - window)
    hi = idx  # exclusive of t: the baseline window is [t-window, t)

    cnt = count_cum[hi] - count_cum[lo]
    s1 = sum_cum[hi] - sum_cum[lo]
    s2 = sumsq_cum[hi] - sumsq_cum[lo]

    ok = cnt >= min_periods
    mean = np.zeros(n, dtype=np.float64)
    mean[ok] = s1[ok] / cnt[ok]
    var = np.zeros(n, dtype=np.float64)
    var[ok] = (s2[ok] - cnt[ok] * mean[ok] ** 2) / (cnt[ok] - 1)
    var = np.clip(var, 0.0, None)  # float round-off can make a ~0 variance slightly negative
    std = np.sqrt(var)

    out = np.full(n, np.nan, dtype=np.float64)
    usable = ok & finite & (std > 0)
    out[usable] = (values[usable] - offset - mean[usable]) / std[usable]
    return out


def expanding_bucket_zscore(values, bucket_id, min_periods=2):
    """Seasonality control: z-score vs earlier same-bucket stamps only; see module docstring."""
    values = np.asarray(values, dtype=np.float64)
    bucket_id = np.asarray(bucket_id)
    if values.ndim != 1 or bucket_id.shape != values.shape:
        raise ValueError("values and bucket_id must be 1-D arrays of the same length")

    out = np.full(values.shape[0], np.nan, dtype=np.float64)
    for bucket in np.unique(bucket_id):
        pos = np.flatnonzero(bucket_id == bucket)
        if pos.size < 2:
            continue
        # window = pos.size makes rolling_zscore's [t-window, t) baseline the full set of
        # STRICTLY EARLIER stamps in this bucket, i.e. expanding rather than trailing.
        out[pos] = rolling_zscore(values[pos], window=pos.size, min_periods=min_periods)
    return out
