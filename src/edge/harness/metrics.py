"""Per-fold metrics (EVAL_PROTOCOL.md sec. 2). Hand-written on numpy; no sklearn, no scipy.

THE ONE THAT MATTERS MOST: `r2_vs_zero`. Out-of-sample R^2 is computed against a ZERO
forecast, not against the test-set mean:

    R2 = 1 - sum((y - yhat)^2) / sum(y^2)          <- this module
    R2 = 1 - sum((y - yhat)^2) / sum((y - ybar)^2) <- WRONG here, and the usual mistake

Using the test mean hands the model the test set's own average return, which it could not
have known. EVAL_PROTOCOL.md sec. 2 calls it "a subtle leak and the most common one in this
literature". On a near-zero-mean series the two are numerically close, which is exactly why
it survives review. `r2_vs_test_mean` is implemented alongside it ONLY so the gap can be
reported as a diagnostic; it is never the headline.

Classification metrics are always returned next to the base rate, because at a 1s horizon
~96% of INTC labels are flat and raw accuracy is meaningless on its own.
"""
import numpy as np


def _finite_pair(y_true, y_pred):
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_pred, dtype=np.float64)
    if y.shape != p.shape:
        raise ValueError(f"shape mismatch: y {y.shape} vs yhat {p.shape}")
    ok = np.isfinite(y) & np.isfinite(p)
    return y[ok], p[ok]


def r2_vs_zero(y_true, y_pred):
    """1 - SSE/sum(y^2). The headline regression metric; a zero forecast scores exactly 0."""
    y, p = _finite_pair(y_true, y_pred)
    if y.size == 0:
        return np.nan
    denom = float(np.sum(y * y))
    if denom <= 0.0:
        return np.nan  # a test block with no variation at all: R2 undefined, not 0
    return 1.0 - float(np.sum((y - p) ** 2)) / denom


def r2_vs_test_mean(y_true, y_pred):
    """Diagnostic ONLY -- the leaky variant, reported to show the size of the difference."""
    y, p = _finite_pair(y_true, y_pred)
    if y.size < 2:
        return np.nan
    denom = float(np.sum((y - y.mean()) ** 2))
    if denom <= 0.0:
        return np.nan
    return 1.0 - float(np.sum((y - p) ** 2)) / denom


def _average_ranks(x):
    """Ranks with ties sharing the mean of the span they cover (hand-written, no scipy)."""
    n = x.shape[0]
    order = np.argsort(x, kind="mergesort")
    ranks = np.empty(n, dtype=np.float64)
    sorted_x = x[order]
    i = 0
    while i < n:
        j = i + 1
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[order[i:j]] = 0.5 * (i + j - 1)
        i = j
    return ranks


def spearman_ic(y_true, y_pred):
    """Information coefficient: Spearman rho between prediction and realized move."""
    y, p = _finite_pair(y_true, y_pred)
    if y.size < 2:
        return np.nan
    ry, rp = _average_ranks(y), _average_ranks(p)
    ry -= ry.mean()
    rp -= rp.mean()
    dy = float(np.sqrt(np.sum(ry * ry)))
    dp = float(np.sqrt(np.sum(rp * rp)))
    if dy <= 0.0 or dp <= 0.0:
        return np.nan  # a constant predictor has no rank variance -- NaN, never 0
    return float(np.sum(ry * rp) / (dy * dp))


def calibration_slope(y_true, y_pred):
    """Slope of realized on predicted. ~1 is calibrated; 0.2 means confidently wrong at scale."""
    y, p = _finite_pair(y_true, y_pred)
    if y.size < 2:
        return np.nan
    pc = p - p.mean()
    var = float(np.sum(pc * pc))
    if var <= 0.0:
        return np.nan
    return float(np.sum(pc * (y - y.mean())) / var)


def _confusion(y_cls, p_cls, classes):
    k = len(classes)
    lookup = {c: i for i, c in enumerate(classes)}
    m = np.zeros((k, k), dtype=np.float64)
    for t, q in zip(y_cls, p_cls):
        m[lookup[t], lookup[q]] += 1.0
    return m


def classification_metrics(y_true_cls, y_pred_cls, classes=(-1.0, 0.0, 1.0)):
    """Balanced accuracy, MCC and the base rate, returned together.

    Balanced accuracy is the mean per-class recall over classes PRESENT in the truth, so a
    class the test block never contains cannot drag the score toward zero. `base_rate` is the
    majority-class share -- the number plain accuracy has to beat to mean anything.
    """
    y, p = _finite_pair(y_true_cls, y_pred_cls)
    out = {
        "n": int(y.size),
        "balanced_accuracy": np.nan,
        "mcc": np.nan,
        "base_rate": np.nan,
        "accuracy": np.nan,
    }
    if y.size == 0:
        return out

    m = _confusion(y, p, list(classes))
    total = float(m.sum())
    out["accuracy"] = float(np.trace(m) / total) if total > 0 else np.nan

    support = m.sum(axis=1)
    present = support > 0
    if np.any(present):
        recalls = np.divide(np.diag(m), support, out=np.zeros_like(support), where=present)
        out["balanced_accuracy"] = float(recalls[present].mean())
        out["base_rate"] = float(support.max() / total)

    # Multiclass MCC (Gorodkin), written out rather than imported.
    t_k = m.sum(axis=1)
    p_k = m.sum(axis=0)
    c = float(np.trace(m))
    s = total
    num = c * s - float(np.dot(t_k, p_k))
    den = np.sqrt(max(s * s - float(np.dot(p_k, p_k)), 0.0)) * np.sqrt(
        max(s * s - float(np.dot(t_k, t_k)), 0.0)
    )
    out["mcc"] = float(num / den) if den > 0 else np.nan
    return out


def auc_binary(y_binary, score):
    """AUC via the rank identity (Mann-Whitney U); ties get half credit. No sklearn."""
    y, s = _finite_pair(y_binary, score)
    pos = y > 0
    n_pos = int(np.count_nonzero(pos))
    n_neg = int(y.size - n_pos)
    if n_pos == 0 or n_neg == 0:
        return np.nan  # AUC is undefined with one class present
    r = _average_ranks(s)
    return float((r[pos].sum() - n_pos * (n_pos - 1) / 2.0) / (n_pos * n_neg))


def fold_summary(values):
    """The distribution EVAL_PROTOCOL.md sec. 1 demands: never a bare mean.

    "A mean IC of 0.03 built from folds of {0.14, 0.02, -0.01, 0.00, 0.00} is a story about
    one day, not a signal." So median and WORST fold travel with every mean this project
    prints.
    """
    v = np.asarray([x for x in values if x is not None], dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {"n_folds": 0, "mean": np.nan, "median": np.nan, "worst": np.nan,
                "best": np.nan, "std": np.nan}
    return {
        "n_folds": int(v.size),
        "mean": float(v.mean()),
        "median": float(np.median(v)),
        "worst": float(v.min()),
        "best": float(v.max()),
        "std": float(v.std(ddof=1)) if v.size > 1 else 0.0,
    }
