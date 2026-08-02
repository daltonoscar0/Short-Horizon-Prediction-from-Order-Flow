"""Inferential statistics for the ladder (EVAL_PROTOCOL.md sec. 2, "Inferential").

Hand-written on numpy: no scipy, no statsmodels. The three tools:

  * Diebold-Mariano between adjacent rungs, with the HAC correction that overlapping
    forecasts make mandatory;
  * block bootstrap CIs on metric DIFFERENCES, block length >= the label horizon;
  * the deflated Sharpe ratio, using the logged trial count.

The deflated Sharpe is the reason CLAUDE.md rule 4 makes the trial log mandatory and requires
logging the failures too. A Sharpe selected as the best of N tries is biased upward by an
amount that grows with N; deflating without an honest N is worse than not deflating, because
it launders the selection through a formula that looks rigorous.

Normal CDF/quantile are computed from `math.erf`/a rational inverse rather than imported,
because scipy is outside the dependency contract.
"""
import math

import numpy as np

EULER_MASCHERONI = 0.5772156649015329


def _norm_cdf(x):
    return 0.5 * (1.0 + math.erf(float(x) / math.sqrt(2.0)))


def _norm_ppf(p):
    """Inverse normal CDF, Acklam's rational approximation (|error| < 1.15e-9)."""
    p = float(p)
    if not 0.0 < p < 1.0:
        return float("nan")
    a = [-3.969683028665376e+01, 2.209460984245205e+02, -2.759285104469687e+02,
         1.383577518672690e+02, -3.066479806614716e+01, 2.506628277459239e+00]
    b = [-5.447609879822406e+01, 1.615858368580409e+02, -1.556989798598866e+02,
         6.680131188771972e+01, -1.328068155288572e+01]
    c = [-7.784894002430293e-03, -3.223964580411365e-01, -2.400758277161838e+00,
         -2.549732539343734e+00, 4.374664141464968e+00, 2.938163982698783e+00]
    d = [7.784695709041462e-03, 3.224671290700398e-01, 2.445134137142996e+00,
         3.754408661907416e+00]
    plow, phigh = 0.02425, 1 - 0.02425
    if p < plow:
        q = math.sqrt(-2 * math.log(p))
        return (((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    if p > phigh:
        q = math.sqrt(-2 * math.log(1 - p))
        return -(((((c[0]*q+c[1])*q+c[2])*q+c[3])*q+c[4])*q+c[5]) / ((((d[0]*q+d[1])*q+d[2])*q+d[3])*q+1)
    q = p - 0.5
    r = q * q
    return (((((a[0]*r+a[1])*r+a[2])*r+a[3])*r+a[4])*r+a[5])*q / (((((b[0]*r+b[1])*r+b[2])*r+b[3])*r+b[4])*r+1)


def _hac_variance(d, lag):
    """Newey-West/Bartlett long-run variance of the mean of `d`."""
    n = d.shape[0]
    dc = d - d.mean()
    gamma0 = float(np.dot(dc, dc)) / n
    total = gamma0
    for k in range(1, int(lag) + 1):
        if k >= n:
            break
        cov = float(np.dot(dc[k:], dc[:-k])) / n
        total += 2.0 * (1.0 - k / (lag + 1.0)) * cov
    return max(total, 0.0)


def diebold_mariano(errors_a, errors_b, lag=None, loss="squared"):
    """DM test that model A and model B have equal predictive accuracy.

    `errors_*` are forecast errors (y - yhat) on the SAME rows, or -- with `loss="linear"` --
    per-observation losses already in the caller's units. Returns the DM statistic and a
    two-sided p-value; the statistic is negative when A has the lower loss (A is better).

    The HAC lag is mandatory rather than optional here: at a 1s horizon on a 100ms grid the
    loss differential is autocorrelated across ~10 neighbouring samples, and the naive
    standard error would overstate significance by roughly sqrt(10). Defaults to the
    conventional floor n^(1/3) when not supplied, but callers who know the label horizon
    should pass it in stamps.

    NOTE ON NESTED MODELS. EVAL_PROTOCOL.md sec. 2 asks for the conditional Giacomini-White
    variant where the models are nested. The ladder's adjacent pairs are largely nested
    (L3's feature set contains L2's), and standard DM is known to be undersized there, so
    a DM p-value between nested rungs should be read as indicative, not decisive. Flagged
    rather than silently reported as if it were exact.
    """
    ea = np.asarray(errors_a, dtype=np.float64)
    eb = np.asarray(errors_b, dtype=np.float64)
    ok = np.isfinite(ea) & np.isfinite(eb)
    ea, eb = ea[ok], eb[ok]
    n = ea.shape[0]
    if n < 8:
        return {"dm_stat": np.nan, "p_value": np.nan, "n": int(n), "lag": np.nan}
    if loss == "squared":
        d = ea ** 2 - eb ** 2
    elif loss == "absolute":
        d = np.abs(ea) - np.abs(eb)
    elif loss == "linear":
        # Inputs are ALREADY losses, not errors. The economic comparison needs this: PnL is
        # not a forecast error, and squaring it would test whether one rung's PnL is larger
        # in MAGNITUDE, which is the wrong hypothesis. Callers pass -pnl per trade.
        d = ea - eb
    else:
        raise ValueError("loss must be 'squared', 'absolute' or 'linear'")
    if lag is None:
        lag = max(1, int(round(n ** (1.0 / 3.0))))
    var = _hac_variance(d, lag)
    if var <= 0:
        # Identical models are the common case here (adjacent rungs can coincide exactly on
        # a fold): the differential is degenerate, so the test is undefined rather than
        # significant. `mean_loss_diff` is still reported so a caller can see it is zero.
        return {"dm_stat": np.nan, "p_value": np.nan, "n": int(n), "lag": int(lag),
                "mean_loss_diff": float(d.mean())}
    stat = float(d.mean() / math.sqrt(var / n))
    p = 2.0 * (1.0 - _norm_cdf(abs(stat)))
    return {"dm_stat": stat, "p_value": p, "n": int(n), "lag": int(lag),
            "mean_loss_diff": float(d.mean())}


def block_bootstrap_ci(values, block_len, n_boot=2000, alpha=0.05, seed=0):
    """Percentile CI for the MEAN of `values` under a circular block bootstrap.

    Block length must be >= the label horizon in samples (EVAL_PROTOCOL.md sec. 2): resampling
    individual observations would destroy the serial dependence that makes the naive CI wrong
    in the first place, and would produce an interval far too narrow.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    n = v.shape[0]
    if n == 0:
        return {"mean": np.nan, "lo": np.nan, "hi": np.nan, "n": 0}
    block_len = max(1, min(int(block_len), n))
    n_blocks = int(math.ceil(n / block_len))
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n, size=(int(n_boot), n_blocks))
    offsets = np.arange(block_len)
    means = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = (starts[i][:, None] + offsets[None, :]).ravel()[:n] % n  # circular
        means[i] = v[idx].mean()
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return {"mean": float(v.mean()), "lo": lo, "hi": hi, "n": int(n),
            "block_len": int(block_len), "n_boot": int(n_boot)}


def expected_max_sharpe(n_trials, var_sharpe):
    """E[max Sharpe] over `n_trials` independent tries under the null (Bailey-Lopez de Prado).

        E[max] ~ sqrt(var) * ((1-g) * z(1 - 1/N) + g * z(1 - 1/(N e)))

    with g the Euler-Mascheroni constant. This is the benchmark the observed Sharpe has to
    clear; it grows with the number of configurations tried, which is why the trial log is
    mandatory.
    """
    n = int(n_trials)
    if n <= 1 or not np.isfinite(var_sharpe) or var_sharpe <= 0:
        return 0.0
    g = EULER_MASCHERONI
    z1 = _norm_ppf(1.0 - 1.0 / n)
    z2 = _norm_ppf(1.0 - 1.0 / (n * math.e))
    return math.sqrt(var_sharpe) * ((1.0 - g) * z1 + g * z2)


def deflated_sharpe(returns, n_trials, periods_per_year=None, seed=0):
    """Deflated Sharpe ratio (Bailey & Lopez de Prado 2014).

    Corrects the observed Sharpe for (a) the number of configurations tried, (b) the
    non-normality of the return series -- skew and kurtosis both inflate a naive Sharpe's
    apparent significance. Returns the DSR as a probability: the chance the true Sharpe is
    positive given the selection effect.

    A DSR below ~0.95 means the strategy is not distinguishable from the best of `n_trials`
    coin flips. An honest `n_trials` is the whole game; passing 1 here silently turns this
    back into an undeflated Sharpe wearing a rigorous name.
    """
    r = np.asarray(returns, dtype=np.float64)
    r = r[np.isfinite(r)]
    n = r.shape[0]
    out = {"sharpe": np.nan, "dsr": np.nan, "n": int(n), "n_trials": int(n_trials),
           "sharpe_threshold": np.nan, "skew": np.nan, "kurtosis": np.nan}
    if n < 8:
        return out
    mu, sd = float(r.mean()), float(r.std(ddof=1))
    if sd <= 0:
        return out
    sharpe = mu / sd
    z = (r - mu) / sd
    skew = float(np.mean(z ** 3))
    kurt = float(np.mean(z ** 4))  # non-excess
    out.update({"sharpe": sharpe, "skew": skew, "kurtosis": kurt})

    # Variance of the Sharpe estimator across trials under the null.
    var_sh = (1.0 - skew * sharpe + 0.25 * (kurt - 1.0) * sharpe ** 2) / (n - 1)
    threshold = expected_max_sharpe(n_trials, max(var_sh, 1e-18))
    out["sharpe_threshold"] = float(threshold)

    denom = math.sqrt(max(1.0 - skew * sharpe + 0.25 * (kurt - 1.0) * sharpe ** 2, 1e-18))
    stat = (sharpe - threshold) * math.sqrt(max(n - 1, 1)) / denom
    out["dsr"] = float(_norm_cdf(stat))
    if periods_per_year:
        out["sharpe_annualised"] = sharpe * math.sqrt(periods_per_year)
    return out
