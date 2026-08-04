"""The L0-L4 ladder (PLAN.md phase 2). Hand-written fits; lightgbm is the only library model.

Ordered so each rung has to beat the one below it. Every rung exposes the same interface --
`fit(X_train, y_train) -> predict(X_test) -> continuous score in label units` -- so the
harness code is one loop and no rung gets a bespoke scoring path.

    L0  zero predictor                      no parameters. Beating this is not news.
    L1  sign(queue imbalance)               no parameters.
    L1m sign(microprice - mid)              no parameters.
    L2  OLS of dmid on best-level OFI       the Cont-Kukanov-Stoikov specification.
    L3  ridge on the full handcrafted set
    L4  gradient boosting on the same set   the real bar.

THE SCALE STEP. L1 emits a direction in {-1, 0, +1}, not a number of ticks, so an R^2
computed on it directly would be measuring units rather than skill. Every rung's raw score is
therefore mapped into label units by ONE parameter fitted through the origin on TRAINING data
only:

    scale = sum(s_train * y_train) / sum(s_train^2)

Applied uniformly to every rung, including the ones that do not need it, so it cannot favour
one. For any least-squares rung it is EXACTLY 1.0 -- OLS residuals are orthogonal to fitted
values, so sum(yhat*y) = sum(yhat^2) -- which is asserted in tests/test_models.py. It
therefore changes nothing for L2/L3 and only puts L1 and L4 on a comparable footing. It is
rank-preserving up to sign, so it never changes the IC.

NaN POLICY. INTERFACE.md sec. 7 puts the drop here, in the harness, per fold and per rung --
never in the family modules -- because which rows are droppable depends on which columns a
rung actually uses. L2 uses one OFI column and must not lose rows to a NaN in an L4-only
column. Every fit reports `n_train`/`n_test` after dropping so the per-fold row counts are
visible rather than absorbed (the phase-1 adversary's F6: `tr_aggr_imb_*` is finite on 1.8%
of AMZN calendar stamps, so rungs using it will differ from rungs that do not by more than
an order of magnitude).
"""
import numpy as np

# The handcrafted set for L3/L4 (PLAN.md's list: multi-level OFI, integrated OFI, queue
# imbalance, microprice deviation, spread, depth, trade-sign imbalance, realized vol,
# time-of-day). Frozen HERE, a priori, and deliberately not the full 144-column table: an
# unrestricted cross product would inflate the trial count for free (RULES.md rule 4) and
# hand a GBM a wide multiple-comparison surface to mine.
L3_FEATURES = (
    # order flow -- best level, three lookbacks, plus the integrated and deep summaries
    "ofi_best_100ms", "ofi_best_1s", "ofi_best_5s",
    "ofi_integrated_1s", "ofi_deep_1s",
    # a thinned multi-level ladder rather than all 30 level x window cells
    "ofi_l2_1s", "ofi_l3_1s", "ofi_l5_1s", "ofi_l10_1s",
    # queue / book imbalance
    "qi_touch", "qi_depth_inv", "qi_l2", "qi_l5",
    # microprice
    "mp_dev_mid", "mp_dev_wmid", "mp_wmid_dev_mid",
    # Trade / aggressor. NOTE the omission of `tr_aggr_imb_1s`: it is a RATIO, so it is NaN
    # whenever its window holds no trades -- 84.7% of INTC calendar stamps and 86.5% of
    # AMZN's. Including it would score L3/L4 on 13% of the sample while L0/L1/L2 kept 100%,
    # which is not a ladder, it is two different experiments. `tr_signed_vol_1s` and
    # `tr_count_1s` are SUMS, legitimately 0.0 on a quiet window (INTERFACE.md sec. 7), and
    # carry the same directional information unnormalised. Chosen on the NaN structure of
    # the columns, before any fold was scored -- not because it improved a result.
    "tr_signed_vol_1s", "tr_count_1s", "tr_eff_spread",
    # state / control
    "st_spread_ticks", "st_depth_touch", "st_depth_total",
    "st_rv_1s", "st_event_rate_1s", "st_slope_bid", "st_slope_ask",
    "st_tod_frac",
)

L2_FEATURES = ("ofi_best_1s",)          # the CKS specification: one regressor
L1_FEATURE = "qi_touch"                 # unfitted sign
L1_MICRO_FEATURE = "mp_dev_mid"         # unfitted sign

RIDGE_LAMBDA = 1.0  # frozen a priori; swept once in the sensitivity table, never tuned per fold


def _fit_scale(score_train, y_train):
    """One-parameter through-origin rescale into label units. Exactly 1.0 for OLS/ridge."""
    s = np.asarray(score_train, dtype=np.float64)
    denom = float(np.sum(s * s))
    if denom <= 0.0:
        return 0.0
    return float(np.sum(s * np.asarray(y_train, dtype=np.float64)) / denom)


class Rung:
    """Base: a rung is a raw score function plus the shared train-only rescale."""

    name = "base"
    features = ()

    def __init__(self):
        self.scale_ = 1.0

    def _raw(self, X):
        raise NotImplementedError

    def _fit_raw(self, X, y):
        pass

    def fit(self, X, y, scale_holdout=None, model_mask=None):
        """Fit the raw score, then the through-origin rescale.

        `scale_holdout` is a boolean mask over the TRAINING rows. Where given, the raw model
        is fitted on `~scale_holdout` and the scale on `scale_holdout` — a strictly causal
        calibration. Where None, both are fitted on the same rows, which is the in-sample
        variant and is retained only as a diagnostic.

        `model_mask` DECOUPLES the two row sets. Where given, the raw model is fitted on
        `model_mask` and the scale on `scale_holdout`, and the two are allowed to OVERLAP.
        That overlap makes the scale partly in-sample and is the accepted cost; what it buys
        is that the calibration rule can be varied without also varying the model's training
        window. Without it, `~scale_holdout` IS the model's training set, so the two channels
        cannot be separated — which is why the `A' - C` control could only ever produce a
        bracket (`FINDINGS.md` sec. 1.1a, 2026-08-03 fourth red team).

        WHY THIS EXISTS (2026-08-02 red team). The in-sample scale solves
        `sum(s*y)/sum(s*s)` against predictions that a high-capacity model has already fitted
        almost perfectly, so it returns essentially no shrinkage. Out of sample the amplitude
        is then far too large and R^2-vs-zero explodes negative — which is exactly the large
        negative protocol-C R^2 the earlier ladder reported as a finding. It was a calibration
        artifact: with a held-out scale, INTC calendar 10s L4 goes from C = -0.5523 to
        C = +0.0151 and its A-C gap from 1.0707 to 0.4920. Under protocol A the artifact never
        bites, because A's "out-of-sample" rows are the temporal neighbours of its training
        rows -- so the in-sample scale inflates the measured optimism rather than the model.
        """
        if scale_holdout is None:
            self._fit_raw(X, y)
            self.scale_ = _fit_scale(self._raw(X), y)
            return self

        hold = np.asarray(scale_holdout, dtype=bool)
        if hold.all() or not hold.any():
            raise ValueError("scale_holdout must leave rows on both sides of the split")
        if model_mask is None:
            fit_rows = ~hold
        else:
            fit_rows = np.asarray(model_mask, dtype=bool)
            if not fit_rows.any():
                raise ValueError("model_mask must select at least one row")
        self._fit_raw(X[fit_rows], y[fit_rows])
        self.scale_ = _fit_scale(self._raw(X[hold]), y[hold])
        return self

    def predict(self, X):
        return self.scale_ * self._raw(X)


class L0Zero(Rung):
    """Constant zero. R^2 vs zero is 0.0 by construction; IC is undefined (no variance)."""

    name = "L0 zero"
    features = ()

    def _raw(self, X):
        return np.zeros(X.shape[0], dtype=np.float64)

    def fit(self, X, y, scale_holdout=None, model_mask=None):
        self.scale_ = 0.0  # nothing to scale; keep the prediction identically zero
        return self


class L1Sign(Rung):
    """Unfitted sign of a single feature. No parameters beyond the shared rescale."""

    def __init__(self, feature, name):
        super().__init__()
        self.features = (feature,)
        self.name = name

    def _raw(self, X):
        return np.sign(X[:, 0])


class L2OLS(Rung):
    """OLS with intercept, normal equations via lstsq (Cont-Kukanov-Stoikov eq. 4 for L2)."""

    name = "L2 OLS on OFI"
    features = L2_FEATURES

    def __init__(self, features=L2_FEATURES, name=None):
        super().__init__()
        self.features = tuple(features)
        if name:
            self.name = name
        self.coef_ = None
        self.intercept_ = 0.0

    def _design(self, X):
        return np.column_stack([np.ones(X.shape[0]), X])

    def _fit_raw(self, X, y):
        beta, *_ = np.linalg.lstsq(self._design(X), y, rcond=None)
        self.intercept_ = float(beta[0])
        self.coef_ = beta[1:]

    def _raw(self, X):
        return self.intercept_ + X @ self.coef_


class L3Ridge(Rung):
    """Ridge on standardised features. Closed form; the standardiser is fit on TRAIN ONLY.

    Centre y and X on training means, solve (X'X + lambda*I) b = X'y on the standardised
    design, and carry the training mean/std forward to transform test rows. A standardiser
    fit on the full sample is the leak this project exists to measure, so it is fit here,
    inside the fold, and nowhere else.
    """

    name = "L3 ridge, full set"
    features = L3_FEATURES

    def __init__(self, features=L3_FEATURES, lam=RIDGE_LAMBDA, name=None):
        super().__init__()
        self.features = tuple(features)
        self.lam = float(lam)
        if name:
            self.name = name
        self.mu_ = None
        self.sigma_ = None
        self.y_mu_ = 0.0
        self.coef_ = None

    def _standardise(self, X):
        return (X - self.mu_) / self.sigma_

    def _fit_raw(self, X, y):
        self.mu_ = X.mean(axis=0)
        sigma = X.std(axis=0, ddof=0)
        # A constant column in this fold carries no information; divide by 1 so it maps to
        # exactly 0 rather than producing inf/NaN and poisoning the whole solve.
        self.sigma_ = np.where(sigma > 0, sigma, 1.0)
        Z = self._standardise(X)
        self.y_mu_ = float(y.mean())
        yc = y - self.y_mu_
        k = Z.shape[1]
        A = Z.T @ Z + self.lam * np.eye(k)
        self.coef_ = np.linalg.solve(A, Z.T @ yc)

    def _raw(self, X):
        return self.y_mu_ + self._standardise(X) @ self.coef_


class L4GBM(Rung):
    """Gradient boosting on the same handcrafted set. The real bar.

    Hyperparameters are frozen a priori (small trees, heavy regularisation, fixed rounds --
    no early stopping, because early stopping on a validation slice carved out of the test
    fold is itself a leak and carving it out of training is one more free parameter). Seeded.
    """

    name = "L4 GBM, full set"
    features = L3_FEATURES

    PARAMS = {
        "objective": "regression",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_data_in_leaf": 200,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "lambda_l2": 1.0,
        "verbosity": -1,
        "deterministic": True,
        "force_row_wise": True,
    }
    N_ROUNDS = 200

    def __init__(self, features=L3_FEATURES, seed=0, name=None):
        super().__init__()
        self.features = tuple(features)
        self.seed = int(seed)
        if name:
            self.name = name
        self.booster_ = None

    def _fit_raw(self, X, y):
        import lightgbm as lgb

        params = dict(self.PARAMS)
        params["seed"] = self.seed
        params["bagging_seed"] = self.seed
        params["feature_fraction_seed"] = self.seed
        params["num_threads"] = 1  # determinism over speed (RULES.md rule 7)
        self.booster_ = lgb.train(
            params, lgb.Dataset(X, label=y), num_boost_round=self.N_ROUNDS
        )

    def _raw(self, X):
        if self.booster_ is None:
            return np.zeros(X.shape[0], dtype=np.float64)
        return np.asarray(self.booster_.predict(X), dtype=np.float64)


def build_ladder(seed=0):
    """The frozen rung list, in ladder order. One place, so every report agrees."""
    return [
        L0Zero(),
        L1Sign(L1_FEATURE, "L1 sign(queue imbalance)"),
        L1Sign(L1_MICRO_FEATURE, "L1m sign(microprice - mid)"),
        L2OLS(),
        L3Ridge(),
        L4GBM(seed=seed),
    ]
