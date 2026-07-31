"""Queue / book imbalance -- FEATURES.md sec. 2: touch I, depth-weighted I_L, sign(I).

Causality: entirely inherited from `book.py`'s `<= t` convention (INTERFACE.md sec. 2/4).
Every column here is a pure function of the single `BookAtStamps` row at each stamp -- there
is no trailing window, so this module has no boundary convention of its own to add.

NaN policy (INTERFACE.md sec. 7), three distinct rules, each with its own test:

* touch/level invalid (`~book.valid` / `~book.valid_l[:, m-1]`) -> that stamp's `qi_touch`,
  `qi_sign`, or `qi_l{m}` is NaN.
* zero total denominator -- `q^b + q^a == 0` at the touch, or (for the depth-weighted
  variants) zero levels contribute after exclusion -- is masked via `np.divide(..., out=nan,
  where=denom > 0)`, never a raw `0/0` under an errstate suppression. Given `book.py`'s
  validity rule already requires a positive size on both sides of a valid level, this branch
  is a strict subset of "no valid level contributed", but the guard is written directly
  against the denominator (not just the validity mask) as defense in depth.
* a level invalid at a given stamp is EXCLUDED from a depth-weighted sum's numerator and
  denominator, not zero-filled: filling it with 0 would assert "no resting size here", which
  is a claim about the book this module cannot make from a missing quote. Excluding it is
  "this level is a missing observation at this stamp".

`qi_sign` maps an exactly-zero touch imbalance to `0.0`, not a coin flip (`np.sign(0.0) ==
0.0`); NaN input propagates through `np.sign` unchanged, so an invalid touch still yields a
NaN sign, never a spurious 0.0.

Levels are iterated over `book.n_levels` (2 on the toy fixture, 10 on the real frames), never
a hardcoded 10, per INTERFACE.md sec. 4.

`EXP_DECAY_LAMBDA = 3.0` is frozen a priori (not fit to any INTC/AMZN data): with
`w_m = exp(-(m-1)/3)` it gives the touch weight 1.0, one full e-folding (a 2.718x decay,
w_4 = e^-1 = 0.368) by level 4, and w_10 = e^-3 = 4.98% at level 10
-- a smooth few-level e-folding that matches the standard prior that resting size far from
the touch carries comparatively little short-horizon information, without collapsing to
either "only the touch matters" (LAMBDA -> 0) or "every level matters equally" (LAMBDA -> inf).
"""
import numpy as np

from edge.features.book import book_at_stamps

# Frozen a priori; see module docstring for the choice of decay rate.
EXP_DECAY_LAMBDA = 3.0


def _safe_ratio(numer, denom, valid_mask=None):
    """`numer / denom`, NaN wherever `denom <= 0` or `valid_mask` is False.

    `np.divide(..., where=...)` leaves the pre-filled NaN output untouched outside the mask,
    so this is never a raw `0/0` computed and then discarded under an errstate suppression.
    """
    out = np.full(denom.shape[0], np.nan, dtype=np.float64)
    mask = denom > 0
    if valid_mask is not None:
        mask = mask & valid_mask
    np.divide(numer, denom, out=out, where=mask)
    return out


def _level_weights(n_levels, kind):
    """1-indexed level weights w_1..w_{n_levels} for one of the three frozen schemes."""
    m = np.arange(1, n_levels + 1, dtype=np.float64)
    if kind == "uniform":
        return np.ones_like(m)
    if kind == "inv":
        return 1.0 / m
    if kind == "exp":
        return np.exp(-(m - 1.0) / EXP_DECAY_LAMBDA)
    raise ValueError(f"unknown level-weight scheme {kind!r}")


def _depth_weighted_ratio(book, weights):
    """I_L for one weight vector: invalid levels excluded from BOTH the sum and the count."""
    # NaN out invalid levels before weighting -- multiplying by a weight must not turn a
    # missing level into a fake zero-size observation.
    bid_masked = np.where(book.valid_l, book.bid_sz_l.astype(np.float64), np.nan)
    ask_masked = np.where(book.valid_l, book.ask_sz_l.astype(np.float64), np.nan)

    # nansum drops the excluded (NaN) levels entirely from both sides; an all-invalid row
    # (zero levels contributing) reduces to 0.0 on both sums, which the safe-ratio's
    # `denom > 0` guard below turns into NaN rather than a 0/0.
    bid_sum = np.nansum(bid_masked * weights[np.newaxis, :], axis=1)
    ask_sum = np.nansum(ask_masked * weights[np.newaxis, :], axis=1)

    return _safe_ratio(bid_sum - ask_sum, bid_sum + ask_sum)


def compute(frame, stamps, book=None):
    """Touch/level/depth-weighted queue imbalance and its sign -- FEATURES.md sec. 2."""
    if book is None:
        book = book_at_stamps(frame, stamps)

    out = {}

    bid_sz = book.bid_sz.astype(np.float64)
    ask_sz = book.ask_sz.astype(np.float64)
    qi_touch = _safe_ratio(bid_sz - ask_sz, bid_sz + ask_sz, book.valid)
    out["qi_touch"] = qi_touch
    out["qi_sign"] = np.sign(qi_touch)  # NaN-in -> NaN-out; exact 0.0 stays 0.0, not +/-1.

    # Loop bound is book.n_levels (a constant <= 10), never NUM_LEVELS: the toy fixture
    # carries 2 levels and a hardcoded 10 would IndexError on it (INTERFACE.md sec. 4).
    for m in range(1, book.n_levels + 1):
        bid_l = book.bid_sz_l[:, m - 1].astype(np.float64)
        ask_l = book.ask_sz_l[:, m - 1].astype(np.float64)
        out[f"qi_l{m}"] = _safe_ratio(bid_l - ask_l, bid_l + ask_l, book.valid_l[:, m - 1])

    out["qi_depth_uniform"] = _depth_weighted_ratio(book, _level_weights(book.n_levels, "uniform"))
    out["qi_depth_inv"] = _depth_weighted_ratio(book, _level_weights(book.n_levels, "inv"))
    out["qi_depth_exp"] = _depth_weighted_ratio(book, _level_weights(book.n_levels, "exp"))

    return out
