"""Order flow imbalance: best-level (CKS), multi-level (XGH), integrated (CCZ) -- FEATURES.md sec. 1.

`e_n` (per-event, per-level contribution) is defined by **differencing consecutive FRAME
rows**, never stamp rows: `e_n = 1{Pb_n>=Pb_{n-1}} qb_n - 1{Pb_n<=Pb_{n-1}} qb_{n-1}
- 1{Pa_n<=Pa_{n-1}} qa_n + 1{Pa_n>=Pa_{n-1}} qa_{n-1}` (Cont-Kukanov-Stoikov). Both
indicators fire at an unchanged price, which is what turns a pure size change (a limit
add or a cancel at an unchanged best) into the signed size delta; when the price moves,
only the "improvement" branch fires and the WHOLE prior queue at the vacated price is
treated as removed -- this is what makes an aggressive order that clears the touch look
like a full-level cancel arithmetically, even though it is a trade. Row 0 has no
predecessor: its contribution is 0.0 (a sum over an empty window is legitimately 0.0,
per INTERFACE.md sec. 7 -- this is not "missing").

Causality: **`<= t`** (INTERFACE.md sec. 2). `e_n` contributions are built once over the
whole frame, then aggregated into each stamp's trailing `(t-w, t]` window via
`windows.prefix_sums` + `window_sum` (`<=stamps.index[i]` never `+1`).

Multi-level OFI (Xu-Gould-Howison) scales each level's raw windowed OFI by that level's
average two-sided depth `(qb+qa)/2`. Integrated OFI (Cont-Cucuringu-Zhang) is the first
PC of the L scaled level-OFIs. Both are FITTED statistics and go through
`windows.refit_blocks` (INTERFACE.md sec. 8): one estimate per `REFIT_BLOCK_NS` block,
computed on **frame rows strictly before the block start**, held constant inside the
block. Block 0 has no prior data, so every `ofi_l*`/`ofi_integrated_*` value in block 0
is NaN (~18,000 calendar stamps on the real frames -- the honest cost of a causal fit).

Design note on the PCA fit sample: the integrated-OFI loadings for block `k` are fit on
the depth-SCALED, per-event (not per-stamp) windowed OFI of frame rows strictly before
block `k`'s start, scaled by block `k`'s OWN depth normaliser (the normaliser that will
also be applied operationally to block `k`'s output stamps). Fitting at per-event
granularity -- literally every frame row treated as its own trailing-window "stamp" --
keeps the fit sample a function of the frame and the block schedule only, never of which
clock (calendar vs event) produced `stamps`; fitting on the stamp array itself would
violate that (INTERFACE.md sec. 8's second failure mode) and would silently differ
between clocks.

Validity: a level's contribution is NaN for one event if either side (bid or ask) is
invalid (LOBSTER sentinel price or zero displayed size, `edge.features.frame`'s per-side
`bid_valid`/`ask_valid`) at that event OR its predecessor -- price direction is then
undefined. This never happens on the 2012-06-21 INTC/AMZN sample (zero invalid rows) but
is exercised by a hand-built sentinel case below. A windowed *sum* silently drops NaN
event contributions (`windows.prefix_sums`'s stated convention); a windowed *ratio*
(depth normaliser, scaled OFI) is NaN if its own count/denominator is degenerate.

`book` supplies only the shared depth cap `book.n_levels`; OFI's own per-event arithmetic
reads `frame.bid_px` / `bid_sz` / `ask_px` / `ask_sz` and the cached per-side
`bid_valid`/`ask_valid` directly (frame-row granularity, not book-at-stamp rows), so
`book`'s array fields are otherwise unused here.
"""
import numpy as np

from edge.features import FEATURE_WINDOWS, WINDOWS_NS
from edge.features.book import book_at_stamps
from edge.features.windows import (
    prefix_sums,
    refit_blocks,
    trailing_bounds,
    trailing_start_index,
    window_sum,
)

# Frozen a priori (INTERFACE.md sec. 8's own worked example uses the same number): a
# 30-minute refit block carries thousands of events on both tickers, so 1000 only ever
# binds on block 0 (fit_end == 0, no prior data at all) -- it is not a second, hidden
# warmup knob past block 0. Applied uniformly to both the depth normaliser and the PCA
# fit; 1000 rows is also comfortably above L<=10 for a numerically stable covariance.
MIN_FIT_ROWS = 1000


def _event_contributions(bid_px, bid_sz, ask_px, ask_sz, bid_valid, ask_valid):
    """(n_events, L) e_n^(m), Cont-Kukanov-Stoikov eq. (FEATURES.md sec. 1). Row 0 is 0.0."""
    n, num_levels = bid_px.shape
    contrib = np.zeros((n, num_levels), dtype=np.float64)
    if n < 2:
        return contrib

    pb, pb0 = bid_px[1:], bid_px[:-1]
    qb, qb0 = bid_sz[1:], bid_sz[:-1]
    pa, pa0 = ask_px[1:], ask_px[:-1]
    qa, qa0 = ask_sz[1:], ask_sz[:-1]

    bid_term = np.where(pb >= pb0, qb, 0.0) - np.where(pb <= pb0, qb0, 0.0)
    ask_term = np.where(pa >= pa0, qa0, 0.0) - np.where(pa <= pa0, qa, 0.0)

    # Joint validity: both sides usable at BOTH events, or the level's contribution for
    # this one event is undefined (NaN), not the whole row.
    pair_valid = bid_valid[1:] & bid_valid[:-1] & ask_valid[1:] & ask_valid[:-1]
    contrib[1:] = np.where(pair_valid, bid_term + ask_term, np.nan)
    return contrib


def _estimate_first_pc(x):
    """First PC of an (n, L) fit matrix: mean-centre, LxL covariance, `np.linalg.eigh`.

    Sign fixed deterministically -- the loading on level 1 (column 0) is forced
    non-negative -- because an eigenvector's sign is otherwise arbitrary and would flip
    the feature's sign at a refit block boundary (INTERFACE.md sec. 8's third failure
    mode). `eigh` (not the general `eig`) because the covariance is symmetric by
    construction, which guarantees real eigenvalues/vectors and a stable ordering.
    """
    x = np.asarray(x, dtype=np.float64)
    centred = x - x.mean(axis=0)
    denom = max(centred.shape[0] - 1, 1)
    cov = (centred.T @ centred) / denom
    _eigvals, eigvecs = np.linalg.eigh(cov)  # ascending eigenvalues
    loadings = eigvecs[:, -1].copy()  # top eigenvector
    if loadings[0] < 0:
        loadings = -loadings
    return loadings


def _divide_where(numerator, denominator, condition):
    """numerator / denominator where `condition` holds, NaN elsewhere; never 0/0 or inf."""
    out = np.full(numerator.shape, np.nan, dtype=np.float64)
    np.divide(numerator, denominator, out=out, where=condition)
    return out


def compute(frame, stamps, book=None):
    """Best-level, multi-level and integrated OFI (FEATURES.md sec. 1); see module docstring."""
    if book is None:
        book = book_at_stamps(frame, stamps)
    num_levels = book.n_levels
    n_stamps = stamps.index.shape[0]

    bid_px = frame.bid_px[:, :num_levels].astype(np.float64)
    bid_sz = frame.bid_sz[:, :num_levels].astype(np.float64)
    ask_px = frame.ask_px[:, :num_levels].astype(np.float64)
    ask_sz = frame.ask_sz[:, :num_levels].astype(np.float64)
    bid_valid = frame.bid_valid[:, :num_levels]
    ask_valid = frame.ask_valid[:, :num_levels]

    contrib = _event_contributions(bid_px, bid_sz, ask_px, ask_sz, bid_valid, ask_valid)
    sum_prefix, _count_prefix = prefix_sums(contrib)  # (n_events+1, L); NaN excluded from sum

    # Depth normaliser input: average two-sided depth at each level, valid rows only.
    depth_valid = bid_valid & ask_valid
    depth_contrib = np.where(depth_valid, (bid_sz + ask_sz) / 2.0, np.nan)
    depth_sum_prefix, depth_count_prefix = prefix_sums(depth_contrib)

    blocks = refit_blocks(frame.ts_ns, stamps.ts_ns)
    fit_end = blocks.fit_end_index  # (n_blocks,), aligned with blocks.block_id
    fit_ok = fit_end >= MIN_FIT_ROWS
    depth_sum = depth_sum_prefix[fit_end]  # (n_blocks, L)
    depth_count = depth_count_prefix[fit_end].astype(np.float64)
    depth_by_block = _divide_where(depth_sum, depth_count, fit_ok[:, None] & (depth_count > 0))

    max_block = int(blocks.block_id.max()) if blocks.block_id.size else -1
    depth_lookup = np.full((max_block + 1, num_levels), np.nan, dtype=np.float64)
    depth_lookup[blocks.block_id] = depth_by_block
    depth_for_stamp = depth_lookup[blocks.stamp_block_id]  # (n_stamps, L)

    event_end = np.arange(frame.n_events, dtype=np.int64) + 1

    out = {}
    for w in FEATURE_WINDOWS:
        window_ns = WINDOWS_NS[w]

        s_start, s_end = trailing_bounds(frame.ts_ns, stamps, window_ns)
        raw_win_stamps = window_sum(sum_prefix, s_start, s_end)  # (n_stamps, L)

        out[f"ofi_best_{w}"] = raw_win_stamps[:, 0].copy()
        out[f"ofi_deep_{w}"] = raw_win_stamps.sum(axis=1)

        if w == "1s":
            for m in range(1, num_levels + 1):
                out[f"ofi_raw_l{m}_1s"] = raw_win_stamps[:, m - 1].copy()

        scaled = _divide_where(raw_win_stamps, depth_for_stamp, depth_for_stamp > 0)
        for m in range(1, num_levels + 1):
            out[f"ofi_l{m}_{w}"] = scaled[:, m - 1].copy()

        e_start = trailing_start_index(frame.ts_ns, frame.ts_ns, window_ns)
        raw_win_events = window_sum(sum_prefix, e_start, event_end)  # (n_events, L)

        integrated = np.full(n_stamps, np.nan, dtype=np.float64)
        for j, block in enumerate(blocks.block_id):
            fe = int(fit_end[j])
            if fe < MIN_FIT_ROWS:
                continue  # block 0 (and any under-warmed block): no fit, stays NaN
            depth_row = depth_lookup[block]
            if not np.all(np.isfinite(depth_row)) or np.any(depth_row <= 0):
                continue  # degenerate fit window (e.g. every level invalid): stays NaN
            fit_sample = raw_win_events[:fe, :] / depth_row[None, :]
            loadings = _estimate_first_pc(fit_sample)
            sel = blocks.stamp_block_id == block
            integrated[sel] = scaled[sel, :] @ loadings
        out[f"ofi_integrated_{w}"] = integrated

    return out
