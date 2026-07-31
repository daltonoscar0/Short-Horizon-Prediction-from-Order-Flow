"""Stoikov (2018) microprice: iterative absorbing/transient Markov estimate; FEATURES.md sec. 3.

`micro` is NOT the weighted mid, and it is NOT the first-order term G^1 either -- the
distinction FEATURES.md sec. 3 says an interviewer will poke at. Over a discretised
(spread bucket, imbalance bucket) state, each event either leaves the mid unchanged
(TRANSIENT, matrix Q) or moves it (ABSORBING, payoff R K and landing distribution R2):

    G^1 = (I - Q)^-1 R K        expected mid move up to the NEXT price change
    B   = (I - Q)^-1 R2         state observed AT that price change
    G^(i+1) = G^1 + B G^i       one more price change per iteration
    micro = mid + G^inf[state]

Stopping at G^1 -- solving `x = R K + Q x` and calling it the microprice -- is the natural
wrong answer: it is a real quantity (the expected move up to the next price change) but it
discards every subsequent change, which is precisely what the outer recursion is for. See
`_micro_adjustment` for why G^inf cannot be obtained by a direct solve.

Causality: standard `<= t` (INTERFACE.md sec. 2) for reading the state at a stamp. The
transition matrices (B, G^1) are a FITTED statistic and go through `windows.refit_blocks`
(INTERFACE.md sec. 8): one estimate per `REFIT_BLOCK_NS` block, estimated on frame rows
STRICTLY BEFORE the block start, held constant inside the block. Block 0 has no prior data
(`fit_end_index[0] == 0`) and is NaN by construction (below `MIN_FIT_ROWS`). A state that is
never visited as a transition source within a block's fit sample gets no estimate -- NaN, not
an imputed neighbour value, per INTERFACE.md sec. 8/7.

The two `(I - Q)^-1` solves go through `_absorb`, which uses `np.linalg.solve` when `I - Q` is
well conditioned (`np.linalg.cond(...) <= COND_THRESHOLD`) and falls back to the bounded
Neumann series `sum_k Q^k M` (frozen `TOL`/`MAX_ITERS`) otherwise. Direct solve is the primary
path because `N_STATES` is small (15) and cheap to invert 14 times; the series exists only for
a degenerate block where a state's entire sampled Q row is a "no move" self-loop.

All four output columns are float64, aligned to `stamps`, and all four are in TICKS:
`mp_micro` is a price LEVEL and `mp_dev_mid`/`mp_dev_wmid`/`mp_wmid_dev_mid` are price
DIFFERENCES, and INTERFACE.md sec. 1 puts both in ticks via `book.ticks()`. The internal
arithmetic stays in raw integer units and converts exactly once, at the return.
"""
import numpy as np

from edge.features.book import book_at_stamps, ticks
from edge.features.windows import refit_blocks

# --- frozen, a-priori state discretisation (FEATURES.md sec. 3 / INTERFACE.md sec. 8: "N_IMB
# and N_SPREAD are free parameters and must be FROZEN A PRIORI ... NOT by looking at which
# value gave a better correlation"). ---

# Imbalance I = q^b / (q^b + q^a) in [0, 1], split into 5 equal-width buckets (quintiles): a
# small, round number from Stoikov's own construction (coarse imbalance buckets, not a
# calibrated grid), picked so a 30-minute fit window still puts a healthy sample in every
# bucket rather than fragmenting the state space.
N_IMB = 5

# Spread in ticks, split into 3 buckets: exactly 1 tick, exactly 2 ticks, and >=3 ticks
# (the top bucket absorbs everything wider). INTC is a large-tick name that sits at 1 tick
# almost always (its mass concentrates in bucket 0); AMZN is small-tick and ranges wider, so
# a fixed, small bucket count with an absorbing top bucket is what keeps the state space
# identical -- and non-degenerate -- across both tickers without a per-ticker calibration.
N_SPREAD = 3

N_STATES = N_SPREAD * N_IMB  # 15

# A block's fit sample must give a non-trivial count to up to N_STATES**2 = 225 (s, s')
# transition cells; 5000 events is a floor of ~22 average transitions per cell in the sparsest
# case. This also automatically excludes block 0 (fit_end_index[0] == 0 always -- see
# windows.refit_blocks / INTERFACE.md sec. 8) without a special case. Frozen a priori, not
# tuned against any downstream correlation.
MIN_FIT_ROWS = 5000

# Neumann-series fallback for (I - Q)^-1, only used when the direct solve is ill-conditioned.
TOL = 1e-10
MAX_ITERS = 200

# Outer microprice recursion depth, G^(i+1) = G^1 + B G^i. Frozen at Stoikov's own repo
# value of 6, NOT tuned: `I - B` is singular by construction (B is row-stochastic on visited
# states), so this recursion cannot be replaced by a direct solve and its depth has to be a
# fixed a-priori constant. The TOL early exit means a chain that has already converged stops
# sooner; six is the cap, not a target.
MICRO_ITERS = 6

# Condition-number threshold above which np.linalg.solve is distrusted. float64 machine
# epsilon is ~2.2e-16; a threshold of 1e10 still leaves ~6 significant digits of accuracy
# margin, the conventional cutoff for "numerically singular in double precision", not tuned.
COND_THRESHOLD = 1e10


def _spread_bucket(spread_ticks, valid):
    """Spread-in-ticks -> bucket index in [0, N_SPREAD): {1 tick, 2 ticks, >=3 ticks}."""
    with np.errstate(invalid="ignore"):
        filled = np.where(valid, spread_ticks, 1.0)  # dummy for invalid rows; masked by caller
        bucket = np.floor(filled).astype(np.int64) - 1  # 1 tick -> 0, 2 ticks -> 1, ...
    return np.clip(bucket, 0, N_SPREAD - 1)


def _imbalance_bucket(bid_sz, ask_sz, valid):
    """Touch imbalance q^b/(q^b+q^a) -> bucket index in [0, N_IMB), equal-width in [0, 1]."""
    bs = np.asarray(bid_sz, dtype=np.float64)
    az = np.asarray(ask_sz, dtype=np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        total = bs + az
        safe_total = np.where(valid & (total > 0), total, 1.0)  # avoid 0/0 on invalid rows
        imb = np.where(valid, bs / safe_total, 0.5)  # dummy for invalid rows; masked by caller
    bucket = np.floor(imb * N_IMB).astype(np.int64)
    return np.clip(bucket, 0, N_IMB - 1)


def _state_index(spread_ticks, bid_sz, ask_sz, valid):
    """(spread bucket, imbalance bucket) -> flat state index in [0, N_STATES); undefined
    (dummy) but harmless where `valid` is False -- callers mask invalid rows separately."""
    sb = _spread_bucket(spread_ticks, valid)
    ib = _imbalance_bucket(bid_sz, ask_sz, valid)
    return sb * N_IMB + ib


def _absorb(Q, M):
    """(I - Q)^-1 M -- the absorbing-chain fundamental-matrix solve, Stoikov (2018) eq. 6/7.

    Q is the transient->transient (no mid move) sub-stochastic matrix, so I - Q is
    non-singular whenever every transient state eventually absorbs. Direct solve when well
    conditioned; the bounded Neumann iteration sum_k Q^k M otherwise (a state whose entire
    sampled Q row is a no-move self-loop makes I - Q numerically singular).
    """
    a = np.eye(N_STATES, dtype=np.float64) - Q
    try:
        cond = np.linalg.cond(a)
        if not np.isfinite(cond) or cond > COND_THRESHOLD:
            raise np.linalg.LinAlgError("ill-conditioned (I - Q)")
        return np.linalg.solve(a, M)
    except np.linalg.LinAlgError:
        acc = np.array(M, dtype=np.float64)
        term = np.array(M, dtype=np.float64)
        for _ in range(MAX_ITERS):  # loop over a frozen constant, not over data
            term = Q @ term
            acc = acc + term
            if np.max(np.abs(term)) < TOL:
                break
        return acc


def _micro_adjustment(Q, R2, RK):
    """Stoikov's iterated microprice adjustment G^inf; Stoikov (2018) sec. 3.

        G^1 = (I - Q)^-1 R K            expected mid move up to the NEXT price change
        B   = (I - Q)^-1 R2             transient state -> state observed AT the price change
        G^(i+1) = G^1 + B G^i           each iteration adds one further price change

    `B` is row-stochastic on visited states (it is a distribution over the state seen at
    absorption), so `I - B` is SINGULAR by construction and G^inf must NOT be obtained by a
    direct solve. Stoikov's argument for the recursion converging is that B^i G^1 -> 0 when
    the mid is a martingale under the fitted chain.

    ON THIS DATA THAT PROPERTY DOES NOT HOLD, measured rather than assumed
    (reports/adversary/phase1-lookahead-normalization-2026-07-31.md, F2): |eig(B)| =
    {1.0, 0.9906, 0.272} and the stationary drift pi . G^1 is -1.80 raw units, i.e. -0.018
    ticks per iteration. So G^i carries a persistent linear term plus a transient decaying
    as 0.99^i, the increments plateau instead of vanishing, and THE LIMIT DOES NOT EXIST.

    `MICRO_ITERS = 6` is therefore not "a cap that convergence usually reaches first" -- it
    IS part of the feature's definition, and G^7 would be a materially different column. It
    is still not tuned: 6 is Stoikov's own repo value, frozen a priori before any correlation
    was looked at. The TOL early exit is retained but effectively never fires here. The
    honest reading of `mp_micro` is "the six-step microprice", not "the microprice".

    Consequence to carry into Phase 2: on AMZN (small-tick, median spread 12 ticks) this
    truncation puts `mp_micro` outside [bid, ask] on 0.32% of finite stamps and stretches
    |mp_dev_mid| to 15.8 ticks (p50 0.39, p99 0.70). INTC is clean at 0.00%. The tail is
    meaningless rather than leaky -- block-refit causality is intact -- but it needs
    winsorising or exclusion, with the choice logged.
    """
    g1 = _absorb(Q, RK)
    B = _absorb(Q, R2)
    g = np.array(g1, dtype=np.float64)
    for _ in range(MICRO_ITERS):  # loop over a frozen constant, not over data
        g_next = g1 + B @ g
        if np.max(np.abs(g_next - g)) < TOL:
            g = g_next
            break
        g = g_next
    return g


def _estimate_block_adjustment(state_event, mid_event, valid_event, fit_end):
    """FITTED per-block adjustment vector (N_STATES,), estimated on transitions strictly
    inside frame rows [0, fit_end) -- i.e. only pairs (k, k+1) with k+1 < fit_end.

    A transition (k, k+1) with mid_event[k+1] == mid_event[k] is TRANSIENT: it feeds the
    Q[s_from, s_to] count. A transition with a nonzero mid change is ABSORBING: it feeds BOTH
    the R*K[s_from] payoff sum (the realised mid change) AND the R2[s_from, s_to] count (the
    state observed AT the price change, which is what lets the chain continue past the first
    move). A state never seen as a transition source (count == 0) gets NaN -- see module
    docstring / INTERFACE.md sec. 8.
    """
    m = fit_end - 1  # number of (k, k+1) transitions available strictly inside [0, fit_end)
    if m <= 0:
        return np.full(N_STATES, np.nan, dtype=np.float64)

    s_from = state_event[:m]
    s_to = state_event[1 : m + 1]
    pair_valid = valid_event[:m] & valid_event[1 : m + 1]
    delta_mid = mid_event[1 : m + 1] - mid_event[:m]

    sf = s_from[pair_valid]
    st = s_to[pair_valid]
    dm = delta_mid[pair_valid]
    moved = dm != 0.0

    counts = np.bincount(sf, minlength=N_STATES).astype(np.float64)
    move_sum = np.bincount(sf[moved], weights=dm[moved], minlength=N_STATES)
    stay_counts = np.bincount(
        sf[~moved] * N_STATES + st[~moved], minlength=N_STATES * N_STATES
    ).astype(np.float64).reshape(N_STATES, N_STATES)
    # The absorbing counterpart of stay_counts: where the mid DID move, which state was
    # observed at the move. Without this the chain can only ever see one price change.
    move_counts = np.bincount(
        sf[moved] * N_STATES + st[moved], minlength=N_STATES * N_STATES
    ).astype(np.float64).reshape(N_STATES, N_STATES)

    visited = counts > 0
    rk = np.zeros(N_STATES, dtype=np.float64)
    rk[visited] = move_sum[visited] / counts[visited]
    q = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    q[visited] = stay_counts[visited] / counts[visited, None]
    r2 = np.zeros((N_STATES, N_STATES), dtype=np.float64)
    r2[visited] = move_counts[visited] / counts[visited, None]

    x = _micro_adjustment(q, r2, rk)
    x = np.array(x, dtype=np.float64)  # defensive copy; never mutate a cached array
    x[~visited] = np.nan
    return x


def compute(frame, stamps, book=None):
    """Stoikov microprice deviations (FEATURES.md sec. 3); see module docstring for the algorithm."""
    if book is None:
        book = book_at_stamps(frame, stamps)

    n = stamps.index.shape[0]

    # Per-event touch chain used ONLY to fit (B, G^1); touch-level cached frame properties
    # per INTERFACE.md sec. 4, not a second book_at_events materialisation.
    valid_event = frame.bid_valid[:, 0] & frame.ask_valid[:, 0]
    spread_ticks_event = ticks(frame.spread)
    state_event = _state_index(spread_ticks_event, frame.best_bid_sz, frame.best_ask_sz, valid_event)
    mid_event = frame.mid

    state_stamp = _state_index(book.spread_ticks, book.bid_sz, book.ask_sz, book.valid)

    micro = np.full(n, np.nan, dtype=np.float64)
    blocks = refit_blocks(frame.ts_ns, stamps.ts_ns)
    # Loop over refit blocks (<=14 on the real frames), never over stamps.
    for block, fit_end in zip(blocks.block_id, blocks.fit_end_index):
        if fit_end < MIN_FIT_ROWS:  # block 0 (fit_end == 0) always lands here -> stays NaN
            continue
        adjustment = _estimate_block_adjustment(state_event, mid_event, valid_event, fit_end)
        sel = blocks.stamp_block_id == block
        # NaN propagates automatically: book.mid is NaN wherever ~book.valid, and adjustment
        # is NaN at an unvisited state, so neither case needs an explicit np.where guard.
        micro[sel] = book.mid[sel] + adjustment[state_stamp[sel]]

    # Deviations are differenced in RAW units first (exact integer arithmetic on a ~2.7e5
    # price), then converted once; `micro` itself is a price LEVEL and so is emitted in ticks
    # per INTERFACE.md sec. 1, which is normative over this module's docstring.
    dev_mid = ticks(micro - book.mid)
    dev_wmid = ticks(micro - book.wmid)
    wmid_dev_mid = ticks(book.wmid - book.mid)

    return {
        "mp_micro": ticks(micro),
        "mp_dev_mid": dev_mid,
        "mp_dev_wmid": dev_wmid,
        "mp_wmid_dev_mid": wmid_dev_mid,
    }
