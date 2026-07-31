"""Feature library over the event-indexed frame (docs/DATA_CONTRACT.md); see INTERFACE.md.

Shared constants only -- no computation lives here. Every family module (ofi, queue,
microprice, trade, state, labels) imports from this module instead of re-declaring
magic numbers, so a universe-level fact (tick size, session bounds, the standard
lookback ladder, the refit schedule) is defined exactly once.

Every constant below was re-verified against docs/DATA_CONTRACT.md and the two built
frames on 2026-07-30; the verification commands are recorded in INTERFACE.md.
"""

# Universe fact, not a per-ticker choice: LOBSTER prices are integers in units of
# 0.01 cents (docs/DATA_CONTRACT.md "Price scaling"), so $0.01 = 100 raw units, for
# both INTC and AMZN in this sample. Verified: gcd over all 10 levels of both frames'
# bid and ask prices is exactly 100, and gcd over the best-level spread is 100.
TICK_SIZE_RAW = 100

# Raw price units per dollar (docs/DATA_CONTRACT.md: dollars = price / 10000). Frames
# store raw integers; the conversion is display-time only and is never baked in.
PRICE_SCALE = 10_000

# Free-sample depth for this universe (docs/DATA_CONTRACT.md); multi-level features
# go to L = NUM_LEVELS per FEATURES.md sec. 1.
NUM_LEVELS = 10

NS_PER_MS = 1_000_000
NS_PER_SEC = 1_000_000_000

# Session bounds encoded in every LOBSTER filename for this universe (`_34200000_
# 57600000_`, milliseconds after midnight = 09:30:00 - 16:00:00). Both clocks anchor
# to these; not derived from the data so a partial/gapped day doesn't silently shrink
# the grid. Verified: both frames' first/last ts_ns fall strictly inside these bounds.
SESSION_START_NS = 34_200 * NS_PER_SEC  # 09:30:00.000000000
SESSION_END_NS = 57_600 * NS_PER_SEC  # 16:00:00.000000000

# The lookback/horizon ladder FEATURES.md names explicitly (OFI variants sec. 1,
# label horizons sec. 7): shared so every family module spells "1s" the same way in
# both an ns constant and a column-name suffix.
WINDOWS_NS = {
    "100ms": 100 * NS_PER_MS,
    "1s": 1 * NS_PER_SEC,
    "2s": 2 * NS_PER_SEC,
    "5s": 5 * NS_PER_SEC,
    "10s": 10 * NS_PER_SEC,
}

# Feature lookbacks (FEATURES.md sec. 1: "OFI over multiple lookbacks (100ms, 1s, 5s)").
# Ordered; the column-name suffix is the key itself.
FEATURE_WINDOWS = ("100ms", "1s", "5s")

# Label horizons (FEATURES.md sec. 7). Ordered.
LABEL_HORIZONS = ("1s", "2s", "5s", "10s")

# Event-time label horizons, in stamps of the event clock (FEATURES.md sec. 7's
# "event-time analogues"). Suffix is "_e{N}".
LABEL_HORIZONS_EVENTS = (1, 2, 5, 10)

# Time-of-day bucket width for the seasonality control (FEATURES.md sec. 5). 30 min
# gives 13 buckets over 09:30-16:00, matching reports/data_quality.md's intraday
# profile so the two tables are directly comparable.
TOD_BUCKET_NS = 30 * 60 * NS_PER_SEC

# Refit schedule for any statistic that has to be *fitted* (PCA loadings, Markov
# transition matrices, depth normalisers). Refit once per block, on data strictly
# before the block start; hold the fit constant inside the block. See INTERFACE.md
# "Refit on a schedule" and windows.py's refit_* helpers.
REFIT_BLOCK_NS = 30 * 60 * NS_PER_SEC

# Reserved output-column prefix, one per family module. Families are written in
# parallel and their dicts are merged into one table, so a name collision would
# silently overwrite a column; owning a prefix each makes that impossible. No prefix
# is a prefix of another (asserted in tests/test_interface.py).
FAMILY_PREFIXES = {
    "ofi": "ofi_",
    "queue": "qi_",
    "microprice": "mp_",
    "trade": "tr_",
    "state": "st_",
    "labels": "label_",
}

# The only prefix permitted to look forward in time (INTERFACE.md "Causality"). The
# regression/classification split lives INSIDE it -- label_y_* and label_c_* -- so a
# single grep for "label_" finds every forward-looking column with no exceptions.
LABEL_PREFIX = "label_"
LABEL_REGRESSION_PREFIX = "label_y_"
LABEL_CLASSIFICATION_PREFIX = "label_c_"

# Suffix appended to a seasonality-adjusted twin of a raw column (FEATURES.md sec. 5:
# "produce every result twice -- raw and seasonality-adjusted").
SEASONAL_ADJ_SUFFIX = "_sadj"
