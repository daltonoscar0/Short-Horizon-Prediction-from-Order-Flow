"""Phase 3: latency, fills, transaction costs, PnL, and the break-even cost table.

EVAL_PROTOCOL.md sec. 3: "State every assumption in the README; a stated crude assumption is
respectable, an unstated generous one is not." The assumptions are therefore constants here,
named and justified, rather than numbers buried in a function.
"""

NS_PER_MS = 1_000_000
NS_PER_SEC = 1_000_000_000

# Latency ladder (EVAL_PROTOCOL.md sec. 3): a signal computed from information up to t can
# only act at t + delta. The delta-sensitivity curve is the diagnostic -- "signals that die
# by 100ms are latency-arbitrage, not alpha".
LATENCIES_NS = (1 * NS_PER_MS, 10 * NS_PER_MS, 100 * NS_PER_MS)

# --- Fees -------------------------------------------------------------------------------
#
# ASSUMPTION, FLAGGED. EVAL_PROTOCOL.md sec. 3 asks for "a real Nasdaq take/make schedule
# from the sample period and cite it". These are the commonly-quoted 2012 Nasdaq standard
# rates for tape-C/A securities -- $0.0030/share to remove liquidity, $0.0020/share rebate to
# add it. They are entered here as an ASSUMPTION and have NOT been checked against an
# archived copy of the Nasdaq price list; the exact tier depends on a member firm's monthly
# volume, which no public dataset pins down for an anonymous participant. Anyone reproducing
# this should substitute their own schedule. The break-even cost table is the reason this is
# survivable: it reports the cost at which PnL crosses zero, so a reader who disagrees with
# these two numbers can read their own answer straight off it.
TAKE_FEE_PER_SHARE = 0.0030
MAKE_REBATE_PER_SHARE = 0.0020

# Position constraints (EVAL_PROTOCOL.md sec. 3): "single unit, no inventory accumulation,
# flat at horizon. Keep it boring -- the point is the signal, not a trading system, and
# complexity here only creates places for optimistic bugs."
TRADE_SIZE_SHARES = 100

# NO MARKET IMPACT MODEL. Stated rather than omitted. Ignoring impact biases PnL UPWARD; the
# break-even cost table partially absorbs it, because a strategy that only survives at an
# implausibly low cost per trade is already reported as such.
MARKET_IMPACT_MODEL = None
