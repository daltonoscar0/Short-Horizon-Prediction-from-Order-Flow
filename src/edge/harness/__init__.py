"""Hand-written three-protocol cross-validation harness. THIS PACKAGE IS THE CONTRIBUTION.

CLAUDE.md hard rule 1: no `sklearn.model_selection` anywhere, and `sklearn` is not installed
so the rule is enforced by the environment rather than by discipline. Every splitter, metric
and linear fit here is written on numpy. `lightgbm` is the one sanctioned GBM library (L4).

The three protocols, and what the difference between them measures:

  A  shuffled k-fold          -- what most of the short-horizon LOB literature does
  B  walk-forward, no purge, no embargo
  C  purged walk-forward with embargo  -- correct

  A -> C is TOTAL optimism.
  B -> C isolates the LABEL-OVERLAP component (B is already strictly forward in time, so
        whatever it gains over C comes from labels straddling the fold boundary).
  A -> B is then the residual: training on the future outright.

Frozen constants live here so no module re-declares one and no protocol is silently run at a
different fold count than the protocol it is being compared against.
"""

NS_PER_SEC = 1_000_000_000

# Fold count, shared by ALL THREE protocols. Shared deliberately: if A ran at k=10 while C
# ran at 5 folds, part of the A->C gap would be a difference in training-set size rather
# than in how the data was split, and the headline decomposition would be confounded.
# 5 is the conventional default, frozen a priori, never tuned against a fold result.
N_FOLDS = 5

# Protocol A's k. Equal to N_FOLDS for the reason above; named separately because CLAUDE.md
# calls it out as a frozen choice that needs its own RESULTS.md row.
K_SHUFFLED = N_FOLDS


def default_embargo_ns(horizon_ns):
    """EVAL_PROTOCOL.md sec. 1: E = max(60s, 10*h), reported with a sensitivity table."""
    return max(60 * NS_PER_SEC, 10 * int(horizon_ns))


# Embargo values swept for the required sensitivity table. The frozen headline is
# default_embargo_ns(h); the rest exist to show the result is not an artifact of that pick.
EMBARGO_SWEEP_NS = (0, 1 * NS_PER_SEC, 10 * NS_PER_SEC, 60 * NS_PER_SEC, 300 * NS_PER_SEC)

PROTOCOLS = ("A", "B", "C")
