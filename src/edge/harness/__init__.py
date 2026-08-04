"""Hand-written three-protocol cross-validation harness. THIS PACKAGE IS THE CONTRIBUTION.

RULES.md hard rule 1: no `sklearn.model_selection` anywhere, and `sklearn` is not installed
so the rule is enforced by the environment rather than by discipline. Every splitter, metric
and linear fit here is written on numpy. `lightgbm` is the one sanctioned GBM library (L4).

The three protocols, and what the difference between them measures:

  A  shuffled k-fold          -- what most of the short-horizon LOB literature does
  B  walk-forward, no purge, no embargo
  C  purged walk-forward with embargo  -- correct

  A -> C is TOTAL optimism.
  B -> C is the purge/embargo effect AT A WALK-FORWARD BOUNDARY.
  A -> B is the residual.

CORRECTED 2026-08-02, and the correction matters. This module used to describe B -> C as
"the label-overlap component" and A -> B as "training on the future outright". Both labels
were wrong, and the phase-2 red team proved it with protocol D (below):

  - Training on the future OUTRIGHT, in blocks disjoint from the test set, is worth ~0.01-0.05
    in IC (that is D -> C).
  - Rows temporally ADJACENT to the test rows -- i.e. label overlap -- are worth 0.11-0.36
    (that is A -> D), and they sit INSIDE the A -> B column.
  - B -> C is near zero for an arithmetic reason, not a substantive one: under expanding
    walk-forward, purge+embargo can only ever remove the last h+E of a multi-hour training
    set, which is 0.6-1.1% of the training rows. A column that can touch 1% of the training
    set is not a measurement of label overlap in general.

So a near-zero B -> C does NOT license "label overlap contributes nothing". The opposite is
true; it is just not measurable in that column under this fold geometry.

Frozen constants live here so no module re-declares one and no protocol is silently run at a
different fold count than the protocol it is being compared against.
"""

NS_PER_SEC = 1_000_000_000

# Fold count, shared by ALL THREE protocols. Shared deliberately: if A ran at k=10 while C
# ran at 5 folds, part of the A->C gap would be a difference in training-set size rather
# than in how the data was split, and the headline decomposition would be confounded.
# 5 is the conventional default, frozen a priori, never tuned against a fold result.
N_FOLDS = 5

# Protocol A's k. Equal to N_FOLDS for the reason above; named separately because RULES.md
# calls it out as a frozen choice that needs its own RESULTS.md row.
K_SHUFFLED = N_FOLDS


def default_embargo_ns(horizon_ns):
    """EVAL_PROTOCOL.md sec. 1: E = max(60s, 10*h), reported with a sensitivity table."""
    return max(60 * NS_PER_SEC, 10 * int(horizon_ns))


# Embargo values swept for the required sensitivity table. The frozen headline is
# default_embargo_ns(h); the rest exist to show the result is not an artifact of that pick.
EMBARGO_SWEEP_NS = (0, 1 * NS_PER_SEC, 10 * NS_PER_SEC, 60 * NS_PER_SEC, 300 * NS_PER_SEC)

PROTOCOLS = ("A", "B", "C", "D")

# D is a CONTROL, not one of the three protocols under study. It shares C's test blocks but
# trains on everything outside them, past and future, unpurged -- so it isolates "the model
# saw the future" from "the model saw rows adjacent to the test rows". Added 2026-08-02;
# see reports/adversary/phase2-ladder-2026-08-02.md.
HEADLINE_PROTOCOLS = ("A", "B", "C")
