# RULES — the working discipline for this project

The rules I set before starting and held myself to throughout. They are numbered because the
rest of the repo refers to them by number: `DECISIONS.md`, the red-team reports and several
module docstrings cite "rule 4" or "rule 7" when explaining why something was done a particular
way, or when recording that it was not.

They are here because a study about evaluation protocol is only worth as much as the discipline
around it, and because several of them were violated at some point and the violation is on the
record.

---

**1. Zero-dependency ethos.** stdlib + numpy, plus `pyarrow` for feature storage and one GBM
library. **The cross-validation harness is hand-written** — no `sklearn.model_selection`
anywhere. `sklearn` is not installed, so the rule is enforced by the environment rather than by
discipline. The splitter, the metrics, the ridge solve, the Newey–West HAC, the Spearman rank
transform, the normal quantile function, the block bootstrap and the deflated Sharpe are all
written out. The harness is the contribution; the models are the detail.

**2. Tests alongside code, adversarial ones first.** For the harness specifically: a splitter
that lets a labelled sample straddle a fold boundary must fail a test that exists *before* the
splitter does. `tests/test_splits.py` contains a deliberately broken splitter that the leak
detector is asserted to catch — a leak detector that has never caught a leak is a comment.

**3. Be adversarial about results.** Propose the boring explanation first: an off-by-one in
timestamp alignment, a scaler fitted on the full sample, a base-rate artifact, an accuracy
number that is just class imbalance. Every phase that produced a number got a dedicated
red-team pass, written up in `reports/adversary/` whatever the outcome. Three of those passes
killed a headline this repo had already written down.

**4. No number without** the per-fold distribution, the worst fold, the baseline it beats, and
the running trial count. Every configuration goes in the trial log, **including the failures** —
two discarded harness configurations that invalidated their own runs are in it for that reason.
This rule was missed three times, each time on a newly added metric, and each miss is recorded.

**5. No result is reported on the data used to pick it.** A choice made by looking at a fold
makes that fold training data. Where this was bent — the phase-3 trading deadband, set after
seeing protocol-C output — it is disclosed in the frozen-decisions table rather than presented
as clean.

**6. Scope discipline.** Multi-asset, multi-day and multi-horizon are separate axes. Never
expand two at once.

**7. Determinism.** Every entry point takes `--seed`. Reproducing a table is one command, and
each report echoes in its own header the exact arguments that produced it. Violated once, when
the published command regenerated the ladder *without* the control table its headline numbers
came from; caught by a red-team pass and fixed.

---

## The decision log

`DECISIONS.md` records every choice made without stopping to deliberate: what, why, and the
date. Frozen choices — the deadband α, the embargo E, the fold count for the shuffled protocol —
additionally get a row in the `RESULTS.md` frozen-decisions table with the date they were fixed,
so that a number can never be quietly re-tuned after the fact.
