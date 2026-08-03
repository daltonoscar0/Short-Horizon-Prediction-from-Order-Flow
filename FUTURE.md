# FUTURE — what this harness is for next

Ordered by what would change a conclusion, not by what would be fun. Everything here is a
follow-up *against this harness*: the point of building the splitter first is that a new arm
is a new rung, not a new project.

---

## 1. More days, then more names — in that order

**Why first.** Every number in this repo comes from one session, 2012-06-21. `EVAL_PROTOCOL.md`
sec. 1 says to split on day boundaries and fall back to intraday blocks only if more folds are
needed; this project only ever had the fallback. Two consequences that no amount of care inside
the harness can fix:

- The **same-day vs next-day leak check cannot be run at all**. It is recorded as unavailable,
  not as passed. It is the check most likely to find something the intraday checks cannot.
- The A→C optimism gap is measured across five intraday blocks of one day. Whether it is a
  property of *this literature's protocol* or of *this Thursday* is not decidable from here.

**The scope-discipline rule (CLAUDE.md rule 6) says one axis at a time.** Days first: the same
two names, twenty sessions, folds cut on day boundaries. Only then more names.

**What would change my mind:** if the A→C gap on L4 collapses from ~0.49 to near zero once
folds are day-separated, then what this repo measured is intraday burst duplication, not label
overlap in general, and the headline should say so.

## 2. The order-flow LM arm

Cut from this project because no trained checkpoint exists — `orderflow-lm` stops at
tokenization. This was the original motivation and it is the natural next arm.

It slots in as **L5** (handcrafted + LM states) and **L6** (LM states alone), with the three
controls `docs/spec/PLAN.md` already specifies and which are non-negotiable:

- **untrained control** — identical architecture, random weights, same pooling. If L5 ≈
  untrained-L5, the gain is a random projection of recent history.
- **lag control** — raw lagged features over the identical lookback window the LM sees. The LM
  must beat naive memory, not just the contemporaneous snapshot.
- **dimension control** — the handcrafted model given the same number of random-projection
  features, so a win is not parameter count.

**The leak that matters more than the result.** `EVAL_PROTOCOL.md` sec. 4 is explicit: if the
LM was pretrained on data overlapping the test folds, every downstream number is void. Check
that on day one of that phase, not after a good number arrives. A checkpoint trained on
2012-06-21 cannot be evaluated on 2012-06-21 under any protocol in this repo.

## 3. Close the gaps this project logged rather than fixed

These are known, recorded, and small — but they are places a silent regression could land.

- **Per-column literal tests.** Every formula and every mutation the audits could construct is
  covered, but roughly 35 `ofi_l{m}_{w}` level×window cells, 12 `label_*` columns at the
  2s/5s/10s horizons and 5 `st_*` columns are pinned only structurally. Five `st_*` columns
  survive replacement by zeros, which means no test currently depends on their values.
- ~~**Calibration slope is computed but not reported.**~~ **Done 2026-08-03**, together with
  AUC — and reporting them is what exposed the two errors described in `FINDINGS.md` §1.1a: the
  artifact floor was frozen against the calibration channel, and about half of L4's flagship gap
  is that channel rather than the split. Both are now measured. A first reading of the slope as a
  leakage signature was published and withdrawn the same day; it is mechanically forced by the
  protocol-dependent holdout rule.

  What remains open here is narrower. **The slope is unstable per cell** (L3 ranges
  [−4.74, 43.20] because its denominator is the prediction's variance), so only the medians are
  quotable and a trimmed or variance-weighted aggregation would make the per-cell column
  readable. And **the AUC floor is small on this data for reasons that will not transfer.**
  Protocol A's per-fold AUC is a whole-day pooled AUC while C's is a within-block AUC, so any
  frozen score with a time-varying offset drives the two apart with nothing refitted — the red
  team reached −0.137 that way. Separately, the floor depends on where the frozen model is
  fitted, and at L4's magnitude a whole-day fit sits at an AUC ceiling (~0.996) where two
  protocols cannot differ; the floor is now maximised over four fit windows for that reason.
  Both must be re-measured on any new data rather than assumed. See §7.
- **Giacomini–White for nested rungs.** The DM tests between adjacent rungs are reported and
  flagged as indicative: adjacent rungs are largely nested (L3's feature set contains L2's) and
  standard DM is undersized there. The conditional GW variant is the correct test and is not
  implemented.
- **The passive fill model is dead code in the headline path.** `walk_the_book` and
  `passive_round_trip_pnl` are unit-tested but never called: `simulate` prices every fill at the
  touch for 100 shares. The module docstring's "walk the book beyond displayed size" claim is
  therefore untested against real data, and whether 100 shares exceeds displayed size at the
  touch on AMZN was never checked.
- **The fee schedule is an unverified assumption.** $0.0030/$0.0020 is the commonly-quoted 2012
  Nasdaq standard rate, entered without an archived price list. The gross-based break-even
  column exists so a reader can substitute their own number; an actual citation would be
  better.

## 4. Combinatorial purged CV

López de Prado ch. 12. Gives a *distribution* of backtest paths rather than one walk-forward
path, which is the right way to ask whether the C-protocol ladder ordering is stable or is one
draw. Keep plain walk-forward as the primary — it is the protocol every reader knows — and add
CPCV as the "and I also did", never as the headline.

## 5. Regime splits, once there is enough data to split

High-vol vs low-vol hours; open/close vs midday. Both are one-axis expansions and both are
currently impossible: five intraday folds on one day cannot be subdivided further without
folds too small to score. This is a consequence of item 1, not an independent task.

## 6. The question this repo cannot answer

**Does any of this survive at a horizon where the cost model is not the whole story?** The
phase-1 adversary measured that OFI's content decays inside 100 ms, and the cost model shows
that even perfect foresight of the 1-second mid move loses money once the spread is crossed.
That combination means the honest ladder here is a statement about *prediction*, not about a
tradable strategy. A version of this project aimed at tradability would need passive fills,
queue position, and a horizon chosen by where the signal actually lives — which is a different
project, and should be labelled one.

## 7. The calibration control this repo could not build

`A' - C` changes the calibration holdout rule, but `Rung.fit` fits the raw model on the
complement of that holdout, so it also changes which rows the model trains on. Every number
attributing a share of `A - C` to "the calibration rule" is therefore confounded with a training
window, and the honest statement is a bracket ([+0.138, +0.522] for L4 on R², [+0.137, +0.190]
on raw AUC) rather than a share.

**The clean control** fits the raw model on the *same* random 80% under both arms and moves only
the rows the through-origin scale is fitted on, accepting that the scale rows then overlap the
model's. It is one branch in `score_fold` and one more protocol column. It would collapse the
bracket to a point and is the single cheapest unbuilt thing in this repo.

Two smaller ones alongside it. The artifact floor is now measured over four frozen-model fit
windows because "whole day" turned out to minimise it; **the set of windows is itself arbitrary**
and a floor maximised over a proper search would be the defensible object. And the frozen model
is a *frozen* model — the rungs refit theirs per fold, so the raw model's own refitting is a
third channel no frozen predictor can bound. Nothing here measures it.
