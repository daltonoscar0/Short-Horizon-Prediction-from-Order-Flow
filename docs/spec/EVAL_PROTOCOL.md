# EVAL_PROTOCOL — Edge

This file is the project. The model is a detail; this is the contribution.

---

## 1. Splitting: purged walk-forward

**Why the standard thing fails.** A label at time *t* with horizon *h* depends on the
interval [t, t+h]. Adjacent samples share almost all of that interval, so labels overlap and
neighbouring samples are near-duplicates. Order flow is additionally long-memory
autocorrelated (Lillo–Farmer). Shuffled k-fold on this data is not "slightly optimistic" —
it is meaningless, and a QR will spot it in the first thirty seconds of reading your repo.

**The scheme.**

```
folds: expanding (or rolling) walk-forward, split on day boundaries first, then
       intraday blocks within a day if you need more folds
train:  [0, T_k)
purge:  drop every training sample whose label window [t, t+h] overlaps the test window
embargo: drop training samples in (T_k + test_len, T_k + test_len + E) for the
         *next* fold, with E ≥ the autocorrelation timescale of the features
test:   [T_k, T_k + test_len)
```

Concretely: purge = h (the label horizon) at both edges of every test block; embargo E =
max(60s, 10·h) as a default, and report sensitivity to E in one table. Overnight gaps are
natural embargoes — don't count on that as an excuse to skip intraday embargoes.

**Rules.**
- Never pool folds into one number and report only that. Report the **distribution across
  folds and the worst fold.** A mean IC of 0.03 built from folds of {0.14, 0.02, −0.01,
  0.00, 0.00} is a story about one day, not a signal.
- Any choice made by looking at a fold (layer, pooling, α, lookback) makes that fold
  training data. Keep a permanent dev slice, chosen once, never scored in the headline.
- If you add combinatorial purged CV (López de Prado ch. 12) for backtest-path
  distributions, keep plain walk-forward as the primary — it's the one every interviewer
  knows, and CPCV becomes a nice "and I also did" rather than something you have to defend.

## 2. Metrics

**Statistical**
- out-of-sample R², computed **against a zero forecast**, not against the test-set mean
  (using the test mean is a subtle leak and the most common one in this literature)
- information coefficient: Spearman ρ between prediction and realized move, per fold
- for classification: balanced accuracy, MCC, AUC — always printed next to the base rate
- calibration on the regression: slope of realized on predicted (should be ≈1; if it's 0.2
  your model is confidently wrong at scale)

**Economic**
- simulated PnL after costs, per fold
- **break-even cost**: the per-trade cost at which PnL crosses zero. Report it in ticks and
  in bps. This is your headline number. Nobody can argue with it, it makes the honest case
  and the impressive case at the same time, and it is far more informative than a Sharpe.
- turnover and holding time, so the cost number has context

**Inferential**
- Diebold–Mariano between adjacent ladder rungs (use the conditional Giacomini–White variant
  where models are nested, which they are for L3 vs L5)
- block bootstrap CIs on metric *differences*, block length ≥ the label horizon
- **deflated Sharpe ratio** using the count of configurations you actually tried — which is
  why the trial log in `RESULTS.md` is mandatory and why you log the failures too

## 3. Cost and fill model

State every assumption in the README; a stated crude assumption is respectable, an unstated
generous one is not.

- **Latency**: the signal computed from information up to *t* can only act at *t + Δ*. Run
  Δ ∈ {1ms, 10ms, 100ms}. The Δ-sensitivity curve is one of the most revealing plots in the
  project — signals that die by 100ms are latency-arbitrage, not alpha.
- **Aggressive fills**: cross the spread — pay the half-spread, plus fee. Fill at the
  displayed price only up to displayed size; walk the book beyond it.
- **Passive fills**: the honest simple rule is *fill only if the level trades through your
  price*, which ignores queue position and is therefore pessimistic. If you want queue
  position, model it explicitly (your position = depth ahead at insertion, decremented by
  executions and cancels ahead) and say what you assumed about cancellation priority.
- **Fees**: use a real Nasdaq take/make schedule from the sample period and cite it.
- **No market impact model.** Say so. Note that ignoring impact biases PnL *upward* and that
  the break-even cost table partially absorbs this.
- **Position constraints**: single unit, no inventory accumulation, flat at horizon. Keep it
  boring — the point is the signal, not a trading system, and complexity here only creates
  places for optimistic bugs.

## 4. Leakage checklist — run before reporting any improvement

- [ ] Row alignment: orderbook row *k* is the state **after** message *k*. Verify with the
      known LOBSTER example (a deletion of 100 shares reduces the displayed size at that
      level from line k−1 to line k).
- [ ] Feature at *t* uses only data with timestamp **< t** (or ≤ t with an explicit,
      documented convention — pick one and test it).
- [ ] Normalization statistics come from a trailing window or from training folds only.
      No `fit` on the full array anywhere. Grep for it.
- [ ] Labels do not straddle fold boundaries (test: assert every training label window is
      disjoint from every test label window).
- [ ] No shuffling of time-ordered data at any point in the pipeline.
- [ ] The LM was trained on data that does not include the test folds. If it was — and this
      is the most likely serious leak in the whole project, given the model already exists —
      retrain on a restricted window or restrict test folds to genuinely unseen days.
      **Check this on day one of Phase 3, not after you get a good number.**
- [ ] No target-derived quantity in the features (e.g. a realized-vol feature whose window
      extends past *t*).

## 5. Deliberate leak-detector tests (write these as unit tests)

1. **Label shuffle**: permute labels within a fold; every metric should collapse to chance.
   If it doesn't, you have a leak or a bug in the metric.
2. **Time-shift probe**: shift features forward by one event so they legitimately see the
   future. Performance should jump sharply. If it doesn't move, your features aren't
   aligned to what you think they are.
3. **Purge-off comparison**: run once with purging disabled and report both numbers in the
   writeup. The gap *is* a result — it quantifies how much of the published literature's
   optimism could be an artifact, and it's the single best paragraph you can put in the
   README for the audience you're targeting.
4. **Same-day vs next-day**: train and test within one day, then across days. A big drop is
   expected and worth reporting; no drop is suspicious.
5. **Random-feature control**: swap the LM states for Gaussian noise of the same dimension.
   Any "improvement" that survives this is dimensionality, not information.

## 6. What the writeup must contain

- The ladder table with per-fold distributions.
- The break-even cost table.
- The alpha-decay curve (incremental value vs horizon).
- The three Phase-3 controls, reported whether or not they're flattering.
- A **"what would change my mind"** section: more days, more names, a different tick regime,
  a larger LM, event-time labels — say specifically what result would flip your conclusion.
- A limitations section that names the free-sample data vintage (2012) as a limitation
  before anyone else does.
