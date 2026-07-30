# INTERVIEW — Edge

The project's second output. Keep this file updated *as results come in*, not at the end.

## The 90-second version

> I wanted to know whether learned representations actually add anything over classic
> microstructure features at 1-to-10-second horizons, so I built the baselines properly
> first — order flow imbalance, queue imbalance, microprice deviation — and then added
> hidden states from a transformer I'd trained on LOBSTER message streams. The interesting
> part was the evaluation: labels overlap at those horizons, so I used purged walk-forward
> splits with an embargo, and I report break-even transaction cost rather than a Sharpe.
> [Result sentence.] The gap between the purged and unpurged numbers was [X], which is
> roughly the size of the improvement a lot of published results claim.

Then stop talking. The follow-up questions are the interview.

## Vocabulary you must use correctly and unprompted

order flow imbalance · queue imbalance · microprice · weighted mid · adverse selection ·
flow toxicity · effective spread · realized spread · price impact (temporary vs permanent) ·
queue position · large-tick vs small-tick regime · alpha decay · information coefficient ·
turnover · capacity · event time vs calendar time vs volume time · purging · embargo ·
label overlap · deflated Sharpe · base rate · adverse selection cost of passive fills ·
intraday seasonality · long memory of order flow

## Questions you will get, with the shape of the answer

**"Tell me about a prediction problem you've worked on."** → the 90-second version.

**"Why 1–10 seconds?"** → Long enough that latency isn't the whole story, short enough that
microstructure dominates over drift and news. Also where the published order-flow results
live, so there's a comparison point. And name the constraint: at 100ms your result would be
about infrastructure you don't have.

**"How do you know you're not leaking?"** → The strongest answer in your arsenal. Purging
and embargo, the label-shuffle test, the time-shift probe, the purge-on/purge-off comparison
that quantifies the leak's size. Say that you ran the unpurged version deliberately to
measure how much optimism it buys.

**"What's your Sharpe?"** → Redirect, don't dodge: a Sharpe from one day of 2012 data on one
name is not a number worth quoting, so you report break-even cost instead — at what cost per
trade does the edge vanish. Then give the number. This answer alone separates you from the
overfit-Sharpe pile.

**"Why do you think the LM didn't help?"** (assume this is the result) → Candidates, ranked:
the handcrafted features already encode the state variables the LM would have to learn (OFI
is a sufficient statistic for a lot of it); the LM was trained for next-token prediction,
not return prediction, and the objectives are only weakly aligned; sample size and one-day
data; the untrained-weights control tells you how much was learned versus projected. Then
say what you'd run next with more data.

**"Why do you think the LM *did* help?"** → Be more suspicious than they are. Walk through
the three controls before they ask. If you can't rule out the random-projection explanation,
say so.

**"What would you do with a year of data?"** → Cross-sectional pooling (Sirignano & Cont's
universality result: one model across names can beat per-name models), regime conditioning,
cross-asset transfer, and the deflated Sharpe with a real trial count.

**"What's OFI, exactly?"** → Give the formula, not the intuition. Then the intuition: it
unifies limit orders, cancels, and market orders into one signed quantity, and its slope
against price change is inversely proportional to depth, which is why you normalize by depth.

**"Where does this break?"** → Large-tick vs small-tick divergence; the 100ms latency curve;
opening and closing auctions; the fact that you have no impact model.

## What *not* to do

- Don't claim a Sharpe. Don't claim "alpha."
- Don't present the pooled metric without the fold distribution — if they ask for the worst
  fold and you don't have it, the conversation is over.
- Don't oversell the LM connection. The honest framing — "I transferred surprisal and
  probing methods from my psycholinguistics work to market data, and here's what did and
  didn't transfer" — is more distinctive than any performance claim, and it's the reason
  a computational linguistics background is an asset rather than something to explain away.

## Resume lines

Primary:
> Benchmarked transformer representations against microstructure baselines for short-horizon
> price prediction under purged walk-forward evaluation with transaction-cost-aware PnL.

Variant emphasizing the harness (for QR roles):
> Built a leakage-resistant evaluation harness (purged walk-forward splits, embargo,
> deflated Sharpe, break-even cost analysis) for 1–10s limit order book prediction;
> quantified the optimism introduced by unpurged cross-validation.

Variant emphasizing the transfer (for ML researcher roles):
> Applied surprisal and probing methods from language model interpretability to limit order
> book message streams; tested whether learned representations add incremental value over
> order flow imbalance and microprice baselines.
