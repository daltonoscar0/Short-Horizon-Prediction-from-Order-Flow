# FEATURES, Edge

Notation: at event *n*, best bid price/size $(P^b_n, q^b_n)$, best ask $(P^a_n, q^a_n)$;
level *m* quantities superscripted $(m)$; tick size δ.

Every feature is a pure function of information **strictly at or before** its timestamp.
Every one gets a unit test on a hand-built toy book.

---

## 1. Order flow imbalance (the centrepiece)

**Best-level OFI (Cont-Kukanov-Stoikov).** Per-event contribution:

```
e_n = 1{P^b_n ≥ P^b_{n-1}}·q^b_n  −  1{P^b_n ≤ P^b_{n-1}}·q^b_{n-1}
    − 1{P^a_n ≤ P^a_{n-1}}·q^a_n  +  1{P^a_n ≥ P^a_{n-1}}·q^a_{n-1}
```

`OFI[t_{k-1}, t_k] = Σ e_n` over events in the interval. Note this counts limit orders,
cancellations, and market orders in one number, that unification is the reason it works.

Test the implementation against the three cases by hand: a bid-side limit order add
(+size), a bid-side cancel (−size), an aggressive sell that clears the bid (price
improvement branch). If your OFI doesn't reproduce a near-linear relation with Δmid over
1s windows, something is wrong with the parser, not the market.

**Multi-level OFI.** Same construction per level *m* up to L=10 (Xu-Gould-Howison). Scale
each by average depth at that level so the levels are comparable.

**Integrated OFI.** First principal component of the L multi-level OFIs, fit on a rolling
past window only (Cont-Cucuringu-Zhang). One number; usually the best single feature.

**Variants worth one row each:** signed-volume OFI (weight by size) vs count OFI; OFI over
multiple lookbacks (100ms, 1s, 5s) since order flow is long-memory autocorrelated.

## 2. Queue / book imbalance

```
I = (q^b − q^a) / (q^b + q^a)                    # touch imbalance, ∈ [−1, 1]
I_L = (Σ_m w_m q^b(m) − Σ_m w_m q^a(m)) / (…)   # depth-weighted, w_m = 1/m or exp decay
```

Include `sign(I)` separately, that unfitted sign is baseline L1 and it is a stiffer
opponent than it looks, especially in large-tick names.

## 3. Microprice and price references

```
mid      = (P^b + P^a)/2
wmid     = (P^b·q^a + P^a·q^b)/(q^a + q^b)          # weighted mid (note the crossed weights)
micro    = Stoikov (2018) Markov estimate on (spread bucket, imbalance bucket)
dev_mid  = micro − mid                              # the feature
dev_wmid = micro − wmid                             # how much the Markov correction adds
```

Implement the microprice properly (the iterative absorbing/transient state computation from
Stoikov's repo), not as a synonym for weighted mid. The distinction is a thing an
interviewer can and will poke at.

## 4. Trade / aggressor features

LOBSTER gives the direction explicitly, no Lee-Ready classification needed, and say so,
because it's a detail that shows you read the data spec.

- signed trade volume over lookback; trade count; trade-to-quote-event ratio
- aggressor imbalance = (buy volume − sell volume)/(total volume)
- effective spread of the last trade: `2·|P_trade − mid_before|`
- optional: VPIN-style bucketed toxicity (Easley-López de Prado-O'Hara)

## 5. State / control features

- spread in ticks (integer; regime indicator for large-tick names)
- total depth at touch and to L levels; book slope (regression of cumulative size on
  distance from mid)
- realized volatility over trailing windows (sum of squared mid returns)
- event intensity: events/second, cancel rate, average order size
- **time of day**: intraday seasonality is real and large. Include a bucketed control, and
  produce every result twice, raw and seasonality-adjusted (per-bucket z-score using
  *past days only*). Cont-Kukanov-Stoikov specifically checked their result was robust to
  intraday seasonality; yours should too.

## 6. LM representations (Phase 3)

- hidden states from the order-flow LM at the decision timestamp; document layer and
  pooling and freeze them before touching the test folds
- controls that must be built at the same time, not later: untrained-weights version,
  lagged-raw-feature version over the identical window, random-projection version matched
  on dimension

## 7. Labels

```
y_h  = mid(t+h) − mid(t)              in ticks       # primary
y'_h = micro(t+h) − micro(t)          in ticks       # lower-noise alternative
c_h  = sign(y_h) with deadband α: up if y_h > α, down if < −α, else flat
```

Horizons: 1s, 2s, 5s, 10s in calendar time; and the event-time analogues (N events, and
"k average price changes" following Kolm et al.).

Two warnings. **Deadband choice is a free parameter that can manufacture accuracy**, fix α
by a rule (e.g. α = half a tick, or the median absolute move at that horizon on training
data only), report sensitivity, never tune it against test folds. And **class balance is
horizon-dependent**: at 1s most labels are flat, so raw accuracy is meaningless, report
balanced accuracy / MCC / AUC against the base rate.

## 8. Clocks

Compute everything on both:
1. **Calendar time**, 100ms grid, features as of the last event at or before each stamp.
2. **Event time**, every N book-changing events.

They can disagree substantially, and the disagreement is itself a finding worth a paragraph:
calendar-time results on a quiet instrument are partly measuring the clock rather than the
market.
