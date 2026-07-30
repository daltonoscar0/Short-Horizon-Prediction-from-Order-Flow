# READING LIST — Edge

Ordered by what you need first. Where I give an arXiv ID I've verified it; where I don't,
the title + authors will one-shot on Google Scholar. A wrong ID is worse than no ID.

---

## Tier 1 — read before writing any feature code (the five that matter)

**1. Cont, Kukanov & Stoikov (2014), "The Price Impact of Order Book Events."**
*Journal of Financial Econometrics* 12(1):47–88. Preprint: https://arxiv.org/abs/1011.6402
The origin of OFI. Over short intervals price changes are driven mainly by the imbalance
between supply and demand at the best quotes, and the relation is linear with a slope
inversely proportional to depth. Your L2 baseline is literally this regression — reproduce
it before you do anything else. Also the source of the framing of OFI as an
**adverse-selection** measure, which is the vocabulary you want in interviews.

**2. Stoikov (2018), "The micro-price: a high-frequency estimator of future prices."**
*Quantitative Finance* 18(12):1959–1966. SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2970694
Code + sample data: https://github.com/sstoikov/microprice
The micro-price is a martingale by construction — the "fair" price conditional on book
information — expressible as a mid-price adjustment for spread and imbalance, and
empirically a better short-horizon predictor than mid or weighted-mid. Two uses for you:
as a feature (microprice − mid) and as an alternative *label* (Δmicroprice is less noisy
than Δmid, and saying so out loud signals you understand the noise structure).

**3. Kolm, Turiel & Westray (2023), "Deep order flow imbalance: extracting alpha at
multiple horizons from the limit order book."** *Mathematical Finance* 33(4):1044–1081.
SSRN: https://papers.ssrn.com/sol3/papers.cfm?abstract_id=3900141 · DOI: 10.1111/mafi.12413
The closest published thing to your project, and your key prior: simple off-the-shelf
networks on **stationary order-flow inputs** beat models trained on raw order book states.
Two things to steal: the multi-horizon design, and the finding that the effective horizon
of a stock-specific forecast is about two average price changes — a ready-made prediction
for your own alpha-decay curve to confirm or contradict.

**4. López de Prado (2018), *Advances in Financial Machine Learning*, ch. 7 ("Cross-
Validation in Finance") and ch. 11–12.** Wiley.
Purging, embargo, and combinatorial purged CV. Overlapping labels are exactly your
situation: a 10s-horizon label at time t shares information with every sample in
[t−10s, t+10s], so a naive split leaks. Chapters 7 and 11 are the whole methodological
spine of this project — read them properly, not in summary.
Companion: **Bailey & López de Prado (2014), "The Deflated Sharpe Ratio"** and
**Bailey, Borwein, López de Prado & Zhu, "The Probability of Backtest Overfitting."**
Reference implementation to read (not import): https://github.com/eslazarev/purged-cross-validation

**5. Gould & Bonart (2016), "Queue imbalance as a one-tick-ahead price predictor in a
limit order book."** *Market Microstructure and Liquidity*.
Queue imbalance at the touch predicts the next mid-price move, with the effect strongest in
**large-tick** instruments where discrete queue dynamics dominate. This is why your universe
has one large-tick and one small-tick name: it's a designed contrast, not a data grab.

## Tier 2 — feature engineering and the modern skeptical literature

**6. Xu, Gould & Howison (2018), "Multi-level order-flow imbalance in a limit order book."**
*Market Microstructure and Liquidity* 4(03n04):1950011. Extends OFI past the touch.

**7. Cont, Cucuringu & Zhang (2023), "Cross-impact of order flow imbalance in equity
markets."** *Quantitative Finance* 23(10):1373–1393. Source of **integrated OFI** — collapse
multi-level OFIs into one variable (first principal component), which explains price impact
better than best-level OFI alone. Cheap to implement, good ablation row.

**8. Kercheval & Zhang (2015), "Modelling high-frequency limit order book dynamics with
support vector machines."** *Quantitative Finance* 15(8):1315–1329. The canonical
handcrafted LOB feature set (basic / time-insensitive / time-sensitive). Use it as a
checklist so no reviewer can say you left an obvious feature out.

**9. Briola, Bartolucci & Aste (2024), "Deep Limit Order Book Forecasting."**
https://arxiv.org/abs/2403.09267 — a microstructural guide to *when* forecasting is even
possible, plus a direct critique of the FI-2010 benchmark as too simplistic and prone to
overfitting, and of the field's dependence on non-reproducible proprietary data.

**10. Prata et al. (2024), "LOB-based deep learning models for stock price trend
prediction: a benchmark study."** *Artificial Intelligence Review* 57:116.
The finding you will cite in every interview: despite a proliferation of specialized deep
architectures, performance converges to low values once you test across diverse datasets.
This is the literature-level version of your own likely negative result — it makes your
framing look informed rather than defeatist.

**11. Berti & Kasneci (2025), "TLOB: a novel transformer model with dual attention for
price trend prediction with LOB data."** https://arxiv.org/abs/2502.15757
Read it for the *MLPLOB* result, not the transformer: a simple MLP adapted to LOB matches
or beats state-of-the-art, challenging the necessity of complex architectures. Also
proposes a labelling method that removes horizon bias — relevant to your α deadband choice.
Code: https://github.com/LeonardoBerti00/TLOB

**12. Zhong et al. (2025), "Representation Learning of Limit Order Book: A Comprehensive
Study and Benchmarking."** https://arxiv.org/abs/2505.02139
The nearest neighbour to your actual research question: it argues existing work couples
representation learning to downstream tasks end-to-end and never analyses the
representations on their own. Your project analyses them on their own. Cite it as
motivation, and note the difference in data (China A-share vs LOBSTER US equities).

## Tier 3 — background, skim as needed

- **Sirignano (2019), "Deep learning for limit order books,"** *Quantitative Finance*
  19(4):549–570, arXiv:1601.01987 — and **Sirignano & Cont, "Universal features of price
  formation in financial markets: perspectives from deep learning"** (universality: a
  single model across stocks outperforms per-stock models; a good stretch experiment).
- **Zhang, Zohren & Roberts, "DeepLOB: deep convolutional neural networks for limit order
  books,"** *IEEE Trans. Signal Processing* (2019). The architecture everyone benchmarks
  against; you need it as a reference point even if you never run it.
- **Ntakaris, Magris, Kanniainen, Gabbouj & Iosifidis (2018), "Benchmark dataset for
  mid-price forecasting of limit order book data with machine learning methods,"**
  *Journal of Forecasting* 37:852–866. FI-2010 — know what it is and know its criticisms
  (see #9, #10) so you can explain why you used LOBSTER instead.
- **Easley, López de Prado & O'Hara (2012), "Flow toxicity and liquidity in a
  high-frequency world,"** *Review of Financial Studies* 25(5):1457–1493. VPIN and flow
  toxicity — one optional feature, and the concept of adverse selection made quantitative.
- **Bouchaud, Bonart, Donier & Gould (2018), *Trades, Quotes and Prices: Financial Markets
  Under the Microscope*,** CUP. The reference book. Read the chapters on order flow's long
  memory and on impact; skim the rest.
- **Lillo & Farmer (2004), "The long memory of the efficient market."** Order flow is
  long-memory autocorrelated — the single fact that most justifies purging and embargo, and
  a good answer when someone asks *why* your CV scheme needs to be exotic.
- **Huang, Lehalle & Rosenbaum (2015), "Simulating and analyzing order book data: the
  queue-reactive model."** *JASA*. Feeds directly into your Exchange project; here it's the
  structural model your statistical features are approximating.
- **Diebold & Mariano (1995), "Comparing predictive accuracy."** The test you will use to
  claim one rung beats another. Know the caveat about nested models (use the Giacomini–White
  conditional version if the comparison is nested).
- **Hasbrouck, *Empirical Market Microstructure*** (2007) — for the vocabulary: effective
  spread, realized spread, price impact decomposition, the Roll model.
