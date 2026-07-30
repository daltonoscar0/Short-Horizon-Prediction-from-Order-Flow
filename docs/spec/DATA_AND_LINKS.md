# DATA AND LINKS — Edge

## Data

**LOBSTER** — https://lobsterdata.com
- Data structure spec (read this before writing the parser tests):
  https://data.lobsterdata.com/info/DataStructure.php
- System paper (Huang & Polak, 2011), *LOBSTER: Limit Order Book Reconstruction System*:
  https://data.lobsterdata.com/info/docs/LobsterReport.pdf
- Free academic sample: **AAPL, AMZN, GOOG, INTC, MSFT for 2012-06-21**, 09:30–16:00, levels
  1–10, as message + orderbook CSV pairs. Roughly 300k–600k events per name.
- Message file: `(time, type, order_id, size, price, direction)`, time in seconds after
  midnight; price in units of 0.01 cents (divide by 10,000 for dollars). Orderbook file:
  4·NumLevel columns. **Row *k* of the orderbook is the state after message *k*.**
- Event types: 1 new limit order, 2 partial cancel, 3 full delete, 4 execution of a visible
  limit order, 5 execution of a hidden order, 7 trading halt.

**Universe choice.** Use INTC (large-tick: spread pinned at one tick, queue dynamics
dominate) and AMZN (small-tick in 2012: wide spreads in ticks, price-level dynamics
dominate). That contrast is the designed experiment — queue imbalance should carry the
large-tick name and OFI/microprice the small-tick one, and if your results don't show that
pattern you have a bug or a genuinely interesting finding.

**Data honesty note for the README.** The free sample is a single day from 2012. One day
gives you very few independent folds, so intraday blocks will carry the walk-forward and
you must say so. If Swarthmore's library or the CIL budget can get a LOBSTER academic
subscription for even a month of one ticker, that upgrade is worth more to this project
than any modelling improvement — worth an email to Dakota about the budget.

**Fallback / supplementary sources** if you want more days without a subscription:
- Nasdaq ITCH raw daily samples (free historical samples from Nasdaq's FTP) — you already
  have parsing machinery from orderflow-lm.
- Databento (paid, but has free trial credits and clean MBO data with modern dates).
- FI-2010 (Ntakaris et al.) — use only as a *comparison to the literature*, never as your
  primary evidence; see the Briola and Prata critiques in the reading list.

## Reference code (read, don't import)

- Stoikov's microprice implementation and sample data: https://github.com/sstoikov/microprice
- Purged CV reference implementation: https://github.com/eslazarev/purged-cross-validation
  (implements purging, embargo, CPCV with backtest-path reconstruction, deflated Sharpe —
  read it to check your own harness, then write yours from the book)
- TLOB / MLPLOB: https://github.com/LeonardoBerti00/TLOB (for the MLP-matches-SoTA result)

## Your own material to upload into the Project

- The orderflow-lm **LOBSTER parser** source, and its output schema.
- The order-flow LM's **tokenizer spec** and **training config** — plus, critically, the
  list of dates/tickers it was trained on, so the Phase-3 leakage check is possible.
- The trained model's checkpoint metadata (layers, hidden dim, context length) — Claude
  needs the shapes to write the extraction code without guessing.
- `PLAN.md`, `EVAL_PROTOCOL.md`, `FEATURES.md`, `READING_LIST.md`, `INTERVIEW.md` (these
  files) as Project knowledge.
- Your garden-path paper's probing code, if the LM-state extraction can reuse it — the
  methodological continuity between "probing a language model's hidden states" and "probing
  a market model's hidden states" is the whole story of your candidacy, and reusing the
  actual code makes that concrete rather than rhetorical.

## Firms this is aimed at

Jane Street (ML Researcher, QR), HRT, Citadel, D. E. Shaw, Five Rings, SIG, Two Sigma,
Tower, Point72/Cubist. Keep the repo public and linkable; put the headline table and the
break-even-cost figure above the fold in the README, because that's all most screeners read.
