# RESULTS — Edge

## Headline ladder

Report per fold. Never pool without the distribution.

| date | ticker | clock | horizon | rung | OOS R² (vs zero) | IC (median/worst fold) | bal. acc | break-even cost (ticks) | notes |
|---|---|---|---|---|---|---|---|---|---|
| | | | | L0 constant | | | | | |
| | | | | L1 sign(imbalance) | | | | | |
| | | | | L2 OLS on OFI | | | | | |
| | | | | L3 ridge/logistic, full set | | | | | |
| | | | | L4 GBM, full set | | | | | |
| | | | | L5 full set + LM states | | | | | |
| | | | | L6 LM states only | | | | | |
| | | | | C1 untrained-weights control | | | | | |
| | | | | C2 lagged-raw control | | | | | |
| | | | | C3 random-projection control | | | | | |

## Leakage diagnostics

| test | expected | observed | date |
|---|---|---|---|
| label shuffle | metrics → chance | | |
| time-shift probe (+1 event) | large jump | | |
| purge off vs on | on < off; record the gap | | |
| same-day vs next-day | drop expected | | |

## Trial log (mandatory — feeds the deflated Sharpe)

Every configuration tried, including abandoned ones. If it isn't here, the trial count is
wrong and the deflated Sharpe is a lie.

| # | date | what changed | outcome | kept? |
|---|---|---|---|---|
| 1 | | | | |

## Decisions frozen (and when)

| decision | value | frozen on | evidence |
|---|---|---|---|
| LM layer / pooling | | | dev slice only |
| deadband α | | | rule, not tuning |
| embargo E | | | |
