# Transaction costs, PnL and break-even cost (Phase 3)

`python3 -m edge.costs.run --seed 0 --trials 288`

**Protocol C only.** A PnL number computed under a leaky protocol is not a weaker result, it is a misleading one; Phase 2 established which protocol to believe.

## Stated assumptions

- **Fills are aggressive.** Buying crosses to the ask, selling to the bid, so a round trip pays the FULL spread plus two take fees. Pessimistic by design.
- **Fees**: $0.0030/share to take, $0.0020/share make rebate — the commonly-quoted 2012 Nasdaq standard rates, entered as an ASSUMPTION and *not* verified against an archived price list. The break-even table is the answer to anyone who disagrees: it reports the cost at which PnL crosses zero, so a reader can substitute their own.
- **No market impact model.** Stated, not omitted. Ignoring impact biases PnL UPWARD; the break-even table partially absorbs this.
- **Position**: single unit of 100 shares, no inventory, flat at the horizon. Trades only when |prediction| exceeds the 0.5-tick deadband.
- **Latency**: the signal formed at t acts at t+delta. A signal whose edge is gone by 100ms is latency arbitrage, not alpha — and the phase-1 adversary measured that OFI's content decays inside 100ms, so the 100ms rung is expected to be brutal.

## Break-even cost table (the headline)

The per-share cost at which each rung's PnL crosses zero. Negative means the rung is already underwater before any additional cost — reported, not clipped.

| ticker | clock | h | rung | latency | trades | total PnL $ | break-even (ticks) | break-even (bps) | PnL CI lo | PnL CI hi | DSR |
|---|---|---|---|---|---|---|---|---|---|---|---|
| INTC | calendar | 1s | L0 zero | 1ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | calendar | 1s | L0 zero | 10ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | calendar | 1s | L0 zero | 100ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | calendar | 1s | L1 sign(queue imbalance) | 1ms | 17043 | -27105.80 | -0.7952 | -2.94 | -1.5948 | -1.5862 | 0.000 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | 10ms | 17043 | -27104.80 | -0.7952 | -2.94 | -1.5948 | -1.5862 | 0.000 |
| INTC | calendar | 1s | L1 sign(queue imbalance) | 100ms | 17043 | -27125.80 | -0.7958 | -2.94 | -1.5962 | -1.5872 | 0.000 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | 1ms | 17044 | -27239.40 | -0.7991 | -2.95 | -1.6028 | -1.5939 | 0.000 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | 10ms | 17044 | -27236.40 | -0.7990 | -2.95 | -1.6027 | -1.5935 | 0.000 |
| INTC | calendar | 1s | L1m sign(microprice - mid) | 100ms | 17044 | -27239.40 | -0.7991 | -2.95 | -1.6031 | -1.5935 | 0.000 |
| INTC | calendar | 1s | L2 OLS on OFI | 1ms | 17044 | -27349.40 | -0.8023 | -2.97 | -1.6097 | -1.5999 | 0.000 |
| INTC | calendar | 1s | L2 OLS on OFI | 10ms | 17044 | -27346.40 | -0.8022 | -2.96 | -1.6094 | -1.5996 | 0.000 |
| INTC | calendar | 1s | L2 OLS on OFI | 100ms | 17044 | -27355.40 | -0.8025 | -2.97 | -1.6097 | -1.6001 | 0.000 |
| INTC | calendar | 1s | L3 ridge, full set | 1ms | 17044 | -27134.40 | -0.7960 | -2.94 | -1.5968 | -1.5876 | 0.000 |
| INTC | calendar | 1s | L3 ridge, full set | 10ms | 17044 | -27132.40 | -0.7960 | -2.94 | -1.5967 | -1.5874 | 0.000 |
| INTC | calendar | 1s | L3 ridge, full set | 100ms | 17044 | -27148.40 | -0.7964 | -2.94 | -1.5977 | -1.5883 | 0.000 |
| INTC | calendar | 1s | L4 GBM, full set | 1ms | 17044 | -27196.40 | -0.7978 | -2.95 | -1.6006 | -1.5908 | 0.000 |
| INTC | calendar | 1s | L4 GBM, full set | 10ms | 17044 | -27195.40 | -0.7978 | -2.95 | -1.6005 | -1.5907 | 0.000 |
| INTC | calendar | 1s | L4 GBM, full set | 100ms | 17044 | -27197.40 | -0.7979 | -2.95 | -1.6007 | -1.5911 | 0.000 |
| INTC | calendar | 5s | L0 zero | 1ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | calendar | 5s | L0 zero | 10ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | calendar | 5s | L0 zero | 100ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | calendar | 5s | L1 sign(queue imbalance) | 1ms | 3790 | -5838.00 | -0.7702 | -2.85 | -1.5536 | -1.5262 | 0.000 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | 10ms | 3790 | -5838.00 | -0.7702 | -2.85 | -1.5536 | -1.5262 | 0.000 |
| INTC | calendar | 5s | L1 sign(queue imbalance) | 100ms | 3790 | -5857.00 | -0.7727 | -2.86 | -1.5583 | -1.5309 | 0.000 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | 1ms | 3790 | -5941.00 | -0.7838 | -2.90 | -1.5821 | -1.5516 | 0.000 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | 10ms | 3790 | -5940.00 | -0.7836 | -2.90 | -1.5819 | -1.5510 | 0.000 |
| INTC | calendar | 5s | L1m sign(microprice - mid) | 100ms | 3790 | -5946.00 | -0.7844 | -2.90 | -1.5833 | -1.5541 | 0.000 |
| INTC | calendar | 5s | L2 OLS on OFI | 1ms | 3790 | -6057.00 | -0.7991 | -2.95 | -1.6117 | -1.5838 | 0.000 |
| INTC | calendar | 5s | L2 OLS on OFI | 10ms | 3790 | -6055.00 | -0.7988 | -2.95 | -1.6115 | -1.5832 | 0.000 |
| INTC | calendar | 5s | L2 OLS on OFI | 100ms | 3790 | -6059.00 | -0.7993 | -2.95 | -1.6127 | -1.5844 | 0.000 |
| INTC | calendar | 5s | L3 ridge, full set | 1ms | 3790 | -5843.00 | -0.7708 | -2.85 | -1.5561 | -1.5278 | 0.000 |
| INTC | calendar | 5s | L3 ridge, full set | 10ms | 3790 | -5842.00 | -0.7707 | -2.85 | -1.5557 | -1.5277 | 0.000 |
| INTC | calendar | 5s | L3 ridge, full set | 100ms | 3790 | -5860.00 | -0.7731 | -2.86 | -1.5609 | -1.5321 | 0.000 |
| INTC | calendar | 5s | L4 GBM, full set | 1ms | 3790 | -5981.00 | -0.7891 | -2.92 | -1.5929 | -1.5634 | 0.000 |
| INTC | calendar | 5s | L4 GBM, full set | 10ms | 3790 | -5980.00 | -0.7889 | -2.92 | -1.5926 | -1.5631 | 0.000 |
| INTC | calendar | 5s | L4 GBM, full set | 100ms | 3790 | -5979.00 | -0.7888 | -2.92 | -1.5926 | -1.5634 | 0.000 |
| INTC | event | 1s | L0 zero | 1ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | event | 1s | L0 zero | 10ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | event | 1s | L0 zero | 100ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | event | 1s | L1 sign(queue imbalance) | 1ms | 10035 | -15899.00 | -0.7922 | -2.93 | -1.5899 | -1.5773 | 0.000 |
| INTC | event | 1s | L1 sign(queue imbalance) | 10ms | 9982 | -15865.20 | -0.7947 | -2.94 | -1.5958 | -1.5825 | 0.000 |
| INTC | event | 1s | L1 sign(queue imbalance) | 100ms | 9504 | -15155.40 | -0.7973 | -2.95 | -1.6007 | -1.5883 | 0.000 |
| INTC | event | 1s | L1m sign(microprice - mid) | 1ms | 10035 | -16012.00 | -0.7978 | -2.95 | -1.6027 | -1.5888 | 0.000 |
| INTC | event | 1s | L1m sign(microprice - mid) | 10ms | 9982 | -15959.20 | -0.7994 | -2.95 | -1.6055 | -1.5922 | 0.000 |
| INTC | event | 1s | L1m sign(microprice - mid) | 100ms | 9504 | -15212.40 | -0.8003 | -2.96 | -1.6071 | -1.5942 | 0.000 |
| INTC | event | 1s | L2 OLS on OFI | 1ms | 10035 | -16155.00 | -0.8049 | -2.97 | -1.6167 | -1.6031 | 0.000 |
| INTC | event | 1s | L2 OLS on OFI | 10ms | 9982 | -16066.20 | -0.8048 | -2.97 | -1.6157 | -1.6028 | 0.000 |
| INTC | event | 1s | L2 OLS on OFI | 100ms | 9504 | -15304.40 | -0.8052 | -2.98 | -1.6166 | -1.6041 | 0.000 |
| INTC | event | 1s | L3 ridge, full set | 1ms | 10035 | -15916.00 | -0.7930 | -2.93 | -1.5926 | -1.5795 | 0.000 |
| INTC | event | 1s | L3 ridge, full set | 10ms | 9982 | -15879.20 | -0.7954 | -2.94 | -1.5970 | -1.5843 | 0.000 |
| INTC | event | 1s | L3 ridge, full set | 100ms | 9504 | -15157.40 | -0.7974 | -2.95 | -1.6011 | -1.5883 | 0.000 |
| INTC | event | 1s | L4 GBM, full set | 1ms | 10035 | -15930.00 | -0.7937 | -2.93 | -1.5944 | -1.5807 | 0.000 |
| INTC | event | 1s | L4 GBM, full set | 10ms | 9982 | -15921.20 | -0.7975 | -2.95 | -1.6015 | -1.5883 | 0.000 |
| INTC | event | 1s | L4 GBM, full set | 100ms | 9504 | -15178.40 | -0.7985 | -2.95 | -1.6041 | -1.5901 | 0.000 |
| INTC | event | 5s | L0 zero | 1ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | event | 5s | L0 zero | 10ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | event | 5s | L0 zero | 100ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| INTC | event | 5s | L1 sign(queue imbalance) | 1ms | 3261 | -5013.60 | -0.7687 | -2.84 | -1.5535 | -1.5212 | 0.000 |
| INTC | event | 5s | L1 sign(queue imbalance) | 10ms | 3255 | -5021.00 | -0.7713 | -2.85 | -1.5579 | -1.5276 | 0.000 |
| INTC | event | 5s | L1 sign(queue imbalance) | 100ms | 3211 | -4983.60 | -0.7760 | -2.87 | -1.5682 | -1.5362 | 0.000 |
| INTC | event | 5s | L1m sign(microprice - mid) | 1ms | 3261 | -5096.60 | -0.7814 | -2.89 | -1.5798 | -1.5491 | 0.000 |
| INTC | event | 5s | L1m sign(microprice - mid) | 10ms | 3255 | -5096.00 | -0.7828 | -2.89 | -1.5826 | -1.5521 | 0.000 |
| INTC | event | 5s | L1m sign(microprice - mid) | 100ms | 3211 | -5039.60 | -0.7847 | -2.90 | -1.5854 | -1.5522 | 0.000 |
| INTC | event | 5s | L2 OLS on OFI | 1ms | 3261 | -5197.60 | -0.7969 | -2.95 | -1.6109 | -1.5779 | 0.000 |
| INTC | event | 5s | L2 OLS on OFI | 10ms | 3255 | -5189.00 | -0.7971 | -2.95 | -1.6094 | -1.5782 | 0.000 |
| INTC | event | 5s | L2 OLS on OFI | 100ms | 3211 | -5100.60 | -0.7942 | -2.94 | -1.6058 | -1.5698 | 0.000 |
| INTC | event | 5s | L3 ridge, full set | 1ms | 3261 | -5018.60 | -0.7695 | -2.84 | -1.5519 | -1.5245 | 0.000 |
| INTC | event | 5s | L3 ridge, full set | 10ms | 3255 | -5031.00 | -0.7728 | -2.86 | -1.5592 | -1.5321 | 0.000 |
| INTC | event | 5s | L3 ridge, full set | 100ms | 3211 | -4979.60 | -0.7754 | -2.87 | -1.5679 | -1.5365 | 0.000 |
| INTC | event | 5s | L4 GBM, full set | 1ms | 3261 | -5095.60 | -0.7813 | -2.89 | -1.5808 | -1.5462 | 0.000 |
| INTC | event | 5s | L4 GBM, full set | 10ms | 3255 | -5113.00 | -0.7854 | -2.90 | -1.5883 | -1.5540 | 0.000 |
| INTC | event | 5s | L4 GBM, full set | 100ms | 3211 | -5062.60 | -0.7883 | -2.91 | -1.5940 | -1.5608 | 0.000 |
| AMZN | calendar | 1s | L0 zero | 1ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | calendar | 1s | L0 zero | 10ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | calendar | 1s | L0 zero | 100ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | 1ms | 13986 | -177501.60 | -6.3457 | -2.85 | -12.8681 | -12.4741 | 0.000 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | 10ms | 13986 | -177482.60 | -6.3450 | -2.85 | -12.8672 | -12.4737 | 0.000 |
| AMZN | calendar | 1s | L1 sign(queue imbalance) | 100ms | 13986 | -177399.60 | -6.3420 | -2.85 | -12.8593 | -12.4634 | 0.000 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | 1ms | 15458 | -192975.80 | -6.2419 | -2.80 | -12.6464 | -12.3050 | 0.000 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | 10ms | 15458 | -192985.80 | -6.2423 | -2.80 | -12.6469 | -12.3067 | 0.000 |
| AMZN | calendar | 1s | L1m sign(microprice - mid) | 100ms | 15458 | -193087.80 | -6.2456 | -2.80 | -12.6523 | -12.3105 | 0.000 |
| AMZN | calendar | 1s | L2 OLS on OFI | 1ms | 15458 | -193041.80 | -6.2441 | -2.80 | -12.6516 | -12.3149 | 0.000 |
| AMZN | calendar | 1s | L2 OLS on OFI | 10ms | 15458 | -193074.80 | -6.2451 | -2.80 | -12.6535 | -12.3158 | 0.000 |
| AMZN | calendar | 1s | L2 OLS on OFI | 100ms | 15458 | -192911.80 | -6.2399 | -2.80 | -12.6429 | -12.3031 | 0.000 |
| AMZN | calendar | 1s | L3 ridge, full set | 1ms | 15458 | -192455.80 | -6.2251 | -2.79 | -12.6091 | -12.2717 | 0.000 |
| AMZN | calendar | 1s | L3 ridge, full set | 10ms | 15458 | -192474.80 | -6.2257 | -2.79 | -12.6118 | -12.2712 | 0.000 |
| AMZN | calendar | 1s | L3 ridge, full set | 100ms | 15458 | -192474.80 | -6.2257 | -2.79 | -12.6102 | -12.2746 | 0.000 |
| AMZN | calendar | 1s | L4 GBM, full set | 1ms | 15458 | -192538.80 | -6.2278 | -2.80 | -12.6211 | -12.2756 | 0.000 |
| AMZN | calendar | 1s | L4 GBM, full set | 10ms | 15458 | -192542.80 | -6.2279 | -2.80 | -12.6191 | -12.2739 | 0.000 |
| AMZN | calendar | 1s | L4 GBM, full set | 100ms | 15458 | -192466.80 | -6.2255 | -2.79 | -12.6131 | -12.2735 | 0.000 |
| AMZN | calendar | 5s | L0 zero | 1ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | calendar | 5s | L0 zero | 10ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | calendar | 5s | L0 zero | 100ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | 1ms | 3308 | -41340.80 | -6.2486 | -2.80 | -13.0524 | -11.9139 | 0.000 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | 10ms | 3308 | -41331.80 | -6.2472 | -2.80 | -13.0455 | -11.9124 | 0.000 |
| AMZN | calendar | 5s | L1 sign(queue imbalance) | 100ms | 3308 | -41442.80 | -6.2640 | -2.81 | -13.0685 | -11.9565 | 0.000 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | 1ms | 3674 | -45446.40 | -6.1849 | -2.78 | -12.9007 | -11.8359 | 0.000 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | 10ms | 3674 | -45459.40 | -6.1866 | -2.78 | -12.9102 | -11.8429 | 0.000 |
| AMZN | calendar | 5s | L1m sign(microprice - mid) | 100ms | 3674 | -45615.40 | -6.2079 | -2.79 | -12.9595 | -11.8800 | 0.000 |
| AMZN | calendar | 5s | L2 OLS on OFI | 1ms | 3674 | -44964.40 | -6.1193 | -2.75 | -12.7437 | -11.7201 | 0.000 |
| AMZN | calendar | 5s | L2 OLS on OFI | 10ms | 3674 | -44979.40 | -6.1213 | -2.75 | -12.7469 | -11.7224 | 0.000 |
| AMZN | calendar | 5s | L2 OLS on OFI | 100ms | 3674 | -45112.40 | -6.1394 | -2.76 | -12.7857 | -11.7632 | 0.000 |
| AMZN | calendar | 5s | L3 ridge, full set | 1ms | 3674 | -44883.40 | -6.1082 | -2.74 | -12.7953 | -11.6999 | 0.000 |
| AMZN | calendar | 5s | L3 ridge, full set | 10ms | 3674 | -44911.40 | -6.1121 | -2.74 | -12.7983 | -11.7121 | 0.000 |
| AMZN | calendar | 5s | L3 ridge, full set | 100ms | 3674 | -45023.40 | -6.1273 | -2.75 | -12.8363 | -11.7416 | 0.000 |
| AMZN | calendar | 5s | L4 GBM, full set | 1ms | 3674 | -45050.40 | -6.1310 | -2.75 | -12.7499 | -11.7625 | 0.000 |
| AMZN | calendar | 5s | L4 GBM, full set | 10ms | 3674 | -45094.40 | -6.1370 | -2.75 | -12.7596 | -11.7735 | 0.000 |
| AMZN | calendar | 5s | L4 GBM, full set | 100ms | 3674 | -45213.40 | -6.1532 | -2.76 | -12.7985 | -11.8053 | 0.000 |
| AMZN | event | 1s | L0 zero | 1ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | event | 1s | L0 zero | 10ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | event | 1s | L0 zero | 100ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | event | 1s | L1 sign(queue imbalance) | 1ms | 7477 | -93970.20 | -6.2840 | -2.82 | -12.7801 | -12.3329 | 0.000 |
| AMZN | event | 1s | L1 sign(queue imbalance) | 10ms | 7429 | -93759.40 | -6.3104 | -2.83 | -12.8494 | -12.3910 | 0.000 |
| AMZN | event | 1s | L1 sign(queue imbalance) | 100ms | 7108 | -90320.80 | -6.3535 | -2.85 | -12.9488 | -12.4589 | 0.000 |
| AMZN | event | 1s | L1m sign(microprice - mid) | 1ms | 8251 | -102412.60 | -6.2061 | -2.79 | -12.6380 | -12.1771 | 0.000 |
| AMZN | event | 1s | L1m sign(microprice - mid) | 10ms | 8198 | -102132.80 | -6.2291 | -2.80 | -12.6833 | -12.2424 | 0.000 |
| AMZN | event | 1s | L1m sign(microprice - mid) | 100ms | 7849 | -98480.40 | -6.2734 | -2.82 | -12.7657 | -12.3224 | 0.000 |
| AMZN | event | 1s | L2 OLS on OFI | 1ms | 8251 | -102769.60 | -6.2277 | -2.80 | -12.6771 | -12.2282 | 0.000 |
| AMZN | event | 1s | L2 OLS on OFI | 10ms | 8198 | -102427.80 | -6.2471 | -2.80 | -12.7212 | -12.2797 | 0.000 |
| AMZN | event | 1s | L2 OLS on OFI | 100ms | 7849 | -98535.40 | -6.2769 | -2.82 | -12.7622 | -12.3330 | 0.000 |
| AMZN | event | 1s | L3 ridge, full set | 1ms | 8251 | -101609.60 | -6.1574 | -2.76 | -12.5295 | -12.0766 | 0.000 |
| AMZN | event | 1s | L3 ridge, full set | 10ms | 8198 | -101407.80 | -6.1849 | -2.78 | -12.5919 | -12.1417 | 0.000 |
| AMZN | event | 1s | L3 ridge, full set | 100ms | 7849 | -97739.40 | -6.2262 | -2.79 | -12.6551 | -12.2377 | 0.000 |
| AMZN | event | 1s | L4 GBM, full set | 1ms | 8251 | -101338.60 | -6.1410 | -2.76 | -12.4975 | -12.0595 | 0.000 |
| AMZN | event | 1s | L4 GBM, full set | 10ms | 8198 | -101232.80 | -6.1742 | -2.77 | -12.5604 | -12.1263 | 0.000 |
| AMZN | event | 1s | L4 GBM, full set | 100ms | 7849 | -97531.40 | -6.2130 | -2.79 | -12.6337 | -12.2068 | 0.000 |
| AMZN | event | 5s | L0 zero | 1ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | event | 5s | L0 zero | 10ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | event | 5s | L0 zero | 100ms | 0 | 0.00 | n/a | n/a | n/a | n/a | n/a |
| AMZN | event | 5s | L1 sign(queue imbalance) | 1ms | 2673 | -33126.80 | -6.1966 | -2.78 | -12.9867 | -11.7890 | 0.000 |
| AMZN | event | 5s | L1 sign(queue imbalance) | 10ms | 2668 | -33232.80 | -6.2280 | -2.80 | -13.0570 | -11.8535 | 0.000 |
| AMZN | event | 5s | L1 sign(queue imbalance) | 100ms | 2629 | -32806.40 | -6.2393 | -2.80 | -13.1012 | -11.8577 | 0.000 |
| AMZN | event | 5s | L1m sign(microprice - mid) | 1ms | 2966 | -36582.60 | -6.1670 | -2.77 | -12.9434 | -11.8002 | 0.000 |
| AMZN | event | 5s | L1m sign(microprice - mid) | 10ms | 2961 | -36664.60 | -6.1913 | -2.78 | -12.9871 | -11.8204 | 0.000 |
| AMZN | event | 5s | L1m sign(microprice - mid) | 100ms | 2916 | -36282.60 | -6.2213 | -2.79 | -13.0141 | -11.8229 | 0.000 |
| AMZN | event | 5s | L2 OLS on OFI | 1ms | 2966 | -36213.60 | -6.1048 | -2.74 | -12.8152 | -11.7070 | 0.000 |
| AMZN | event | 5s | L2 OLS on OFI | 10ms | 2961 | -36298.60 | -6.1294 | -2.75 | -12.8271 | -11.7675 | 0.000 |
| AMZN | event | 5s | L2 OLS on OFI | 100ms | 2916 | -35798.60 | -6.1383 | -2.76 | -12.7756 | -11.7376 | 0.000 |
| AMZN | event | 5s | L3 ridge, full set | 1ms | 2966 | -35647.60 | -6.0094 | -2.70 | -12.6343 | -11.4882 | 0.000 |
| AMZN | event | 5s | L3 ridge, full set | 10ms | 2961 | -35797.60 | -6.0448 | -2.71 | -12.6869 | -11.5536 | 0.000 |
| AMZN | event | 5s | L3 ridge, full set | 100ms | 2916 | -35515.60 | -6.0898 | -2.73 | -12.7421 | -11.5911 | 0.000 |
| AMZN | event | 5s | L4 GBM, full set | 1ms | 2966 | -35546.60 | -5.9923 | -2.69 | -12.5347 | -11.4623 | 0.000 |
| AMZN | event | 5s | L4 GBM, full set | 10ms | 2961 | -35675.60 | -6.0242 | -2.70 | -12.5859 | -11.5283 | 0.000 |
| AMZN | event | 5s | L4 GBM, full set | 100ms | 2916 | -35368.60 | -6.0646 | -2.72 | -12.6330 | -11.5820 | 0.000 |

## Reading notes

- A break-even cost BELOW the assumed take fee of $0.0030/share (0.30 ticks) means the rung does not pay for its own fees, before any spread cost.
- The PnL confidence interval is a circular block bootstrap with block length >= the label horizon; resampling individual trades would ignore the serial dependence that makes the naive interval far too narrow.
- Trade counts differ across rungs because the deadband bites differently: a rung whose predictions rarely exceed half a tick simply does not trade, which is a result rather than a missing row.
