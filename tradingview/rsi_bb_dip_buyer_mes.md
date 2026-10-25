# RSI-BB Dip Buyer (MES)

Pine script: `rsi_bb_dip_buyer_mes.pine`. Built from ChartArt's public
"Bollinger + RSI, Double Strategy (by ChartArt) v1.1" (TradingView, Pine v2),
ported to Pine v6 with a direction filter, an optional trend filter, Tradeify
costs and risk-based sizing with a stop loss.

## Rules

- **Entry (long):** RSI(6) crosses above 50 **and** close crosses back above
  the lower Bollinger Band (200, 2.0). The order is a stop at the band, which
  fills at the next bar's open because price is already above it.
- **Exit:** the short signal (RSI(6) crosses below 50 and close crosses below
  the upper band) closes the long. In "Long only" mode it goes flat and never
  opens a short.
- **Stop:** 150 points from the fill ($750 per MES before costs).
- Trades last days to weeks: average 11.6 calendar days, median 8.1
  (about 340 bars of 24 minutes), longest 70. They are held overnight and over
  weekends.

## Settings to use

| Setting | Value |
|---|---|
| Chart | `MES1!`, **24 minutes**, **ETH**, **B-ADJ on** |
| Trade direction | Long only |
| RSI / Bollinger | 6 / 200, 2.0 |
| Trend filter | **Off** |
| Stop mode | Points, **150** |
| Risk per trade / Max contracts | 1000 / 1 |
| Commission / slippage (Properties) | $0.91 per contract per side / 1 tick |

These are the script's input defaults. $1.82 round trip is Tradeify's
published MES rate (help.tradeify.co, "Trading Commission Fees").

## Backtest results

TradingView Deep Backtesting, May 2019 – Oct 25 2026, 1 MES, costs included.
"Max DD" includes losses on open trades, which is what a prop-firm drawdown
limit measures.

### Variants tested (24m, long only)

| Test | Stop | Trend filter | Trades | Closed PnL | PF | Max DD |
|---|---|---|---|---|---|---|
| 0 | none | off | 83 | $21,698 | 2.66 | −$5,710 |
| **A** | **150 pts** | **off** | **110** | **$21,979** | **1.90** | **−$3,530** |
| B | 200 pts | off | 104 | $21,514 | 1.88 | −$4,780 |
| C | 2% | off | 126 | $20,039 | 1.73 | −$3,673 |
| D | 3% | off | — | — | — | — |
| E | 150 pts | daily SMA 200 | 72 | $12,708 | 1.80 | −$3,879 |

- **No stop** has the highest PF but held single trades through drawdowns of
  up to −$5,340 (Covid 2020) and −$3,777 (Apr 2025).
- **150 points** keeps nearly all the profit: a stopped trade frees the
  strategy to buy again lower, which caught the Mar 23 2020 (+$2,094) and
  Apr 7 2025 (+$2,572) rebounds.
- **Wider or percent stops** cost more per loss without avoiding the losing
  streaks. Test D (3%) stopped trading after Aug 2025: once 3% of price
  exceeded the $1,000 risk budget, the size rounded to 0. Through Aug 2025 it
  trailed Test A (PF 1.86 vs 1.94, DD −$3,963 vs −$3,530).
- **The trend filter hurts.** The 32 trades it removed made +$8,447 (21
  winners), including the biggest rebound trades, while the stop streaks
  happened just above the SMA.

### Timeframe check (Test A settings)

| Chart | Trades | Closed PnL | PF | Max DD | Worst losing streak |
|---|---|---|---|---|---|
| 20m | 123 | $19,262 | 1.66 | −$4,642 | 3 |
| 24m | 110 | $21,979 | 1.90 | −$3,530 | 4 |
| 30m | 83 | $12,845 | 1.50 | −$4,825 | 5 |

| Year | 20m | 24m | 30m |
|---|---|---|---|
| 2019 | +$348 | +$71 | +$455 |
| 2020 | +$1,575 | +$4,193 | −$1,892 |
| 2021 | +$5,216 | +$4,720 | +$5,825 |
| 2022 | −$1,139 | +$628 | −$1,766 |
| 2023 | +$2,589 | +$1,923 | +$1,085 |
| 2024 | +$2,452 | +$3,825 | +$3,825 |
| 2025 | +$5,503 | +$4,100 | +$4,574 |
| 2026 | +$2,719 | +$2,519 | +$739 |

Profitable on all three charts, but 24m is the best of the three. **Plan
around PF about 1.6, about $2.5k a year and a $4.5–5k drawdown per MES.**
Bear markets like 2022 are the weak spot.

## Earlier findings (original ChartArt script)

- Original always-in-market version (long and short), 12m chart, 412 trades:
  longs +$22.9k, shorts −$1.3k. The short side is dropped.
- 12m / 20m / 24m with a 100-point stop: PF 1.36 / 1.57 / 1.81. Too tight;
  it was stopped out on 36–44% of trades.

## Notes and caveats

- **Use B-ADJ (back-adjusted) data.** Without it, trades held through a
  quarterly contract roll count the roll gap as profit. On the no-stop test
  that inflated PnL by about $2.6k.
- **Percent stops are computed on back-adjusted prices**, which run higher than
  real prices in early years. In 2019, "2%" was about 2.6% of the real price.
- **Position size:** a Tradeify 50k account (about $2k drawdown limit) is too
  small. Three stops in a row (−$2,259) happened on every timeframe. Use a
  drawdown limit of about $6–7k or more per MES, or about $15–20k of capital
  per MES in a personal account.
- **Margin:** the script uses TradingView's default 100% margin. 1 MES fits a
  $50k account, but more contracts can be silently rejected for margin.
  Lower the margin in Properties before sizing up.
- **Rolls:** trades are often open through a quarterly roll. Live, close and
  reopen in the next contract.
- No alerts or webhook are in the script.
- The script was not compile-tested outside TradingView; the results above
  come from the user's own TradingView runs (trade-list CSV exports).
