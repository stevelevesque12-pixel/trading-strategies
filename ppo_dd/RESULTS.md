# PPO - DD optimization (long side, NQ)

Simulator: `ppo_dd/sim.py`, a bar-by-bar port of `tradingview/ppo_dd.pine` that follows
TradingView's fill rules (market orders fill at the next open, resting TP/SL fill on touch,
the OHLC path rule decides when both are hit in one bar). Costs: $2.50/contract/side and
1 tick of slippage on market and stop fills. All $ figures are per **1 NQ base lot**
($20/pt). Divide by 10 for MNQ.

- `python -m ppo_dd.optimize`: NQ 1m, fitted on 2023–2024, checked on 2025 (Dec 2025 data end).
- `python -m ppo_dd.stress15`: NQ 15m 2016–2026 (chart TF = 15m, approximate), fitted on
  2016–2022 (2018 Q4, 2020, 2022 bear), checked on 2023–2026.

"maxDD" is the worst mark-to-market drawdown, including open losses on a live basket.
For this strategy that matters more than win rate, which is ~99% for almost every setting.

## Findings

| Config | 1m 2023–25 net | 1m maxDD | ret/DD | 15m 2016–26 net | 15m maxDD | ret/DD | SL hits | max lots |
|---|---|---|---|---|---|---|---|---|
| Script defaults (1.05 ratios) | −$3k | $20k | — | — | — | — | 0 | 8 |
| **R1 (new defaults)** | **$670k** | **$203k** | **3.30** | **$1.47M** | **$249k** | **5.92** | **0** | **16** |
| Same, max DD 7 | $880k | $472k | 1.86 | $1.73M | $685k | 2.53 | 0 | 64 |
| Script 2/1.5× ladder, tuned | $908k | $527k | 1.72 | — | — | — | 0 | 50 |
| Linear ladder (+1 lot/DD) | $454k | $99k | 4.60 | $286k | $348k | 0.82 | 11 | 7 |

R1 = EMA1 15m len 300 ratio 1.01, EMA2 chart(1m) len 300 ratio 1.0025, base TP 0.135%,
DD distances unchanged, multipliers `1,1,...` (basket doubles per DD), Max DD Times 5,
SL 5% armed at final DD. R1 made money in every calendar year on both datasets.

- The 1.05 ratios almost never fire on NQ: 0 trades in 2023–24.
- The script's 2/1.5× multiplier ladder grows 1 → 3 → 8 → 20 → 50 → 126 → 381 lots. A
  2023–24 fit looks great, but 2025 (April) shows $0.5–1M drawdowns.
- A linear ladder doesn't pull the average down fast enough. In 2022 it hit the stop
  7 times and lost $306k.
- Neighbors of R1: SL 3% is clearly worse (stops out in selloffs), and DD-distance scale
  0.75 is worse. Max DD Times 5–6, DD scale 1.0–1.25 and SL 5–8% are all fine.

## Caveats

- The NQ files are a back-adjusted continuous contract, so % thresholds are slightly off
  versus the real contract (worse the further back you go).
- This is a model of TradingView, not TradingView itself. Confirm in the Strategy Tester.
- Only the long side was optimized. The short side is still on the script defaults.
