# MACD Strategy 1 (MACD / signal-line crossover) on MES

Rule: MACD(12,26,9) crosses above signal → buy; crosses below → sell.
Tested as stop-and-reverse, 1 MES contract, fill at the **next bar's open**
after the signal bar closes. Costs: $0.62/side commission + 1 tick/side
slippage = **$3.74 round turn**.

Two session modes:
- **24h** – always in the market through the full Globex session.
- **RTH** – only take signals from bars closing 09:30–15:45 ET, flatten at 15:55 ET.

Reproduce: `python -m macd_cross.run_timeframes --mode {24h,rth} [--common-window]`

## Longest available history per timeframe

1m–4m are built from the 1m file (Aug 2–25 2026), 5m/10m from the 5m file
(May 10–Aug 25 2026), 15m from the 15m file (Sep 30 2025–Aug 25 2026).

| TF | 24h trades | 24h gross | 24h net | 24h PF | RTH trades | RTH win % | RTH gross | RTH net | RTH PF |
|---|---|---|---|---|---|---|---|---|---|
| 15s | – | – | – | – | – | – | – | – | – |
| 30s | – | – | – | – | – | – | – | – | – |
| 45s | – | – | – | – | – | – | – | – | – |
| 1m | 1909 | -2,434 | -9,573 | 0.46 | 537 | 27.0 | -1,506 | -3,515 | 0.49 |
| 2m | 1041 | -2,270 | -6,163 | 0.49 | 253 | 34.0 | +45 | -901 | 0.75 |
| 3m | 678 | -1,645 | -4,181 | 0.55 | 168 | 31.0 | -249 | -877 | 0.71 |
| 4m | 502 | -1,650 | -3,527 | 0.55 | 112 | 40.2 | -110 | -529 | 0.77 |
| 5m | 1719 | -2,429 | -8,858 | 0.77 | 429 | 39.2 | +861 | -743 | 0.94 |
| 10m | 840 | -196 | -3,338 | 0.87 | 235 | 39.1 | +459 | -420 | 0.95 |
| 15m | 1658 | -2,660 | -8,861 | 0.87 | 508 | 39.4 | +160 | -1,740 | 0.92 |

(PF = net profit factor. Dollar figures are for 1 MES contract.)

## Same window for every timeframe (Aug 2–25 2026, all from the 1m file, RTH)

| TF | trades | net | PF |
|---|---|---|---|
| 1m | 537 | -3,515 | 0.49 |
| 2m | 253 | -901 | 0.75 |
| 3m | 168 | -877 | 0.71 |
| 4m | 112 | -529 | 0.77 |
| 5m | 99 | -568 | 0.73 |
| 10m | 53 | -1,153 | 0.35 |
| 15m | 38 | -792 | 0.52 |

## Takeaways

- **No timeframe is profitable after costs**, in either session mode.
- Before costs the crossover is about break-even at best (RTH 2m/5m/10m/15m
  have a slightly positive gross). Costs then decide it: at 1m, costs
  ($2.0k RTH / $7.1k 24h) are larger than the gross loss.
- Faster timeframes are clearly worse because they whipsaw more: net PF
  rises from ~0.5 at 1m to ~0.9 at 5m–15m, so 15s/30s/45s would very
  likely be worse still.
- RTH-only beats 24h on every timeframe; the overnight session adds mostly
  whipsaw.
- Samples are short (3.5 weeks for 1m–4m) and cover one market regime.
  Treat this as a screen, not a verdict.

## + Confluence: 200-EMA trend filter

Buy crosses count only when the signal bar closes above EMA(200), sell
crosses only when it closes below. A cross against the trend closes the
open position but doesn't reverse it (you go flat).
`python -m macd_cross.run_timeframes --mode rth --trend-ema 200`

Same data windows as the first table.

| TF | 24h trades | 24h net | 24h PF | RTH trades | RTH net | RTH PF | RTH max DD | Base RTH PF |
|---|---|---|---|---|---|---|---|---|
| 1m | 970 | -4,510 | 0.45 | 275 | -1,586 | 0.50 | 1,809 | 0.49 |
| 2m | 537 | -2,376 | 0.58 | 134 | -197 | 0.88 | 538 | 0.75 |
| 3m | 343 | -1,458 | 0.67 | 85 | -140 | 0.89 | 388 | 0.71 |
| 4m | 244 | -768 | 0.77 | 56 | -14 | 0.98 | 255 | 0.77 |
| 5m | 878 | -4,485 | 0.76 | 217 | **+213** | **1.03** | 1,291 | 0.94 |
| 10m | 411 | -1,248 | 0.89 | 116 | **+260** | **1.06** | 911 | 0.95 |
| 15m | 855 | -5,613 | 0.83 | 253 | -799 | 0.93 | 2,985 | 0.92 |

- In RTH the filter roughly halves the number of trades and improves PF on
  every timeframe from 2m up. 5m and 10m turn slightly positive after costs.
- The edge is thin (about $1–2 per trade, PF 1.03–1.06) and comes from ~3.5
  months of data. That's well within noise; don't read it as a real edge
  without a longer test.
- 24h is still negative everywhere.

## 15s / 30s / 45s: not tested

The repo has no sub-minute MES data (the finest file is 1m), and splitting
1m bars into seconds would mean making up prices. The runner supports it
once a file exists:
`python -m macd_cross.run_timeframes --seconds-data path/to/mes_1s_or_5s.parquet`
(any resolution that evenly divides 15s, e.g. 1s or 5s).
