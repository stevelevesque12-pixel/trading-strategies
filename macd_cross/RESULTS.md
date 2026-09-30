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

## + Confluence: session VWAP (RTH only)

Buy crosses count only when the signal bar closes above session VWAP
(anchored at 09:30 ET, reset daily, hlc3 × volume); sell crosses only
when it closes below. Crosses on the wrong side of VWAP close the open
position (go flat) but don't reverse it.
`python -m macd_cross.run_timeframes --mode rth --vwap`

| TF | Trades | Win % | Net | PF | Max DD | Base PF | EMA200 PF |
|---|---|---|---|---|---|---|---|
| 1m | 288 | 27.1 | -1,842 | 0.48 | 2,065 | 0.49 | 0.50 |
| 2m | 146 | 32.2 | -396 | 0.80 | 736 | 0.75 | 0.88 |
| 3m | 97 | 30.9 | -335 | 0.81 | 664 | 0.71 | 0.89 |
| 4m | 64 | 35.9 | -214 | 0.83 | 570 | 0.77 | 0.98 |
| 5m | 258 | 37.2 | -620 | 0.93 | 1,997 | 0.94 | 1.03 |
| 10m | 157 | 37.6 | -693 | 0.89 | 1,763 | 0.95 | 1.06 |
| 15m | 372 | 36.3 | -2,129 | 0.88 | 3,613 | 0.92 | 0.93 |

- VWAP is negative after costs on every timeframe.
- It helps a bit on 2m–4m (PF +0.05 to +0.10 vs base) but does nothing
  for 5m–15m, where it's slightly worse than base.
- The 200-EMA filter beats VWAP on every timeframe.
- Why, probably: early in the session VWAP sits right on top of price, so
  it doesn't separate trend from chop in the first hour, which is when
  most of the crosses happen.
- On the common Aug 2–25 window, 5m + VWAP was +$178 (PF 1.17) but on only
  57 trades; over the full 3.5 months at 5m it's negative.

## 15s / 30s / 45s: not tested

The repo has no sub-minute MES data (the finest file is 1m), and splitting
1m bars into seconds would mean making up prices. The runner supports it
once a file exists:
`python -m macd_cross.run_timeframes --seconds-data path/to/mes_1s_or_5s.parquet`
(any resolution that evenly divides 15s, e.g. 1s or 5s).
