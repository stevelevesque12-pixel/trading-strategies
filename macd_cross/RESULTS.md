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

# MACD Strategy 2: zero-line cross (RTH only)

Two readings, both stop-and-reverse, RTH only, same fills/costs as above:
- **zero_cross** – MACD line crosses above 0 → buy, below 0 → sell
  (= EMA12/EMA26 crossover).
- **zero_cross_hist** – the "MACD + zero line combination": long when
  MACD > 0 **and** histogram > 0 (MACD above signal) first both hold,
  short when both are < 0. Whichever condition arrives second triggers.

`python -m macd_cross.run_timeframes --mode rth --rule {zero_cross,zero_cross_hist} --timeframes 1min,2min,3min,4min,5min,6min,7min,8min,9min,10min --common-window`

## Same window for all (Aug 2–25 2026, 1m source)

| TF | ZC trades | ZC net | ZC PF | ZC+hist trades | ZC+hist net | ZC+hist PF | Strategy 1 PF |
|---|---|---|---|---|---|---|---|
| 1m | 205 | -1,089 | 0.64 | 200 | -517 | 0.83 | 0.49 |
| 2m | 100 | -233 | 0.86 | 96 | +53 | 1.03 | 0.75 |
| 3m | 67 | -359 | 0.77 | 68 | +44 | 1.03 | 0.71 |
| 4m | 56 | -392 | 0.74 | 56 | -406 | 0.76 | 0.77 |
| 5m | 42 | -335 | 0.73 | 42 | +49 | 1.03 | 0.73 |
| 6m | 33 | -10 | 0.99 | 36 | -128 | 0.90 | 0.74 |
| 7m | 29 | -233 | 0.74 | 33 | +52 | 1.05 | 0.70 |
| 8m | 28 | -356 | 0.65 | 32 | -32 | 0.97 | 0.52 |
| 9m | 22 | -175 | 0.75 | 29 | -107 | 0.88 | 0.34 |
| 10m | 23 | +110 | 1.15 | 29 | +75 | 1.08 | 0.35 |

## Longer history for 5m / 10m (May 10–Aug 25 2026, 5m source)

| TF | ZC trades | ZC net | ZC PF | ZC+hist trades | ZC+hist net | ZC+hist PF | Strategy 1 PF |
|---|---|---|---|---|---|---|---|
| 5m | 192 | -43 | 0.99 | 213 | -919 | 0.90 | 0.94 |
| 10m | 100 | -42 | 0.99 | 126 | -756 | 0.88 | 0.95 |

## Takeaways

- Both Strategy 2 readings beat Strategy 1 on almost every timeframe,
  mainly because they trade about half as often and so pay half the costs.
- The combo rule is roughly break-even on 2m/3m/5m/7m/10m over Aug 2–25,
  but each result is only 30–100 trades and ±$50. That's noise.
- The one longer test contradicts it: on 3.5 months of 5m/10m data the
  combo rule loses (PF ~0.9), while plain zero_cross is dead flat (PF 0.99).
- Net: nothing here is a demonstrated edge. The best honest summary is that
  zero-line crosses are break-even before being tuned, not profitable.

# MACD Strategy 4: price / MACD divergence (RTH only)

Code: `macd_cross/divergence.py`.
- Swing low = lowest low of the 3 bars either side (so it's only known 3
  bars later, and that's when it's used). Swing highs mirror it.
- Bullish divergence: swing low below the previous swing low while the
  MACD line at the new swing is higher. Bearish mirrors it. This is the
  pattern in the ES weekly example (price higher highs, MACD lower highs).
- **divergence** (raw): enter at the next open once the divergence is known.
- **divergence_confirmed**: treat it as a warning; enter only on the next
  MACD/signal cross in that direction within 10 bars.
- Exits: stop 1 tick beyond the divergence swing, opposite MACD/signal
  cross, or 15:55 flatten. One position at a time.

`python -m macd_cross.run_timeframes --rule {divergence,divergence_confirmed} --common-window --timeframes 1min,...,10min`

## Same window for all (Aug 2–25 2026)

| TF | Raw trades | Raw net | Raw PF | Confirmed trades | Confirmed net | Confirmed PF |
|---|---|---|---|---|---|---|
| 1m | 118 | -718 | 0.49 | 52 | -413 | 0.40 |
| 2m | 53 | -527 | 0.34 | 22 | -109 | 0.57 |
| 3m | 42 | -315 | 0.59 | 15 | -22 | 0.91 |
| 4m | 37 | -147 | 0.75 | 14 | -111 | 0.50 |
| 5m | 27 | +112 | 1.26 | 6 | -32 | 0.63 |
| 6m | 20 | +20 | 1.05 | 4 | +34 | 1.75 |
| 7m | 16 | -40 | 0.88 | 3 | +63 | inf |
| 8m | 16 | -150 | 0.59 | 5 | -87 | 0.19 |
| 9m | 13 | -82 | 0.70 | 7 | -89 | 0.18 |
| 10m | 11 | -55 | 0.77 | 5 | -169 | 0.07 |

## Longer history

| TF (window) | Raw trades | Raw net | Raw PF | Raw max DD | Confirmed trades | Confirmed net | Confirmed PF |
|---|---|---|---|---|---|---|---|
| 5m (May–Aug 2026) | 94 | +476 | 1.21 | 580 | 23 | -32 | 0.94 |
| 10m (May–Aug 2026) | 46 | +315 | 1.26 | 287 | 16 | +143 | 1.29 |
| 15m (Oct 2025–Aug 2026) | 99 | -159 | 0.96 | 810 | – | – | – |

## Robustness check (raw, net PF by swing width k)

| TF | k=2 | k=3 (default) | k=4 | k=5 |
|---|---|---|---|---|
| 5m | 0.85 | 1.21 | 1.37 | 1.19 |
| 10m | 1.06 | 1.26 | 1.04 | 0.66 |
| 15m | 1.15 | 0.96 | 0.95 | 0.86 |

(max_gap 30/60/120 made no difference; swings are always closer than 30 bars.)

15m by month (k=3): -25, -180, -384, +155, +321, +248, +184, +143, -449, -62, -110
(Oct 2025 → Aug 2026). It made money Jan–May and lost the rest.

## Takeaways

- Raw divergence on 5m/10m is the only MACD rule so far that is positive
  over 3.5 months after costs (PF 1.21 / 1.26), with small drawdowns.
- It isn't robust yet: the same rule on 11 months of 15m is slightly
  negative, results swing with the swing width k, and the profitable
  stretch is concentrated in Jan–May 2026.
- Waiting for a signal-line cross ("confirmed") cuts trades by ~75% and
  isn't better. Samples are too small to judge.
- 1m–4m lose, as with the other MACD rules.

## Strategy 4 across MCL, MGC, MES, MNQ

Same rules and RTH window (09:30–15:55 ET) for all four; costs $0.62/side
+ 1 tick/side. `python -m macd_cross.run_timeframes --symbol MCL --rule divergence ...`

### Longer history, net $ (PF)

5m/10m = May 10–Aug 25 2026, 15m = Sep 30 2025–Aug 25 2026.

| Rule | TF | MCL | MGC | MES | MNQ |
|---|---|---|---|---|---|
| raw | 5m | -40 (0.98) | +633 (1.22) | +476 (1.21) | -2,901 (0.59) |
| raw | 10m | +563 (1.90) | -1,058 (0.54) | +315 (1.26) | +444 (1.10) |
| raw | 15m | -669 (0.74) | +489 (1.11) | -159 (0.96) | -127 (0.98) |
| confirmed | 5m | +143 (1.20) | +1,070 (1.84) | -32 (0.94) | -50 (0.98) |
| confirmed | 10m | -183 (0.53) | -394 (0.38) | +143 (1.29) | -966 (0.66) |
| confirmed | 15m | -179 (0.82) | -707 (0.73) | +1,364 (1.66) | +1,705 (1.49) |

Trades per cell: raw 40–99, confirmed 13–51.

### Aug 2–25 2026, 1m–10m, raw, net $ (PF)

| TF | MCL | MGC | MES | MNQ |
|---|---|---|---|---|
| 1m | -337 (0.62) | -234 (0.90) | -718 (0.49) | -375 (0.85) |
| 2m | -47 (0.92) | -108 (0.91) | -527 (0.34) | -416 (0.79) |
| 3m | -107 (0.63) | -575 (0.51) | -315 (0.59) | -96 (0.91) |
| 4m | -136 (0.55) | +2 (1.00) | -147 (0.75) | -162 (0.86) |
| 5m | -12 (0.95) | -278 (0.55) | +112 (1.26) | -633 (0.50) |
| 6m | -146 (0.39) | -456 (0.37) | +20 (1.05) | -675 (0.57) |
| 7m | +48 (1.29) | -11 (0.98) | -40 (0.88) | +384 (1.52) |
| 8m | -7 (0.91) | +141 (1.39) | -150 (0.59) | -94 (0.89) |
| 9m | +15 (1.11) | +132 (1.41) | -82 (0.70) | +277 (1.38) |
| 10m | +119 (2.92) | +314 (1.97) | -55 (0.77) | +642 (1.90) |

(7m–10m are 6–22 trades each.)

### Takeaways

- No timeframe/variant is positive on all four instruments. On the longer
  data, 6 of 12 raw cells and 5 of 12 confirmed cells are positive, which
  is what you'd expect from chance.
- The 5m/10m edge seen on MES doesn't carry over: MNQ 5m raw is the worst
  result (-$2.9k), MGC 10m raw loses $1k.
- 1m–4m lose on nearly every instrument, same as on MES alone.
- The one possibly interesting pattern: confirmed 15m on the two equity
  indexes, MES +$1,364 (PF 1.66) and MNQ +$1,705 (PF 1.49) over 11
  months, ~40–50 trades each. But MES and MNQ move together, so that's
  closer to one result than two, and the same rule loses on MCL/MGC.
- MCL/MGC use the equity RTH window here, not their own busiest hours
  (crude ~09:00–14:30, gold ~08:20–13:30 ET), which may hurt them.

## 15s / 30s / 45s: not tested

The repo has no sub-minute MES data (the finest file is 1m), and splitting
1m bars into seconds would mean making up prices. The runner supports it
once a file exists:
`python -m macd_cross.run_timeframes --seconds-data path/to/mes_1s_or_5s.parquet`
(any resolution that evenly divides 15s, e.g. 1s or 5s).
