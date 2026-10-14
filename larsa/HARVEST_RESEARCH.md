# LARSA v14 Gear 2 ("harvest") — signal research

Reproduce: `python -m larsa.harvest_research` (needs `pyarrow`).

## Setup

- Data: `sample_data/real_multi_instrument/real_{es,nq}_15m_2016-*.parquet`, resampled to
  1h bars (and RTH daily bars for the daily rule). No YM in the repo.
- Regime: a pandas re-implementation of `detect_regime()` (EMA 12/26/50/200, Wilder ADX/ATR).
  It's close to the QC version but not identical.
- Signals act on a bar's close and fill at the **next bar's open**.
- Costs: round trip of 0.5 pt for ES and 1.0 pt for NQ, per contract.
- Periods: **train** 2016-06 to 2021 (variants were chosen by looking at this), **test** 2022 to 2024,
  **holdout** 2025-01 to 2026-08 (after the QC backtest's end date).

## Results

Cells show net points per trade / profit factor, after costs.

| Sym | Rule | Trades/wk | Train | Test | Holdout |
|---|---|---|---|---|---|
| ES | v14 as written | 3.68 | -0.63 / 0.87 | -1.03 / 0.88 | -2.87 / 0.77 |
| ES | v14 + re-arm after stop | 3.48 | -0.68 / 0.86 | -0.92 / 0.89 | -3.08 / 0.75 |
| ES | no regime gate | 7.01 | -0.29 / 0.95 | -0.22 / 0.98 | -0.81 / 0.94 |
| ES | long-only | 1.77 | -0.56 / 0.88 | -2.51 / 0.73 | -3.77 / 0.70 |
| ES | long-only, above 200h EMA | 1.24 | +0.32 / 1.08 | -2.81 / 0.70 | +1.13 / 1.14 |
| ES | RTH-only (10-15 ET), flat by 16 | 1.60 | -1.06 / 0.76 | -2.36 / 0.73 | -4.41 / 0.59 |
| ES | z 2.0 entry | 2.20 | -0.69 / 0.86 | -0.91 / 0.90 | -0.75 / 0.94 |
| ES | lookback 50 | 1.58 | -0.49 / 0.91 | -0.95 / 0.91 | -1.59 / 0.89 |
| ES | time stop 6 bars | 3.72 | -0.84 / 0.81 | -1.15 / 0.86 | -2.36 / 0.79 |
| ES | long-only, no regime gate, uptrend | 1.93 | +0.91 / 1.22 | -1.64 / 0.82 | +4.27 / 1.61 |
| ES | **Daily RSI(2)<10 dip-buy, close > SMA200, exit > SMA5** | 0.14 | +12.70 / 1.78 | +17.33 / 1.91 | +41.28 / 5.63 |
| ES | same + 3×ATR stop | 0.15 | +9.62 / 1.49 | +13.97 / 1.63 | +41.28 / 5.63 |
| NQ | v14 as written | 3.59 | -5.20 / 0.74 | -5.79 / 0.85 | -0.22 / 1.00 |
| NQ | v14 + re-arm after stop | 3.37 | -5.54 / 0.73 | -4.68 / 0.88 | -0.95 / 0.98 |
| NQ | no regime gate | 6.98 | -1.52 / 0.93 | +1.02 / 1.02 | +2.27 / 1.03 |
| NQ | long-only | 1.70 | -7.11 / 0.66 | -12.97 / 0.68 | -4.52 / 0.91 |
| NQ | long-only, above 200h EMA | 1.17 | -5.17 / 0.73 | -13.09 / 0.67 | +5.41 / 1.13 |
| NQ | RTH-only (10-15 ET), flat by 16 | 1.59 | -7.51 / 0.61 | -11.62 / 0.67 | -17.67 / 0.65 |
| NQ | z 2.0 entry | 2.24 | -5.19 / 0.75 | -13.11 / 0.71 | +1.15 / 1.02 |
| NQ | lookback 50 | 1.60 | -6.42 / 0.72 | -3.79 / 0.92 | +6.30 / 1.10 |
| NQ | time stop 6 bars | 3.62 | -5.31 / 0.72 | -5.14 / 0.85 | -2.01 / 0.96 |
| NQ | long-only, no regime gate, uptrend | 1.89 | +1.00 / 1.06 | -10.48 / 0.76 | +16.46 / 1.36 |
| NQ | **Daily RSI(2)<10 dip-buy, close > SMA200, exit > SMA5** | 0.14 | +42.14 / 2.01 | +116.79 / 2.82 | +105.40 / 2.03 |
| NQ | same + 3×ATR stop | 0.15 | +17.44 / 1.27 | +116.27 / 2.80 | +120.93 / 2.39 |

## Findings

1. **The hourly z-score fade has no edge.** Every hourly variant loses money in at least one
   period on at least one instrument. None is positive across all three periods on both. The
   SIDEWAYS gate makes it worse, not better. SIDEWAYS covers about 30% of hours, not 80%.
2. **The daily RSI(2) dip-buy is the only rule that's positive everywhere.** It's positive in
   all three periods on both instruments. It also holds when the parameters move: RSI below
   5/10/15/25 with an exit above SMA 3 or 5 is positive in all 24 period-instrument cells. An
   exit above SMA 10 turns negative in train. This is a long-published rule (Connors), not one
   fitted to this data.
3. **Stops don't help it.** 2, 3 and 4×ATR stops cut the edge in most cells. The worst losses
   (NQ −1442 pts, early 2020) gap through any stop. Risk control has to come from position size.
4. **It's small and infrequent.** About 7–10 trades per year per instrument, held around
   5 days, so it's in the market about 15% of the time. Per contract it made about $5–8k/yr (ES)
   and $6–23k/yr (NQ). It's long equity beta in uptrends, so it is **not** anti-correlated
   with Gear 1.

## Expected size in v14

`larsa_v14.py` sizes each leg at `0.01 / corr_factor / daily_vol` of harvest equity, about
0.25–0.35× harvest equity per instrument while in a trade, capped at 0.5×. On $19M of harvest
equity that's roughly 15–25 ES-equivalent contracts per instrument. Scaling the per-contract
numbers above gives about **3–6% a year on the harvest bucket**, not the 20–30% in the original
v14 header. A tested signal for that higher return still doesn't exist.

## Not verified here

- The QC algorithm itself. `larsa_v14.py` was only syntax-checked locally because
  `AlgorithmImports` isn't available. Run it in QuantConnect before trusting it.
- QC builds daily bars for futures over the full session, while this research used RTH daily
  bars. Expect small differences.
- YM wasn't tested because there's no YM data in the repo.
