# Carver fast trend following (long/short, trend strength): local backtest

This is a port of the QuantConnect algorithm `FuturesFastTrendFollowingLongAndShortWithTrendStrenthAlgorithm`
with `slow_ema_span_exponent = 7` (QC default is 6) and otherwise default parameters: EWMAC 32/128, forecast scalar 2.79,
cap +/-20, sigma span 32, 70/30 blend with the 3-year average vol,
tau 0.2, buffer 0.1, IDM from Carver Table 16, $1M starting capital (compounding).
It runs on daily session bars built from the repo's 15-minute data. A signal at day t's close fills at
day t+1's open. Costs are commission plus 1 tick of slippage per side, plus an estimated roll cost.

Regenerate with `python -m carver_trend.run_all`.

**Coverage.** "Every major future" isn't possible here because the environment has no
QuantConnect/LEAN and no outside data access. The tests use every futures series in `sample_data/`:
ES, NQ, GC and SI with 10 years of data, and MES, MNQ, MYM, M2K, MGC, SIL and MCL with about 11 months.
Rates (ZN/ZB), FX (6E/6J), grains, softs and full-size energies aren't in the repo. Running the
original algorithm on QuantConnect is still the way to get the full universe.

**Data caveat.** The continuous series look unadjusted (not Panama back-adjusted the way QC is), so
roll gaps show up as small fake returns.

![equity](equity_curves.png)

### Contract specs used

| symbol   | class        |   multiplier |   tick |   cost/side $ |   rolls/yr |
|:---------|:-------------|-------------:|-------:|--------------:|-----------:|
| ES       | Equity index |         50   |  0.25  |         14.75 |          4 |
| NQ       | Equity index |         20   |  0.25  |          7.25 |          4 |
| GC       | Metals       |        100   |  0.1   |         12.25 |          6 |
| SI       | Metals       |       5000   |  0.005 |         27.25 |          5 |
| MES      | Equity index |          5   |  0.25  |          1.87 |          4 |
| MNQ      | Equity index |          2   |  0.25  |          1.12 |          4 |
| MYM      | Equity index |          0.5 |  1     |          1.12 |          4 |
| M2K      | Equity index |          5   |  0.1   |          1.12 |          4 |
| MGC      | Metals       |         10   |  0.1   |          1.62 |          6 |
| SIL      | Metals       |       1000   |  0.005 |          5.62 |          5 |
| MCL      | Energy       |        100   |  0.01  |          1.62 |         12 |


## 1. Full history (2016 data, trading after the QC 3-year warm-up)

Each instrument alone (IDM 1, weight 1) and as the 4-instrument portfolio (IDM 1.56, weight 0.25).

|                         | start      | end        |   years |   total_return_pct |   cagr_pct |   ann_vol_pct |   sharpe |   max_dd_pct |   calmar |   costs_usd |   contracts_per_year |   pct_days_long |
|:------------------------|:-----------|:-----------|--------:|-------------------:|-----------:|--------------:|---------:|-------------:|---------:|------------:|---------------------:|----------------:|
| ES                      | 2019-10-14 | 2026-08-25 |    6.86 |               51.9 |       6.28 |          23.5 |     0.37 |        -32.8 |     0.19 |       13799 |                   74 |            78.8 |
| NQ                      | 2019-10-14 | 2026-08-25 |    6.86 |               99.2 |      10.56 |          24.5 |     0.52 |        -29.9 |     0.35 |        4656 |                   49 |            79.9 |
| GC                      | 2019-10-08 | 2026-08-25 |    6.88 |              233.5 |      19.13 |          26.4 |     0.78 |        -49.9 |     0.38 |       12089 |                   59 |            77.3 |
| SI                      | 2019-10-08 | 2026-08-25 |    6.88 |               14.8 |       2.03 |          20.7 |     0.2  |        -49.7 |     0.04 |       10732 |                   30 |            72.1 |
| PORTFOLIO (ES+NQ+GC+SI) | 2019-10-08 | 2026-08-25 |    6.88 |              150.2 |      14.26 |          25.9 |     0.63 |        -34.9 |     0.41 |       18484 |                   90 |            77.1 |

Portfolio gross P&L by instrument (USD): ES +133,000, NQ +227,000, GC +852,000, SI +308,000

## 2. QC's window: last 5 years (2021-08-25 -> 2026-08-25)

|                         | start      | end        |   years |   total_return_pct |   cagr_pct |   ann_vol_pct |   sharpe |   max_dd_pct |   calmar |   costs_usd |   contracts_per_year |   pct_days_long |
|:------------------------|:-----------|:-----------|--------:|-------------------:|-----------:|--------------:|---------:|-------------:|---------:|------------:|---------------------:|----------------:|
| ES                      | 2021-08-25 | 2026-08-25 |       5 |               11   |       2.11 |          21.9 |     0.2  |        -28.3 |     0.07 |        6388 |                   48 |            75.5 |
| NQ                      | 2021-08-25 | 2026-08-25 |       5 |                9.2 |       1.77 |          22.6 |     0.19 |        -28.6 |     0.06 |        1752 |                   26 |            74.7 |
| GC                      | 2021-08-25 | 2026-08-25 |       5 |              195.5 |      24.2  |          26.6 |     0.93 |        -33.6 |     0.72 |        7240 |                   48 |            77.4 |
| SI                      | 2021-08-25 | 2026-08-25 |       5 |               22.3 |       4.11 |          20.9 |     0.29 |        -42.6 |     0.1  |        6988 |                   26 |            67.5 |
| PORTFOLIO (ES+NQ+GC+SI) | 2021-08-25 | 2026-08-25 |       5 |               81.1 |      12.61 |          25.4 |     0.58 |        -40.8 |     0.31 |        8145 |                   53 |            73.8 |

## 3. Micro futures: adds Dow (MYM), Russell (M2K) and crude oil (MCL)

Only about 11 months of data, so the warm-up is 120 calendar days instead of 3 years, and the vol blend uses the history available so far. Treat these as **indicative only**: a few months of trend-following P&L is noise.

|                      | start      | end        |   years |   total_return_pct |   cagr_pct |   ann_vol_pct |   sharpe |   max_dd_pct |   calmar |   costs_usd |   contracts_per_year |   pct_days_long |
|:---------------------|:-----------|:-----------|--------:|-------------------:|-----------:|--------------:|---------:|-------------:|---------:|------------:|---------------------:|----------------:|
| MES                  | 2026-01-29 | 2026-08-25 |    0.57 |               -4.2 |      -7.18 |          19.2 |    -0.28 |        -11.6 |    -0.62 |         672 |                  407 |            82.6 |
| MNQ                  | 2026-01-29 | 2026-08-25 |    0.57 |               -8   |     -13.65 |          23.8 |    -0.48 |        -18.5 |    -0.74 |         180 |                  193 |            71.1 |
| MYM                  | 2026-01-29 | 2026-08-25 |    0.57 |               -1.2 |      -2.16 |          17   |    -0.04 |        -11.1 |    -0.2  |         552 |                  544 |            85.2 |
| M2K                  | 2026-01-29 | 2026-08-25 |    0.57 |                0.2 |       0.32 |          18.4 |     0.11 |         -7.3 |     0.04 |         716 |                  620 |            95.3 |
| MGC                  | 2026-01-29 | 2026-08-25 |    0.57 |              -24   |     -38.31 |          20.2 |    -2.23 |        -24.1 |    -1.59 |         232 |                  146 |            58.1 |
| SIL                  | 2026-01-29 | 2026-08-25 |    0.57 |              -19.2 |     -31.28 |          20.2 |    -1.7  |        -19.4 |    -1.61 |         176 |                   30 |            64.2 |
| MCL                  | 2026-01-29 | 2026-08-25 |    0.57 |                0.2 |       0.41 |          17.2 |     0.11 |        -16.2 |     0.03 |         876 |                  330 |            75.7 |
| PORTFOLIO (7 micros) | 2026-01-29 | 2026-08-25 |    0.57 |              -15.6 |     -25.79 |          24   |    -1.08 |        -17.3 |    -1.49 |         951 |                  643 |            76.1 |

## 4. Sensitivity: `slow_ema_span_exponent` 4-7 (ES+NQ+GC+SI portfolio, full history)

|              | start      | end        |   years |   total_return_pct |   cagr_pct |   ann_vol_pct |   sharpe |   max_dd_pct |   calmar |   costs_usd |   contracts_per_year |   pct_days_long |
|:-------------|:-----------|:-----------|--------:|-------------------:|-----------:|--------------:|---------:|-------------:|---------:|------------:|---------------------:|----------------:|
| EWMAC 4/16   | 2019-06-18 | 2026-08-25 |    7.19 |              -16.7 |      -2.51 |          27.5 |     0.05 |        -59   |    -0.04 |       52777 |                  379 |            63   |
| EWMAC 8/32   | 2019-07-04 | 2026-08-25 |    7.14 |               40.8 |       4.91 |          26.7 |     0.31 |        -57.5 |     0.09 |       31437 |                  199 |            65.8 |
| EWMAC 16/64  | 2019-08-05 | 2026-08-25 |    7.06 |               70.9 |       7.89 |          25.7 |     0.42 |        -54.8 |     0.14 |       21378 |                  119 |            70.8 |
| EWMAC 32/128 | 2019-10-08 | 2026-08-25 |    6.88 |              150.2 |      14.26 |          25.9 |     0.63 |        -34.9 |     0.41 |       18484 |                   90 |            77.1 |
