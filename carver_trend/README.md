# Adjusted trend (Carver) on all major US futures

`quantconnect_main.py` is the QuantConnect/LEAN algorithm as given. Its `universe.py`,
`alpha.py` and `portfolio.py` modules weren't provided and LEAN's futures data isn't
reachable from here, so `backtest.py` re-implements the strategy in pandas from the
book ("Advanced Futures Trading Strategies", Rob Carver) with the same parameters:
6 EWMAC filters (2/8 ... 64/256), forecast cap 20, sigma span 32, 20% risk target,
3-year sigma blend, 10% buffer, $1M, 3-year warm-up.

"Adjusted" is taken to mean Strategy 13's volatility-regime forecast multiplier.
`--no-regime` gives the plain multi-trend version (Strategy 9) for comparison.

Data: Carver's own back-adjusted daily prices from `robcarver17/pysystemtrade`
(38 USD futures: equity indices, VIX, Treasuries, FX, energies, metals, ags), ending
2024-03-28. `fetch_data.py` rebuilds `sample_data/carver_daily/`.

    python carver_trend/backtest.py [--start 2019-03-28] [--no-regime] [--fractional]
    python carver_trend/report.py     # writes RESULTS.md and equity.png

Results: [RESULTS.md](RESULTS.md).
