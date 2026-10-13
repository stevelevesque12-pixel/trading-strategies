# Micro-only low-vol futures portfolio: backtest results

Run: `python -m risk_parity.run --sweep` (needs `pyarrow` for the parquet files).
Prop rules simulated: $50K Phidias-style swing evaluation with a +$4,000 target and a
$2,500 end-of-day trailing drawdown, starting a new evaluation every 5 sessions.
Costs: $1.25/side/contract plus 1 tick of slippage per side.

Default config: MES/MNQ/MCL/MGC, long only above the 100-day SMA, one equity slot,
$1,000 risk per market at a 2x daily-ATR stop, $2,000 max total open risk,
$1,000 weekly portfolio stop, rebalanced Mondays 10:00 ET.

| | micro (real MES/MNQ/MCL/MGC, Sep 2025-Aug 2026) | proxy (ES/NQ/GC at micro size, 2016-2026, no crude) |
|---|---|---|
| Sessions traded (after warm-up) | 131 | 2,548 |
| $ / year | ~$10.8K | ~$5.9K |
| Sharpe (daily) | 2.09 | 1.27 |
| Max EOD drawdown | $3,358 | $5,961 |
| Worst week | -$1,040 | -$1,415 |
| Weeks invested | 33% | 85% |
| Eval pass / fail / still open | 7% / 48% / 44% | 43% / 55% / 2% |
| Median sessions to pass | 7.5 | 63 |

Proxy P&L by year: 2017 +8.2K, 2018 +6.8K, 2019 +6.3K, 2020 +9.4K, 2021 -0.8K,
2022 -1.0K, 2023 +7.3K, 2024 +11.0K, 2025 +12.8K, 2026 YTD -0.3K.

Main constraint: at 2026 volatility, one micro with a 2x ATR stop risks about $800 (MES, MCL)
or $2,100 (MNQ, MGC). Inside a $2,500 trailing drawdown the portfolio can only hold
about 1-2 micros, so MNQ and MGC were almost never affordable in the micro run.
The 11-month micro sample is too short to trust; its first ~100 sessions are
trend-filter warm-up.
