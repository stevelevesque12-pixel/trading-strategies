# SuperTrend / ATR strategy

Trend-following with pending STOP-order entries, an ATR stop and an ATR trailing stop.

| Component | Implementation |
|---|---|
| Entry logic | SuperTrend(10, 3.0) flips direction on the entry timeframe |
| Market-structure filter | Higher-timeframe SuperTrend (4x the entry TF) must agree |
| Volatility filter | ATR(14) / SMA(ATR, 100) must be at least 0.8 (optional max) |
| Order type | Pending stop at the flip bar's high + 1 tick (low - 1 tick for shorts). Expires after 3 bars, or is cancelled if SuperTrend flips back. |
| Stop-loss | Fill price - **4.2** x ATR at the signal |
| Trailing stop | Once the trade is 0.6 ATR in profit, trail the best price by **0.6** x current ATR. Only tightens. |
| Profit target | None |

The trail starts only after the trade is 0.6 ATR in profit, the way a classic
MetaTrader trailing stop works. If it started on the entry bar, a 0.6 ATR trail
would replace the 4.2 ATR stop right away and the 4.2 would never matter. Use
`--trail-activation 0` to start the trail immediately.

Files: `indicators.py` (streaming ATR/SuperTrend that match TradingView's
`ta.atr`/`ta.supertrend`), `strategy.py` (rules), `backtest.py` (fill
simulation + CLI). The TradingView port is `tradingview/supertrend_atr.pine`,
and the tests are in `tests/test_supertrend_atr.py`.

```
python -m supertrend_atr.backtest \
  --data sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet \
  --symbol NQ --compare 5min,15min,30min,1h --session rth      # or --session full
```

Fill model: bar OHLC only. A stop that gaps fills at the open. Each stop fill
(entry or exit) costs 1 tick of slippage, and every trade pays $4 round-trip
commission. If the entry and the initial stop are both inside the fill bar's
range, the trade is counted as stopped out. The trail is recalculated at each
bar's close and applies from the next bar.

## Results: NQ 1-minute data, Dec 2022 - Dec 2025, 1 contract, net of costs

Default parameters, nothing optimized.

| Mode | TF | Trades | Win % | PF | Net $ | Max DD $ |
|---|---|---|---|---|---|---|
| RTH (flatten 15:55) | 5m | 380 | 67.6 | 0.91 | -8,930 | 16,307 |
| RTH | 15m | 221 | 67.9 | 1.10 | +9,048 | 17,722 |
| RTH | 30m | 160 | 61.3 | 0.83 | -15,722 | 28,404 |
| RTH | 1h | 59 | 59.3 | 1.00 | -126 | 21,865 |
| 24h, hold overnight | 5m | 1054 | 68.6 | 0.96 | -8,356 | 15,019 |
| 24h | 15m | 355 | 69.3 | 0.95 | -8,138 | 27,785 |
| 24h | 30m | 241 | 70.5 | 1.21 | +21,177 | 14,092 |
| 24h | 1h | 128 | 71.9 | 1.50 | +32,082 | 23,146 |
| 24h | 4h | 32 | 62.5 | 0.82 | -5,661 | 25,439 |

**There's no reliable edge here.** Most trades win, but the losses are large:
the stop is 4.2 ATR away and the trail sits 0.6 ATR behind, so the average loss
is about 2x the average win. Profitability flips between neighboring
timeframes. The best run (24h, 1h) lost $8.1k in 2023 and made $9.5k in 2024
and $30.7k in 2025. Without the HTF and volatility filters, RTH 5m drops to
-$67k (PF 0.82), so the filters do cut out a lot of bad trades. Don't trade
this live without walk-forward testing and forward-testing first.
