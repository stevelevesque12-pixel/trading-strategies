# Futures order-flow imbalance (QuantConnect port)

`order_flow/strategy.py` ports the QuantConnect `FuturesOrderFlow` algorithm
(SMA(100) + 10-bar price entropy + bid/ask size imbalance, 1 contract, ±1%
TP/SL) so it runs on this repo's OHLCV data. See the module docstring for
the rules and the two caveats below.

```bash
pip install pyarrow   # parquet support
python -m order_flow.run                    # as written (incl. entropy-MA bug)
python -m order_flow.run --fix-entropy-ma   # MA divided by 20, as intended
```

## Caveats

- **No bid/ask sizes in our data.** The imbalance is approximated from OHLCV
  by splitting volume on close location, which reduces to `CLV > 0.75`
  (bid) / `CLV < 0.25` (ask). It doesn't measure resting quote size.
- **Bug in the original `Entropy` indicator:** `_get_entropy_ma` divides a
  20-value sum by 10, so the "MA" is 2x the real mean. `entropy > ma` almost
  never holds, so **the fade branch (close < SMA) never fires**. Every
  trade comes from "close > SMA, follow the imbalance".

## Results: NQ 1m RTH, 2022-12-26 to 2025-12-11, $14.50/RT cost

| Variant | Trades | Win % | PF | Net $ | Max DD $ | Long $ | Short $ |
|---|---|---|---|---|---|---|---|
| As written (bug) | 576 | 53.5 | 1.10 | +131k | 94k | +159k | -28k |
| Entropy-MA fixed | 639 | 51.6 | 1.02 | +32k | 89k | +123k | -91k |

Median hold is about 23 hours: a 1% NQ move takes a day or more, so this is
really an overnight swing strategy with a 1-minute trigger.

**These results show no edge beyond NQ's 2023-25 uptrend.** All the profit
comes from longs, and shorts lose in both variants. Two baselines with the
same ±1% exits and about the same trade count (200 runs each):

- random direction: mean -$6k, 95th percentile +$130k
- always long: mean +$93k (5th to 95th percentile: +$9k to +$180k)

The as-written result is at the top edge of the random-direction range and
inside the always-long range. The fixed version does worse than always-long.

Exits are checked on 1m closes only (as in QC), so overnight gaps overshoot
the 1% level. The average exit is 1.13-1.15% from entry.
