# Morning Dip Limit

- `morning_dip_limit.py`: the reference implementation. It expects 1-second NQ bars, one file per day.
- `backtest_1m.py`: runs the reference on the 3-year NQ 1-minute file, using its `resolution=60` mode.
- `../tradingview/morning_dip_limit.pine`: the TradingView port, which also sends TradersPost alerts.

The rule: on 3-minute candles between 09:00 and 11:00 CT, when ER(15) <= 0.35, rest a buy limit at
close - 1 x ATR(14). Fills need price to trade 1 tick through. The target is the signal candle's
midpoint and the stop is 1.5 x ATR below the entry. Orders cancel after 9 minutes, positions exit
after 15 minutes, and everything is flat at 11:00 CT. The full spec is in the docstring of
`morning_dip_limit.py`.

## Backtest on 1-minute NQ (2022-12-27 to 2025-12-11, 765 sessions, 1 contract)

There's no 1-second history here, so this is coarser than the rule as specified. The 1 s order
latency is dropped, and fills and exits are resolved on 1-minute bars, with the adverse move first.
Costs are $2.25 per side and 1 tick of slippage on stops and market exits.

| Grid offset | Fills | Net $ | PF | Win % | Max DD $ | Net without top 3 days |
|---|---|---|---|---|---|---|
| 0 s   | 1786 | +5,923  | 1.02 | 58 | 21,407 | -2,915  |
| 60 s  | 1752 | -31,434 | 0.92 | 57 | 45,818 | -41,469 |
| 120 s | 1765 | -29,333 | 0.93 | 57 | 43,925 | -41,872 |

Net by year at offset 0: 2023 -$1,991, 2024 -$581, 2025 +$9,000.

Read: about 2.3 fills per session, and 28-29% of orders fill. The average target is about
+$440 and the average stop about -$750. The small profit at offset 0 comes from a few days and
is gone at the 60 s and 120 s offsets. On this data the rule has no edge that survives a change
in candle phase. A 1-second test could still differ: 1-minute bars can't resolve which of the
target and stop came first inside a minute.

Reproduce (needs `pyarrow`; `numba` is optional and makes it faster):

```
python -m morning_dip.backtest_1m --offsets 0 60 120 --out trades/
```
