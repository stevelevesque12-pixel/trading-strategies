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

## Against the author's trade lists (Jan–Sep 2026, 1-second data)

The author ran 3-minute candles at offsets 0–162 s (in 18 s steps) and 5-minute candles at offsets
0–240 s. The 5-minute runs also scale the timers: orders cancel after 15 minutes and positions exit
after 25 minutes (`--tf 300 --ttl 900 --hold 1500`).

His results: 3-minute, every offset profitable, PF 1.14–1.45 (net +$17.6k to +$53.1k per offset).
5-minute, PF 1.51–2.08 (net +$23.9k to +$46.6k).

**The port matches his trades.** On 2026-08-03 to 08-25, using MNQ 1-minute bars (same price
scale as NQ), the 3-minute offset-0 run gives the same signal times as his trades. Limit, stop and
target prices are within a tick or two, and ATR/ER nearly match. At 5 minutes, 60 of his 70
trades from those dates line up across the five offsets. The remaining differences come from
resolving fills and exits on 1-minute rather than 1-second bars, and they go both ways (3 min, offset
0: +$721 here vs −$901 for him).

**The difference is the period, not the code.** The same rules on 2023–2025 NQ:

| Run | Offset | Fills | Net $ | PF | 2023 | 2024 | 2025 |
|---|---|---|---|---|---|---|---|
| 5 min | 0 s   | 676 | +28     | 1.00 | +5,891 | −1,826 | −2,978  |
| 5 min | 60 s  | 677 | +12,778 | 1.08 | +8,254 | −3,065 | +8,843  |
| 5 min | 120 s | 631 | −4,890  | 0.97 | +5,447 | +488   | −10,466 |
| 5 min | 180 s | 659 | −4,290  | 0.97 | +2,462 | −1,882 | −4,408  |
| 5 min | 240 s | 653 | +1,532  | 1.01 | +6,256 | −4,040 | −26     |

Combined with the 3-minute table above, 2023–2025 is about break-even before any edge. The 2026
profits come from a single nine-month period of high NQ volatility. That is a regime result until
it shows up in other years.

Reproduce (needs `pyarrow`; `numba` is optional and makes it faster):

```
python -m morning_dip.backtest_1m --offsets 0 60 120 --out trades/
```
