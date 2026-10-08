"""
Sweep the LH/LL strategy over hold periods 1..N on real data.

    python -m lhll.run --data sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet \
        --timeframe 1D --instrument NQ
    python -m lhll.run --data sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet \
        --timeframe 5min --instrument NQ

Timeframe '1D' builds daily bars from the regular session (09:30-16:00 ET)
and lets trades hold across days. Any intraday timeframe also uses RTH bars
only and never holds overnight (see lhll.strategy).
"""

import argparse

import pandas as pd

from backtest.data import load_1m_csv
from failed2s.instruments import INSTRUMENTS
from lhll.strategy import backtest_lhll, baseline_avg_points, summarize


def build_bars(raw: pd.DataFrame, timeframe: str) -> tuple:
    rth = raw.between_time("09:30", "15:59")
    if timeframe.upper() in ("1D", "D", "DAILY"):
        agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        daily = rth.groupby(rth.index.date).agg(agg).dropna()
        daily.index = pd.DatetimeIndex(daily.index)
        return daily, None
    # offset so bins start at :30 (an hourly bar is 09:30-10:30, not 09:00-10:00)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    bars = rth.resample(timeframe, label="left", closed="left", offset="30min").agg(agg).dropna()
    return bars, pd.Series(bars.index.date, index=bars.index)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--timeframe", default="1D", help="1D, or a pandas rule like 5min/15min/1h")
    p.add_argument("--direction", choices=["long", "short"], default="long")
    p.add_argument("--max-hold", type=int, default=10)
    p.add_argument("--cost", type=float, default=0.0, help="round-trip cost in points per trade")
    p.add_argument("--instrument", default="NQ")
    p.add_argument("--start", help="only use bars on/after this date")
    args = p.parse_args(argv)

    raw = load_1m_csv(args.data)
    if args.start:
        raw = raw[raw.index >= pd.Timestamp(args.start, tz=raw.index.tz)]
    bars, session = build_bars(raw, args.timeframe)
    point_value = INSTRUMENTS[args.instrument].point_value

    print(f"{args.data}\n{args.timeframe} bars: {len(bars)}  {bars.index[0]} -> {bars.index[-1]}  "
          f"direction={args.direction} cost={args.cost}pts {args.instrument}=${point_value}/pt\n")
    rows = []
    for n in range(1, args.max_hold + 1):
        trades = backtest_lhll(bars, n, args.direction, args.cost, session)
        base = baseline_avg_points(bars, n, args.direction, session) - args.cost
        s = summarize(trades, point_value, base)
        rows.append({"hold": n, **s})
    out = pd.DataFrame(rows).set_index("hold")
    print(out.drop(columns=["total_$"]).to_string())


if __name__ == "__main__":
    main()
