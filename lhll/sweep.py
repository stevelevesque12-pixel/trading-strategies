"""
Run the LH/LL strategy over many intraday timeframes x hold periods 1..N
and print one matrix per statistic (rows = timeframe, columns = hold bars).

    python -m lhll.sweep --data sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet
"""

import argparse

import pandas as pd

from backtest.data import load_1m_csv
from lhll.run import build_bars
from lhll.strategy import backtest_lhll, baseline_avg_points, summarize

DEFAULT_TFS = "1min,2min,3min,5min,10min,15min,20min,30min,45min,1h,90min,2h,3h"


def sweep(raw, timeframes, max_hold, direction="long", cost=0.0, consecutive=1):
    rows = []
    for tf in timeframes:
        bars, session = build_bars(raw, tf)
        for n in range(1, max_hold + 1):
            trades = backtest_lhll(bars, n, direction, cost, session, consecutive)
            base = baseline_avg_points(bars, n, direction, session) - cost
            s = summarize(trades, 1.0, base)
            rows.append({"tf": tf, "hold": n, **s})
    return pd.DataFrame(rows)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--timeframes", default=DEFAULT_TFS)
    p.add_argument("--max-hold", type=int, default=10)
    p.add_argument("--direction", choices=["long", "short"], default="long")
    p.add_argument("--consecutive", type=int, default=1, help="enter on the Nth LH/LL bar in a row")
    p.add_argument("--cost", type=float, default=0.0, help="round-trip cost in points per trade")
    args = p.parse_args(argv)

    raw = load_1m_csv(args.data)
    tfs = args.timeframes.split(",")
    df = sweep(raw, tfs, args.max_hold, args.direction, args.cost, args.consecutive)
    print(f"{args.data}  {raw.index[0].date()} -> {raw.index[-1].date()}  "
          f"direction={args.direction} consecutive={args.consecutive} cost={args.cost}pts\n")
    for col, label in [("trades", "Trades"), ("avg_pts", "Avg points per trade"),
                       ("pf", "Profit factor"), ("t_stat", "t-stat of avg trade"),
                       ("edge_vs_base_pts", "Edge vs drift (points)"), ("edge_t", "t-stat of edge vs drift")]:
        m = df.pivot(index="tf", columns="hold", values=col).reindex(tfs)
        print(f"{label}\n{m.to_string()}\n")


if __name__ == "__main__":
    main()
