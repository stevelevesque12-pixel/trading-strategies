"""
Backtest the order-flow imbalance port on the repo's datasets.

    python -m order_flow.run                         # NQ 1m, 2022-12 -> 2025-12
    python -m order_flow.run --fix-entropy-ma
    python -m order_flow.run --data <parquet> --point-value 5 --cost 1.5
"""

import argparse
from dataclasses import replace

import pandas as pd

from backtest.data import load_1m_csv
from .strategy import OrderFlowParams, run, summarize

DEFAULT_DATA = "sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--point-value", type=float, default=20.0, help="NQ=20, ES=50, YM=5, RTY=50; micros /10")
    ap.add_argument("--cost", type=float, default=14.5, help="$ per round turn (default: ~$4.50 comm + 1 tick slip each side on NQ)")
    ap.add_argument("--tp", type=float, default=0.01)
    ap.add_argument("--sl", type=float, default=0.01)
    ap.add_argument("--fix-entropy-ma", action="store_true")
    ap.add_argument("--trades-csv", default=None)
    args = ap.parse_args()

    df = load_1m_csv(args.data)
    p = replace(OrderFlowParams(), take_profit_pct=args.tp, stop_loss_pct=args.sl, fix_entropy_ma=args.fix_entropy_ma)
    trades = run(df, p)
    for k, v in summarize(trades, args.point_value, args.cost).items():
        print(f"{k:>16}: {v}")
    if args.trades_csv:
        pd.DataFrame([vars(t) | {"points": t.points} for t in trades]).to_csv(args.trades_csv, index=False)


if __name__ == "__main__":
    main()
