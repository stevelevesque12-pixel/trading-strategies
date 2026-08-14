"""CLI: run the volume-confirmed scalping strategy across bar timeframes and compare results.

Usage:
    python -m backtest.compare_volume_scalp --data sample_data/sample_1min.csv --symbol MES
    python -m backtest.compare_volume_scalp --data sample_data/sample_1min.csv --symbol MNQ --timeframes 1min,3min,5min
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager

from .metrics import compute_metrics
from .report import print_comparison_table, write_comparison_csv, write_trades_csv
from .run_volume_scalp import add_common_args, build_strategy
from .volume_engine import VolumeScalpBacktestEngine

DEFAULT_TIMEFRAMES = ["1min", "3min", "5min"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the volume-confirmed scalping strategy across bar timeframes")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close,volume)")
    add_common_args(parser)
    parser.add_argument(
        "--timeframes", default=None,
        help=f"Comma-separated bar timeframes to compare (default: {','.join(DEFAULT_TIMEFRAMES)})",
    )
    parser.add_argument("--out-prefix", default="trades_volume_scalp", help="Per-timeframe trade logs are written to <prefix>_<timeframe>.csv")
    parser.add_argument("--summary-out", default="volume_scalp_timeframe_comparison.csv")
    args = parser.parse_args()

    timeframes = [t.strip() for t in args.timeframes.split(",")] if args.timeframes else DEFAULT_TIMEFRAMES

    instrument = INSTRUMENTS[args.symbol]
    rows = []

    for timeframe in timeframes:
        strategy = build_strategy(instrument, args)
        risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
        engine = VolumeScalpBacktestEngine(
            instrument=instrument, strategy=strategy, risk=risk, timeframe=timeframe, contracts=args.contracts
        )
        trades = engine.run(args.data)

        trades_path = f"{args.out_prefix}_{timeframe}.csv"
        write_trades_csv(trades, trades_path)

        metrics = compute_metrics(trades)
        metrics["pair"] = timeframe  # reuse the "pair" column name so report.py's fixed field list still applies
        rows.append(metrics)

    print(f"Symbol: {args.symbol}  Contracts: {args.contracts}  Target R: {args.target_r}\n")
    print_comparison_table(rows)
    write_comparison_csv(rows, args.summary_out)

    print(f"\nPer-timeframe trade logs: {args.out_prefix}_<timeframe>.csv")
    print(f"Comparison summary: {args.summary_out}")


if __name__ == "__main__":
    main()
