"""CLI: run all three timeframe pairs on the same data and compare results.

Usage:
    python -m backtest.compare --data path/to/1min.csv --symbol MES
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import PAIRS, Failed2sStrategy

from .engine import BacktestEngine
from .metrics import compute_metrics
from .report import print_comparison_table, write_comparison_csv, write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the Failed-2s strategy across all timeframe pairs")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV")
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=3)
    parser.add_argument("--target-r", type=float, default=1.0)
    parser.add_argument("--out-prefix", default="trades", help="Per-pair trade logs are written to <prefix>_<pair>.csv")
    parser.add_argument("--summary-out", default="pair_comparison.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    rows = []

    for pair_name, pair in PAIRS.items():
        strategy = Failed2sStrategy(tick_size=instrument.tick_size, target_r=args.target_r)
        risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
        engine = BacktestEngine(pair=pair, instrument=instrument, strategy=strategy, risk=risk, contracts=args.contracts)
        trades = engine.run(args.data)

        trades_path = f"{args.out_prefix}_{pair_name}.csv"
        write_trades_csv(trades, trades_path)

        metrics = compute_metrics(trades)
        metrics["pair"] = pair_name
        rows.append(metrics)

    print(f"Symbol: {args.symbol}  Contracts: {args.contracts}  Target R: {args.target_r}\n")
    print_comparison_table(rows)
    write_comparison_csv(rows, args.summary_out)

    print(f"\nPer-pair trade logs: {args.out_prefix}_<pair>.csv")
    print(f"Comparison summary: {args.summary_out}")


if __name__ == "__main__":
    main()
