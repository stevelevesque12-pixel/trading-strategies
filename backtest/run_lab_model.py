"""
CLI entrypoint: backtest the Lab Model strategy over synchronized NQ + ES
1-minute OHLCV CSVs.

Usage:
    python -m backtest.run_lab_model --nq-data path/to/nq_1min.csv --es-data path/to/es_1min.csv
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from lab_model.strategy import LabModelStrategy

from .lab_model_engine import LabModelEngine
from .metrics import compute_metrics
from .report import write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Lab Model strategy (NQ, ES for SMT)")
    parser.add_argument("--nq-data", required=True, help="Path to NQ 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--es-data", required=True, help="Path to ES 1-minute OHLCV CSV, timestamps aligned to the same clock as --nq-data")
    parser.add_argument("--execution-tf", default="1min", choices=["1min", "3min", "5min"])
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--stop-buffer-ticks", type=int, default=4, help="Extra ticks beyond the swept/recent H/L before the stop sits")
    parser.add_argument("--exec-swing-strength", type=int, default=2, help="Fractal window (bars) confirming swings on the execution timeframe -- drives LLT/target selection")
    parser.add_argument(
        "--breakeven-at-r",
        type=float,
        default=0.25,
        help="Fraction of the way from entry to target that moves the stop to breakeven; pass a negative value to disable",
    )
    parser.add_argument("--out", default="trades_lab_model.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS["NQ"]
    strategy = LabModelStrategy(tick_size=instrument.tick_size, stop_buffer_ticks=args.stop_buffer_ticks, exec_swing_strength=args.exec_swing_strength)
    breakeven = args.breakeven_at_r if args.breakeven_at_r >= 0 else None
    engine = LabModelEngine(
        strategy=strategy,
        instrument=instrument,
        execution_tf=args.execution_tf,
        contracts=args.contracts,
        breakeven_at_r=breakeven,
    )
    trades = engine.run(args.nq_data, args.es_data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    print(f"Execution TF: {args.execution_tf}  Contracts: {args.contracts}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
