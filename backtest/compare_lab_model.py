"""
CLI: run the Lab Model strategy across execution timeframes (1m/3m/5m) and
compare results -- this is "which chart Kane would pick for the iFVG" made
into an empirical comparison instead of a discretionary, bar-by-bar choice.

Usage:
    python -m backtest.compare_lab_model --nq-data path/to/nq_1min.csv --es-data path/to/es_1min.csv
    python -m backtest.compare_lab_model --nq-data ... --es-data ... --execution-tfs 1min,5min
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from lab_model.strategy import LabModelStrategy

from .lab_model_engine import LabModelEngine
from .metrics import compute_metrics
from .report import print_comparison_table, write_comparison_csv, write_trades_csv

ALL_EXECUTION_TFS = ["1min", "3min", "5min"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare the Lab Model strategy across execution timeframes")
    parser.add_argument("--nq-data", required=True, help="Path to NQ 1-minute OHLCV CSV")
    parser.add_argument("--es-data", required=True, help="Path to ES 1-minute OHLCV CSV, aligned to the same clock as --nq-data")
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument(
        "--breakeven-at-r",
        type=float,
        default=0.5,
        help="Fraction of the way from entry to target that moves the stop to breakeven; pass a negative value to disable",
    )
    parser.add_argument("--out-prefix", default="trades_lab_model", help="Per-timeframe trade logs are written to <prefix>_<tf>.csv")
    parser.add_argument("--summary-out", default="lab_model_tf_comparison.csv")
    parser.add_argument(
        "--execution-tfs", default=None,
        help=f"Comma-separated subset of execution timeframes to run (default: all). Choices: {','.join(ALL_EXECUTION_TFS)}",
    )
    args = parser.parse_args()

    if args.execution_tfs:
        tfs = [t.strip() for t in args.execution_tfs.split(",")]
        unknown = [t for t in tfs if t not in ALL_EXECUTION_TFS]
        if unknown:
            parser.error(f"Unknown execution timeframe(s) {unknown}; choices are {ALL_EXECUTION_TFS}")
    else:
        tfs = list(ALL_EXECUTION_TFS)

    instrument = INSTRUMENTS["NQ"]
    breakeven = args.breakeven_at_r if args.breakeven_at_r >= 0 else None
    rows = []

    for tf in tfs:
        strategy = LabModelStrategy(tick_size=instrument.tick_size)
        engine = LabModelEngine(
            strategy=strategy, instrument=instrument, execution_tf=tf, contracts=args.contracts, breakeven_at_r=breakeven
        )
        trades = engine.run(args.nq_data, args.es_data)

        trades_path = f"{args.out_prefix}_{tf}.csv"
        write_trades_csv(trades, trades_path)

        metrics = compute_metrics(trades)
        metrics["pair"] = tf  # reuses report.py's generic "pair" column for the timeframe label
        rows.append(metrics)

    print(f"Symbol: NQ  Contracts: {args.contracts}  Breakeven at: {breakeven}\n")
    print_comparison_table(rows)
    write_comparison_csv(rows, args.summary_out)

    print(f"\nPer-timeframe trade logs: {args.out_prefix}_<tf>.csv")
    print(f"Comparison summary: {args.summary_out}")


if __name__ == "__main__":
    main()
