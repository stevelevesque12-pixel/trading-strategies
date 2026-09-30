"""CLI: backtest the RSI 80-20 strategy across one or more entry timeframes.

Usage:
    python -m backtest.run_rsi8020 --data path/to/1min.csv --symbol MES
    python -m backtest.run_rsi8020 --data path/to/5min.parquet --timeframes 5min,15min --target-r 1.5
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import TimeframePair
from rsi_8020.strategy import RSI8020Strategy

from .engine import BacktestEngine
from .metrics import compute_metrics
from .report import print_comparison_table, write_comparison_csv, write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the RSI 80-20 strategy")
    parser.add_argument("--data", required=True, help="Path to OHLCV CSV/parquet (resolution must be <= the finest timeframe)")
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--timeframes", default="1min,5min,15min", help="Comma-separated pandas rules, e.g. 5min,15min")
    parser.add_argument("--rsi-length", type=int, default=14)
    parser.add_argument("--overbought", type=float, default=80.0)
    parser.add_argument("--oversold", type=float, default=20.0)
    parser.add_argument("--lookback", type=int, default=50, help="Candles for the first low/high (default 50)")
    parser.add_argument("--max-setup-bars", type=int, default=50, help="Drop a setup this many candles after its first low/high")
    parser.add_argument("--stop-buffer-ticks", type=int, default=2)
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=3)
    parser.add_argument("--target-r", type=float, default=3.0)
    parser.add_argument("--out-prefix", default="trades_rsi8020", help="Per-timeframe trade logs go to <prefix>_<tf>.csv")
    parser.add_argument("--summary-out", default="rsi8020_comparison.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    rows = []

    for tf in [t.strip() for t in args.timeframes.split(",") if t.strip()]:
        strategy = RSI8020Strategy(
            tick_size=instrument.tick_size,
            rsi_length=args.rsi_length,
            overbought=args.overbought,
            oversold=args.oversold,
            lookback=args.lookback,
            max_setup_bars=args.max_setup_bars,
            stop_buffer_ticks=args.stop_buffer_ticks,
            target_r=args.target_r,
        )
        risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
        pair = TimeframePair(f"rsi-{tf}", tf, tf)
        engine = BacktestEngine(pair=pair, instrument=instrument, strategy=strategy, risk=risk, contracts=args.contracts)
        trades = engine.run(args.data)

        write_trades_csv(trades, f"{args.out_prefix}_{tf}.csv")
        metrics = compute_metrics(trades)
        metrics["pair"] = tf
        rows.append(metrics)

    print(
        f"Symbol: {args.symbol}  RSI({args.rsi_length}) {args.oversold:g}/{args.overbought:g}  "
        f"Contracts: {args.contracts}  Target R: {args.target_r}\n"
    )
    print_comparison_table(rows)
    write_comparison_csv(rows, args.summary_out)
    print(f"\nPer-timeframe trade logs: {args.out_prefix}_<tf>.csv")
    print(f"Comparison summary: {args.summary_out}")


if __name__ == "__main__":
    main()
