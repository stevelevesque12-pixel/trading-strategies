"""CLI entrypoint: backtest the Hurst-gated VWAP band-fade strategy over a 1-minute OHLCV CSV.

Usage:
    python -m backtest.run_hurst_vwap --data path/to/1min.csv --symbol MES
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from hurst_vwap.strategy import HurstVWAPStrategy

from .hurst_vwap_engine import HurstVWAPBacktestEngine
from .metrics import compute_metrics
from .report import write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Hurst-gated VWAP band-fade strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--entry-sd", type=float, default=2.5, help="VWAP stdev band that triggers an entry")
    parser.add_argument("--stop-sd", type=float, default=3.0, help="VWAP stdev band the hard stop sits just outside of (must be > entry-sd)")
    parser.add_argument("--stop-buffer-ticks", type=int, default=2)
    parser.add_argument("--hurst-window", type=int, default=30, help="Rolling Hurst window, in bars (minutes, since this runs at 1-minute resolution)")
    parser.add_argument("--hurst-threshold", type=float, default=0.5, help="Only trade while the rolling Hurst estimate is below this (range-bound regime)")
    parser.add_argument("--warmup-bars", type=int, default=20)
    parser.add_argument("--min-stdev-ticks", type=float, default=2.0)
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=6)
    parser.add_argument("--out", default="trades.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    strategy = HurstVWAPStrategy(
        tick_size=instrument.tick_size,
        entry_sd=args.entry_sd,
        stop_sd=args.stop_sd,
        stop_buffer_ticks=args.stop_buffer_ticks,
        hurst_window=args.hurst_window,
        hurst_threshold=args.hurst_threshold,
        warmup_bars=args.warmup_bars,
        min_stdev_ticks=args.min_stdev_ticks,
    )
    engine = HurstVWAPBacktestEngine(instrument=instrument, strategy=strategy, risk=risk, contracts=args.contracts)
    trades = engine.run(args.data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    print(f"Symbol: {args.symbol}  Contracts: {args.contracts}  Hurst window: {args.hurst_window}min  Hurst threshold: {args.hurst_threshold}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
