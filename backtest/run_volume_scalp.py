"""CLI entrypoint: backtest the volume-confirmed scalping strategy over a 1-minute OHLCV CSV.

Usage:
    python -m backtest.run_volume_scalp --data sample_data/sample_1min.csv --symbol MES
    python -m backtest.run_volume_scalp --data sample_data/sample_1min.csv --symbol MNQ --timeframe 5min

To compare multiple bar timeframes on the same data, use
`python -m backtest.compare_volume_scalp` instead.
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from volume_scalp.strategy import VolumeScalpStrategy

from .metrics import compute_metrics
from .report import write_trades_csv
from .volume_engine import VolumeScalpBacktestEngine


def build_strategy(instrument, args) -> VolumeScalpStrategy:
    return VolumeScalpStrategy(
        tick_size=instrument.tick_size,
        stop_buffer_ticks=args.stop_buffer_ticks,
        target_r=args.target_r,
        volume_window=args.volume_window,
        breakout_window=args.breakout_window,
        delta_window=args.delta_window,
        breakout_rvol_threshold=args.breakout_rvol,
        climax_rvol_threshold=args.climax_rvol,
        climax_wick_pct=args.climax_wick_pct,
        require_vwap_alignment=not args.no_vwap_filter,
        enable_breakout=not args.no_breakout,
        enable_climax_fade=not args.no_climax_fade,
    )


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=6)
    parser.add_argument("--target-r", type=float, default=1.5)
    parser.add_argument("--stop-buffer-ticks", type=int, default=2)
    parser.add_argument("--volume-window", type=int, default=20, help="Bars used for the RVOL baseline")
    parser.add_argument("--breakout-window", type=int, default=10, help="Bars used for the breakout channel")
    parser.add_argument("--delta-window", type=int, default=5, help="Bars summed for the volume-delta confirmation")
    parser.add_argument("--breakout-rvol", type=float, default=1.5, help="Min RVOL to qualify a breakout")
    parser.add_argument("--climax-rvol", type=float, default=3.0, help="Min RVOL to qualify a climax fade")
    parser.add_argument("--climax-wick-pct", type=float, default=0.5, help="Min wick fraction of range for a climax fade")
    parser.add_argument("--no-vwap-filter", action="store_true", help="Disable the VWAP trend-alignment filter on breakout entries")
    parser.add_argument("--no-breakout", action="store_true", help="Disable the volume-breakout entry setup")
    parser.add_argument("--no-climax-fade", action="store_true", help="Disable the volume-climax fade entry setup")


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the volume-confirmed scalping strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close,volume)")
    parser.add_argument("--timeframe", default="1min", help="Bar timeframe to trade, resampled from the 1-minute CSV (e.g. 1min, 3min, 5min)")
    add_common_args(parser)
    parser.add_argument("--out", default="trades_volume_scalp.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    strategy = build_strategy(instrument, args)
    engine = VolumeScalpBacktestEngine(
        instrument=instrument, strategy=strategy, risk=risk, timeframe=args.timeframe, contracts=args.contracts
    )
    trades = engine.run(args.data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    print(f"Symbol: {args.symbol}  Timeframe: {args.timeframe}  Contracts: {args.contracts}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
