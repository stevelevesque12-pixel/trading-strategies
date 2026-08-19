"""CLI entrypoint: backtest the Donchian+MACD+Volume breakout strategy over a 1-minute OHLCV CSV.

Usage:
    python -m backtest.run_donchian_macd_vol --data path/to/1min.csv --timeframe 5min --symbol MES
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from donchian_macd_vol.strategy import DonchianMacdVolumeStrategy

from .donchian_macd_vol_engine import DonchianMacdVolBacktestEngine
from .metrics import compute_metrics
from .report import write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Donchian+MACD+Volume breakout strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--timeframe", default="5min", choices=["1min", "5min", "15min"], help="Bar interval the strategy evaluates entries on")
    parser.add_argument("--donchian-period", type=int, default=10, help="10 on the default 5min timeframe was tuned against real 2019 NAS100 data to land around 2-3 trades/day")
    parser.add_argument("--macd-fast", type=int, default=12)
    parser.add_argument("--macd-slow", type=int, default=26)
    parser.add_argument("--macd-signal", type=int, default=9)
    parser.add_argument("--volume-period", type=int, default=20)
    parser.add_argument("--volume-mult", type=float, default=1.5)
    parser.add_argument("--target-r", type=float, default=1.5)
    parser.add_argument("--stop-buffer-ticks", type=int, default=2)
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=6)
    parser.add_argument("--out", default="trades.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    strategy = DonchianMacdVolumeStrategy(
        tick_size=instrument.tick_size,
        donchian_period=args.donchian_period,
        macd_fast=args.macd_fast,
        macd_slow=args.macd_slow,
        macd_signal=args.macd_signal,
        volume_period=args.volume_period,
        volume_mult=args.volume_mult,
        target_r=args.target_r,
        stop_buffer_ticks=args.stop_buffer_ticks,
    )
    engine = DonchianMacdVolBacktestEngine(
        timeframe=args.timeframe, instrument=instrument, strategy=strategy, risk=risk, contracts=args.contracts
    )
    trades = engine.run(args.data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    n_days = len({t.entry_time.date() for t in trades}) or 1
    print(f"Timeframe: {args.timeframe}  Donchian period: {args.donchian_period}  Symbol: {args.symbol}  Contracts: {args.contracts}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"  trades_per_active_day: {round(metrics.get('num_trades', 0) / n_days, 2)}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
