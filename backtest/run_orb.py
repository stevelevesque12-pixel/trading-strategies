"""CLI entrypoint: backtest the ORB (Opening Range Breakout) strategy over a 1-minute OHLCV CSV.

Usage:
    python -m backtest.run_orb --data path/to/1min.csv --or-minutes 15 --symbol MES
"""

import argparse
from datetime import time

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import SessionConfig
from orb.strategy import ORBStrategy

from .metrics import compute_metrics
from .orb_engine import ORBBacktestEngine
from .report import write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the ORB (Opening Range Breakout) strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--timeframe", default="5min", choices=["1min", "5min"], help="Bar interval the strategy evaluates the OR/ATR/breakout on")
    parser.add_argument("--or-minutes", type=int, default=15, help="Opening Range length in minutes from session open (e.g. 5, 15, 30, 60)")
    parser.add_argument("--atr-period", type=int, default=14, help="Trailing ATR lookback (bars, carried across session boundaries)")
    parser.add_argument("--atr-mult", type=float, default=0.2, help="Breakout velocity buffer, in multiples of ATR")
    parser.add_argument("--target-r", type=float, default=1.0)
    parser.add_argument("--stop-buffer-ticks", type=int, default=0)
    parser.add_argument("--no-breakout-after", default="10:15", help="HH:MM (ET) -- cancel unfilled breakout orders after this time")
    parser.add_argument("--flatten-at", default="15:45", help="HH:MM (ET) -- force-flatten any open position at/after this time")
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=1, help="ORB is one-attempt-per-day by design; raised only if you widen the strategy")
    parser.add_argument("--out", default="trades.csv")
    args = parser.parse_args()

    no_breakout_h, no_breakout_m = (int(x) for x in args.no_breakout_after.split(":"))
    flatten_h, flatten_m = (int(x) for x in args.flatten_at.split(":"))
    session = SessionConfig(no_entry_after=time(no_breakout_h, no_breakout_m), flatten_at=time(flatten_h, flatten_m))

    instrument = INSTRUMENTS[args.symbol]
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    strategy = ORBStrategy(
        or_minutes=args.or_minutes,
        atr_period=args.atr_period,
        atr_mult=args.atr_mult,
        target_r=args.target_r,
        stop_buffer_ticks=args.stop_buffer_ticks,
        tick_size=instrument.tick_size,
        session=session,
    )
    engine = ORBBacktestEngine(
        timeframe=args.timeframe, instrument=instrument, strategy=strategy, risk=risk, session=session, contracts=args.contracts
    )
    trades = engine.run(args.data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    print(f"Timeframe: {args.timeframe}  OR: {args.or_minutes}min  Symbol: {args.symbol}  Contracts: {args.contracts}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
