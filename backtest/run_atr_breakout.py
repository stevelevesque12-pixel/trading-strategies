"""CLI entrypoint: backtest the ATR-everything breakout strategy over a 1-minute OHLCV CSV.

Usage:
    python -m backtest.run_atr_breakout --data path/to/1min.csv --timeframe 5min --symbol MES
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from atr_breakout.strategy import ATRBreakoutStrategy

from .atr_breakout_engine import ATRBreakoutBacktestEngine
from .metrics import compute_metrics
from .report import write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the ATR-everything breakout strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--timeframe", default="5min", choices=["1min", "5min"], help="Bar interval the strategy evaluates entries/ATR on")
    parser.add_argument("--atr-period-long", type=int, default=20)
    parser.add_argument("--atr-period-short", type=int, default=5)
    parser.add_argument("--entry-atr-mult", type=float, default=2.5, help="Breakout distance from the bar's open, in multiples of ATR_long")
    parser.add_argument("--stop-atr-mult", type=float, default=0.5, help="Stop distance from entry, in multiples of ATR_long (a fraction)")
    parser.add_argument("--target-atr-mult", type=float, default=None, help="Target distance from entry, in multiples of ATR_long. Omit for no fixed target (default -- ride to stop/session-flatten); set e.g. 2.0 to A/B test a take-profit")
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=6)
    parser.add_argument("--out", default="trades.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    strategy = ATRBreakoutStrategy(
        atr_period_long=args.atr_period_long,
        atr_period_short=args.atr_period_short,
        entry_atr_mult=args.entry_atr_mult,
        stop_atr_mult=args.stop_atr_mult,
        target_atr_mult=args.target_atr_mult,
    )
    engine = ATRBreakoutBacktestEngine(
        timeframe=args.timeframe, instrument=instrument, strategy=strategy, risk=risk, contracts=args.contracts
    )
    trades = engine.run(args.data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    print(f"Timeframe: {args.timeframe}  Symbol: {args.symbol}  Contracts: {args.contracts}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
