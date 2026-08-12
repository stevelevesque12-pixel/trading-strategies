"""CLI entrypoint: backtest the Failed-2s strategy over a 1-minute OHLCV CSV.

Usage:
    python -m backtest.run --data path/to/1min.csv --pair 5m-1h --symbol MES

To compare all three timeframe pairs at once, use `python -m backtest.compare` instead.
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import PAIRS, Failed2sStrategy

from .engine import BacktestEngine
from .metrics import compute_metrics
from .report import write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Failed-2s strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--pair", required=True, choices=list(PAIRS.keys()))
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=3)
    parser.add_argument("--target-r", type=float, default=1.0)
    parser.add_argument("--out", default="trades.csv")
    args = parser.parse_args()

    pair = PAIRS[args.pair]
    instrument = INSTRUMENTS[args.symbol]
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    strategy = Failed2sStrategy(tick_size=instrument.tick_size, target_r=args.target_r)
    engine = BacktestEngine(pair=pair, instrument=instrument, strategy=strategy, risk=risk, contracts=args.contracts)
    trades = engine.run(args.data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    print(f"Pair: {args.pair}  Symbol: {args.symbol}  Contracts: {args.contracts}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
