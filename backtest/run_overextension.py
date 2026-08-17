"""CLI entrypoint: backtest the VWAP overextension strategy over a 1-minute OHLCV CSV.

Usage:
    python -m backtest.run_overextension --data path/to/1min.csv --timeframe 5min --symbol MNQ
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from overextension.strategy import OverextensionStrategy

from .metrics import compute_metrics
from .overextension_engine import OverextensionBacktestEngine
from .report import write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the VWAP overextension strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--timeframe", default="5min", choices=["1min", "5min"], help="Bar interval the strategy evaluates z-score/VWAP on")
    parser.add_argument("--symbol", default="MNQ", choices=list(INSTRUMENTS.keys()), help="High-beta tech futures (MNQ/NQ) are the intended use case, but any instrument works")
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=6)
    parser.add_argument("--entry-z", type=float, default=2.0, help="Standard deviations from VWAP considered 'overextended'")
    parser.add_argument("--confirm-z", type=float, default=0.5, help="Recovery (in stdevs) off the extreme required before firing")
    parser.add_argument("--max-z", type=float, default=4.0, help="Skip episodes more extended than this -- likely a trend day, not a fade")
    parser.add_argument("--reversion-target-pct", type=float, default=1.0, help="Fraction of the distance back to VWAP taken as target (1.0 = full reversion)")
    parser.add_argument("--warmup-bars", type=int, default=20, help="Bars into the session before z-score signals are trusted")
    parser.add_argument("--out", default="trades.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    strategy = OverextensionStrategy(
        tick_size=instrument.tick_size,
        entry_z=args.entry_z,
        confirm_z=args.confirm_z,
        max_z=args.max_z,
        reversion_target_pct=args.reversion_target_pct,
        warmup_bars=args.warmup_bars,
    )
    engine = OverextensionBacktestEngine(
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
