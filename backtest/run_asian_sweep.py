"""CLI entrypoint: backtest the Asian-range liquidity-sweep strategy over a
1-minute OHLCV CSV (needs overnight/pre-market coverage, not RTH-only data --
see the strategy's docstring for the box/sweep/entry session windows).

Usage:
    python -m backtest.run_asian_sweep --data path/to/1min.csv --symbol NQ
"""

import argparse

from asian_sweep.strategy import AsianSweepStrategy
from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager

from .asian_sweep_engine import AsianSweepEngine
from .metrics import compute_metrics
from .report import write_trades_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Asian-range liquidity-sweep strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume]), with overnight coverage")
    parser.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--entry-tf", default="15min", help="Bar resolution the strategy runs on (pandas resample rule)")
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=2)
    parser.add_argument("--target-r", type=float, default=2.0)
    parser.add_argument("--target-mode", default="fixed_r", choices=["fixed_r", "liquidity"])
    parser.add_argument("--stop-mode", default="structural", choices=["structural", "box_extreme"])
    parser.add_argument("--require-fvg", action="store_true", help="Require an FVG alongside the MSS to confirm (default: MSS alone is enough)")
    parser.add_argument("--out", default="trades_asian_sweep.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    strategy = AsianSweepStrategy(
        tick_size=instrument.tick_size,
        target_r=args.target_r,
        target_mode=args.target_mode,
        stop_mode=args.stop_mode,
        require_fvg=args.require_fvg,
    )
    engine = AsianSweepEngine(
        instrument=instrument, strategy=strategy, risk=risk, entry_tf=args.entry_tf, contracts=args.contracts
    )
    trades = engine.run(args.data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    print(f"Symbol: {args.symbol}  Entry TF: {args.entry_tf}  Contracts: {args.contracts}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
