"""
CLI entrypoint: backtest the Fair Value Theory (FVT) strategy over a
1-minute OHLCV CSV.

Usage:
    python -m backtest.run_fair_value --data sample_data/real_nas100_2016_2020.csv --symbol NQ
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from fair_value.strategy import DEFAULT_WINDOWS, FairValueStrategy

from .fair_value_engine import FairValueBacktestEngine
from .metrics import compute_metrics
from .report import write_trades_csv

WINDOW_CHOICES = [w.name for w in DEFAULT_WINDOWS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest JJ Simon's Fair Value Theory strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--target-risk", type=float, default=1000.0, help="Target $ risk per trade used for contract sizing")
    parser.add_argument("--min-contracts", type=int, default=1)
    parser.add_argument("--max-contracts", type=int, default=3)
    parser.add_argument("--fixed-contracts", type=int, default=None, help="Skip $-risk sizing and always trade this many contracts")
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=3)
    parser.add_argument("--atr-period", type=int, default=14)
    parser.add_argument("--swing-strength", type=int, default=2)
    parser.add_argument("--min-mss-body-pct", type=float, default=0.5)
    parser.add_argument("--max-counter-wick-pct", type=float, default=0.20)
    parser.add_argument("--restrict-to-first-hour", action="store_true", help="PDF's optional 'skip 2nd hour of window' filter")
    parser.add_argument(
        "--windows", default=",".join(WINDOW_CHOICES),
        help=f"Comma-separated subset of session windows to trade (default: all). Choices: {','.join(WINDOW_CHOICES)}",
    )
    parser.add_argument("--out", default="trades_fair_value.csv")
    args = parser.parse_args()

    window_names = [w.strip() for w in args.windows.split(",")]
    unknown = [w for w in window_names if w not in WINDOW_CHOICES]
    if unknown:
        parser.error(f"Unknown window(s) {unknown}; choices are {WINDOW_CHOICES}")
    windows = [w for w in DEFAULT_WINDOWS if w.name in window_names]

    instrument = INSTRUMENTS[args.symbol]
    strategy = FairValueStrategy(
        tick_size=instrument.tick_size,
        atr_period=args.atr_period,
        swing_strength=args.swing_strength,
        min_mss_body_pct=args.min_mss_body_pct,
        max_counter_wick_pct=args.max_counter_wick_pct,
        windows=windows,
        restrict_to_first_hour=args.restrict_to_first_hour,
    )
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    engine = FairValueBacktestEngine(
        instrument=instrument,
        strategy=strategy,
        risk=risk,
        target_risk=args.target_risk,
        min_contracts=args.min_contracts,
        max_contracts=args.max_contracts,
        fixed_contracts=args.fixed_contracts,
    )
    trades = engine.run(args.data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    print(f"Symbol: {args.symbol}  Target risk/trade: ${args.target_risk:,.0f}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
