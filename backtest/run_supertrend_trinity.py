"""CLI entrypoint: backtest the Trinity SuperTrend strategy over a 1-minute OHLCV CSV.

Usage:
    python -m backtest.run_supertrend_trinity --data path/to/1min.csv --tf1 15min --tf2 1h --tf3 4h --symbol MES
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import SessionConfig
from supertrend_trinity.strategy import TrinitySuperTrendStrategy

from .metrics import compute_metrics
from .report import write_trades_csv
from .supertrend_trinity_engine import TrinitySuperTrendBacktestEngine


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Trinity SuperTrend strategy")
    parser.add_argument("--data", required=True, help="Path to 1-minute OHLCV CSV (timestamp,open,high,low,close[,volume])")
    parser.add_argument("--tf1", default="15min", help="Entry timeframe (ST1) -- pandas resample rule, e.g. 5min, 15min")
    parser.add_argument("--tf2", default="1h", help="ST2 timeframe -- pandas resample rule, e.g. 1h")
    parser.add_argument("--tf3", default="4h", help="ST3 timeframe -- pandas resample rule, e.g. 4h")
    parser.add_argument("--atr-period1", type=int, default=10)
    parser.add_argument("--mult1", type=float, default=1.0)
    parser.add_argument("--atr-period2", type=int, default=10)
    parser.add_argument("--mult2", type=float, default=1.0)
    parser.add_argument("--atr-period3", type=int, default=10)
    parser.add_argument("--mult3", type=float, default=1.0)
    parser.add_argument("--entry-mode", default="triple", choices=["single", "double", "triple"])
    parser.add_argument("--sl-mult", type=float, default=1.0)
    parser.add_argument("--tp-mult", type=float, default=1.0)
    parser.add_argument("--use-htf-atr", action="store_true", default=True)
    parser.add_argument("--no-htf-atr", dest="use_htf_atr", action="store_false")
    parser.add_argument("--trailing", action="store_true", default=False, help="Use a trailing stop instead of a fixed one")
    parser.add_argument("--trail-source", default="atr", choices=["atr", "percent"])
    parser.add_argument("--trail-atr-period", type=int, default=14)
    parser.add_argument("--trail-atr-mult", type=float, default=2.0)
    parser.add_argument("--trail-pct", type=float, default=1.5)
    parser.add_argument("--flatten-at", default=None, help="HH:MM (ET) -- force-flatten at/after this time. Omit for genuine swing behavior (no forced flatten, the default -- see strategy.py's module docstring)")
    parser.add_argument("--symbol", default="MES", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--daily-loss-limit", type=float, default=1000.0)
    parser.add_argument("--max-daily-trades", type=int, default=6)
    parser.add_argument("--out", default="trades.csv")
    args = parser.parse_args()

    session = None
    if args.flatten_at:
        from datetime import time
        h, m = (int(x) for x in args.flatten_at.split(":"))
        session = SessionConfig(flatten_at=time(h, m))

    instrument = INSTRUMENTS[args.symbol]
    risk = RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades)
    strategy = TrinitySuperTrendStrategy(
        atr_period1=args.atr_period1,
        mult1=args.mult1,
        atr_period2=args.atr_period2,
        mult2=args.mult2,
        atr_period3=args.atr_period3,
        mult3=args.mult3,
        entry_mode=args.entry_mode,
        sl_mult=args.sl_mult,
        tp_mult=args.tp_mult,
        use_htf_atr=args.use_htf_atr,
        trailing=args.trailing,
        trail_source=args.trail_source,
        trail_atr_period=args.trail_atr_period,
        trail_atr_mult=args.trail_atr_mult,
        trail_pct=args.trail_pct,
        session=session,
    )
    engine = TrinitySuperTrendBacktestEngine(
        tf1=args.tf1, tf2=args.tf2, tf3=args.tf3, instrument=instrument, strategy=strategy, risk=risk, contracts=args.contracts
    )
    trades = engine.run(args.data)

    write_trades_csv(trades, args.out)

    metrics = compute_metrics(trades)
    print(f"TF: {args.tf1}/{args.tf2}/{args.tf3}  Mode: {args.entry_mode}  Symbol: {args.symbol}  Contracts: {args.contracts}")
    for k, v in metrics.items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
