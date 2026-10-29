"""CLI: backtest structure_scalp.stop_entry (structure bias + stop-order entries).

The live version is 1m bias / 5s entries. With 1-minute history only, run the
same 12:1 ratio one level up (12m bias / 1m entries) -- or a pair that the data
supports directly, like 15m bias / 1m entries:

    python -m backtest.scalp_run --data sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet \\
        --symbol NQ --bias-tf 12min --entry-tf 1min

If you ever get real 5-second bars into the same CSV/parquet schema, pass
--entry-tf 5s --bias-tf 1min to test the live timeframes directly.
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from structure_scalp.stop_entry import StopEntryScalpStrategy

from .metrics import compute_metrics
from .report import write_trades_csv
from .scalp_engine import StopEntryBacktest


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the structure + stop-entry scalp")
    parser.add_argument("--data", required=True)
    parser.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--bias-tf", default="12min")
    parser.add_argument("--entry-tf", default="1min")
    parser.add_argument("--target-r", type=float, default=1.0)
    parser.add_argument("--swing-strength-bias", type=int, default=2)
    parser.add_argument("--swing-strength-entry", type=int, default=2)
    parser.add_argument("--slippage-ticks", type=float, default=0.0)
    parser.add_argument("--commission-per-side", type=float, default=0.0)
    parser.add_argument("--max-daily-trades", type=int, default=10_000)
    parser.add_argument("--daily-loss-limit", type=float, default=float("inf"))
    parser.add_argument("--out", default="trades_stop_entry.csv")
    args = parser.parse_args()

    inst = INSTRUMENTS[args.symbol]
    strat = StopEntryScalpStrategy(
        tick_size=inst.tick_size,
        target_r=args.target_r,
        swing_strength_bias=args.swing_strength_bias,
        swing_strength_entry=args.swing_strength_entry,
    )
    engine = StopEntryBacktest(
        inst,
        entry_tf=args.entry_tf,
        bias_tf=args.bias_tf,
        strategy=strat,
        risk=RiskManager(daily_loss_limit=args.daily_loss_limit, max_daily_trades=args.max_daily_trades),
        slippage_ticks=args.slippage_ticks,
        commission_per_side=args.commission_per_side,
    )
    trades = engine.run(args.data)
    write_trades_csv(trades, args.out)

    days = len({t.entry_time.date() for t in trades})
    print(f"{args.bias_tf} bias / {args.entry_tf} entries  {args.symbol}  slippage={args.slippage_ticks}t")
    for k, v in compute_metrics(trades).items():
        print(f"  {k}: {v}")
    if days:
        print(f"  trading_days: {days}  trades_per_day: {len(trades) / days:.1f}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
