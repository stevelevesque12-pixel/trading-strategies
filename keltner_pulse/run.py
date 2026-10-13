"""CLI entrypoint: backtest NQ Keltner Pulse on 15-minute bars.

Usage:
    python -m keltner_pulse.run \
        --data sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet

--data can be native 15m bars or anything finer (1m, 5m); it's resampled to
--timeframe either way.
"""

import argparse

from backtest.data import load_1m_csv, resample_ohlc
from backtest.metrics import compute_metrics
from backtest.report import write_trades_csv
from failed2s.instruments import INSTRUMENTS

from .strategy import KeltnerPulseBacktest, KeltnerPulseConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the NQ Keltner Pulse M15 strategy")
    parser.add_argument("--data", required=True, help="OHLCV CSV or parquet (15m or finer)")
    parser.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--timeframe", default="15min")
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--start", help="First date to trade (YYYY-MM-DD)")
    parser.add_argument("--end", help="Last date to trade (YYYY-MM-DD)")
    parser.add_argument("--commission-rt", type=float, default=0.0, help="Round-turn commission per contract ($)")
    parser.add_argument("--slippage-ticks", type=float, default=0.0, help="Adverse ticks on every stop-type fill")
    parser.add_argument("--out", default="keltner_pulse_trades.csv")
    args = parser.parse_args()

    df = resample_ohlc(load_1m_csv(args.data), args.timeframe)
    if args.start:
        df = df[df.index >= args.start]
    if args.end:
        df = df[df.index < args.end]

    cfg = KeltnerPulseConfig(commission_rt=args.commission_rt, slippage_ticks=args.slippage_ticks)
    trades = KeltnerPulseBacktest(INSTRUMENTS[args.symbol], cfg, args.contracts).run(df)
    write_trades_csv(trades, args.out)

    print(f"Symbol: {args.symbol}  TF: {args.timeframe}  Bars: {len(df)}  {df.index[0]} -> {df.index[-1]}")
    for k, v in compute_metrics(trades).items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
