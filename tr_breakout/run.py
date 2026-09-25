"""
Backtest the TR1 breakout.

    python -m tr_breakout.run --data sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet

NQ/ES full-size price data is fine as input: sizing and P&L always use the
MNQ point value ($2) unless --point-value says otherwise.
"""

import argparse

import pandas as pd

from backtest.data import load_1m_csv
from tr_breakout.strategy import Config, run_backtest, summarize, trades_frame


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--session", default="rth", choices=["rth", "eth"])
    ap.add_argument("--point-value", type=float, default=2.0)
    ap.add_argument("--trades-out", default=None)
    args = ap.parse_args()

    cfg = Config(session=args.session, point_value=args.point_value)
    df = load_1m_csv(args.data)
    trades = run_backtest(df, cfg)
    print("Overall:")
    for k, v in summarize(trades, cfg).items():
        print(f"  {k:>14}: {v}")

    tf = trades_frame(trades)
    if tf.empty:
        return
    tf["year"] = pd.to_datetime(tf["day"]).dt.year
    start_eq = tf.groupby("year")["equity_after"].first() - tf.groupby("year")["pnl"].first()
    by_year = pd.DataFrame({
        "trades": tf.groupby("year").size(),
        "win_%": (tf.groupby("year")["pnl"].apply(lambda s: (s > 0).mean() * 100)).round(1),
        "avg_R": tf.groupby("year")["r_multiple"].mean().round(3),
        "return_%": ((tf.groupby("year")["equity_after"].last() / start_eq - 1) * 100).round(1),
        "median_TR1": tf.groupby("year")["tr1"].median().round(1),
    })
    print("\nBy year:")
    print(by_year.to_string())
    print("\nExit reasons:")
    print(tf["exit_reason"].value_counts().to_string())
    if args.trades_out:
        tf.to_csv(args.trades_out, index=False)
        print(f"\nTrade log -> {args.trades_out}")


if __name__ == "__main__":
    main()
