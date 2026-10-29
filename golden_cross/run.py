"""Backtest golden-cross gold longs on GC 15m (2016-2026).

    python -m golden_cross.run [--fill-time 05:45] [--cost 10]
"""

import argparse
import datetime as dt

import pandas as pd

from golden_cross.strategy import daily_bars, fill_prices, run

DATA = "sample_data/real_multi_instrument/real_gc_15m_2016-05-26_2026-08-25.parquet"


def summarize(trades, daily, point_value):
    pnl = [t.pnl_dollars for t in trades]
    eq = pd.Series(pnl).cumsum()
    dd = (eq.cummax().clip(lower=0) - eq).max() if len(eq) else 0
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    first, last = daily.index[0], daily.index[-1]
    bh_pts = daily["close"].iloc[-1] - daily["close"].iloc[260]
    in_mkt = sum(t.days_held for t in trades) / (len(daily) - 260)
    return {
        "trades": len(trades),
        "win_rate_pct": round(100 * len(wins) / len(trades), 1),
        "total_pnl_per_GC": round(sum(pnl), 0),
        "total_points": round(sum(t.points for t in trades), 1),
        "profit_factor": round(sum(wins) / -sum(losses), 2) if losses else float("inf"),
        "avg_win": round(sum(wins) / len(wins), 0) if wins else 0,
        "avg_loss": round(sum(losses) / len(losses), 0) if losses else 0,
        "max_closed_trade_dd": round(dd, 0),
        "time_in_market_pct": round(100 * in_mkt, 1),
        "buy_hold_points_same_window": round(bh_pts, 1),
        "data": f"{first} .. {last}",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DATA)
    ap.add_argument("--fill-time", default="05:45")
    ap.add_argument("--point-value", type=float, default=100.0)
    ap.add_argument("--cost", type=float, default=10.0, help="$ per round turn (commission + slippage)")
    a = ap.parse_args()

    intraday = pd.read_parquet(a.data)
    daily = daily_bars(intraday)
    daily = daily[pd.Series([d.weekday() < 5 for d in daily.index], index=daily.index)]  # drop Sunday-only stubs
    fills = fill_prices(intraday, dt.time.fromisoformat(a.fill_time)).reindex(daily.index)
    trades = run(daily, fills, a.point_value, a.cost)

    for t in trades:
        print(f"{t.entry_date}  {t.entry_price:8.1f} -> {t.exit_date}  {t.exit_price:8.1f}  "
              f"{t.points:+8.1f} pts  ${t.pnl_dollars:+10,.0f}  {t.days_held:4d}d")
    print()
    for k, v in summarize(trades, daily, a.point_value).items():
        print(f"{k:28s} {v}")


if __name__ == "__main__":
    main()
