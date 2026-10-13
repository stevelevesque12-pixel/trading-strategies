"""
Backtest the MA pullback strategy on the repo's MCL data.

    python -m ma_pullback.run                      # 30m from the 15m file, defaults
    python -m ma_pullback.run --tf 1h --swing-strength 3 --swing-buffer 2 --longs
    python -m ma_pullback.run --stop-mode ticks --stop-ticks 30 --target-r 1.5   # the original fixed stop
"""

import argparse

import pandas as pd

from backtest.data import load_1m_csv, resample_ohlc

from .strategy import MAPullbackConfig, run

DEFAULT_DATA = "sample_data/real_multi_instrument/real_mcl_15m_2025-09-30_2026-08-25.parquet"


def summarize(trades) -> dict:
    if not trades:
        return {"trades": 0}
    df = pd.DataFrame([t.__dict__ for t in trades])
    equity = df["pnl_dollars"].cumsum()
    wins = df[df["pnl_dollars"] > 0]["pnl_dollars"].sum()
    losses = -df[df["pnl_dollars"] < 0]["pnl_dollars"].sum()
    return {
        "trades": len(df),
        "win_rate": round((df["pnl_dollars"] > 0).mean(), 3),
        "net_usd": round(df["pnl_dollars"].sum(), 2),
        "avg_r": round(df["r_multiple"].mean(), 3),
        "profit_factor": round(wins / losses, 2) if losses else float("inf"),
        "max_drawdown_usd": round((equity.cummax() - equity).max(), 2),
        "median_risk_ticks": round(((df["entry_price"] - df["stop_price"]).abs()).median() * 100),
        "max_risk_ticks": round(((df["entry_price"] - df["stop_price"]).abs()).max() * 100),
        "filled_at_fast_ma": int((df["filled_at"] == "fast").sum()),
        "filled_at_slow_ma": int((df["filled_at"] == "slow").sum()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--tf", default="30min", help="pandas rule, e.g. 30min, 1h")
    ap.add_argument("--fast", type=int, default=9)
    ap.add_argument("--slow", type=int, default=20)
    ap.add_argument("--ma-type", default="SMA", choices=["SMA", "EMA"])
    ap.add_argument("--stop-mode", default="swing", choices=["swing", "ticks"])
    ap.add_argument("--swing-strength", type=int, default=2)
    ap.add_argument("--swing-buffer", type=int, default=0, help="ticks beyond the swing")
    ap.add_argument("--stop-ticks", type=int, default=30)
    ap.add_argument("--target-r", type=float, default=1.0)
    ap.add_argument("--expiry-bars", type=int, default=0)
    ap.add_argument("--longs", action="store_true", help="also trade the mirrored long side")
    a = ap.parse_args()

    bars = resample_ohlc(load_1m_csv(a.data), a.tf)
    cfg = MAPullbackConfig(fast_len=a.fast, slow_len=a.slow, ma_type=a.ma_type, stop_mode=a.stop_mode,
                           swing_strength=a.swing_strength, swing_buffer_ticks=a.swing_buffer, stop_ticks=a.stop_ticks,
                           target_r=a.target_r, expiry_bars=a.expiry_bars, enable_longs=a.longs)
    trades = run(bars, cfg)
    print(f"{len(bars)} {a.tf} bars, {bars.index[0]} -> {bars.index[-1]}")
    for k, v in summarize(trades).items():
        print(f"  {k}: {v}")
    if a.longs:
        for side in ("short", "long"):
            print(side, summarize([t for t in trades if t.direction == side]))


if __name__ == "__main__":
    main()
