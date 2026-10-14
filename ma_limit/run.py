"""Backtest the 9/20 SMA limit-at-the-20 strategy.

    python -m ma_limit.run sample_data/real_multi_instrument/real_mcl_15m_2025-09-30_2026-08-25.parquet
"""

import argparse

import pandas as pd

from failed2s.instruments import INSTRUMENTS
from .strategy import Params, backtest, summarize


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="parquet with bars finer than --tf (used for fills/exits)")
    ap.add_argument("--symbol", default="MCL")
    ap.add_argument("--tf", default="30min")
    ap.add_argument("--sl", type=float, default=2.0, help="stop loss in ATRs")
    ap.add_argument("--rr", type=float, default=1.0)
    ap.add_argument("--x", type=int, default=5, help="candles the limit stays working after the setup close")
    ap.add_argument("--mode", choices=["chart", "realtime"], default="chart",
                    help="chart: entry/stop from the touching candle (matches hand-marked trades); realtime: no look-ahead")
    ap.add_argument("--hours", help='only take setups in this ET window, e.g. "09:00-14:30" or "18:00-02:00"')
    a = ap.parse_args()

    inst = INSTRUMENTS[a.symbol]
    df = pd.read_parquet(a.path).tz_convert("America/New_York")
    df.columns = [c.lower() for c in df.columns]
    df = df[[c for c in ("open", "high", "low", "close", "volume") if c in df]]
    hours = tuple(a.hours.split("-")) if a.hours else None
    p = Params(atr_mult=a.sl, hours=hours, rr=a.rr, expiry_bars=a.x, live_mode=a.mode, tick_size=inst.tick_size, point_value=inst.point_value)
    trades = backtest(df, a.tf, p)
    print(summarize(trades))


if __name__ == "__main__":
    main()
