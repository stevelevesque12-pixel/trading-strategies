"""ma_limit flat by 16:30 on MNQ / MES / MGC, 2016-2026 (15m NQ/ES/GC data priced as the micros, Tradeify fees).
Settings are chosen on 2016-06..2020-12 only and judged on 2021-01..2026-08.
  python -m research.ma_limit_micros   -> research/out/ma_limit_micros.csv
"""
import glob
import itertools
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from failed2s.instruments import INSTRUMENTS
from ma_limit.strategy import Params, backtest

SYMS = {"MNQ": "nq", "MES": "es", "MGC": "gc"}
SPLIT = pd.Timestamp("2021-01-01", tz="America/New_York")
HOURS = {"10-15": ("10:00", "15:00"), "0930-15": ("09:30", "15:00"), "0930-12": ("09:30", "12:00"),
         "03-15": ("03:00", "15:00"), "18-15": ("18:00", "15:00")}
GRID = list(itertools.product(SYMS, HOURS, (1.5, 2.0, 3.0), (0.75, 1.0, 1.5, 2.0), (1, 2)))
OUT = Path(__file__).parent / "out"
_bars = {}


def bars(sym):
    if sym not in _bars:
        f = glob.glob(f"sample_data/real_multi_instrument/real_{SYMS[sym]}_15m_*.parquet")[0]
        d = pd.read_parquet(f).tz_convert("America/New_York")
        d.columns = [c.lower() for c in d.columns]
        _bars[sym] = d[["open", "high", "low", "close"]]
    return _bars[sym]


def stats(df, prefix):
    if df.empty:
        return {f"{prefix}_n": 0}
    pnl, r = df.pnl_dollars, df.r_multiple_net
    w, l = pnl[pnl > 0], pnl[pnl <= 0]
    eq = pnl.cumsum(); dd = (eq.cummax().clip(lower=0) - eq).max()
    return {f"{prefix}_n": len(df), f"{prefix}_win": (pnl > 0).mean(), f"{prefix}_net": pnl.sum(), f"{prefix}_per": pnl.mean(),
            f"{prefix}_pf": w.sum() / -l.sum() if l.sum() else np.inf, f"{prefix}_dd": dd, f"{prefix}_avgR": r.mean(),
            f"{prefix}_t": r.mean() / r.std() * len(r) ** 0.5 if len(r) > 1 else 0.0}


def run(cfg):
    sym, hk, sl, rr, x = cfg
    inst = INSTRUMENTS[sym]
    p = Params(expiry_bars=x, live_mode="realtime", atr_mult=sl, rr=rr, hours=HOURS[hk], flatten_at="16:30",
               tick_size=inst.tick_size, point_value=inst.point_value, commission_rt=inst.round_trip_fee)
    t = backtest(bars(sym), "30min", p)
    df = pd.DataFrame([vars(x) for x in t])
    df["r_multiple_net"] = df.pnl_dollars / (df.risk0 * inst.point_value)
    row = dict(sym=sym, hours=hk, sl=sl, rr=rr, x=x)
    row |= stats(df[df.entry_time < SPLIT], "is") | stats(df[df.entry_time >= SPLIT], "oos")
    for y, g in df.groupby(df.entry_time.dt.year):
        row[f"y{y}"] = g.pnl_dollars.sum()
    return row


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    with Pool(4) as pool:
        rows = []
        for i, r in enumerate(pool.imap_unordered(run, GRID), 1):
            rows.append(r)
            if i % 20 == 0:
                print(f"{i}/{len(GRID)}", flush=True)
                pd.DataFrame(rows).to_csv(OUT / "ma_limit_micros.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "ma_limit_micros.csv", index=False)
    print("done")
