"""Final failed 2 scalper: 1m, quiet failed 2 bar (range < 0.6 ATR), stop-entry trigger, stop >= 6 pts, 0.5R.

Costs: harsh fills (2-tick entry slippage, 1-tick stop/EOD slippage, target must trade 1 tick through),
priced per 10 MNQ ($1.24 RT each = $12.40) and per 1 NQ ($4 RT). One position at a time.
Run: python f2final.py   (≈1 min; needs numba, pyarrow)
"""
import warnings
import numpy as np
import pandas as pd
import f2lab as L

warnings.filterwarnings("ignore")
MAX_RNG_ATR, MIN_RISK, TGT = 0.6, 6.0, 0.5


def trades(m, pv, rt):
    L.PV, L.RT = pv, rt
    t = L.with_entry(m, L.signals(m, 1), "trigger", expiry_bars=1, thru_ticks=1, entry_slip_ticks=2)
    t = t[(t.rng_atr < MAX_RNG_ATR) & (t.risk >= MIN_RISK)].sort_values("j")
    return L.nonoverlap(t, TGT)


def report(x, scale=1.0, extra=0.0):
    p = x[f"p{TGT}"] * scale - extra
    d = p.groupby(x.time.dt.date).sum(); eq = p.cumsum()
    return dict(trades=len(p), WR=round(100 * (p > 0).mean(), 1), PF=round(L.pf(p), 2), net=round(p.sum()),
                per_day=round(d.mean()), maxDD=round((eq - eq.cummax()).min()), worst_day=round(d.min()),
                win_days=f"{(d > 0).mean():.0%}")


if __name__ == "__main__":
    nq = trades(L.M1(L.load_1m()), 20.0, 4.0)
    nq = nq[nq.time.dt.year >= 2023]
    print("NQ 1m Jan 2023 - Dec 2025, per 1 NQ:  ", report(nq))
    print("                 per 10 MNQ:         ", report(nq, extra=12.4 - 4.0))
    for y, g in nq.groupby(nq.time.dt.year):
        print(f"  {y} per 10 MNQ:", report(g, extra=8.4))
    mnq = trades(L.M1(L.load_1m("real_mnq_1m_2026-08-02_2026-08-25.parquet")), 2.0, 1.24)
    print("MNQ 1m Aug 2026 (out of sample), per 10 MNQ:", report(mnq, scale=10))
