"""Python replica of strat-failed-2.pine (defaults: Both dirs, magnitude target, ribbon on, FTFC off)."""
import sys
import numpy as np
import pandas as pd

from pathlib import Path
DATA = str(Path(__file__).resolve().parents[1] / "sample_data" / "real_multi_instrument") + "/"
TICK, PV, COMM = 0.25, 2.0, 0.62  # MNQ tick, $/pt, commission per side
NY = "America/New_York"


def load(fname):
    d = pd.read_parquet(DATA + fname).tz_convert(NY)
    return d[["open", "high", "low", "close", "volume"]]


def resample(d, n, offset=0):
    if n == 1:
        return d
    origin = pd.Timestamp("2020-01-01 18:00", tz=NY) + pd.Timedelta(minutes=offset)  # CME session start, like TradingView
    r = d.resample(f"{n}min", origin=origin).agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    return r.dropna()


def in_sess(idx, start, end):
    m = idx.hour * 60 + idx.minute
    return (m >= start) & (m < end)


def run(d, ribbon=True, check_prev=False, direction="Both", allow8=False, vwap=None, tmode="mag", minR=0.0, fast=8, slow=21):
    o, h, l, c = (d[k].to_numpy() for k in ("open", "high", "low", "close"))
    ef = d["close"].ewm(span=fast, adjust=False).mean().to_numpy()
    es = d["close"].ewm(span=slow, adjust=False).mean().to_numpy()
    top, bot = np.maximum(ef, es), np.minimum(ef, es)
    up, dn = ef > es, ef < es
    if vwap is not None:
        top = bot = vwap
        up = dn = np.ones(len(vwap), bool)
    if allow8:
        top, bot = es, es  # only the 21 EMA must stay untouched
    idx = d.index
    entry_ok = in_sess(idx, 9 * 60 + 30, 15 * 60 + 45)
    flat_ok = in_sess(idx, 15 * 60 + 55, 16 * 60)
    n = len(d)
    trades = []
    pos = 0; entry = stop = tgt = 0.0; etime = None
    pending = None  # (dir, stop, tgt) to fill at next open
    flat_pending = False
    for i in range(1, n):
        # --- fills at this bar's open ---
        if flat_pending and pos != 0:
            px = o[i] - pos * TICK
            trades.append((etime, pos, entry, px, "EOD")); pos = 0
        flat_pending = False
        if pending is not None:
            pdir, pstop, ptgt = pending; pending = None
            pos = pdir; entry = o[i] + pdir * TICK; stop, tgt = pstop, ptgt; etime = idx[i]
        # --- exits intrabar ---
        if pos != 0:
            px = None; why = None
            if pos == 1:
                if o[i] <= stop: px, why = o[i] - TICK, "SL"
                elif o[i] >= tgt: px, why = o[i], "TP"
                else:
                    hit_s, hit_t = l[i] <= stop, h[i] >= tgt
                    if hit_s and hit_t:
                        high_first = (h[i] - o[i]) < (o[i] - l[i])  # TV emulator: nearer extreme first
                        px, why = (tgt, "TP") if high_first else (stop - TICK, "SL")
                    elif hit_s: px, why = stop - TICK, "SL"
                    elif hit_t: px, why = tgt, "TP"
            else:
                if o[i] >= stop: px, why = o[i] + TICK, "SL"
                elif o[i] <= tgt: px, why = o[i], "TP"
                else:
                    hit_s, hit_t = h[i] >= stop, l[i] <= tgt
                    if hit_s and hit_t:
                        low_first = (o[i] - l[i]) < (h[i] - o[i])
                        px, why = (tgt, "TP") if low_first else (stop + TICK, "SL")
                    elif hit_s: px, why = stop + TICK, "SL"
                    elif hit_t: px, why = tgt, "TP"
            if px is not None:
                trades.append((etime, pos, entry, px, why)); pos = 0
        # --- signals at this bar's close ---
        if pos != 0 and flat_ok[i]:
            flat_pending = True
        if pos != 0 or not entry_ok[i]:
            continue
        f2d = l[i] < l[i-1] and h[i] <= h[i-1] and c[i] > o[i]
        f2u = h[i] > h[i-1] and l[i] >= l[i-1] and c[i] < o[i]
        if f2d and direction != "Short":
            rl = (not ribbon) or (up[i] and l[i] > top[i] and (not check_prev or l[i-1] > top[i-1]))
            if rl:
                st_ = l[i] - TICK; rk = c[i] - st_
                if tmode == "mag":
                    if h[i-1] > c[i] and (minR == 0 or h[i-1] - c[i] >= minR * rk):
                        pending = (1, st_, h[i-1])
                else:
                    pending = (1, st_, c[i] + tmode * rk)
        elif f2u and direction != "Long":
            rs = (not ribbon) or (dn[i] and h[i] < bot[i] and (not check_prev or h[i-1] < bot[i-1]))
            if rs:
                st_ = h[i] + TICK; rk = st_ - c[i]
                if tmode == "mag":
                    if l[i-1] < c[i] and (minR == 0 or c[i] - l[i-1] >= minR * rk):
                        pending = (-1, st_, l[i-1])
                else:
                    pending = (-1, st_, c[i] - tmode * rk)
    t = pd.DataFrame(trades, columns=["time", "dir", "entry", "exit", "why"])
    t["pnl"] = (t["exit"] - t["entry"]) * t["dir"] * PV - 2 * COMM
    return t


def stats(t):
    if len(t) == 0:
        return dict(trades=0)
    w, ls = t.pnl[t.pnl > 0], t.pnl[t.pnl <= 0]
    eq = t.pnl.cumsum(); dd = (eq - eq.cummax().clip(lower=0)).min()
    return dict(trades=len(t), net=round(t.pnl.sum()), wr=round(100 * len(w) / len(t), 1),
                pf=round(w.sum() / -ls.sum(), 2) if ls.sum() < 0 else np.inf,
                avgW=round(w.mean(), 1) if len(w) else 0, avgL=round(ls.mean(), 1) if len(ls) else 0,
                maxDD=round(dd), longs=int((t.dir == 1).sum()), shorts=int((t.dir == -1).sum()))


if __name__ == "__main__":
    rows = []
    m1 = load("real_mnq_1m_2026-08-02_2026-08-25.parquet")
    m5 = load("real_mnq_5m_2026-05-10_2026-08-25.parquet")
    for ribbon in (True, False):
        for n in (1, 2, 3, 4, 5):
            rows.append(dict(set="Aug2-25 (1m src)", tf=f"{n}m", ribbon=ribbon, **stats(run(resample(m1, n), ribbon=ribbon))))
        t = run(m5, ribbon=ribbon)
        rows.append(dict(set="May10-Aug25 (5m)", tf="5m", ribbon=ribbon, **stats(t)))
        mid = t.time.iloc[len(t)//2] if len(t) else None
        if ribbon:
            half = pd.Timestamp("2026-07-02", tz=NY)
            rows.append(dict(set="  H1 May10-Jul1", tf="5m", ribbon=ribbon, **stats(t[t.time < half])))
            rows.append(dict(set="  H2 Jul2-Aug25", tf="5m", ribbon=ribbon, **stats(t[t.time >= half])))
    pd.set_option("display.width", 200)
    print(pd.DataFrame(rows).to_string(index=False))


def vwap(d, anchor="eth"):
    tp = (d.high + d.low + d.close) / 3
    if anchor == "eth":  # CME session starts 18:00 ET (TradingView default)
        key = (d.index - pd.Timedelta(hours=18)).normalize()
    else:  # RTH: reset 09:30 ET
        key = (d.index - pd.Timedelta(hours=9, minutes=30)).normalize()
    pv = (tp * d.volume).groupby(key).cumsum(); v = d.volume.groupby(key).cumsum()
    return (pv / v).to_numpy()
