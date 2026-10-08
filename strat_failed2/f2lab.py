"""Failed 2 research lab on 1m NQ.

Builds every failed 2 signal for an n-minute timeframe (any bar offset), attaches
features (ribbon, FTFC, sweeps, VWAP, time, bar shape...), and simulates each trade on
the underlying 1m bars for several fixed-R targets (stop-first if both hit in one minute).

Fill model: signal on TF bar close, entry at next 1m open +1 tick, stop 1 tick beyond the
failed 2 wick (exit with 1 tick slippage), target = close +/- m * (close - stop) as a limit,
flatten 15:55 ET. Entries only on bars opening 09:30-15:44 ET. 1 NQ, $20/pt, $4 RT.
"""
import numpy as np
import pandas as pd
from numba import njit
from pathlib import Path

DATA = Path(__file__).resolve().parents[1] / "sample_data" / "real_multi_instrument"
NY = "America/New_York"
TICK, PV, RT = 0.25, 20.0, 4.0
TARGETS = (0.5, 0.75, 1.0, 1.5, 2.0, 3.0)


def load_1m(fname="real_nq_1m_2022-12-26_2025-12-11.parquet"):
    d = pd.read_parquet(DATA / fname).tz_convert(NY)
    return d[["open", "high", "low", "close", "volume"]].astype(float)


class M1:
    """1m arrays + per-minute day-level context."""

    def __init__(self, d):
        self.d = d
        idx = d.index
        self.o, self.h, self.l, self.c, self.v = (d[k].to_numpy() for k in ("open", "high", "low", "close", "volume"))
        self.mod = (idx.hour * 60 + idx.minute).to_numpy()
        # trading date: CME session starting 18:00 belongs to next day
        tdate = (idx + pd.Timedelta(hours=6)).normalize().tz_localize(None)
        self.tdate = tdate.to_numpy()
        self.epoch_min = ((idx.tz_convert("UTC").tz_localize(None) - pd.Timestamp("2000-01-01")) // pd.Timedelta(minutes=1)).to_numpy()
        rth = (self.mod >= 570) & (self.mod < 960)
        self.rth = rth
        df = pd.DataFrame({"o": self.o, "h": self.h, "l": self.l, "c": self.c, "v": self.v, "td": self.tdate, "rth": rth, "mod": self.mod})
        g = df.groupby("td")
        # RTH open / running RTH high-low (inclusive of current minute)
        r = df[rth]
        rg = r.groupby("td")
        day_open = rg.o.first()
        self.day_open = df.td.map(day_open).to_numpy()
        run_hi = np.full(len(df), np.nan); run_lo = np.full(len(df), np.nan)
        run_hi[rth] = rg.h.cummax().to_numpy(); run_lo[rth] = rg.l.cummin().to_numpy()
        self.run_hi, self.run_lo = run_hi, run_lo
        # overnight (pre-RTH, same trading date) high/low
        on = df[df["mod"].lt(570) | df["mod"].ge(1080)]
        self.on_hi = df.td.map(on.groupby("td").h.max()).to_numpy()
        self.on_lo = df.td.map(on.groupby("td").l.min()).to_numpy()
        # prior RTH day high/low/close
        rh, rl, rc = rg.h.max(), rg.l.min(), rg.c.last()
        self.pdh = df.td.map(rh.shift(1)).to_numpy(); self.pdl = df.td.map(rl.shift(1)).to_numpy()
        self.pdc = df.td.map(rc.shift(1)).to_numpy()
        # opening range 09:30-09:59
        orr = df[(df["mod"] >= 570) & (df["mod"] < 600)].groupby("td")
        self.or_hi = df.td.map(orr.h.max()).to_numpy(); self.or_lo = df.td.map(orr.l.min()).to_numpy()
        # RTH-anchored VWAP and ETH VWAP (cumulative incl. current minute)
        tp = (df.h + df.l + df.c) / 3
        pv = (tp * df.v); vw = np.full(len(df), np.nan)
        vw[rth] = (pv[rth].groupby(df.td[rth]).cumsum() / df.v[rth].groupby(df.td[rth]).cumsum().replace(0, np.nan)).to_numpy()
        self.vwap_rth = vw
        self.vwap_eth = (pv.groupby(df.td).cumsum() / df.v.groupby(df.td).cumsum().replace(0, np.nan)).to_numpy()
        # slow 1m EMAs (~2h, ~8h) as higher-timeframe trend
        cs = pd.Series(self.c)
        self.e120 = cs.ewm(span=120, adjust=False).mean().to_numpy()
        self.e480 = cs.ewm(span=480, adjust=False).mean().to_numpy()
        # daily trend: prior RTH close vs 20-day mean of RTH closes
        self.d20 = df.td.map(rc.rolling(20).mean().shift(1)).to_numpy()
        # 60m bar open, clock aligned to :30 (TheStrat RTH hours)
        hkey = (self.epoch_min - 30) // 60
        self.h60_open = pd.Series(self.o).groupby(hkey).transform("first").to_numpy()
        # week open (first minute of the CME week)
        wk = pd.Series(tdate).dt.to_period("W-FRI").astype(str).to_numpy()
        self.wk_open = pd.Series(self.o).groupby(wk).transform("first").to_numpy()
        # daily ATR proxy: mean RTH range of prior 14 days
        self.adr = df.td.map((rh - rl).rolling(14).mean().shift(1)).to_numpy()


def tf_bars(m, n, offset=0):
    """Aggregate 1m into n-minute bars aligned to 18:00 ET + offset. Returns dict of arrays."""
    # 18:00 ET == 22:00 or 23:00 UTC; align on local wall clock instead
    loc = m.d.index.tz_localize(None)
    lm = ((loc - pd.Timestamp("2000-01-01 18:00")) // pd.Timedelta(minutes=1)).to_numpy() - offset
    key = lm // n
    brk = np.flatnonzero(np.diff(key)) + 1
    first = np.r_[0, brk]; last = np.r_[brk - 1, len(key) - 1]
    o = m.o[first]; c = m.c[last]
    h = np.maximum.reduceat(m.h, first); l = np.minimum.reduceat(m.l, first); v = np.add.reduceat(m.v, first)
    return dict(first=first, last=last, o=o, h=h, l=l, c=c, v=v, mod=m.mod[first], n=n)


@njit(cache=True)
def _sim(o, h, l, mod, tdate, j, d, stop, tgt, tick):
    """Return (exit_px, exit_idx) for one trade entering at o[j]."""
    entry = o[j] + d * tick
    td = tdate[j]
    k = j
    N = len(o)
    while k < N:
        if tdate[k] != td or mod[k] >= 955:
            return o[k] - d * tick, k
        if d == 1:
            if k == j:
                if o[k] <= stop: return o[k] - tick, k
                if o[k] >= tgt: return o[k], k
            elif o[k] <= stop: return o[k] - tick, k
            elif o[k] >= tgt: return o[k], k
            if l[k] <= stop: return stop - tick, k
            if h[k] >= tgt: return tgt, k
        else:
            if k == j:
                if o[k] >= stop: return o[k] + tick, k
                if o[k] <= tgt: return o[k], k
            elif o[k] >= stop: return o[k] + tick, k
            elif o[k] <= tgt: return o[k], k
            if h[k] >= stop: return stop + tick, k
            if l[k] <= tgt: return tgt, k
        k += 1
    return o[N - 1], N - 1


@njit(cache=True)
def sim_all(o, h, l, mod, tdate, J, D, S, Cl, mults, tick):
    n = len(J); K = len(mults)
    px = np.empty((n, K)); ix = np.empty((n, K), np.int64)
    for i in range(n):
        for k in range(K):
            risk = D[i] * (Cl[i] - S[i])
            tgt = Cl[i] + D[i] * mults[k] * risk
            px[i, k], ix[i, k] = _sim(o, h, l, mod, tdate, J[i], D[i], S[i], tgt, tick)
    return px, ix


def ema(x, span):
    return pd.Series(x).ewm(span=span, adjust=False).mean().to_numpy()


def signals(m, n, offset=0, targets=TARGETS):
    b = tf_bars(m, n, offset)
    o, h, l, c, v = b["o"], b["h"], b["l"], b["c"], b["v"]
    N = len(c)
    ph, pl = np.r_[np.nan, h[:-1]], np.r_[np.nan, l[:-1]]
    f2d = (l < pl) & (h <= ph) & (c > o)
    f2u = (h > ph) & (l >= pl) & (c < o)
    bmod = b["mod"]
    ok = (bmod >= 570) & (bmod < 945) & (b["last"] + 1 < len(m.o))
    ok &= np.r_[False, m.tdate[b["first"][1:]] == m.tdate[b["first"][:-1]]]  # prior bar same session
    sig = np.flatnonzero((f2d | f2u) & ok)
    i = sig
    d = np.where(f2d[i], 1, -1)
    j = b["last"][i] + 1
    keep = (m.tdate[j] == m.tdate[b["last"][i]]) & (m.mod[j] < 955)
    i, d, j = i[keep], d[keep], j[keep]
    stop = np.where(d == 1, l[i] - TICK, h[i] + TICK)
    px, ix = sim_all(m.o, m.h, m.l, m.mod, m.tdate, j, d, stop, c[i], np.array(targets, float), TICK)
    entry = m.o[j] + d * TICK
    pnl = (px - entry[:, None]) * d[:, None] * PV - RT
    t = pd.DataFrame({"time": m.d.index[b["first"][i]], "dir": d, "j": j, "entry": entry, "stop": stop,
                      "_h": h[i], "_l": l[i], "_ph": ph[i], "_pl": pl[i]})
    for k, tm in enumerate(targets):
        t[f"p{tm}"] = pnl[:, k]; t[f"x{tm}"] = ix[:, k]
    # ---------------- features ----------------
    jl = b["last"][i]  # last 1m index of the signal bar (info available at signal)
    cc = c[i]
    risk = d * (cc - stop)
    t["risk"] = risk
    tr = np.maximum(h, np.r_[np.nan, c[:-1]]) - np.minimum(l, np.r_[np.nan, c[:-1]])
    atr = pd.Series(np.nan_to_num(tr, nan=h[0] - l[0])).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()  # ta.atr(14) (RMA)
    t["rng_atr"] = (h[i] - l[i]) / atr[i - 1]
    t["risk_atr"] = risk / atr[i - 1]
    t["risk_adr"] = risk / m.adr[jl]
    e8, e21 = ema(c, 8), ema(c, 21)
    top, bot = np.maximum(e8, e21), np.minimum(e8, e21)
    t["rib_align"] = np.where(d == 1, e8[i] > e21[i], e8[i] < e21[i])
    t["rib_strict"] = t.rib_align & np.where(d == 1, l[i] > top[i], h[i] < bot[i])
    t["rib_close"] = t.rib_align & np.where(d == 1, cc > top[i], cc < bot[i])
    t["rib_against"] = np.where(d == 1, e8[i] < e21[i], e8[i] > e21[i])  # counter-trend reversal
    e50 = ema(c, 50)
    t["ema50"] = d * (cc - e50[i]) > 0
    t["htf120"] = d * (cc - m.e120[jl]) > 0
    t["htf480"] = d * (cc - m.e480[jl]) > 0
    t["htf_slope"] = d * (m.e120[jl] - m.e120[np.maximum(jl - 30, 0)]) > 0
    t["vwap_rth"] = d * (cc - m.vwap_rth[jl]) > 0
    t["vwap_eth"] = d * (cc - m.vwap_eth[jl]) > 0
    t["ftfc_day"] = d * (cc - m.day_open[jl]) > 0
    t["ftfc_60"] = d * (cc - m.h60_open[jl]) > 0
    t["ftfc_wk"] = d * (cc - m.wk_open[jl]) > 0
    t["ftfc_pdc"] = d * (cc - m.pdc[jl]) > 0
    t["daily_up"] = d * (m.pdc[jl] - m.d20[jl]) > 0
    # sweeps: the failed 2 wick took out a reference level and closed back
    ext = np.where(d == 1, l[i], h[i])
    first1 = b["first"][i]
    prev = np.maximum(first1 - 1, 0)
    rlo = np.where(m.rth[prev] & (m.tdate[prev] == m.tdate[first1]), m.run_lo[prev], np.nan)
    rhi = np.where(m.rth[prev] & (m.tdate[prev] == m.tdate[first1]), m.run_hi[prev], np.nan)
    t["sw_lod"] = np.where(d == 1, ext < rlo, ext > rhi)  # new high/low of day, then failed
    t["sw_on"] = np.where(d == 1, ext < m.on_lo[jl], ext > m.on_hi[jl])
    t["sw_pd"] = np.where(d == 1, ext < m.pdl[jl], ext > m.pdh[jl])
    t["sw_or"] = (bmod[i] >= 600) & np.where(d == 1, ext < m.or_lo[jl], ext > m.or_hi[jl])
    t["sw_any"] = t.sw_lod | t.sw_on | t.sw_pd | t.sw_or
    # location in day range (0 = at the extreme against the trade, 1 = at the other end)
    dr = (rhi - rlo)
    t["loc_day"] = np.where(d == 1, (cc - rlo) / dr, (rhi - cc) / dr)
    t["sweep_ticks"] = np.where(d == 1, pl[i] - l[i], h[i] - ph[i]) / TICK
    t["close_loc"] = np.where(d == 1, (cc - l[i]) / (h[i] - l[i]), (h[i] - cc) / (h[i] - l[i]))
    t["body"] = np.abs(cc - o[i]) / (h[i] - l[i])
    # previous bar scenario (bar i-1 relative to i-2): 2 in the trap direction, 1 (inside), 3 (outside)
    pph, ppl = np.r_[np.nan, np.nan, h[:-2]], np.r_[np.nan, np.nan, l[:-2]]
    p2u = (h > ph) & (l >= pl); p2d = (l < pl) & (h <= ph); p1 = (h <= ph) & (l >= pl)
    prv2u = np.r_[False, p2u[:-1]][i]; prv2d = np.r_[False, p2d[:-1]][i]; prv1 = np.r_[False, p1[:-1]][i]
    t["prev_same"] = np.where(d == 1, prv2d, prv2u)  # e.g. 2D-2D(failed): continuation trap
    t["prev_opp"] = np.where(d == 1, prv2u, prv2d)
    t["prev_in"] = prv1
    t["relvol"] = v[i] / pd.Series(v).rolling(20).mean().shift(1).to_numpy()[i]
    t["tod"] = bmod[i]
    t["dow"] = t.time.dt.dayofweek
    t["n"] = n; t["off"] = offset
    return t.reset_index(drop=True)


def nonoverlap(t, tm):
    """Keep signals taken in order while flat (exit index of the previous trade < entry)."""
    keep = np.zeros(len(t), bool); last = -1
    J, X = t.j.to_numpy(), t[f"x{tm}"].to_numpy()
    for k in range(len(t)):
        if J[k] > last:
            keep[k] = True; last = X[k]
    return t[keep]


def pf(p):
    p = np.asarray(p); w = p[p > 0].sum(); ls = -p[p < 0].sum()
    return w / ls if ls > 0 else np.inf


def summary(t, tm):
    p = t[f"p{tm}"]
    return dict(trades=len(p), net=round(p.sum()), pf=round(pf(p), 2), wr=round(100 * (p > 0).mean(), 1),
                avg=round(p.mean(), 1) if len(p) else 0)


# ---------------- alternative entries ----------------
@njit(cache=True)
def _sim_from(o, h, l, mod, tdate, k0, entry, d, stop, tgt, tick, intrabar, thru=0.0):
    """Manage a position opened at minute k0 (intrabar=True: filled during k0, so only the stop is checked there)."""
    td = tdate[k0]; N = len(o); k = k0
    while k < N:
        if tdate[k] != td or mod[k] >= 955:
            return o[k] - d * tick, k
        if not (intrabar and k == k0):
            if d == 1:
                if o[k] <= stop: return o[k] - tick, k
                if o[k] >= tgt: return o[k], k
            else:
                if o[k] >= stop: return o[k] + tick, k
                if o[k] <= tgt: return o[k], k
        if d == 1:
            if l[k] <= stop: return stop - tick, k
            if not (intrabar and k == k0) and h[k] >= tgt + thru: return tgt, k
        else:
            if h[k] >= stop: return stop + tick, k
            if not (intrabar and k == k0) and l[k] <= tgt - thru: return tgt, k
        k += 1
    return o[N - 1], N - 1


@njit(cache=True)
def sim_entry(o, h, l, mod, tdate, J, D, S, PX, KIND, EXP, mults, tick, thru=0.0, eslip=1.0):
    """KIND 0 = market at o[J]; 1 = limit at PX; 2 = stop-entry at PX. Order lives until minute EXP
    (exclusive) and is cancelled if the stop level trades first. Returns fill idx (-1 none), entry px, exits."""
    n = len(J); K = len(mults)
    fi = np.full(n, -1, np.int64); ep = np.full(n, np.nan)
    px = np.full((n, K), np.nan); ix = np.full((n, K), -1, np.int64)
    for i in range(n):
        d = D[i]; stop = S[i]; j = J[i]; td = tdate[j]
        k = j; got = False; intrabar = False; e = 0.0
        if KIND[i] == 0:
            got = True; e = o[j] + d * tick * eslip
        else:
            while k < EXP[i] and k < len(o) and tdate[k] == td and mod[k] < 955:
                lim = PX[i]
                if KIND[i] == 1:  # limit: long buys at/below lim
                    if d == 1:
                        if o[k] <= lim:
                            if o[k] <= stop: break
                            got = True; e = o[k]; break
                        if l[k] <= lim: got = True; e = lim; intrabar = True; break
                    else:
                        if o[k] >= lim:
                            if o[k] >= stop: break
                            got = True; e = o[k]; break
                        if h[k] >= lim: got = True; e = lim; intrabar = True; break
                else:  # stop-entry: long buys at/above lim; cancel if stop trades first
                    if d == 1:
                        if o[k] >= lim: got = True; e = o[k] + tick * eslip; break
                        if l[k] <= stop: break
                        if h[k] >= lim: got = True; e = lim + tick * eslip; intrabar = True; break
                    else:
                        if o[k] <= lim: got = True; e = o[k] - tick * eslip; break
                        if h[k] >= stop: break
                        if l[k] <= lim: got = True; e = lim - tick * eslip; intrabar = True; break
                k += 1
        if not got: continue
        risk = d * (e - stop)
        if risk <= 0: continue
        fi[i] = k; ep[i] = e
        for q in range(K):
            tgt = e + d * mults[q] * risk
            px[i, q], ix[i, q] = _sim_from(o, h, l, mod, tdate, k, e, d, stop, tgt, tick, intrabar, thru)
    return fi, ep, px, ix


def with_entry(m, t, kind, expiry_bars=2, thru_ticks=0, entry_slip_ticks=1, stop_from_entry=False):
    """Re-simulate signals `t` (from signals()) with another entry. kind: market|retest|mid|trigger."""
    t = t.copy()
    J = t.j.to_numpy(); D = t.dir.to_numpy(); S = t.stop.to_numpy()
    lo, hi = t["_l"].to_numpy(), t["_h"].to_numpy()
    if kind == "market":
        K = np.zeros(len(t), np.int64); PX = np.zeros(len(t))
    elif kind == "retest":  # back to the broken prior-bar level
        K = np.ones(len(t), np.int64); PX = np.where(D == 1, t["_pl"], t["_ph"])
    elif kind == "mid":
        K = np.ones(len(t), np.int64); PX = (lo + hi) / 2
    elif kind == "trigger":  # break of the failed 2 bar's other end
        K = np.full(len(t), 2, np.int64); PX = np.where(D == 1, hi, lo)
    PX = np.round(PX / TICK) * TICK
    EXP = J + expiry_bars * t.n.to_numpy()
    fi, ep, px, ix = sim_entry(m.o, m.h, m.l, m.mod, m.tdate, J, D, S, PX, K, EXP, np.array(TARGETS, float), TICK,
                               thru_ticks * TICK, float(entry_slip_ticks))
    t["j"] = fi; t["entry"] = ep
    for q, tm in enumerate(TARGETS):
        t[f"p{tm}"] = (px[:, q] - ep) * D * PV - RT; t[f"x{tm}"] = ix[:, q]
    t["risk"] = D * (ep - S)
    return t[fi >= 0].reset_index(drop=True)
