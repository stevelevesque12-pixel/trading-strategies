"""Offline research for LARSA v14's Gear 2 ("harvest mode").

Rebuilds the harvest signal on hourly bars made from the repo's ES/NQ 15m
data and compares variants across three periods:

    train    2016-06 .. 2021-12   (variants were picked by looking at this)
    test     2022-01 .. 2024-12   (the v14 backtest window's tail, untouched while choosing)
    holdout  2025-01 .. 2026-08   (after the QC backtest's end date)

Fills are at the NEXT bar's open (no same-bar lookahead). Costs are a round
trip in points per contract: ES 0.5 (2 ticks), NQ 1.0 (4 ticks), covering
commission + slippage. Results are per 1 contract.

    python -m larsa.harvest_research
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "sample_data" / "real_multi_instrument"
POINT_VALUE = {"ES": 50.0, "NQ": 20.0}
COST_PTS = {"ES": 0.5, "NQ": 1.0}
PERIODS = {
    "train": ("2016-06-01", "2022-01-01"),
    "test": ("2022-01-01", "2025-01-01"),
    "holdout": ("2025-01-01", "2026-09-01"),
}


def load_15m(sym: str) -> pd.DataFrame:
    path = next(DATA.glob(f"real_{sym.lower()}_15m_2016-*.parquet"))
    return pd.read_parquet(path).tz_convert("America/New_York")


def to_hourly(df: pd.DataFrame) -> pd.DataFrame:
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample("1h").agg(agg).dropna(subset=["close"])


def to_daily_rth(df: pd.DataFrame) -> pd.DataFrame:
    rth = df.between_time("09:30", "15:45")
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    return rth.resample("1D").agg(agg).dropna(subset=["close"])


def _ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def _wilder(s, n):
    return s.ewm(alpha=1 / n, adjust=False).mean()


def regime(h: pd.DataFrame) -> pd.Series:
    """Approximation of LARSA's detect_regime(): 0 BULL, 1 BEAR, 2 SIDEWAYS, 3 HIGH_VOL."""
    c, hi, lo = h["close"], h["high"], h["low"]
    tr = pd.concat([hi - lo, (hi - c.shift()).abs(), (lo - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14)
    up, dn = hi.diff(), -lo.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), h.index)
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), h.index)
    pdi, mdi = 100 * _wilder(pdm, 14) / atr, 100 * _wilder(mdm, 14) / atr
    adx = _wilder(100 * (pdi - mdi).abs() / (pdi + mdi), 14)
    e12, e26, e50, e200 = (_ema(c, n) for n in (12, 26, 50, 200))
    ratio = atr / atr.rolling(50).mean()
    bull_stack = (e12 > e26) & (e26 > e50) & (e50 > e200)
    bear_stack = (e200 > e50) & (e50 > e26) & (e26 > e12)
    bull = (c > e200) & (e12 > e26) & (adx > np.where(bull_stack, 14, 18))
    bear = (c < e200) & (e12 < e26) & (adx > np.where(bear_stack, 14, 18))
    out = pd.Series(2, h.index)
    out[bear] = 1
    out[bull] = 0
    out[ratio >= 1.5] = 3
    return out


def zscore(c: pd.Series, n: int, include_current: bool = True) -> pd.Series:
    ref = c if include_current else c.shift(1)
    mu, sd = ref.rolling(n).mean(), ref.rolling(n).std(ddof=0)
    return (c - mu) / sd


@dataclass
class Rule:
    name: str
    lookback: int = 20
    z_entry: float = 1.5
    z_exit: float = 0.3
    z_stop: float = 2.8
    sideways_only: bool = True
    long_only: bool = False
    rearm: bool = False          # after a stop, wait for |z| < z_entry before re-entering
    rth_only: bool = False       # entries 10:00-15:00 ET; flat by 16:00
    max_bars: int = 0            # time stop (0 = off)
    trend_filter: bool = False   # longs only when close > 200-bar EMA
    exit_on_regime: bool = True


def run_hourly(h: pd.DataFrame, reg: pd.Series, r: Rule):
    """Event loop over hourly bars. Signals on bar i's close, fills at bar i+1's open."""
    c = h["close"]
    z = zscore(c, r.lookback).values
    side = (reg == 2).values
    o = h["open"].values
    hour = h.index.hour.values
    up = (c > _ema(c, 200)).values
    n = len(h)
    trades = []  # (entry_time, exit_time, dir, pts)
    pos, ent, ent_i, armed_l, armed_s = 0, 0.0, 0, True, True
    for i in range(n - 1):
        zi = z[i]
        if np.isnan(zi):
            continue
        if not armed_s and zi < r.z_entry:
            armed_s = True
        if not armed_l and zi > -r.z_entry:
            armed_l = True
        in_window = (10 <= hour[i] < 15) if r.rth_only else True
        if pos != 0:
            stop = (pos < 0 and zi > r.z_stop) or (pos > 0 and zi < -r.z_stop)
            tp = (pos < 0 and zi < r.z_exit) or (pos > 0 and zi > -r.z_exit)
            reg_exit = r.sideways_only and r.exit_on_regime and not side[i]
            t_exit = r.max_bars and (i - ent_i) >= r.max_bars
            eod = r.rth_only and hour[i] >= 15
            if stop or tp or reg_exit or t_exit or eod:
                px = o[i + 1]
                trades.append((h.index[ent_i], h.index[i + 1], pos, pos * (px - ent)))
                if stop and r.rearm:
                    armed_s, armed_l = (False, armed_l) if pos < 0 else (armed_s, False)
                pos = 0
            continue
        if r.sideways_only and not side[i]:
            continue
        if not in_window:
            continue
        if zi > r.z_entry and armed_s and not r.long_only:
            pos, ent, ent_i = -1, o[i + 1], i + 1
        elif zi < -r.z_entry and armed_l and (not r.trend_filter or up[i]):
            pos, ent, ent_i = 1, o[i + 1], i + 1
    return pd.DataFrame(trades, columns=["entry", "exit", "dir", "pts"])


def run_daily_rsi2(d: pd.DataFrame, rsi_entry=10.0, exit_ma=5, trend_ma=200, stop_atr=0.0):
    """Connors-style daily dip buy: RSI(2) < rsi_entry and close > SMA(trend_ma);
    exit when close > SMA(exit_ma), or (if stop_atr) a close below
    entry - stop_atr * ATR(14) at entry. Signal at close, fill at next RTH open."""
    c = d["close"]
    tr = pd.concat([d["high"] - d["low"], (d["high"] - c.shift()).abs(),
                    (d["low"] - c.shift()).abs()], axis=1).max(axis=1)
    atr = _wilder(tr, 14).values
    cv = c.values
    delta = c.diff()
    gain = _wilder(delta.clip(lower=0), 2)
    loss = _wilder(-delta.clip(upper=0), 2)
    rsi = (100 - 100 / (1 + gain / loss)).values
    trend = (c > c.rolling(trend_ma).mean()).values
    ex = (c > c.rolling(exit_ma).mean()).values
    o = d["open"].values
    trades, pos, ent, ent_i, stop_px = [], 0, 0.0, 0, -np.inf
    for i in range(len(d) - 1):
        if pos and (ex[i] or cv[i] < stop_px):
            trades.append((d.index[ent_i], d.index[i + 1], 1, o[i + 1] - ent))
            pos = 0
        elif not pos and trend[i] and rsi[i] < rsi_entry:
            pos, ent, ent_i = 1, o[i + 1], i + 1
            stop_px = ent - stop_atr * atr[i] if stop_atr else -np.inf
    return pd.DataFrame(trades, columns=["entry", "exit", "dir", "pts"])


def summarize(trades: pd.DataFrame, sym: str, start: str, end: str) -> dict:
    t = trades[(trades["entry"] >= pd.Timestamp(start, tz="America/New_York"))
               & (trades["entry"] < pd.Timestamp(end, tz="America/New_York"))]
    years = (pd.Timestamp(end) - pd.Timestamp(start)).days / 365.25
    if t.empty:
        return {"n": 0, "per_wk": 0.0, "win": np.nan, "net_pts": np.nan, "usd_yr": 0.0, "pf": np.nan, "maxdd_usd": 0.0}
    net = t["pts"] - COST_PTS[sym]
    usd = net * POINT_VALUE[sym]
    eq = usd.cumsum()
    gains, losses = usd[usd > 0].sum(), -usd[usd < 0].sum()
    return {
        "n": len(t),
        "per_wk": len(t) / years / 52,
        "win": float((net > 0).mean()),
        "net_pts": float(net.mean()),
        "usd_yr": float(usd.sum() / years),
        "pf": float(gains / losses) if losses > 0 else np.inf,
        "maxdd_usd": float((eq.cummax() - eq).max()),
    }


RULES = [
    Rule("v14 as written"),
    Rule("v14 + re-arm after stop", rearm=True),
    Rule("no regime gate", sideways_only=False, rearm=True),
    Rule("long-only", long_only=True, rearm=True),
    Rule("long-only, above 200h EMA", long_only=True, rearm=True, trend_filter=True),
    Rule("RTH-only (10-15 ET), flat by 16", rth_only=True, rearm=True),
    Rule("z 2.0 entry", z_entry=2.0, z_stop=3.2, rearm=True),
    Rule("lookback 50", lookback=50, rearm=True),
    Rule("time stop 6 bars", max_bars=6, rearm=True),
    Rule("long-only, no regime gate, uptrend", long_only=True, sideways_only=False,
         trend_filter=True, rearm=True),
]


def main():
    pd.set_option("display.width", 200)
    rows = []
    for sym in ["ES", "NQ"]:
        m15 = load_15m(sym)
        h = to_hourly(m15)
        reg = regime(h)
        for r in RULES:
            tr = run_hourly(h, reg, r)
            for p, (s, e) in PERIODS.items():
                rows.append({"sym": sym, "rule": r.name, "period": p, **summarize(tr, sym, s, e)})
        d = to_daily_rth(m15)
        for label, kw in [("DAILY RSI(2)<10 dip-buy > SMA200", {}),
                          ("  same + 3xATR catastrophic stop", {"stop_atr": 3.0})]:
            tr = run_daily_rsi2(d, **kw)
            for p, (s, e) in PERIODS.items():
                rows.append({"sym": sym, "rule": label, "period": p, **summarize(tr, sym, s, e)})
    out = pd.DataFrame(rows)
    piv = out.pivot_table(index=["sym", "rule"], columns="period",
                          values=["net_pts", "usd_yr", "per_wk", "pf"], sort=False)
    piv = piv.reindex(columns=["train", "test", "holdout"], level=1)
    print(piv.round(2).to_string())
    return out


if __name__ == "__main__":
    main()
