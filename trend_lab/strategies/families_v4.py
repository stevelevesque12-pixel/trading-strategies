"""
Batch 4: iterations 2-3 found that 15m "dip inside a higher-timeframe trend"
entries (trend_dip_rsi, trend_dip_atr) are the only region whose top
in-sample configs survive out-of-sample. Breakouts/crossovers don't.
This batch: (a) union of both dip triggers for more trades per week (the
Lucid $3k target needs frequency), (b) regime filters on top (ADX /
efficiency ratio / volatility percentile), (c) a weekly-VWAP dip variant.
"""

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import COMMON_SPACE, atr, cached, common_exits, ema, htf_ema, swing_stop
from .families_v3 import TREND_SPACE, _trend


def _regime(df, p, a):
    """Optional regime gate: trend strength and a volatility band."""
    ok = np.ones(len(df), bool)
    if p.get("adx_min"):
        adx = cached(df, ("adx", 14), lambda: ind.adx(df, 14).to_numpy())
        ok &= adx >= p["adx_min"]
    if p.get("er_min"):
        er = cached(df, ("er", 20), lambda: ind.efficiency_ratio(df["close"], 20).to_numpy())
        ok &= er >= p["er_min"]
    if p.get("vol_band"):
        lo_q, hi_q = p["vol_band"]
        pct = cached(df, ("atr_pct", 14, 500),
                     lambda: pd.Series(a).rolling(500, min_periods=100).rank(pct=True).to_numpy())
        ok &= (pct >= lo_q) & (pct <= hi_q)
    return ok


REGIME_SPACE = {
    "adx_min": [0, 15, 20, 25],
    "er_min": [0, 0.1, 0.2],
    "vol_band": [None, (0.2, 1.0), (0.0, 0.8), (0.2, 0.9)],
}


def _dip_triggers(df, p, a):
    h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
    f = ema(df, p["fast"])
    db = p["dip_bars"]
    atr_dip_l = pd.Series(lo <= f - p["dip_k"] * a).rolling(db, min_periods=1).max().to_numpy() > 0
    atr_dip_s = pd.Series(h >= f + p["dip_k"] * a).rolling(db, min_periods=1).max().to_numpy() > 0
    r = cached(df, ("rsi", p["rsi_n"]), lambda: ind.rsi(df["close"], p["rsi_n"]).fillna(50).to_numpy())
    rsi_dip_l = pd.Series(r < p["rsi_lo"]).rolling(db, min_periods=1).max().to_numpy() > 0
    rsi_dip_s = pd.Series(r > 100 - p["rsi_lo"]).rolling(db, min_periods=1).max().to_numpy() > 0
    mode = p["trigger"]
    if mode == "atr":
        dl, ds = atr_dip_l, atr_dip_s
    elif mode == "rsi":
        dl, ds = rsi_dip_l, rsi_dip_s
    elif mode == "either":
        dl, ds = atr_dip_l | rsi_dip_l, atr_dip_s | rsi_dip_s
    else:  # both
        dl, ds = atr_dip_l & rsi_dip_l, atr_dip_s & rsi_dip_s
    return dl & (c > np.roll(h, 1)), ds & (c < np.roll(lo, 1))


DIP_SPACE = TREND_SPACE | REGIME_SPACE | {
    "fast": [9, 13, 21],
    "dip_k": [0.25, 0.5, 1.0],
    "dip_bars": [2, 4, 6],
    "rsi_n": [2, 3, 5],
    "rsi_lo": [15, 25, 35],
    "trigger": ["atr", "rsi", "either", "both"],
    "swing_lb": [3, 5, 8],
}


# --------------------------------------------------------------------------
# 13. Unified dip-in-trend with regime gate
# --------------------------------------------------------------------------
class TrendDipCombo:
    name = "trend_dip_combo"
    description = ("Higher-TF trend (EMA side + slope) and base EMA side; dip trigger = ATR distance below a fast "
                   "EMA and/or a short-RSI oversold read; enter on a close above the prior high. Optional "
                   "ADX / efficiency-ratio / volatility-percentile regime gate.")
    space = DIP_SPACE

    def generate(self, df, p):
        a = atr(df)
        up, dn = _trend(df, p, a)
        tl, ts = _dip_triggers(df, p, a)
        reg = _regime(df, p, a)
        long, short = up & tl & reg, dn & ts & reg
        stop = np.where(long, swing_stop(df, p["swing_lb"], 0.1, a, 1), swing_stop(df, p["swing_lb"], 0.1, a, -1))
        return {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)


# --------------------------------------------------------------------------
# 14. Long-only / short-only variants of the combo (crude trended down most of
#     the 15m sample -- check whether the edge is just one side)
# --------------------------------------------------------------------------
class TrendDipOneSide:
    name = "trend_dip_oneside"
    description = "Same as trend_dip_combo but restricted to one side, to test whether the edge is symmetric."
    space = DIP_SPACE | {"side": ["long", "short"]}

    def generate(self, df, p):
        sig = TrendDipCombo().generate(df, p)
        if p["side"] == "long":
            sig["short"] = np.zeros(len(df), bool)
        else:
            sig["long"] = np.zeros(len(df), bool)
        return sig


# --------------------------------------------------------------------------
# 15. Dip to the weekly anchored VWAP in a weekly trend
# --------------------------------------------------------------------------
class WeeklyVWAPDip:
    name = "weekly_vwap_dip"
    description = ("Weekly VWAP (anchored at Sunday's open) as the trend reference: when price has been above a "
                   "rising weekly VWAP, buy a touch of the VWAP band that closes back up. Mirror for shorts.")
    space = {k: v for k, v in COMMON_SPACE.items() if k not in ("htf_len",)} | {
        "htf_len": [0, 20, 50],
        "band_atr": [0.0, 0.5, 1.0],
        "slope_bars": [8, 24],
        "stop_k": [0.5, 1.0, 1.5],
    }

    def generate(self, df, p):
        a = atr(df)
        o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))

        def wvwap():
            td = pd.to_datetime(pd.Series(df["trade_day"].to_numpy(), index=df.index))
            week = (td - pd.to_timedelta(td.dt.weekday, unit="D")).dt.date.to_numpy()
            tp = (df["high"] + df["low"] + df["close"]) / 3
            vol = df["volume"].replace(0, 1.0)
            return ((tp * vol).groupby(week).cumsum() / vol.groupby(week).cumsum()).to_numpy()

        v = cached(df, ("wvwap",), wvwap)
        sl = v - np.roll(v, p["slope_bars"])
        band = p["band_atr"] * a
        long = (sl > 0) & (lo <= v + band) & (c > v) & (c > o)
        short = (sl < 0) & (h >= v - band) & (c < v) & (c < o)
        if p["htf_len"]:
            he = htf_ema(df, p["htf_rule"], p["htf_len"])
            long, short = long & (c > he), short & (c < he)
        stop = np.where(long, c - np.minimum(lo, v), np.maximum(h, v) - c) + p["stop_k"] * a
        return {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)


FAMILIES = [TrendDipCombo(), TrendDipOneSide(), WeeklyVWAPDip()]
