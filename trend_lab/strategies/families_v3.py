"""
Batch 3: iteration 2 showed "buy the dip inside a higher-timeframe trend"
(trend_dip_rsi) was the first family whose top in-sample configs held up
out-of-sample. This batch explores that region harder (different dip
measures, more trades per week) and adds two level-based trend ideas.
"""

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import COMMON_SPACE, atr, cached, common_exits, ema, htf_bias, htf_ema, swing_stop


def _trend(df, p, a):
    """Shared trend definition: HTF EMA side + HTF EMA slope + base EMA side."""
    c = df["close"].to_numpy()
    he = htf_ema(df, p["htf_rule"], p["htf_len"])
    hslope = he - np.roll(he, p.get("slope_bars", 12))
    e = ema(df, p["base_ema"])
    up = (c > he) & (hslope > 0) & (c > e)
    dn = (c < he) & (hslope < 0) & (c < e)
    return up, dn


TREND_SPACE = {k: v for k, v in COMMON_SPACE.items() if k != "htf_len"} | {
    "htf_len": [20, 50],
    "slope_bars": [4, 12, 24],
    "base_ema": [20, 50, 100],
}


# --------------------------------------------------------------------------
# 10. Dip measured as distance below the fast EMA in ATRs
# --------------------------------------------------------------------------
class TrendDipATR:
    name = "trend_dip_atr"
    description = ("Uptrend on the higher timeframe; a dip is a low at least k ATR below a fast EMA within "
                   "the last few bars; enter when a bar closes back above the prior high. Stop under the dip.")
    space = TREND_SPACE | {
        "fast": [9, 13, 21],
        "dip_k": [0.25, 0.5, 1.0, 1.5],
        "dip_bars": [2, 4, 6],
        "swing_lb": [3, 5, 8],
    }

    def generate(self, df, p):
        a = atr(df)
        h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
        f = ema(df, p["fast"])
        up, dn = _trend(df, p, a)
        dipped = pd.Series(lo <= f - p["dip_k"] * a).rolling(p["dip_bars"], min_periods=1).max().to_numpy() > 0
        popped = pd.Series(h >= f + p["dip_k"] * a).rolling(p["dip_bars"], min_periods=1).max().to_numpy() > 0
        long = up & dipped & (c > np.roll(h, 1))
        short = dn & popped & (c < np.roll(lo, 1))
        stop = np.where(long, swing_stop(df, p["swing_lb"], 0.1, a, 1), swing_stop(df, p["swing_lb"], 0.1, a, -1))
        return {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)


# --------------------------------------------------------------------------
# 11. N consecutive counter-trend closes, then a with-trend close
# --------------------------------------------------------------------------
class TrendCountback:
    name = "trend_countback"
    description = ("Uptrend on the higher timeframe; after N consecutive lower closes (a pullback), enter on "
                   "the first higher close. Stop under the pullback low.")
    space = TREND_SPACE | {
        "n_down": [2, 3, 4],
        "swing_lb": [3, 5, 8],
        "exit_bars": [None, 8, 16],
    }

    def generate(self, df, p):
        a = atr(df)
        c = df["close"].to_numpy()
        up, dn = _trend(df, p, a)
        d = np.sign(np.diff(c, prepend=c[0]))
        k = p["n_down"]
        downs = pd.Series(d < 0).rolling(k).sum().shift(1).to_numpy() == k
        ups = pd.Series(d > 0).rolling(k).sum().shift(1).to_numpy() == k
        long = up & downs & (d > 0)
        short = dn & ups & (d < 0)
        stop = np.where(long, swing_stop(df, p["swing_lb"], 0.1, a, 1), swing_stop(df, p["swing_lb"], 0.1, a, -1))
        sig = {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)
        sig["max_bars"] = p["exit_bars"]
        return sig


# --------------------------------------------------------------------------
# 12. Prior-day high/low break in the HTF trend direction, entered on retest
# --------------------------------------------------------------------------
class PDHLRetest:
    name = "pdhl_retest"
    description = ("Crude respects prior-day high/low. In an HTF uptrend, after price closes above the prior "
                   "trading day's high, buy the first pullback that holds that level. Mirror for shorts.")
    space = {k: v for k, v in COMMON_SPACE.items() if k != "htf_len"} | {
        "htf_len": [0, 20, 50],
        "retest_atr": [0.1, 0.25, 0.5],
        "stop_k": [0.5, 1.0, 1.5],
        "break_bars": [8, 16, 32],
    }

    def generate(self, df, p):
        a = atr(df)
        h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))

        def levels():
            g = df["trade_day"]
            dh = df.groupby(g)["high"].max().shift(1)
            dl = df.groupby(g)["low"].min().shift(1)
            return g.map(dh).to_numpy(dtype=float), g.map(dl).to_numpy(dtype=float)

        pdh, pdl = cached(df, ("pdhl",), levels)
        bias = htf_bias(df, p["htf_rule"], p["htf_len"])
        broke_up = pd.Series(c > pdh).rolling(p["break_bars"], min_periods=1).max().to_numpy() > 0
        broke_dn = pd.Series(c < pdl).rolling(p["break_bars"], min_periods=1).max().to_numpy() > 0
        tol = p["retest_atr"] * a
        long = broke_up & (lo <= pdh + tol) & (c > pdh) & (c > np.roll(c, 1))
        short = broke_dn & (h >= pdl - tol) & (c < pdl) & (c < np.roll(c, 1))
        if p["htf_len"]:
            long, short = long & (bias > 0), short & (bias < 0)
        stop = np.where(long, c - pdh, pdl - c) + p["stop_k"] * a
        return {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)


FAMILIES = [TrendDipATR(), TrendCountback(), PDHLRetest()]
