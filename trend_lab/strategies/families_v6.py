"""
Batch 6: more dip-in-trend mechanisms (the only region that has survived
walk-forward so far), aimed at adding *uncorrelated* trades to the
trend_dip_atr + trend_dip_rsi portfolio, plus a trend-day gate.
"""

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import atr, cached, common_exits, ema, swing_stop
from .families_v3 import TREND_SPACE, _trend


class TrendDipStoch:
    name = "trend_dip_stoch"
    description = ("Higher-TF trend; slow stochastic %K crosses back above %D from below an oversold level, "
                   "with a close above the prior high. Stop under the recent swing.")
    space = TREND_SPACE | {
        "k_len": [5, 9, 14],
        "d_len": [3],
        "os": [10, 20, 30],
        "swing_lb": [3, 5, 8],
    }

    def generate(self, df, p):
        a = atr(df)
        h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))

        def stoch():
            hh = df["high"].rolling(p["k_len"]).max()
            ll = df["low"].rolling(p["k_len"]).min()
            k = (100 * (df["close"] - ll) / (hh - ll).replace(0, np.nan)).rolling(3).mean()
            return k.to_numpy(), k.rolling(p["d_len"]).mean().to_numpy()

        k, d = cached(df, ("stoch", p["k_len"], p["d_len"]), stoch)
        up, dn = _trend(df, p, a)
        kp, dp = np.roll(k, 1), np.roll(d, 1)
        with np.errstate(invalid="ignore"):
            long = up & (kp < p["os"]) & (kp <= dp) & (k > d) & (c > np.roll(h, 1))
            short = dn & (kp > 100 - p["os"]) & (kp >= dp) & (k < d) & (c < np.roll(lo, 1))
        stop = np.where(long, swing_stop(df, p["swing_lb"], 0.1, a, 1), swing_stop(df, p["swing_lb"], 0.1, a, -1))
        return {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)


class FibPullback:
    name = "fib_pullback"
    description = ("Higher-TF trend; measure the latest impulse leg (lowest low to highest high of the last N "
                   "bars, low first). Buy when price retraces into the 38-62% zone and closes back up.")
    space = TREND_SPACE | {
        "leg_n": [12, 24, 48],
        "zone": [(0.382, 0.618), (0.5, 0.786), (0.236, 0.5)],
        "min_leg_atr": [2.0, 3.0, 5.0],
    }

    def generate(self, df, p):
        a = atr(df)
        o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        n = p["leg_n"]
        hh = pd.Series(h).rolling(n).max().shift(1).to_numpy()
        ll = pd.Series(lo).rolling(n).min().shift(1).to_numpy()
        hi_age = pd.Series(h).rolling(n).apply(lambda x: n - 1 - np.argmax(x), raw=True).shift(1).to_numpy()
        lo_age = pd.Series(lo).rolling(n).apply(lambda x: n - 1 - np.argmin(x), raw=True).shift(1).to_numpy()
        leg = hh - ll
        z0, z1 = p["zone"]
        up, dn = _trend(df, p, a)
        with np.errstate(invalid="ignore"):
            big = leg >= p["min_leg_atr"] * a
            up_leg = big & (lo_age > hi_age)      # low came first, then the high: an up-leg
            dn_leg = big & (hi_age > lo_age)
            in_zone_l = (lo <= hh - z0 * leg) & (lo >= hh - z1 * leg)
            in_zone_s = (h >= ll + z0 * leg) & (h <= ll + z1 * leg)
            long = up & up_leg & in_zone_l & (c > o)
            short = dn & dn_leg & in_zone_s & (c < o)
            stop = np.where(long, c - (hh - z1 * leg), (ll + z1 * leg) - c) + 0.1 * a
        return {"long": long, "short": short, "stop_dist": np.nan_to_num(stop, nan=-1)} | common_exits(p, a)


class TrendDipATRDayGate:
    name = "trend_dip_atr_daygate"
    description = ("trend_dip_atr, but only on 'trend days': the session's efficiency ratio so far (net move / "
                   "path since the 18:00 open) must exceed a threshold in the trade's direction.")
    space = TREND_SPACE | {
        "fast": [13],
        "dip_k": [0.5, 1.0],
        "dip_bars": [4, 6],
        "swing_lb": [5, 8],
        "day_er": [0.0, 0.1, 0.2, 0.3],
    }

    def generate(self, df, p):
        from .families_v3 import TrendDipATR
        sig = TrendDipATR().generate(df, p)
        c = df["close"].to_numpy()
        g = df["trade_day"].to_numpy()

        def day_er():
            s = pd.Series(c)
            first = s.groupby(g).transform("first").to_numpy()
            path = s.diff().abs().fillna(0).groupby(g).cumsum().to_numpy()
            net = c - first
            return net / np.where(path > 0, path, np.nan)

        er = np.nan_to_num(cached(df, ("day_er",), day_er))
        sig["long"] = sig["long"] & (er >= p["day_er"])
        sig["short"] = sig["short"] & (er <= -p["day_er"])
        return sig


FAMILIES = [TrendDipStoch(), FibPullback(), TrendDipATRDayGate()]
