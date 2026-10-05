"""
Batch 10: trend-definition sensitivity for Engine A. Same dip trigger, stop and
exits as the validated trend_dip_atr settings; only the higher-timeframe trend
filter changes. A real "buy dips in a trend" edge should not depend on one
indicator's definition of trend.
"""

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import atr, cached, common_exits, ema, swing_stop
from .cross_market_params import ENGINE_A_PARAMS


def _htf_series(df, rule, fn, key):
    return cached(df, ("htf_generic", rule, key), lambda: ind.htf(df, rule, fn).to_numpy())


class TrendDipATRTrendVar:
    name = "trend_dip_atr_trendvar"
    description = ("Engine A with alternative 1h trend definitions: EMA50 side+slope (baseline), Supertrend(10,3) "
                   "direction, close in the upper/lower half of the 20-bar Donchian channel, or 20-bar "
                   "linear-regression slope sign.")
    space = {"trend_mode": ["ema", "supertrend", "donchian", "linreg"], "require_base_ema": [True, False]}

    def generate(self, df, q):
        p = dict(ENGINE_A_PARAMS)
        a = atr(df)
        h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
        rule = p["htf_rule"]
        mode = q["trend_mode"]
        if mode == "ema":
            he = _htf_series(df, rule, lambda x: ind.ema(x["close"], p["htf_len"]), "ema50")
            sl = he - np.roll(he, p["slope_bars"])
            up, dn = (c > he) & (sl > 0), (c < he) & (sl < 0)
        elif mode == "supertrend":
            d = _htf_series(df, rule, lambda x: ind.supertrend(x, 10, 3.0)[1], "st")
            up, dn = d > 0, d < 0
        elif mode == "donchian":
            mid = _htf_series(df, rule, lambda x: (x["high"].rolling(20).max() + x["low"].rolling(20).min()) / 2,
                              "dmid")
            up, dn = c > mid, c < mid
        else:
            def slope(x):
                y = x["close"]
                w = np.arange(20) - 9.5
                return y.rolling(20).apply(lambda v: float(np.dot(v, w)), raw=True)
            s = _htf_series(df, rule, slope, "lr20")
            up, dn = s > 0, s < 0
        up, dn = np.nan_to_num(up).astype(bool), np.nan_to_num(dn).astype(bool)
        if q["require_base_ema"]:
            e = ema(df, p["base_ema"])
            up, dn = up & (c > e), dn & (c < e)
        f = ema(df, p["fast"])
        db = p["dip_bars"]
        dipped = pd.Series(lo <= f - p["dip_k"] * a).rolling(db, min_periods=1).max().to_numpy() > 0
        popped = pd.Series(h >= f + p["dip_k"] * a).rolling(db, min_periods=1).max().to_numpy() > 0
        long = up & dipped & (c > np.roll(h, 1))
        short = dn & popped & (c < np.roll(lo, 1))
        stop = np.where(long, swing_stop(df, p["swing_lb"], 0.1, a, 1), swing_stop(df, p["swing_lb"], 0.1, a, -1))
        return {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)


FAMILIES = [TrendDipATRTrendVar()]
