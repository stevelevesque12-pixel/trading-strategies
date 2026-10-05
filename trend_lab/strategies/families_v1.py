"""
First batch of trend-following families for MCL.

Each family: SPACE (param -> choices) and generate(df, p) -> signal dict
(see trend_lab/sim.py for the contract).
"""

import numpy as np

from .. import indicators as ind
from .base import COMMON_SPACE, apply_bias, atr, cached, common_exits, ema, htf_bias, swing_stop


# --------------------------------------------------------------------------
# 1. Supertrend flip + ADX strength + higher-timeframe EMA bias
# --------------------------------------------------------------------------
class SupertrendADX:
    name = "supertrend_adx"
    description = ("Enter on a Supertrend direction flip when ADX confirms trend strength and price is on the "
                   "right side of a higher-timeframe EMA. Exit on the opposite flip, trail, or target.")
    space = COMMON_SPACE | {
        "st_n": [7, 10, 14],
        "st_mult": [2.0, 2.5, 3.0, 3.5, 4.0],
        "adx_min": [0, 15, 20, 25, 30],
        "stop_mode": ["line", "atr1.5", "atr2"],
        "exit_on_flip": [True, False],
    }

    def generate(self, df, p):
        line, d = cached(df, ("st", p["st_n"], p["st_mult"]),
                         lambda: tuple(x.to_numpy() for x in ind.supertrend(df, p["st_n"], p["st_mult"])))
        a = atr(df)
        adx = cached(df, ("adx", 14), lambda: ind.adx(df, 14).to_numpy())
        c = df["close"].to_numpy()
        prev = np.roll(d, 1)
        prev[0] = d[0]
        strong = adx >= p["adx_min"]
        long, short = (d == 1) & (prev == -1) & strong, (d == -1) & (prev == 1) & strong
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        if p["stop_mode"] == "line":
            stop = np.abs(c - line) + 0.1 * a
        else:
            stop = float(p["stop_mode"][3:]) * a
        sig = {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)
        if p["exit_on_flip"]:
            sig["exit_long"], sig["exit_short"] = d == -1, d == 1
        return sig


# --------------------------------------------------------------------------
# 2. EMA-stack pullback continuation
# --------------------------------------------------------------------------
class EMAPullback:
    name = "ema_pullback"
    description = ("Trend = fast EMA above a rising slow EMA. Wait for a pullback that tags the fast EMA, then "
                   "enter when a bar closes back above the prior bar's high. Stop under the pullback swing.")
    space = COMMON_SPACE | {
        "fast": [9, 13, 21],
        "slow": [34, 50, 89],
        "touch_atr": [0.0, 0.25, 0.5],
        "touch_bars": [3, 5, 8],
        "swing_lb": [3, 5, 8],
        "exit_slow": [True, False],
    }

    def generate(self, df, p):
        a = atr(df)
        f, s = ema(df, p["fast"]), ema(df, p["slow"])
        h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
        slope = s - np.roll(s, 5)
        up, dn = (f > s) & (slope > 0), (f < s) & (slope < 0)
        tb = p["touch_bars"]
        import pandas as pd
        touched_up = pd.Series(lo <= f + p["touch_atr"] * a).rolling(tb, min_periods=1).max().to_numpy() > 0
        touched_dn = pd.Series(h >= f - p["touch_atr"] * a).rolling(tb, min_periods=1).max().to_numpy() > 0
        ph, pl = np.roll(h, 1), np.roll(lo, 1)
        long = up & touched_up & (c > ph) & (c > f)
        short = dn & touched_dn & (c < pl) & (c < f)
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        stop = np.where(long, swing_stop(df, p["swing_lb"], 0.1, a, 1), swing_stop(df, p["swing_lb"], 0.1, a, -1))
        sig = {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)
        if p["exit_slow"]:
            sig["exit_long"], sig["exit_short"] = c < s, c > s
        return sig


# --------------------------------------------------------------------------
# 3. Donchian breakout filtered by Kaufman efficiency ratio
# --------------------------------------------------------------------------
class DonchianER:
    name = "donchian_er"
    description = ("Close through the prior N-bar high/low while the efficiency ratio says the market is "
                   "trending (not chopping). ATR stop; exit on a close through the shorter M-bar channel.")
    space = COMMON_SPACE | {
        "n": [20, 30, 40, 55, 80],
        "m": [10, 15, 20],
        "er_n": [10, 20, 30],
        "er_min": [0.0, 0.2, 0.3, 0.4],
        "stop_k": [1.0, 1.5, 2.0, 2.5],
    }

    def generate(self, df, p):
        a = atr(df)
        c = df["close"].to_numpy()
        up, dn = cached(df, ("don", p["n"]), lambda: tuple(x.to_numpy() for x in ind.donchian(df, p["n"])))
        xu, xd = cached(df, ("don", p["m"]), lambda: tuple(x.to_numpy() for x in ind.donchian(df, p["m"])))
        er = cached(df, ("er", p["er_n"]), lambda: ind.efficiency_ratio(df["close"], p["er_n"]).to_numpy())
        ok = er >= p["er_min"]
        long, short = (c > up) & ok, (c < dn) & ok
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        return {"long": long, "short": short, "stop_dist": p["stop_k"] * a,
                "exit_long": c < xd, "exit_short": c > xu} | common_exits(p, a)


# --------------------------------------------------------------------------
# 4. Opening-range breakout (NYMEX 09:00 open) with VWAP + HTF trend filter
# --------------------------------------------------------------------------
class ORBTrend:
    name = "orb_trend"
    description = ("Build the range from the 09:00 ET NYMEX open; take the first close outside it before "
                   "noon, only if price is on the trend side of session VWAP (and optional HTF EMA).")
    space = {k: v for k, v in COMMON_SPACE.items() if k != "session"} | {
        "session": ["us"],
        "orb_min": [15, 30, 45, 60],
        "stop_mode": ["mid", "opposite", "atr1.5"],
        "vwap_filter": [True, False],
        "last_entry": [11 * 60, 12 * 60, 13 * 60],
    }

    def generate(self, df, p):
        import pandas as pd
        a = atr(df)
        c = df["close"].to_numpy()
        vwap = cached(df, ("vwap",), lambda: ind.session_vwap(df).to_numpy())
        tf = int(pd.Series(df.index).diff().median().total_seconds() / 60)
        mod = df.index.hour * 60 + df.index.minute
        close_mod = mod + tf
        in_or = (mod >= 9 * 60) & (close_mod <= 9 * 60 + p["orb_min"])
        g = df["trade_day"]
        orh = df["high"].where(in_or).groupby(g).transform("max").to_numpy()
        orl = df["low"].where(in_or).groupby(g).transform("min").to_numpy()
        after = (close_mod > 9 * 60 + p["orb_min"]) & (close_mod <= p["last_entry"])
        brk_up = after & (c > orh)
        brk_dn = after & (c < orl)
        # only the first breakout each day, each direction
        first_up = pd.Series(brk_up).groupby(g.to_numpy()).cumsum().to_numpy() == 1
        first_dn = pd.Series(brk_dn).groupby(g.to_numpy()).cumsum().to_numpy() == 1
        long, short = brk_up & first_up, brk_dn & first_dn
        if p["vwap_filter"]:
            long, short = long & (c > vwap), short & (c < vwap)
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        mid = (orh + orl) / 2
        if p["stop_mode"] == "mid":
            stop = np.abs(c - mid)
        elif p["stop_mode"] == "opposite":
            stop = np.where(long, c - orl, orh - c)
        else:
            stop = 1.5 * a
        return {"long": long, "short": short, "stop_dist": stop + 0.02} | common_exits(p, a)


# --------------------------------------------------------------------------
# 5. Session-VWAP trend pullback
# --------------------------------------------------------------------------
class VWAPTrend:
    name = "vwap_trend"
    description = ("Trend day = price held above a rising session VWAP for N bars. Enter when a pullback "
                   "into the VWAP band prints a bullish close. Stop a few ATRs through VWAP.")
    space = COMMON_SPACE | {
        "hold_bars": [6, 12, 20],
        "slope_bars": [6, 12],
        "band_atr": [0.25, 0.5, 1.0],
        "stop_k": [0.75, 1.0, 1.5],
        "exit_cross": [True, False],
    }

    def generate(self, df, p):
        import pandas as pd
        a = atr(df)
        o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        v = cached(df, ("vwap",), lambda: ind.session_vwap(df).to_numpy())
        hb = p["hold_bars"]
        above = pd.Series(lo > v - 0.1 * a).rolling(hb).sum().to_numpy() >= hb - 2
        below = pd.Series(h < v + 0.1 * a).rolling(hb).sum().to_numpy() >= hb - 2
        sl = v - np.roll(v, p["slope_bars"])
        long = above & (sl > 0) & (lo <= v + p["band_atr"] * a) & (c > o) & (c > v)
        short = below & (sl < 0) & (h >= v - p["band_atr"] * a) & (c < o) & (c < v)
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        stop = np.where(long, c - v, v - c) + p["stop_k"] * a
        sig = {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)
        if p["exit_cross"]:
            sig["exit_long"], sig["exit_short"] = c < v - 0.5 * a, c > v + 0.5 * a
        return sig


FAMILIES = [SupertrendADX(), EMAPullback(), DonchianER(), ORBTrend(), VWAPTrend()]
