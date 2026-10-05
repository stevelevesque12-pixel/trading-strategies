"""
Batch 5: new trend mechanisms rather than more dip variants.

  - nr_break        volatility compression (NR4/NR7 / inside bar) breaking in the trend direction
  - impulse_retrace a strong trend-direction impulse bar, entered on a pullback into its body
  - ribbon_trend    strictly aligned & fanned EMA ribbon, entered on a reclaim of the fast EMA
  - intraday_mom    "intraday momentum": the day's move from the 09:00 open to a decision time
                    predicts the move into settlement (14:30 ET), documented for crude futures
"""

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import COMMON_SPACE, apply_bias, atr, cached, common_exits, ema, htf_bias, swing_stop


def _tf_minutes(df):
    return int(pd.Series(df.index).diff().median().total_seconds() / 60)


class NRBreak:
    name = "nr_break"
    description = ("A narrow-range bar (smallest range of the last N, or an inside bar) in a higher-TF trend; "
                   "enter when a later bar closes beyond it in the trend direction. Stop at the other side.")
    space = {k: v for k, v in COMMON_SPACE.items() if k != "htf_len"} | {
        "htf_len": [20, 50],
        "nr": [0, 4, 7],          # 0 = inside bar
        "within": [1, 2, 3],      # break must come within this many bars
        "stop_buf": [0.0, 0.25],
    }

    def generate(self, df, p):
        a = atr(df)
        h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
        rng = h - lo
        if p["nr"]:
            setup = rng <= pd.Series(rng).rolling(p["nr"]).min().to_numpy()
        else:
            setup = (h <= np.roll(h, 1)) & (lo >= np.roll(lo, 1))
        # carry the setup bar's high/low forward `within` bars
        sh = pd.Series(np.where(setup, h, np.nan)).ffill(limit=p["within"]).shift(1).to_numpy()
        sl = pd.Series(np.where(setup, lo, np.nan)).ffill(limit=p["within"]).shift(1).to_numpy()
        long = np.nan_to_num(c > sh, nan=False).astype(bool)
        short = np.nan_to_num(c < sl, nan=False).astype(bool)
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), True)
        stop = np.where(long, c - sl, sh - c) + p["stop_buf"] * a
        return {"long": long, "short": short, "stop_dist": np.nan_to_num(stop, nan=-1)} | common_exits(p, a)


class ImpulseRetrace:
    name = "impulse_retrace"
    description = ("A trend-direction impulse bar (range ≥ k·ATR, closing near its extreme) marks aggressive "
                   "participation; buy the first pullback into its body that closes back up. Stop below the bar.")
    space = {k: v for k, v in COMMON_SPACE.items() if k != "htf_len"} | {
        "htf_len": [0, 20, 50],
        "impulse_k": [1.25, 1.75, 2.5],
        "close_pct": [0.7, 0.8],
        "retrace": [0.38, 0.5, 0.62],
        "within": [2, 4, 8],
    }

    def generate(self, df, p):
        a = atr(df)
        o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
        rng = h - lo
        pos = (c - lo) / np.where(rng > 0, rng, np.nan)
        big = rng >= p["impulse_k"] * np.roll(a, 1)
        up_imp = big & (pos >= p["close_pct"]) & (c > o)
        dn_imp = big & (pos <= 1 - p["close_pct"]) & (c < o)
        lvl_up = pd.Series(np.where(up_imp, h - p["retrace"] * rng, np.nan)).ffill(limit=p["within"]).shift(1).to_numpy()
        low_up = pd.Series(np.where(up_imp, lo, np.nan)).ffill(limit=p["within"]).shift(1).to_numpy()
        lvl_dn = pd.Series(np.where(dn_imp, lo + p["retrace"] * rng, np.nan)).ffill(limit=p["within"]).shift(1).to_numpy()
        hi_dn = pd.Series(np.where(dn_imp, h, np.nan)).ffill(limit=p["within"]).shift(1).to_numpy()
        with np.errstate(invalid="ignore"):
            long = (lo <= lvl_up) & (c > lvl_up) & (c > o) & (lo > low_up)
            short = (h >= lvl_dn) & (c < lvl_dn) & (c < o) & (h < hi_dn)
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        stop = np.where(long, c - low_up, hi_dn - c) + 0.05 * a
        return {"long": long, "short": short, "stop_dist": np.nan_to_num(stop, nan=-1)} | common_exits(p, a)


class RibbonTrend:
    name = "ribbon_trend"
    description = ("EMA ribbon (fast < mid < slow) strictly stacked and fanning out = strong trend. Enter when "
                   "price dips to the mid EMA and closes back above the fast EMA. Exit when the stack breaks.")
    space = COMMON_SPACE | {
        "ribbon": [(8, 21, 55), (5, 13, 34), (10, 30, 90)],
        "fan_atr": [0.0, 0.25, 0.5],
        "swing_lb": [3, 5],
        "exit_stack": [True, False],
    }

    def generate(self, df, p):
        a = atr(df)
        h, lo, c = (df[k].to_numpy() for k in ("high", "low", "close"))
        f, m, s = (ema(df, n) for n in p["ribbon"])
        up = (f > m) & (m > s) & (m - s >= p["fan_atr"] * a) & (s > np.roll(s, 3))
        dn = (f < m) & (m < s) & (s - m >= p["fan_atr"] * a) & (s < np.roll(s, 3))
        touch_up = pd.Series(lo <= m).rolling(3, min_periods=1).max().to_numpy() > 0
        touch_dn = pd.Series(h >= m).rolling(3, min_periods=1).max().to_numpy() > 0
        long = up & touch_up & (c > f) & (np.roll(c, 1) <= np.roll(f, 1))
        short = dn & touch_dn & (c < f) & (np.roll(c, 1) >= np.roll(f, 1))
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        stop = np.where(long, swing_stop(df, p["swing_lb"], 0.1, a, 1), swing_stop(df, p["swing_lb"], 0.1, a, -1))
        sig = {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)
        if p["exit_stack"]:
            sig["exit_long"], sig["exit_short"] = f < m, f > m
        return sig


class IntradayMomentum:
    name = "intraday_mom"
    description = ("Intraday momentum: if the move from the 09:00 ET open to a decision time exceeds k·ATR, "
                   "ride it into the 14:30 settlement. One trade per day. ATR stop.")
    space = {
        "session": ["ny"],
        "decide": [10 * 60 + 30, 11 * 60 + 30, 12 * 60 + 30, 13 * 60],
        "move_k": [1.0, 2.0, 3.0, 4.0],
        "stop_k": [1.0, 1.5, 2.0, 3.0],
        "htf_rule": ["60min", "240min"],
        "htf_len": [0, 20, 50],
        "trail_k": [None, 2.0, 3.0],
        "target_r": [None, 1.5, 2.0],
        "be_r": [None, 1.0],
    }

    def generate(self, df, p):
        a = atr(df)
        c = df["close"].to_numpy()
        tf = _tf_minutes(df)
        mod = (df.index.hour * 60 + df.index.minute).to_numpy()
        g = df["trade_day"].to_numpy()
        open9 = cached(df, ("open9",), lambda: pd.Series(np.where(mod == 9 * 60, df["open"].to_numpy(), np.nan))
                       .groupby(g).transform("max").to_numpy())
        at = (mod + tf) == p["decide"]
        mv = c - open9
        long = at & (mv >= p["move_k"] * a)
        short = at & (mv <= -p["move_k"] * a)
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        return {"long": long, "short": short, "stop_dist": p["stop_k"] * a} | common_exits(p, a)


FAMILIES = [NRBreak(), ImpulseRetrace(), RibbonTrend(), IntradayMomentum()]
