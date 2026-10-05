"""
Batch 8: crude-specific engines meant to ADD uncorrelated trades to the MCL
Trend Dip portfolio (trend_dip_atr + trend_dip_rsi), not replace it.
"""

import numpy as np
import pandas as pd

from .base import atr, common_exits, htf_bias
from .families_v3 import TREND_SPACE, TrendDipATR


def _mod(df):
    return (df.index.hour * 60 + df.index.minute).to_numpy()


class EIAMomentum:
    name = "eia_momentum"
    description = ("Wednesday EIA crude inventory report (10:30 ET): if the release bar moves >= k x ATR and "
                   "agrees with the higher-TF trend, join it at the close of the bar; ride with an ATR stop.")
    sim_overrides = {"eia_filter": False}  # this engine trades the release on purpose
    space = {
        "session": ["us"],
        "k": [0.5, 1.0, 1.5, 2.0],
        "confirm_bars": [1, 2],      # 1 = the 10:30 bar itself, 2 = the 10:30 + 10:45 bars combined
        "stop_k": [1.0, 1.5, 2.0],
        "htf_rule": ["60min", "240min"],
        "htf_len": [0, 20, 50],
        "trail_k": [None, 2.0, 3.0],
        "target_r": [None, 1.5, 2.0, 3.0],
        "be_r": [None, 1.0],
    }

    def generate(self, df, p):
        a = atr(df)
        o, c = df["open"].to_numpy(), df["close"].to_numpy()
        mod = _mod(df)
        wed = (df.index.weekday == 2)
        tf = int(pd.Series(df.index).diff().median().total_seconds() / 60)
        start = 10 * 60 + 30
        end_bar_start = start + (p["confirm_bars"] - 1) * tf
        at = wed & (mod == end_bar_start)
        k = p["confirm_bars"]
        move = c - np.roll(o, k - 1)
        prev_a = np.roll(a, k)
        long = at & (move >= p["k"] * prev_a)
        short = at & (move <= -p["k"] * prev_a)
        if p["htf_len"]:
            b = htf_bias(df, p["htf_rule"], p["htf_len"])
            long, short = long & (b > 0), short & (b < 0)
        return {"long": long, "short": short, "stop_dist": p["stop_k"] * a} | common_exits(p, a)


class TrendDipATRWindow:
    name = "trend_dip_atr_window"
    description = ("Engine A (trend_dip_atr) restricted to an explicit entry window -- e.g. London (02:00-08:00 ET) "
                   "-- to add trades from sessions the US-hours engines never see. Session 'all' (flat 16:40 ET).")
    space = {k: v for k, v in TREND_SPACE.items() if k != "session"} | {
        "session": ["all"],
        "fast": [9, 13, 21],
        "dip_k": [0.5, 1.0, 1.5],
        "dip_bars": [4, 6],
        "swing_lb": [5, 8],
        "window": [(2 * 60, 8 * 60), (19 * 60, 2 * 60), (3 * 60, 9 * 60)],
    }

    def generate(self, df, p):
        sig = TrendDipATR().generate(df, p)
        tf = int(pd.Series(df.index).diff().median().total_seconds() / 60)
        cm = (_mod(df) + tf) % (24 * 60)
        w0, w1 = p["window"]
        inw = (cm >= w0) & (cm <= w1) if w0 < w1 else (cm >= w0) | (cm <= w1)
        sig["long"] = sig["long"] & inw
        sig["short"] = sig["short"] & inw
        return sig


FAMILIES = [EIAMomentum(), TrendDipATRWindow()]
