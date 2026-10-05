"""
Batch 7: classic, published trend mechanisms -- the ones most likely to be
real behaviors across futures markets (screened with trend_lab/xm_screen.py
on 10y GC/ES/NQ/SI before trusting them on MCL).
"""

import numpy as np
import pandas as pd

from .base import COMMON_SPACE, apply_bias, atr, cached, common_exits, htf_bias


def _mod(df):
    return (df.index.hour * 60 + df.index.minute).to_numpy()


def _tf(df):
    return int(pd.Series(df.index).diff().median().total_seconds() / 60)


def _daily_range(df):
    """Previous trading day's high-low range, broadcast to bars."""
    g = df["trade_day"]
    rng = (df.groupby(g)["high"].max() - df.groupby(g)["low"].min()).shift(1)
    return g.map(rng).to_numpy(dtype=float)


def _anchor_open(df, anchor_min):
    """Open of the first bar at/after anchor time each trading day (NaN before it)."""
    mod = _mod(df)
    g = df["trade_day"].to_numpy()
    first = pd.Series(np.where(mod == anchor_min, df["open"].to_numpy(), np.nan)).groupby(g).transform("max")
    started = pd.Series(mod >= anchor_min).groupby(g).cummax().to_numpy() & (mod < 17 * 60)
    return np.where(started, first.to_numpy(), np.nan)


class VolBreakout:
    name = "vol_breakout"
    description = ("Volatility breakout (Williams-style): go with price when it travels k x the previous day's "
                   "range away from the session anchor open. Stop back at the anchor (or ATR), ride to the flatten.")
    space = {k: v for k, v in COMMON_SPACE.items()} | {
        "anchor": [8 * 60, 9 * 60, 9 * 60 + 30],
        "k": [0.2, 0.3, 0.4, 0.6],
        "stop_mode": ["anchor", "atr1.5", "atr2.5"],
        "last_entry": [11 * 60, 13 * 60],
    }

    def generate(self, df, p):
        a = atr(df)
        c = df["close"].to_numpy()
        mod = _mod(df)
        tf = _tf(df)
        g = df["trade_day"].to_numpy()
        dr = cached(df, ("drange",), lambda: _daily_range(df))
        ao = cached(df, ("anchor_open", p["anchor"]), lambda: _anchor_open(df, p["anchor"]))
        up_lvl, dn_lvl = ao + p["k"] * dr, ao - p["k"] * dr
        win = ((mod + tf) <= p["last_entry"]) & (mod >= p["anchor"])
        with np.errstate(invalid="ignore"):
            brk_up = win & (c > up_lvl)
            brk_dn = win & (c < dn_lvl)
        first_up = pd.Series(brk_up).groupby(g).cumsum().to_numpy() == 1
        first_dn = pd.Series(brk_dn).groupby(g).cumsum().to_numpy() == 1
        long, short = brk_up & first_up, brk_dn & first_dn
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        if p["stop_mode"] == "anchor":
            stop = np.abs(c - ao) + 0.1 * a
        else:
            stop = float(p["stop_mode"][3:]) * a
        return {"long": long, "short": short, "stop_dist": np.nan_to_num(stop, nan=-1)} | common_exits(p, a)


class TSMom:
    name = "tsmom"
    description = ("Intraday time-series momentum: when the return over the last L bars exceeds z x its "
                   "volatility, enter in that direction; exit when the signal turns or at the flatten.")
    space = {k: v for k, v in COMMON_SPACE.items()} | {
        "lookback": [8, 16, 32, 64],
        "z": [0.5, 1.0, 1.5, 2.0],
        "stop_k": [1.5, 2.0, 3.0],
        "exit_flip": [True, False],
    }

    def generate(self, df, p):
        a = atr(df)
        c = df["close"].to_numpy()
        L = p["lookback"]
        ret = c - np.roll(c, L)
        vol = a * np.sqrt(L)
        s = ret / np.where(vol > 0, vol, np.nan)
        sp = np.roll(s, 1)
        with np.errstate(invalid="ignore"):
            long = (s > p["z"]) & ~(sp > p["z"])
            short = (s < -p["z"]) & ~(sp < -p["z"])
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        sig = {"long": long, "short": short, "stop_dist": p["stop_k"] * a} | common_exits(p, a)
        if p["exit_flip"]:
            with np.errstate(invalid="ignore"):
                sig["exit_long"], sig["exit_short"] = s < 0, s > 0
        return sig


class OvernightRangeBreak:
    name = "on_range_break"
    description = ("After the US anchor time, the first close beyond the overnight (18:00 to anchor) high or low, "
                   "in the direction of the higher-TF trend. Stop at the overnight midpoint or ATR.")
    space = {k: v for k, v in COMMON_SPACE.items() if k != "htf_len"} | {
        "htf_len": [0, 20, 50],
        "anchor": [8 * 60, 9 * 60, 9 * 60 + 30],
        "stop_mode": ["mid", "atr1.5", "atr2"],
        "last_entry": [11 * 60, 13 * 60],
    }

    def generate(self, df, p):
        a = atr(df)
        c = df["close"].to_numpy()
        mod = _mod(df)
        tf = _tf(df)
        g = df["trade_day"].to_numpy()

        def on_levels():
            on = (mod >= 18 * 60) | (mod < p["anchor"])
            hi = df["high"].where(on).groupby(g).transform("max").to_numpy()
            lo = df["low"].where(on).groupby(g).transform("min").to_numpy()
            return hi, lo

        oh, ol = cached(df, ("on_levels", p["anchor"]), on_levels)
        win = (mod >= p["anchor"]) & ((mod + tf) <= p["last_entry"])
        brk_up, brk_dn = win & (c > oh), win & (c < ol)
        first_up = pd.Series(brk_up).groupby(g).cumsum().to_numpy() == 1
        first_dn = pd.Series(brk_dn).groupby(g).cumsum().to_numpy() == 1
        long, short = brk_up & first_up, brk_dn & first_dn
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        if p["stop_mode"] == "mid":
            stop = np.abs(c - (oh + ol) / 2) + 0.1 * a
        else:
            stop = float(p["stop_mode"][3:]) * a
        return {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)


FAMILIES = [VolBreakout(), TSMom(), OvernightRangeBreak()]
