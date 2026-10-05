"""
Batch 2: trend families built around crude-specific behavior and
"buy the dip inside a trend" entries (higher win rate, which suits a
trailing-drawdown prop account better than low-win breakout systems).
"""

import numpy as np
import pandas as pd

from .. import indicators as ind
from .base import COMMON_SPACE, apply_bias, atr, cached, common_exits, ema, htf_bias, swing_stop


def _tf_minutes(df):
    return int(pd.Series(df.index).diff().median().total_seconds() / 60)


# --------------------------------------------------------------------------
# 6. RSI dip in a higher-timeframe trend (triple-screen style)
# --------------------------------------------------------------------------
class TrendDipRSI:
    name = "trend_dip_rsi"
    description = ("Higher-TF trend from a 1h/4h EMA plus a rising base-TF EMA; enter when a short RSI "
                   "dips oversold (rallies overbought for shorts) and then turns back. Stop under the dip.")
    space = {k: v for k, v in COMMON_SPACE.items() if k != "htf_len"} | {
        "htf_len": [20, 50],
        "base_ema": [20, 50, 100],
        "rsi_n": [2, 3, 5],
        "rsi_lo": [10, 20, 30],
        "swing_lb": [3, 5, 8],
        "exit_rsi": [None, 70, 80],
    }

    def generate(self, df, p):
        a = atr(df)
        c, h, lo = (df[k].to_numpy() for k in ("close", "high", "low"))
        r = cached(df, ("rsi", p["rsi_n"]), lambda: ind.rsi(df["close"], p["rsi_n"]).fillna(50).to_numpy())
        e = ema(df, p["base_ema"])
        bias = htf_bias(df, p["htf_rule"], p["htf_len"])
        rp = np.roll(r, 1)
        up = (bias > 0) & (c > e) & (e > np.roll(e, 3))
        dn = (bias < 0) & (c < e) & (e < np.roll(e, 3))
        long = up & (rp < p["rsi_lo"]) & (r >= p["rsi_lo"]) & (c > np.roll(h, 1))
        short = dn & (rp > 100 - p["rsi_lo"]) & (r <= 100 - p["rsi_lo"]) & (c < np.roll(lo, 1))
        stop = np.where(long, swing_stop(df, p["swing_lb"], 0.1, a, 1), swing_stop(df, p["swing_lb"], 0.1, a, -1))
        sig = {"long": long, "short": short, "stop_dist": stop} | common_exits(p, a)
        if p["exit_rsi"]:
            sig["exit_long"], sig["exit_short"] = r > p["exit_rsi"], r < 100 - p["exit_rsi"]
        return sig


# --------------------------------------------------------------------------
# 7. Overnight-trend continuation into the NY session
# --------------------------------------------------------------------------
class OvernightContinuation:
    name = "overnight_cont"
    description = ("If Globex (18:00 to 08:59 ET) trended cleanly (high efficiency ratio, big net move vs ATR), "
                   "join that direction on the first pullback-and-resume after the 09:00 open. ATR trail exit.")
    space = {k: v for k, v in COMMON_SPACE.items() if k not in ("session",)} | {
        "session": ["us"],
        "er_min": [0.15, 0.25, 0.35],
        "move_atr": [2.0, 4.0, 6.0],
        "pb_ema": [9, 20],
        "last_entry": [11 * 60, 12 * 60 + 30],
        "stop_k": [1.0, 1.5, 2.0],
    }

    def generate(self, df, p):
        a = atr(df)
        c, o = df["close"].to_numpy(), df["open"].to_numpy()
        tf = _tf_minutes(df)
        mod = (df.index.hour * 60 + df.index.minute).to_numpy()
        g = df["trade_day"].to_numpy()

        def overnight():
            on = (mod >= 18 * 60) | (mod < 9 * 60)
            s = pd.DataFrame({"g": g, "c": c, "o": o, "on": on})
            s = s[s.on]
            first = s.groupby("g")["o"].first()
            last = s.groupby("g")["c"].last()
            path = s.groupby("g")["c"].apply(lambda x: x.diff().abs().sum())
            return (last - first), (last - first).abs() / path.replace(0, np.nan)

        net, er = cached(df, ("overnight",), overnight)
        net_b = pd.Series(g).map(net).to_numpy(dtype=float)
        er_b = pd.Series(g).map(er).fillna(0).to_numpy(dtype=float)
        close_mod = mod + tf
        window = (close_mod > 9 * 60) & (close_mod <= p["last_entry"])
        strong = (er_b >= p["er_min"]) & (np.abs(net_b) >= p["move_atr"] * a)
        e = ema(df, p["pb_ema"])
        dipped = pd.Series(df["low"].to_numpy() <= e).rolling(4, min_periods=1).max().to_numpy() > 0
        popped = pd.Series(df["high"].to_numpy() >= e).rolling(4, min_periods=1).max().to_numpy() > 0
        long = window & strong & (net_b > 0) & dipped & (c > e) & (c > o)
        short = window & strong & (net_b < 0) & popped & (c < e) & (c < o)
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        return {"long": long, "short": short, "stop_dist": p["stop_k"] * a} | common_exits(p, a)


# --------------------------------------------------------------------------
# 8. Hull MA turn with ADX
# --------------------------------------------------------------------------
class HullTurn:
    name = "hull_turn"
    description = ("Enter when a Hull MA turns up (down) while ADX is above threshold and price is beyond a "
                   "slow EMA. Exit when the Hull turns back.")
    space = COMMON_SPACE | {
        "hma_n": [16, 24, 36, 55],
        "slow": [50, 100, 200],
        "adx_min": [0, 18, 25],
        "stop_k": [1.0, 1.5, 2.0],
        "exit_turn": [True, False],
    }

    def generate(self, df, p):
        a = atr(df)
        c = df["close"].to_numpy()
        hm = cached(df, ("hma", p["hma_n"]), lambda: ind.hma(df["close"], p["hma_n"]).to_numpy())
        adx = cached(df, ("adx", 14), lambda: ind.adx(df, 14).to_numpy())
        s = ema(df, p["slow"])
        d = np.sign(hm - np.roll(hm, 1))
        dp = np.roll(d, 1)
        ok = adx >= p["adx_min"]
        long = (d > 0) & (dp <= 0) & (c > s) & ok
        short = (d < 0) & (dp >= 0) & (c < s) & ok
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        sig = {"long": long, "short": short, "stop_dist": p["stop_k"] * a} | common_exits(p, a)
        if p["exit_turn"]:
            sig["exit_long"], sig["exit_short"] = d < 0, d > 0
        return sig


# --------------------------------------------------------------------------
# 9. Keltner breakout with volume surge
# --------------------------------------------------------------------------
class KeltnerVolume:
    name = "keltner_volume"
    description = ("Close outside a Keltner channel (EMA ± k·ATR) on volume well above its recent average: "
                   "an institutional push. Stop back at the channel midline; trail with ATR.")
    space = COMMON_SPACE | {
        "kc_n": [20, 34],
        "kc_k": [1.5, 2.0, 2.5],
        "vol_mult": [1.0, 1.5, 2.0],
        "stop_mode": ["mid", "atr1", "atr1.5"],
    }

    def generate(self, df, p):
        a = atr(df)
        c = df["close"].to_numpy()
        m = ema(df, p["kc_n"])
        v = df["volume"].to_numpy()
        va = cached(df, ("vavg", 20), lambda: df["volume"].rolling(20, min_periods=5).mean().shift(1).to_numpy())
        surge = v >= p["vol_mult"] * va
        long = (c > m + p["kc_k"] * a) & surge
        short = (c < m - p["kc_k"] * a) & surge
        long, short = apply_bias(long, short, htf_bias(df, p["htf_rule"], p["htf_len"]), p["htf_len"])
        stop = np.abs(c - m) if p["stop_mode"] == "mid" else float(p["stop_mode"][3:]) * a
        return {"long": long, "short": short, "stop_dist": stop,
                "exit_long": c < m, "exit_short": c > m} | common_exits(p, a)


FAMILIES = [TrendDipRSI(), OvernightContinuation(), HullTurn(), KeltnerVolume()]
