"""Shared helpers for strategy families."""

import numpy as np
import pandas as pd

from .. import indicators as ind

_CACHE = {}


def cached(df: pd.DataFrame, key, fn):
    """Memoize an indicator per dataframe -- the optimizer re-uses them across thousands of configs."""
    k = (id(df), key)
    if k not in _CACHE:
        _CACHE[k] = fn()
    return _CACHE[k]


def atr(df, n=14):
    return cached(df, ("atr", n), lambda: ind.atr(df, n).to_numpy())


def ema(df, n, col="close"):
    return cached(df, ("ema", n, col), lambda: ind.ema(df[col], n).to_numpy())


def htf_ema(df, rule, n):
    """Higher-timeframe EMA of close, lookahead-safe."""
    return cached(df, ("htf_ema", rule, n), lambda: ind.htf(df, rule, lambda h: ind.ema(h["close"], n)).to_numpy())


def htf_bias(df, rule, n):
    """+1 / -1 / 0 : base close vs higher-TF EMA (0 = filter disabled)."""
    if not n:
        return np.zeros(len(df))
    e = htf_ema(df, rule, n)
    c = df["close"].to_numpy()
    return np.where(np.isnan(e), 0, np.sign(c - e))


def apply_bias(long, short, bias, enabled):
    if not enabled:
        return long, short
    return long & (bias > 0), short & (bias < 0)


def rolling_min(x, n):
    return pd.Series(x).rolling(n, min_periods=1).min().to_numpy()


def rolling_max(x, n):
    return pd.Series(x).rolling(n, min_periods=1).max().to_numpy()


def swing_stop(df, lookback, buf_atr, a, direction):
    """Stop distance to the extreme of the last `lookback` bars, plus an ATR buffer."""
    c = df["close"].to_numpy()
    if direction > 0:
        return c - rolling_min(df["low"].to_numpy(), lookback) + buf_atr * a
    return rolling_max(df["high"].to_numpy(), lookback) - c + buf_atr * a


def common_exits(p, a):
    """Trail / target / breakeven parameters shared by most families."""
    out = {"target_r": p.get("target_r"), "be_r": p.get("be_r")}
    if p.get("trail_k"):
        out["trail_dist"] = p["trail_k"] * a
    return out


COMMON_SPACE = {
    "session": ["ny", "us", "all"],
    "trail_k": [None, 1.5, 2.0, 2.5, 3.0, 4.0],
    "target_r": [None, 1.5, 2.0, 3.0, 4.0],
    "be_r": [None, 1.0, 1.5],
    "htf_rule": ["60min", "240min"],
    "htf_len": [0, 20, 50],
}
