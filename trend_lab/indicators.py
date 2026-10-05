"""Vectorized indicators. Every value at index i uses only bars 0..i (no lookahead)."""

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def wma(s: pd.Series, n: int) -> pd.Series:
    w = np.arange(1, n + 1, dtype=float)
    return s.rolling(n, min_periods=n).apply(lambda x: np.dot(x, w) / w.sum(), raw=True)


def hma(s: pd.Series, n: int) -> pd.Series:
    """Hull moving average: low-lag trend line."""
    half = max(int(n / 2), 1)
    root = max(int(np.sqrt(n)), 1)
    return wma(2 * wma(s, half) - wma(s, n), root)


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    return pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    ).max(axis=1)


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    """Wilder ATR."""
    return true_range(df).ewm(alpha=1.0 / n, adjust=False).mean()


def adx(df: pd.DataFrame, n: int = 14) -> pd.Series:
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = np.where((up > down) & (up > 0), up, 0.0)
    minus_dm = np.where((down > up) & (down > 0), down, 0.0)
    tr = true_range(df).ewm(alpha=1.0 / n, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False).mean() / tr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1.0 / n, adjust=False).mean() / tr
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    return dx.ewm(alpha=1.0 / n, adjust=False).mean().fillna(0.0)


def supertrend(df: pd.DataFrame, n: int = 10, mult: float = 3.0):
    """Returns (line, direction) where direction is +1 (up-trend) or -1."""
    a = atr(df, n).to_numpy()
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy()
    close = df["close"].to_numpy()
    upper = hl2 + mult * a
    lower = hl2 - mult * a
    fu = upper.copy()
    fl = lower.copy()
    direction = np.ones(len(df))
    line = np.zeros(len(df))
    for i in range(1, len(df)):
        fu[i] = upper[i] if (upper[i] < fu[i - 1] or close[i - 1] > fu[i - 1]) else fu[i - 1]
        fl[i] = lower[i] if (lower[i] > fl[i - 1] or close[i - 1] < fl[i - 1]) else fl[i - 1]
        if direction[i - 1] == 1:
            direction[i] = -1 if close[i] < fl[i] else 1
        else:
            direction[i] = 1 if close[i] > fu[i] else -1
        line[i] = fl[i] if direction[i] == 1 else fu[i]
    return pd.Series(line, index=df.index), pd.Series(direction, index=df.index)


def donchian(df: pd.DataFrame, n: int):
    """Channel of the *previous* n bars (so a close above `upper` is a true breakout)."""
    return df["high"].rolling(n).max().shift(1), df["low"].rolling(n).min().shift(1)


def efficiency_ratio(s: pd.Series, n: int) -> pd.Series:
    """Kaufman efficiency ratio: |net move| / path length over n bars. 1 = pure trend."""
    net = (s - s.shift(n)).abs()
    path = s.diff().abs().rolling(n).sum()
    return (net / path.replace(0, np.nan)).fillna(0.0)


def rsi(s: pd.Series, n: int = 14) -> pd.Series:
    d = s.diff()
    up = d.clip(lower=0).ewm(alpha=1.0 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1.0 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn.replace(0, np.nan))


def session_vwap(df: pd.DataFrame) -> pd.Series:
    """VWAP reset at each CME trading day (18:00 ET open)."""
    tp = (df["high"] + df["low"] + df["close"]) / 3
    vol = df["volume"].replace(0, 1.0)
    g = df["trade_day"]
    return (tp * vol).groupby(g).cumsum() / vol.groupby(g).cumsum()


def macd_hist(s: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    m = ema(s, fast) - ema(s, slow)
    return m - ema(m, signal)


def htf(df: pd.DataFrame, rule: str, fn) -> pd.Series:
    """
    Compute fn(htf_bars) on a higher timeframe and map it back onto `df` with
    no lookahead: a higher-TF value becomes visible only once its bar has
    closed, i.e. at the close of the base bar that ends at/after it.
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    h = df[list(agg)].resample(rule, label="left", closed="left").agg(agg).dropna()
    vals = fn(h)
    close_time = h.index + pd.Timedelta(rule)
    base_tf = df.index.to_series().diff().median()
    base_close = df.index + base_tf
    out = pd.Series(vals.to_numpy(), index=close_time)
    mapped = out.reindex(out.index.union(base_close)).ffill().reindex(base_close)
    return pd.Series(mapped.to_numpy(), index=df.index)
