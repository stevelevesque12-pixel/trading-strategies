"""Pluggable strategy components.

A strategy = 1 trend-direction component + 2 confirmation components + 1 regime
(volatility / chop) filter + ATR bands. Every component maps (Market, params) to
an int8 array per bar:

  * trend / confirm: +1 bullish, -1 bearish, 0 neutral
  * regime:          0 = sideways (no trade), 1 = trending (small size), 2 = strong trend (full size)

New strategy families are added by registering a new component here; the
research loop picks them up automatically.
"""

import numpy as np

from . import indicators as ind

TREND = {}
CONFIRM = {}
REGIME = {}

# param spaces: name -> list of candidate values (the optimiser samples from these)


def _reg(table, name, space):
    def deco(fn):
        table[name] = (fn, space)
        return fn
    return deco


def _sign(x):
    return np.sign(np.nan_to_num(x)).astype(np.int8)


# ------------------------------------------------------------------ trend
@_reg(TREND, "ema_slope", {"n": [50, 80, 100, 150, 200, 250, 300]})
def trend_ema_slope(m, n):
    e = ind.ema(m.c, n)
    slope = e - np.roll(e, 3)
    return (((m.c > e) & (slope > 0)).astype(np.int8) - ((m.c < e) & (slope < 0)).astype(np.int8))


@_reg(TREND, "ema_cross", {"fast": [10, 20, 30, 50], "slow": [80, 100, 150, 200, 300]})
def trend_ema_cross(m, fast, slow):
    return _sign(ind.ema(m.c, fast) - ind.ema(m.c, slow))


@_reg(TREND, "supertrend", {"n": [10, 14, 20, 30], "mult": [2.0, 2.5, 3.0, 4.0, 5.0]})
def trend_supertrend(m, n, mult):
    return ind.supertrend_dir(m.h, m.l, m.c, n, mult).astype(np.int8)


@_reg(TREND, "hma_slope", {"n": [55, 80, 100, 150, 200]})
def trend_hma_slope(m, n):
    hm = ind.hma(m.c, n)
    return _sign(hm - np.roll(hm, 2))


@_reg(TREND, "kama_slope", {"n": [10, 20, 30, 50]})
def trend_kama_slope(m, n):
    k = ind.kama(m.c, n)
    return _sign(k - np.roll(k, 3))


@_reg(TREND, "donchian", {"n": [40, 60, 100, 150, 200]})
def trend_donchian(m, n):
    hi = np.roll(ind.rolling_max(m.h, n), 1)
    lo = np.roll(ind.rolling_min(m.l, n), 1)
    raw = np.where(m.c > hi, 1, np.where(m.c < lo, -1, 0)).astype(np.int8)
    # persist last breakout direction
    idx = np.where(raw != 0, np.arange(len(raw)), 0)
    np.maximum.accumulate(idx, out=idx)
    return raw[idx]


# ------------------------------------------------------------------ confirmations
@_reg(CONFIRM, "macd", {"fast": [8, 12], "slow": [21, 26], "sig": [5, 9]})
def conf_macd(m, fast, slow, sig):
    return _sign(ind.macd_hist(m.c, fast, slow, sig))


@_reg(CONFIRM, "rsi", {"n": [9, 14, 21], "band": [0, 5, 10]})
def conf_rsi(m, n, band):
    r = ind.rsi(m.c, n)
    return ((r > 50 + band).astype(np.int8) - (r < 50 - band).astype(np.int8))


@_reg(CONFIRM, "dmi", {"n": [10, 14, 20]})
def conf_dmi(m, n):
    p, q, _ = ind.dmi_adx(m.h, m.l, m.c, n)
    return _sign(p - q)


@_reg(CONFIRM, "roc", {"n": [5, 10, 20, 40]})
def conf_roc(m, n):
    return _sign(m.c - np.roll(m.c, n))


@_reg(CONFIRM, "vwap", {})
def conf_vwap(m):
    return _sign(m.c - ind.session_vwap(m.h, m.l, m.c, m.v, m.day_id))


@_reg(CONFIRM, "fast_ema", {"n": [9, 20, 34]})
def conf_fast_ema(m, n):
    return _sign(m.c - ind.ema(m.c, n))


# ------------------------------------------------------------------ regime filters
def _bucket(x, lo, hi, higher_is_trend=True):
    x = np.nan_to_num(x, nan=-1e9 if higher_is_trend else 1e9)
    if higher_is_trend:
        return ((x >= lo).astype(np.int8) + (x >= hi).astype(np.int8))
    return ((x <= hi).astype(np.int8) + (x <= lo).astype(np.int8))


@_reg(REGIME, "adx", {"n": [14, 20], "lo": [18, 20, 22, 25], "hi": [28, 30, 35, 40]})
def reg_adx(m, n, lo, hi):
    _, _, a = ind.dmi_adx(m.h, m.l, m.c, n)
    return _bucket(a, lo, hi)


@_reg(REGIME, "chop", {"n": [14, 20, 30], "lo": [35, 38.2, 42], "hi": [50, 55, 61.8]})
def reg_chop(m, n, lo, hi):
    # chop <= hi: trending (1); chop <= lo: strong trend (2); else sideways (0)
    return _bucket(ind.choppiness(m.h, m.l, m.c, n), lo, hi, higher_is_trend=False)


@_reg(REGIME, "er", {"n": [10, 20, 30], "lo": [0.15, 0.2, 0.25, 0.3], "hi": [0.35, 0.4, 0.5]})
def reg_er(m, n, lo, hi):
    return _bucket(ind.efficiency_ratio(m.c, n), lo, hi)


@_reg(REGIME, "atr_ratio", {"fast": [7, 14], "slow": [50, 100], "lo": [0.8, 0.9, 1.0], "hi": [1.1, 1.2, 1.4]})
def reg_atr_ratio(m, fast, slow, lo, hi):
    r = ind.atr(m.h, m.l, m.c, fast) / ind.atr(m.h, m.l, m.c, slow)
    return _bucket(r, lo, hi)


@_reg(REGIME, "bbw_rank", {"n": [20], "look": [100, 200, 500], "lo": [0.3, 0.4, 0.5], "hi": [0.7, 0.8]})
def reg_bbw_rank(m, n, look, lo, hi):
    mid = ind.sma(m.c, n)
    sd = np.sqrt(np.maximum(ind.sma(m.c ** 2, n) - mid ** 2, 0))
    bw = 4 * sd / mid
    return _bucket(ind.percent_rank(np.nan_to_num(bw), look), lo, hi)


# ------------------------------------------------------------------ batch 2 (added during the research loop)
@_reg(TREND, "linreg_slope", {"n": [50, 100, 150, 200]})
def trend_linreg_slope(m, n):
    # sign of the least-squares slope over n bars (via rolling sums)
    x = np.arange(len(m.c), dtype=np.float64)
    sx, sy = ind.rolling_sum(x, n), ind.rolling_sum(m.c, n)
    sxy, sxx = ind.rolling_sum(x * m.c, n), ind.rolling_sum(x * x, n)
    return _sign((n * sxy - sx * sy) / (n * sxx - sx * sx))


@_reg(CONFIRM, "heikin", {"smooth": [1, 3]})
def conf_heikin(m, smooth):
    hc = (m.o + m.h + m.l + m.c) / 4
    ho = np.empty_like(hc)
    ho[0] = m.o[0]
    for i in range(1, len(hc)):  # cheap enough once per cache key
        ho[i] = (ho[i - 1] + hc[i - 1]) / 2
    d = hc - ho
    return _sign(ind.ema(d, smooth) if smooth > 1 else d)


@_reg(CONFIRM, "obv_slope", {"n": [10, 20, 50]})
def conf_obv_slope(m, n):
    obv = np.cumsum(np.sign(np.diff(m.c, prepend=m.c[0])) * m.v)
    e = ind.ema(obv, n)
    return _sign(e - np.roll(e, 3))


@_reg(REGIME, "atr_rank", {"n": [14], "look": [200, 500, 1000], "lo": [0.3, 0.4, 0.5], "hi": [0.7, 0.8, 0.9]})
def reg_atr_rank(m, n, look, lo, hi):
    a = ind.atr(m.h, m.l, m.c, n) / m.c
    return _bucket(ind.percent_rank(a, look), lo, hi)
