"""Vectorised / numba indicator primitives. All return float64 numpy arrays aligned to the input."""

import numpy as np
from numba import njit


def ema(x: np.ndarray, n: int) -> np.ndarray:
    return _ema(np.asarray(x, dtype=np.float64), 2.0 / (n + 1.0))


def rma(x: np.ndarray, n: int) -> np.ndarray:
    """Wilder smoothing (what TradingView's ta.rma / ta.atr use)."""
    return _ema(np.asarray(x, dtype=np.float64), 1.0 / n)


@njit(cache=True)
def _ema(x, alpha):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        v = x[i]
        out[i] = out[i - 1] + alpha * (v - out[i - 1]) if v == v else out[i - 1]
    return out


def sma(x: np.ndarray, n: int) -> np.ndarray:
    c = np.cumsum(np.insert(np.asarray(x, dtype=np.float64), 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = (c[n:] - c[:-n]) / n
    return out


def wma(x: np.ndarray, n: int) -> np.ndarray:
    w = np.arange(1, n + 1, dtype=np.float64)
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = np.convolve(x, w[::-1], mode="valid") / w.sum()
    return out


def hma(x: np.ndarray, n: int) -> np.ndarray:
    half = max(2, n // 2)
    raw = 2 * wma(x, half) - wma(x, n)
    raw = np.where(np.isnan(raw), x, raw)
    return wma(raw, max(2, int(np.sqrt(n))))


def true_range(h, l, c):
    pc = np.roll(c, 1)
    pc[0] = c[0]
    return np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))


def atr(h, l, c, n: int) -> np.ndarray:
    return rma(true_range(h, l, c), n)


def rolling_max(x, n):
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).max(axis=1)
    return out


def rolling_min(x, n):
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        out[n - 1:] = sliding_window_view(x, n).min(axis=1)
    return out


def rolling_sum(x, n):
    c = np.cumsum(np.insert(np.asarray(x, dtype=np.float64), 0, 0.0))
    out = np.full(len(x), np.nan)
    out[n - 1:] = c[n:] - c[:-n]
    return out


def dmi_adx(h, l, c, n: int):
    """Returns (+DI, -DI, ADX), Wilder-smoothed."""
    up = np.diff(h, prepend=h[0])
    dn = -np.diff(l, prepend=l[0])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = rma(true_range(h, l, c), n)
    tr = np.where(tr == 0, np.nan, tr)
    pdi = 100 * rma(pdm, n) / tr
    ndi = 100 * rma(ndm, n) / tr
    s = pdi + ndi
    dx = 100 * np.abs(pdi - ndi) / np.where(s == 0, np.nan, s)
    adx = rma(np.nan_to_num(dx), n)
    return np.nan_to_num(pdi), np.nan_to_num(ndi), adx


def rsi(c, n: int):
    d = np.diff(c, prepend=c[0])
    g = rma(np.maximum(d, 0), n)
    ls = rma(np.maximum(-d, 0), n)
    rs = g / np.where(ls == 0, 1e-12, ls)
    return 100 - 100 / (1 + rs)


def macd_hist(c, fast: int, slow: int, sig: int):
    m = ema(c, fast) - ema(c, slow)
    return m - ema(m, sig)


def choppiness(h, l, c, n: int):
    """Choppiness Index: ~100 = sideways, ~0 = trending. Classic thresholds 61.8 / 38.2."""
    tr_sum = rolling_sum(true_range(h, l, c), n)
    rng = rolling_max(h, n) - rolling_min(l, n)
    rng = np.where(rng <= 0, np.nan, rng)
    return 100 * np.log10(tr_sum / rng) / np.log10(n)


def efficiency_ratio(c, n: int):
    """Kaufman ER: |net move| / path length over n bars, 0 (noise) .. 1 (straight line)."""
    net = np.abs(c - np.roll(c, n))
    path = rolling_sum(np.abs(np.diff(c, prepend=c[0])), n)
    out = net / np.where(path == 0, np.nan, path)
    out[:n] = np.nan
    return out


@njit(cache=True)
def _supertrend(h, l, c, atr_v, mult):
    n = len(c)
    direction = np.zeros(n)
    up = np.zeros(n)
    dn = np.zeros(n)
    for i in range(n):
        mid = (h[i] + l[i]) / 2.0
        bu = mid - mult * atr_v[i]
        bd = mid + mult * atr_v[i]
        if i == 0:
            up[i], dn[i], direction[i] = bu, bd, 1.0
            continue
        up[i] = max(bu, up[i - 1]) if c[i - 1] > up[i - 1] else bu
        dn[i] = min(bd, dn[i - 1]) if c[i - 1] < dn[i - 1] else bd
        if direction[i - 1] < 0 and c[i] > dn[i - 1]:
            direction[i] = 1.0
        elif direction[i - 1] > 0 and c[i] < up[i - 1]:
            direction[i] = -1.0
        else:
            direction[i] = direction[i - 1]
    return direction


def supertrend_dir(h, l, c, n: int, mult: float):
    return _supertrend(h, l, c, atr(h, l, c, n), float(mult))


@njit(cache=True)
def _kama(c, n, fast, slow):
    out = np.empty_like(c)
    out[: n] = c[: n]
    fsc = 2.0 / (fast + 1.0)
    ssc = 2.0 / (slow + 1.0)
    for i in range(n, len(c)):
        path = 0.0
        for j in range(i - n + 1, i + 1):
            path += abs(c[j] - c[j - 1])
        er = abs(c[i] - c[i - n]) / path if path > 0 else 0.0
        sc = (er * (fsc - ssc) + ssc) ** 2
        out[i] = out[i - 1] + sc * (c[i] - out[i - 1])
    return out


def kama(c, n: int, fast: int = 2, slow: int = 30):
    return _kama(np.asarray(c, dtype=np.float64), n, fast, slow)


@njit(cache=True)
def _session_vwap(tp, v, day):
    out = np.empty_like(tp)
    pv = 0.0
    vv = 0.0
    for i in range(len(tp)):
        if i == 0 or day[i] != day[i - 1]:
            pv = 0.0
            vv = 0.0
        w = v[i] if v[i] > 0 else 1.0
        pv += tp[i] * w
        vv += w
        out[i] = pv / vv
    return out


def session_vwap(h, l, c, v, day_id):
    return _session_vwap((h + l + c) / 3.0, np.asarray(v, dtype=np.float64), day_id)


def percent_rank(x, n: int):
    """Fraction of the last n values that are <= the current one (0..1)."""
    from numpy.lib.stride_tricks import sliding_window_view
    out = np.full(len(x), np.nan)
    if len(x) >= n:
        w = sliding_window_view(x, n)
        out[n - 1:] = (w <= w[:, -1:]).mean(axis=1)
    return out
