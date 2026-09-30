"""
MACD Strategy 1: crossing of the MACD line and its signal line.
MACD Strategy 2: crossing of the zero line (see `zero_cross_signals`).

- MACD line   = EMA(close, fast) - EMA(close, slow)
- Signal line = EMA(MACD, signal)
- MACD crosses the signal line from below -> buy signal (+1)
- MACD crosses the signal line from above -> sell signal (-1)

A cross is only acknowledged on a *closed* bar; the backtest fills it at
the next bar's open, so nothing here looks ahead.
"""

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class MACDParams:
    fast: int = 12
    slow: int = 26
    signal: int = 9


def macd(close: pd.Series, params: MACDParams = MACDParams()) -> pd.DataFrame:
    """Standard (TradingView-equivalent) MACD: EMAs with alpha = 2/(n+1)."""
    ema_fast = close.ewm(span=params.fast, adjust=False).mean()
    ema_slow = close.ewm(span=params.slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=params.signal, adjust=False).mean()
    return pd.DataFrame(
        {"macd": macd_line, "signal": signal_line, "hist": macd_line - signal_line},
        index=close.index,
    )


def crossover_signals(close: pd.Series, params: MACDParams = MACDParams()) -> pd.Series:
    """
    +1 on the bar where MACD closes above the signal line after being at or
    below it on the previous bar, -1 for the mirror image, 0 otherwise.
    The first `slow + signal` bars are suppressed while the EMAs warm up.
    """
    hist = macd(close, params)["hist"]
    above = hist > 0
    prev_above = above.shift(1, fill_value=False)
    below = hist < 0
    prev_below = below.shift(1, fill_value=False)

    sig = pd.Series(0, index=close.index, dtype=int)
    sig[above & ~prev_above] = 1
    sig[below & ~prev_below] = -1
    sig.iloc[: params.slow + params.signal] = 0
    return sig


def _transitions(state: pd.Series, warmup: int) -> pd.Series:
    """+1/-1 on the bar `state` first becomes +1/-1; 0 otherwise."""
    prev = state.shift(1, fill_value=0)
    sig = state.where((state != prev) & (state != 0), 0).astype(int)
    sig.iloc[:warmup] = 0
    return sig


def zero_cross_signals(close: pd.Series, params: MACDParams = MACDParams(), confirm_hist: bool = False) -> pd.Series:
    """
    MACD Strategy 2 -- zero-line cross.

    confirm_hist=False: +1 when the MACD line closes above 0 after being
    at/below it (i.e. EMA(fast) crosses above EMA(slow)), -1 for the mirror.

    confirm_hist=True ("MACD + zero line combination"): +1 on the bar where
    MACD > 0 AND histogram > 0 (MACD above its signal line) first both hold,
    -1 when MACD < 0 AND histogram < 0 first both hold. Whichever happens
    second -- the zero cross or the signal cross -- triggers the entry.
    """
    m = macd(close, params)
    warmup = params.slow + params.signal
    if not confirm_hist:
        # A cross needs the previous bar on the other side (or at 0).
        up = (m["macd"] > 0) & (m["macd"].shift(1) <= 0)
        dn = (m["macd"] < 0) & (m["macd"].shift(1) >= 0)
        sig = pd.Series(0, index=close.index, dtype=int)
        sig[up] = 1
        sig[dn] = -1
        sig.iloc[:warmup] = 0
        return sig
    state = pd.Series(0, index=close.index, dtype=int)
    state[(m["macd"] > 0) & (m["hist"] > 0)] = 1
    state[(m["macd"] < 0) & (m["hist"] < 0)] = -1
    return _transitions(state, warmup)


SIGNAL_RULES = {
    "signal_cross": lambda c, p: crossover_signals(c, p),
    "zero_cross": lambda c, p: zero_cross_signals(c, p),
    "zero_cross_hist": lambda c, p: zero_cross_signals(c, p, confirm_hist=True),
}
