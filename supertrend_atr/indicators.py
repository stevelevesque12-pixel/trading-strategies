"""
Streaming (bar-by-bar) ATR and SuperTrend, matching TradingView's
`ta.atr` / `ta.supertrend` so the Python backtest and the Pine port in
`tradingview/supertrend_atr.pine` agree bar for bar.

Streaming rather than vectorized so the same objects drive both the
backtester and any live runner -- each `update()` only sees closed bars.
"""

from collections import deque
from typing import Optional

from failed2s.bars import Bar


class ATR:
    """Wilder ATR (RMA of true range), seeded with an SMA of the first `period` TRs like ta.atr."""

    def __init__(self, period: int = 14):
        self.period = period
        self.value: Optional[float] = None
        self._prev_close: Optional[float] = None
        self._seed: list = []

    def update(self, bar: Bar) -> Optional[float]:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(bar.high - bar.low, abs(bar.high - self._prev_close), abs(bar.low - self._prev_close))
        self._prev_close = bar.close

        if self.value is None:
            self._seed.append(tr)
            if len(self._seed) == self.period:
                self.value = sum(self._seed) / self.period
                self._seed = []
        else:
            self.value = (self.value * (self.period - 1) + tr) / self.period
        return self.value


class SMA:
    def __init__(self, period: int):
        self.period = period
        self._window: deque = deque(maxlen=period)
        self._sum = 0.0

    def update(self, x: float) -> Optional[float]:
        if len(self._window) == self.period:
            self._sum -= self._window[0]
        self._window.append(x)
        self._sum += x
        return self.value

    @property
    def value(self) -> Optional[float]:
        return self._sum / self.period if len(self._window) == self.period else None


class SuperTrend:
    """
    SuperTrend(factor, atr_period) on hl2, same band-ratchet and flip rules
    as ta.supertrend. `direction` is +1 (uptrend, line below price) or -1
    (downtrend, line above price); None until the ATR has warmed up.
    `flipped` is True only on the bar the direction changed.
    """

    def __init__(self, atr_period: int = 10, factor: float = 3.0):
        self.factor = factor
        self.atr = ATR(atr_period)
        self.upper: Optional[float] = None
        self.lower: Optional[float] = None
        self.line: Optional[float] = None
        self.direction: Optional[int] = None
        self.flipped = False
        self._prev_close: Optional[float] = None

    def update(self, bar: Bar) -> Optional[int]:
        prev_atr = self.atr.value
        atr = self.atr.update(bar)
        prev_close = self._prev_close
        self._prev_close = bar.close
        self.flipped = False

        if atr is None:
            return None

        hl2 = (bar.high + bar.low) / 2.0
        upper = hl2 + self.factor * atr
        lower = hl2 - self.factor * atr
        prev_upper, prev_lower, prev_line = self.upper, self.lower, self.line

        if prev_lower is not None and not (lower > prev_lower or prev_close < prev_lower):
            lower = prev_lower
        if prev_upper is not None and not (upper < prev_upper or prev_close > prev_upper):
            upper = prev_upper

        prev_direction = self.direction
        if prev_atr is None or prev_line is None:
            direction = -1  # ta.supertrend starts in the "line above price" state
        elif prev_line == prev_upper:
            direction = 1 if bar.close > upper else -1
        else:
            direction = -1 if bar.close < lower else 1

        self.upper, self.lower = upper, lower
        self.direction = direction
        self.line = lower if direction == 1 else upper
        self.flipped = prev_direction is not None and direction != prev_direction
        return direction
