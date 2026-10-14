"""
Streaming indicators that follow QuantConnect LEAN's definitions, so the port
of the QC algorithm in `strategy.py` computes the same values bar for bar
(up to EMA seeding, see `EMA`).

Each indicator exposes `update(x) -> None`, `value`, `previous` and
`is_ready`, mirroring LEAN's `Current.Value` / `Previous.Value` / `IsReady`.
"""

from collections import deque
from typing import Deque, Optional


class EMA:
    """Exponential moving average, k = 2 / (n + 1), seeded with the SMA of the
    first n inputs (LEAN's current behaviour). Ready after n inputs."""

    def __init__(self, period: int):
        self.period = period
        self.k = 2.0 / (period + 1)
        self.samples = 0
        self._seed_sum = 0.0
        self.value: Optional[float] = None
        self.previous: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return self.samples >= self.period

    def update(self, x: float) -> None:
        self.samples += 1
        self.previous = self.value
        if self.samples <= self.period:
            self._seed_sum += x
            self.value = self._seed_sum / self.samples
        else:
            self.value = x * self.k + self.value * (1.0 - self.k)


class SMA:
    def __init__(self, period: int):
        self.period = period
        self._window: Deque[float] = deque(maxlen=period)
        self.value: Optional[float] = None
        self.previous: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return len(self._window) == self.period

    def update(self, x: float) -> None:
        self._window.append(x)
        self.previous = self.value
        self.value = sum(self._window) / len(self._window)


class RSI:
    """Relative Strength Index with Wilder smoothing (LEAN's default
    MovingAverageType.Wilders): SMA seed over the first n changes, then
    avg = (avg * (n - 1) + x) / n. Ready once n changes have been seen."""

    def __init__(self, period: int):
        self.period = period
        self._prev_close: Optional[float] = None
        self._changes = 0
        self._avg_gain = 0.0
        self._avg_loss = 0.0
        self.value: Optional[float] = None
        self.previous: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return self._changes >= self.period

    def update(self, close: float) -> None:
        if self._prev_close is None:
            self._prev_close = close
            return
        change = close - self._prev_close
        self._prev_close = close
        gain, loss = max(change, 0.0), max(-change, 0.0)
        self._changes += 1
        n = self.period
        if self._changes <= n:
            self._avg_gain += (gain - self._avg_gain) / self._changes
            self._avg_loss += (loss - self._avg_loss) / self._changes
        else:
            self._avg_gain = (self._avg_gain * (n - 1) + gain) / n
            self._avg_loss = (self._avg_loss * (n - 1) + loss) / n
        if not self.is_ready:
            return
        self.previous = self.value
        if self._avg_loss == 0.0:
            self.value = 100.0
        else:
            rs = self._avg_gain / self._avg_loss
            self.value = 100.0 - 100.0 / (1.0 + rs)


class Momentum:
    """close - close n bars ago. Ready after n + 1 inputs."""

    def __init__(self, period: int):
        self.period = period
        self._window: Deque[float] = deque(maxlen=period + 1)
        self.value: Optional[float] = None
        self.previous: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return len(self._window) == self.period + 1

    def update(self, x: float) -> None:
        self._window.append(x)
        if self.is_ready:
            self.previous = self.value
            self.value = self._window[-1] - self._window[0]


class UltimateOscillator:
    """Larry Williams' Ultimate Oscillator over three windows (default 7/14/28).

    BP = close - min(low, prev_close), TR = max(high, prev_close) - min(low, prev_close),
    UO = 100 * (4*A1 + 2*A2 + A3) / 7 where Ai = sum(BP, ni) / sum(TR, ni).
    Needs a previous close, so the first bar only seeds it."""

    def __init__(self, p1: int = 7, p2: int = 14, p3: int = 28):
        self.periods = (p1, p2, p3)
        self._prev_close: Optional[float] = None
        self._bp: Deque[float] = deque(maxlen=p3)
        self._tr: Deque[float] = deque(maxlen=p3)
        self.value: Optional[float] = None
        self.previous: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return len(self._bp) == self.periods[2]

    def update(self, high: float, low: float, close: float) -> None:
        if self._prev_close is None:
            self._prev_close = close
            return
        true_low = min(low, self._prev_close)
        self._bp.append(close - true_low)
        self._tr.append(max(high, self._prev_close) - true_low)
        self._prev_close = close
        if not self.is_ready:
            return
        bp, tr = list(self._bp), list(self._tr)
        avgs = []
        for n in self.periods:
            tr_sum = sum(tr[-n:])
            avgs.append(sum(bp[-n:]) / tr_sum if tr_sum else 0.0)
        self.previous = self.value
        self.value = 100.0 * (4 * avgs[0] + 2 * avgs[1] + avgs[2]) / 7.0


class Slope:
    """`indicator - Delay(1).of(indicator)`: the one-update change in a source
    indicator's value. `value` is the latest change, `previous` the one before
    it. Call `update()` right after the source updates."""

    def __init__(self, source):
        self.source = source
        self._last: Optional[float] = None
        self.value: Optional[float] = None
        self.previous: Optional[float] = None

    @property
    def is_ready(self) -> bool:
        return self.value is not None

    def update(self) -> None:
        if not self.source.is_ready:
            return
        cur = self.source.value
        if self._last is not None:
            self.previous = self.value
            self.value = cur - self._last
        self._last = cur
