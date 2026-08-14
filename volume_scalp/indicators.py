"""
Volume-based indicators for the volume scalp strategy: session VWAP,
relative volume (RVOL), a close-location-value (CLV) volume-delta proxy,
and a rolling breakout channel.

These all operate on plain OHLCV bars (failed2s.bars.Bar) -- no bid/ask or
tick data -- which is what every free/cheap data source available to this
repo (and most retail futures feeds) actually provides. The volume-delta
proxy is a standard approximation used when true buy/sell tick volume
isn't available (the same idea behind the classic Accumulation/
Distribution Line and Chaikin Money Flow indicators): a bar that closes
near its high is assumed to have been net bought, a bar that closes near
its low is assumed net sold, scaled by that bar's volume. It is *not* real
order flow -- a large trade printed mid-range would be misclassified, and
it says nothing about what happened intra-bar -- but it's a reasonable,
well-established proxy from OHLCV alone, and it's what most retail
"volume delta" scalping tools compute when they don't have exchange tick
data either.
"""

from collections import deque
from typing import Deque, Optional

from failed2s.bars import Bar


def clv(bar: Bar) -> float:
    """Close Location Value in [-1, 1]: +1 = closed at the high, -1 = closed at the low."""
    rng = bar.range
    if rng <= 0:
        return 0.0
    return ((bar.close - bar.low) - (bar.high - bar.close)) / rng


def bar_volume_delta(bar: Bar) -> float:
    """CLV-weighted volume: a proxy for net buying (+) / selling (-) volume in the bar."""
    return clv(bar) * bar.volume


class SessionVWAP:
    """Volume-weighted average price, anchored to (reset at) the start of each session.

    Updates include the current bar (there's no lookahead concern here --
    VWAP at a bar's close is only ever computed from that bar's own known
    OHLCV plus everything before it).
    """

    def __init__(self):
        self._cum_pv = 0.0
        self._cum_vol = 0.0
        self.value: Optional[float] = None

    def update(self, bar: Bar) -> None:
        typical_price = (bar.high + bar.low + bar.close) / 3.0
        self._cum_pv += typical_price * bar.volume
        self._cum_vol += bar.volume
        self.value = self._cum_pv / self._cum_vol if self._cum_vol > 0 else None

    def reset(self) -> None:
        self._cum_pv = 0.0
        self._cum_vol = 0.0
        self.value = None


class RollingVolume:
    """Rolling average volume over the trailing `window` bars, for relative volume (RVOL).

    Call `relvol()` *before* `update()` for a given bar -- the baseline a
    bar is judged against must never include that bar's own volume, or a
    genuine spike would inflate its own average and dampen its own signal.
    """

    def __init__(self, window: int = 20):
        self.window = window
        self._vols: Deque[float] = deque(maxlen=window)

    @property
    def average(self) -> float:
        return sum(self._vols) / len(self._vols) if self._vols else 0.0

    def relvol(self, current_volume: float) -> float:
        avg = self.average
        return current_volume / avg if avg > 0 else 0.0

    def update(self, volume: float) -> None:
        self._vols.append(volume)


class RollingDelta:
    """Sum of the CLV-based volume-delta proxy over the trailing `window` bars.

    Unlike RollingVolume/RollingChannel, this is meant to be updated with
    the current bar's own delta *before* being read -- it's a momentum
    confirmation ("does the last `window` bars' worth of order flow,
    including this bar, support the move?"), not a baseline the current
    bar is compared against, so including it is the point.
    """

    def __init__(self, window: int = 5):
        self.window = window
        self._deltas: Deque[float] = deque(maxlen=window)

    @property
    def value(self) -> float:
        return sum(self._deltas)

    def update(self, delta: float) -> None:
        self._deltas.append(delta)


class RollingChannel:
    """Highest-high / lowest-low over the trailing `window` bars, excluding the current bar.

    Returns None for both until `window` bars have printed, so a breakout
    can't fire off a partially-formed channel. Like RollingVolume, read
    `.high`/`.low` before calling `update()` for the current bar.
    """

    def __init__(self, window: int = 10):
        self.window = window
        self._bars: Deque[Bar] = deque(maxlen=window)

    @property
    def high(self) -> Optional[float]:
        if len(self._bars) < self.window:
            return None
        return max(b.high for b in self._bars)

    @property
    def low(self) -> Optional[float]:
        if len(self._bars) < self.window:
            return None
        return min(b.low for b in self._bars)

    def update(self, bar: Bar) -> None:
        self._bars.append(bar)
