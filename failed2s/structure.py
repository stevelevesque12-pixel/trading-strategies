"""Swing structure, Market Structure Shift (MSS), and Fair Value Gap (FVG) detection."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional

from .bars import Bar


@dataclass
class Swing:
    timestamp: object
    price: float
    kind: Literal["high", "low"]


class SwingTracker:
    """
    Rolling fractal swing-point tracker. A swing high/low is confirmed once
    `strength` bars have printed on both sides without invalidating it
    (classic N-bar fractal, confirmed with an N-bar lag).
    """

    def __init__(self, strength: int = 2):
        self.strength = strength
        self._buffer: Deque[Bar] = deque(maxlen=strength * 2 + 1)
        self.last_swing_high: Optional[Swing] = None
        self.last_swing_low: Optional[Swing] = None

    def update(self, bar: Bar) -> None:
        self._buffer.append(bar)
        if len(self._buffer) < self.strength * 2 + 1:
            return

        bars = list(self._buffer)
        pivot = bars[self.strength]
        left = bars[: self.strength]
        right = bars[self.strength + 1:]

        if all(pivot.high > b.high for b in left) and all(pivot.high > b.high for b in right):
            self.last_swing_high = Swing(pivot.timestamp, pivot.high, "high")

        if all(pivot.low < b.low for b in left) and all(pivot.low < b.low for b in right):
            self.last_swing_low = Swing(pivot.timestamp, pivot.low, "low")


def detect_mss(
    bar: Bar,
    swing: Optional[Swing],
    direction: Literal["long", "short"],
    min_body_pct: float = 0.5,
) -> bool:
    """
    Market Structure Shift: close breaks the last confirmed opposite swing
    point with a strong-bodied candle in the direction of the break.
    """
    if swing is None or bar.body_pct < min_body_pct:
        return False
    if direction == "long":
        return swing.kind == "high" and bar.close > swing.price and bar.is_bullish
    return swing.kind == "low" and bar.close < swing.price and bar.is_bearish


def detect_fvg(bar1: Bar, bar2: Bar, bar3: Bar) -> Optional[Literal["bullish", "bearish"]]:
    """3-candle Fair Value Gap. `bar2` is the displacement candle. Returns 'bullish'/'bearish'/None."""
    if bar1.high < bar3.low:
        return "bullish"
    if bar1.low > bar3.high:
        return "bearish"
    return None
