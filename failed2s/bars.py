"""Bar classification and Failed-2 detection (TheStrat methodology)."""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Optional


class BarType(str, Enum):
    INSIDE = "1"
    DIRECTIONAL_UP = "2U"
    DIRECTIONAL_DOWN = "2D"
    OUTSIDE = "3"


@dataclass
class Bar:
    timestamp: Any
    open: float
    high: float
    low: float
    close: float
    volume: float = 0.0

    @property
    def is_bullish(self) -> bool:
        return self.close > self.open

    @property
    def is_bearish(self) -> bool:
        return self.close < self.open

    @property
    def body(self) -> float:
        return abs(self.close - self.open)

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def body_pct(self) -> float:
        return self.body / self.range if self.range else 0.0


def classify_bar(prev: Bar, curr: Bar) -> BarType:
    """Classify `curr` relative to `prev` per TheStrat: 1 (inside), 2U/2D (directional), 3 (outside)."""
    broke_up = curr.high > prev.high
    broke_down = curr.low < prev.low
    if broke_up and broke_down:
        return BarType.OUTSIDE
    if broke_up:
        return BarType.DIRECTIONAL_UP
    if broke_down:
        return BarType.DIRECTIONAL_DOWN
    return BarType.INSIDE


def detect_failed_2(prev: Bar, curr: Bar, require_reclaim: bool = True) -> Optional[str]:
    """
    Detect a Failed-2: a directional (2) bar that reverses and closes against
    its own break -- a failed breakout / liquidity sweep.

    require_reclaim=True additionally requires the close to give back the
    broken level (not just close red/green), for a stricter signal.

    Returns "F2U", "F2D", or None.
    """
    bar_type = classify_bar(prev, curr)

    if bar_type == BarType.DIRECTIONAL_UP and curr.is_bearish:
        if not require_reclaim or curr.close < prev.high:
            return "F2U"

    if bar_type == BarType.DIRECTIONAL_DOWN and curr.is_bullish:
        if not require_reclaim or curr.close > prev.low:
            return "F2D"

    return None
