"""
Fair Value Theory structure primitives.

Swing tracking and Market Structure Break/Shift (MSB/BOS) detection are the
same fractal logic already used by the Failed-2s strategy, so they're reused
directly rather than re-implemented. This module only adds the
displacement-candle test, which is specific to this strategy.
"""

from failed2s.bars import Bar
from failed2s.structure import Swing, SwingTracker, detect_mss  # noqa: F401 (re-exported)


def is_displacement_candle(bar: Bar, direction: str, max_counter_wick_pct: float = 0.20) -> bool:
    """
    Mechanical reading of "closes decisively (small counter-wick)": the wick
    on the side opposing the move must be less than `max_counter_wick_pct`
    of the candle's full high-low range.
    """
    if bar.range <= 0:
        return False

    if direction == "long":
        if not bar.is_bullish:
            return False
        counter_wick = bar.open - bar.low
    elif direction == "short":
        if not bar.is_bearish:
            return False
        counter_wick = bar.high - bar.open
    else:
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    return (counter_wick / bar.range) <= max_counter_wick_pct
