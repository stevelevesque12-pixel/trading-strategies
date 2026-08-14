"""
Premium/discount zone tracking, balanced/imbalanced state, and Logical
Liquidity Targets (LLT), per the "Lab Model" deck.

Operationalized rules (the source deck is a discretionary/visual system, so
these are concrete, deterministic stand-ins -- see README.md for the
reasoning):

  - A "recent price leg" is the range between the two most recently
    confirmed swing points of opposite kind (reusing `failed2s.structure`'s
    fractal `SwingTracker`, extended here to keep the full swing history
    rather than just the latest one).
  - Premium = upper half of that leg, discount = lower half, split at the
    midpoint.
  - "Balanced" means price has traded back through the leg's midpoint since
    the leg's second point confirmed. This is tracked per-leg (identified by
    its two swing timestamps), so a freshly-formed new leg starts
    unbalanced again even if the previous leg had balanced.
  - LLT ("Logical Liquidity Target"): the deck's own wording -- "the first
    high or low through the premium/discount midline" -- is read literally
    here as the nearest pre-existing swing point (of the opposite kind to
    the leg's end, i.e. the kind price would be sweeping toward) that
    already sits on the far side of the midpoint. That matches the example
    charts, where the LLT line sits just past the midline rather than at
    the extreme of the visible range. Falls back to the leg's own start
    point if no earlier qualifying swing exists.
"""

import pandas as pd

from failed2s.bars import Bar
from failed2s.structure import Swing, SwingTracker
from dataclasses import dataclass
from typing import List, Literal, Optional


def four_hour_origin(df: pd.DataFrame) -> pd.Timestamp:
    """
    Anchor for 4-hour resampling so bins close on 02:00/06:00/10:00/14:00/
    18:00/22:00 in the frame's own tz (matching the deck's "10am 4hr
    candle") instead of pandas' default UTC-midnight-aligned boundaries.

    Caveat: pandas bins tz-aware data by fixed elapsed time from the origin,
    not by local wall-clock, so a 4h bin that straddles a DST transition can
    drift by an hour from what a wall-clock-anchored charting platform would
    show for that one day. Same class of caveat as the plain UTC-anchored
    4H resampling already flagged in the main README for failed2s.
    """
    first_day = df.index[0].normalize()
    return first_day - pd.Timedelta(hours=6)  # previous day's 18:00 local


@dataclass
class Leg:
    start: Swing
    end: Swing

    @property
    def high(self) -> float:
        return max(self.start.price, self.end.price)

    @property
    def low(self) -> float:
        return min(self.start.price, self.end.price)

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0

    @property
    def direction(self) -> Literal["up", "down"]:
        return "up" if self.end.price > self.start.price else "down"


class ZoneTracker:
    """Tracks confirmed swing history on one timeframe/symbol and derives the current leg, its balanced state, and its LLT."""

    def __init__(self, swing_strength: int = 2, max_history: int = 200):
        self.swing_strength = swing_strength
        self.max_history = max_history
        self._swings = SwingTracker(strength=swing_strength)
        self.history: List[Swing] = []
        self._balanced_legs: set = set()

    def update(self, bar: Bar) -> None:
        prev_high, prev_low = self._swings.last_swing_high, self._swings.last_swing_low
        self._swings.update(bar)

        if self._swings.last_swing_high is not None and self._swings.last_swing_high is not prev_high:
            self.history.append(self._swings.last_swing_high)
        if self._swings.last_swing_low is not None and self._swings.last_swing_low is not prev_low:
            self.history.append(self._swings.last_swing_low)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

        leg = self.current_leg
        if leg is None:
            return
        key = (leg.start.timestamp, leg.end.timestamp)
        if leg.direction == "down" and bar.high >= leg.mid:
            self._balanced_legs.add(key)
        elif leg.direction == "up" and bar.low <= leg.mid:
            self._balanced_legs.add(key)

    @property
    def current_leg(self) -> Optional[Leg]:
        """The most recent confirmed swing paired with the nearest earlier swing of the opposite kind."""
        if len(self.history) < 2:
            return None
        end = self.history[-1]
        for s in reversed(self.history[:-1]):
            if s.kind != end.kind:
                return Leg(start=s, end=end)
        return None

    @property
    def is_balanced(self) -> bool:
        leg = self.current_leg
        if leg is None:
            return False
        return (leg.start.timestamp, leg.end.timestamp) in self._balanced_legs

    @property
    def llt(self) -> Optional[float]:
        leg = self.current_leg
        if leg is None:
            return None

        if leg.direction == "down":
            # Reversal targets upward: nearest pre-existing swing HIGH above the midpoint.
            candidates = [s for s in self.history if s.kind == "high" and s.price >= leg.mid]
            best = min(candidates, key=lambda s: s.price) if candidates else None
        else:
            # Reversal targets downward: nearest pre-existing swing LOW below the midpoint.
            candidates = [s for s in self.history if s.kind == "low" and s.price <= leg.mid]
            best = max(candidates, key=lambda s: s.price) if candidates else None

        return best.price if best is not None else leg.start.price
