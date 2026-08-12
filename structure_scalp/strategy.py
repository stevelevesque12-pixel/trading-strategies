"""
Structure-alignment scalping strategy.

Three tiers, reusing the same swing/MSS primitives as failed2s/structure.py:
  - 15-minute structure direction: a persisting state that flips "long" on a
    bullish Market Structure Shift (a strong-bodied close through the last
    confirmed swing high) and flips "short" on the bearish mirror. Stays
    that way until the opposite break happens -- it does not reset every bar.
  - 1-minute structure direction: the same state machine, independently.
  - 5-second entry: while the 15m and 1m directions currently agree
    (aligned), enter on every fresh 5s MSS in that same direction.

Unlike failed2s.strategy.Failed2sStrategy, alignment here is a persisting
state, not a one-shot bias consumed by the first trade -- this strategy
re-enters on every qualifying 5s MSS for as long as alignment holds and no
position is open (the caller/engine is responsible for that flat-position
gate, same as Failed2sStrategy). That's what drives the higher frequency.

Data note: backtesting the 5-second entry timeframe needs real 5-second (or
finer) historical bars, which no free source available to this repo
provides (Yahoo Finance's finest granularity is 1-minute). See
backtest/scalp_engine.py's docstring for what that means in practice.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig
from failed2s.structure import SwingTracker, detect_mss


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


class StructureTracker:
    """Persisting structure direction for one timeframe, driven by MSS events."""

    def __init__(self, swing_strength: int = 2, min_body_pct: float = 0.5):
        self.swing_strength = swing_strength
        self.min_body_pct = min_body_pct
        self.swings = SwingTracker(strength=swing_strength)
        self.direction: Optional[Literal["long", "short"]] = None

    def update(self, bar: Bar) -> None:
        self.swings.update(bar)
        if detect_mss(bar, self.swings.last_swing_high, "long", self.min_body_pct):
            self.direction = "long"
        elif detect_mss(bar, self.swings.last_swing_low, "short", self.min_body_pct):
            self.direction = "short"

    def reset(self) -> None:
        self.swings = SwingTracker(strength=self.swing_strength)
        self.direction = None


class StructureScalpStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        stop_buffer_ticks: int = 2,
        target_r: float = 1.0,
        min_mss_body_pct: float = 0.5,
        swing_strength_15m: int = 2,
        swing_strength_1m: int = 2,
        swing_strength_5s: int = 2,
        session: Optional[SessionConfig] = None,
    ):
        self.tick_size = tick_size
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.target_r = target_r
        self.min_mss_body_pct = min_mss_body_pct
        self.session = session or SessionConfig()

        self.struct_15m = StructureTracker(swing_strength_15m, min_mss_body_pct)
        self.struct_1m = StructureTracker(swing_strength_1m, min_mss_body_pct)
        self.entry_swings = SwingTracker(strength=swing_strength_5s)
        self._entry_swing_strength = swing_strength_5s

        self._current_date = None

    def reset(self) -> None:
        """Clear all state -- called automatically on a new session day."""
        self.struct_15m.reset()
        self.struct_1m.reset()
        self.entry_swings = SwingTracker(strength=self._entry_swing_strength)

    def _roll_session(self, ts) -> None:
        d = ts.date()
        if self._current_date is None:
            self._current_date = d
        elif d != self._current_date:
            self._current_date = d
            self.reset()

    def _in_entry_window(self, ts) -> bool:
        t = ts.time()
        return self.session.session_start <= t < self.session.no_entry_after

    def on_15m_bar(self, bar: Bar) -> None:
        self._roll_session(bar.timestamp)
        self.struct_15m.update(bar)

    def on_1m_bar(self, bar: Bar) -> None:
        self._roll_session(bar.timestamp)
        self.struct_1m.update(bar)

    def on_5s_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)
        self.entry_swings.update(bar)

        direction = self.struct_15m.direction
        if direction is None or direction != self.struct_1m.direction:
            return None
        if not self._in_entry_window(bar.timestamp):
            return None

        swing = self.entry_swings.last_swing_high if direction == "long" else self.entry_swings.last_swing_low
        if not detect_mss(bar, swing, direction, self.min_mss_body_pct):
            return None

        entry_price = bar.close
        if direction == "long":
            stop_price = swing.price - self.stop_buffer
            risk = entry_price - stop_price
            target_price = entry_price + risk * self.target_r
        else:
            stop_price = swing.price + self.stop_buffer
            risk = stop_price - entry_price
            target_price = entry_price - risk * self.target_r

        if risk <= 0:
            return None

        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=f"15m+1m aligned {direction}, 5s MSS through {swing.price}",
        )
