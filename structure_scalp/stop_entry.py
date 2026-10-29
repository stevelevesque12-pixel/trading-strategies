"""
Structure bias + stop-order entries on the lower timeframe.

This is the version the user actually trades by hand ("1m direction /
structure, 5s entries on stops, 1:1" -- see their trade log), as opposed to
structure_scalp/strategy.py's anchored-VWAP-retest limit entry.

Long case (short is the exact mirror):
  - Bias: the higher timeframe's StructureTracker direction (1m in the live
    version). Flips long on a bullish MSS, short on a bearish one.
  - While the bias is long and we're flat, a buy-stop rests one tick above
    the last confirmed entry-timeframe swing high (5s live). Protective stop
    sits `stop_buffer_ticks` below the last confirmed entry-timeframe swing
    low. Target = entry + target_r * risk.
  - The order is only placed while it sits above the current close (a
    resting stop, not a chase), and each swing high can trigger at most one
    trade. It is re-priced every bar as new swings confirm, and pulled when
    the bias flips, the swing low that defines the stop gets taken out, or
    the entry window closes.

Timeframe-agnostic: the engine (backtest/scalp_engine.py) decides which
bars count as "bias" and "entry". With only 1-minute history available, the
same 12:1 ratio can be studied as 12m bias / 1m entries; a true 1m/5s run
needs real 5-second bars.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig
from failed2s.structure import SwingTracker

from .strategy import StructureTracker


@dataclass
class StopOrder:
    direction: Literal["long", "short"]
    trigger: float  # stop-entry price
    stop_price: float
    target_r: float
    swing_ts: object  # timestamp of the swing being broken -- one trade per swing

    def risk(self) -> float:
        return abs(self.trigger - self.stop_price)

    def target_for(self, fill_price: float) -> float:
        # Target is measured from the actual fill so a gap-through fill keeps 1:1
        # relative to where the position really started.
        r = abs(fill_price - self.stop_price)
        return fill_price + r * self.target_r if self.direction == "long" else fill_price - r * self.target_r


class StopEntryScalpStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        target_r: float = 1.0,
        entry_offset_ticks: int = 1,
        stop_buffer_ticks: int = 1,
        min_mss_body_pct: float = 0.5,
        swing_strength_bias: int = 2,
        swing_strength_entry: int = 2,
        min_risk_ticks: int = 2,
        max_risk_ticks: Optional[int] = None,
        session: Optional[SessionConfig] = None,
    ):
        self.tick_size = tick_size
        self.target_r = target_r
        self.entry_offset = entry_offset_ticks * tick_size
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.min_risk = min_risk_ticks * tick_size
        self.max_risk = max_risk_ticks * tick_size if max_risk_ticks else None
        self.min_mss_body_pct = min_mss_body_pct
        self.session = session or SessionConfig()

        self._bias_strength = swing_strength_bias
        self._entry_strength = swing_strength_entry
        self.bias = StructureTracker(swing_strength_bias, min_mss_body_pct)
        self.entry_swings = SwingTracker(strength=swing_strength_entry)
        self._used_swings: set = set()
        self._current_date = None

    def reset(self) -> None:
        self.bias = StructureTracker(self._bias_strength, self.min_mss_body_pct)
        self.entry_swings = SwingTracker(strength=self._entry_strength)
        self._used_swings = set()

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

    def on_bias_bar(self, bar: Bar) -> None:
        self._roll_session(bar.timestamp)
        self.bias.update(bar)

    def on_entry_bar(self, bar: Bar) -> Optional[StopOrder]:
        """Update with a closed entry-timeframe bar; return the order to rest for the next bar (or None)."""
        self._roll_session(bar.timestamp)
        self.entry_swings.update(bar)

        direction = self.bias.direction
        if direction is None or not self._in_entry_window(bar.timestamp):
            return None

        hi, lo = self.entry_swings.last_swing_high, self.entry_swings.last_swing_low
        if hi is None or lo is None:
            return None

        if direction == "long":
            breakout, protect = hi, lo
            trigger = hi.price + self.entry_offset
            stop = lo.price - self.stop_buffer
            if bar.close >= trigger or bar.low <= stop:
                return None
        else:
            breakout, protect = lo, hi
            trigger = lo.price - self.entry_offset
            stop = hi.price + self.stop_buffer
            if bar.close <= trigger or bar.high >= stop:
                return None

        if breakout.timestamp in self._used_swings:
            return None

        risk = abs(trigger - stop)
        if risk < self.min_risk or (self.max_risk is not None and risk > self.max_risk):
            return None

        return StopOrder(direction, trigger, stop, self.target_r, breakout.timestamp)

    def mark_filled(self, order: StopOrder) -> None:
        self._used_swings.add(order.swing_ts)
