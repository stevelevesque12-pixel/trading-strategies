"""
JJ Simon's Fair Value Theory (FVT) strategy engine, for NQ.

Cascade, per the PDF ("JJ Simon's Fair Value Theory NQ Strategy"):
  1. Two intraday windows: 9:30-11:00 and 14:00-15:00 NY time. The window's
     open price is the "fair value" anchor (9:30 open / 2pm price).
  2. First ~10-15m of a window ("continuation" phase): look for a
     displacement candle + BOS/MSB *away* from fair value.
     Rest of the window ("reversion" phase): look for a displacement candle
     + BOS/MSB *back toward* fair value.
  3. Entry: market order on the signal bar's close. No trade management.
  4. Stop/target distance is picked from an ATR bucket (fixed 1.5R either
     way): >20 ATR -> 50/75, 7-20 ATR -> 25/37.5, <7 ATR -> 16.5/24.75.

Two interpretation calls the PDF leaves implicit, made explicit here:
  - "Displacement candle" is defined mechanically as counter-wick <= 20% of
    the candle's high-low range (the PDF's own "more mechanical definition").
  - The ATR is computed on the entry timeframe itself (1-minute bars, the
    strategy's stated general parameter) rather than a daily ATR: the given
    bucket thresholds (7 / 20 points) match a 1-minute NQ true range, not a
    daily one (which runs well into the hundreds of points).

Deliberately not implemented (the PDF itself flags these as discretionary /
unconfirmed, not core mechanical rules): 2nd-attempt re-entries, VWAP as a
discretionary confluence filter, 8:30am news reversions, and tagging extra
continuation trades after price returns to fair value.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import time
from typing import Deque, List, Literal, Optional

from failed2s.bars import Bar

from .atr import RollingATR
from .structure import SwingTracker, detect_mss, is_displacement_candle


@dataclass(frozen=True)
class SessionWindow:
    name: str
    start: time
    end: time
    entry_start: time  # first bar eligible to trade (lets a window skip its own open)
    continuation_minutes: int = 15


DEFAULT_WINDOWS: List[SessionWindow] = [
    # Avoid the first 3 minutes after the 9:30 open, per the PDF.
    SessionWindow("AM", time(9, 30), time(11, 0), time(9, 33), continuation_minutes=15),
    SessionWindow("PM", time(14, 0), time(15, 0), time(14, 0), continuation_minutes=15),
]

# (atr_upper_exclusive_bound, sl_points, tp_points) checked low-to-high; the
# last row with atr_upper_bound=None is the catch-all "above 20" bucket.
ATR_BUCKETS = [
    (7.0, 16.5, 24.75),
    (20.0, 25.0, 37.5),
    (None, 50.0, 75.0),
]


def atr_to_stop_target(atr: float) -> "tuple[float, float]":
    for upper, sl, tp in ATR_BUCKETS:
        if upper is None or atr < upper:
            return sl, tp
    raise AssertionError("unreachable: ATR_BUCKETS must end with an open-ended bucket")


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    atr: float
    window: str
    phase: Literal["continuation", "reversion"]
    reason: str


class FairValueStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        atr_period: int = 14,
        swing_strength: int = 2,
        min_mss_body_pct: float = 0.5,
        max_counter_wick_pct: float = 0.20,
        windows: Optional[List[SessionWindow]] = None,
        restrict_to_first_hour: bool = False,
    ):
        self.tick_size = tick_size
        self.swing_strength = swing_strength
        self.min_mss_body_pct = min_mss_body_pct
        self.max_counter_wick_pct = max_counter_wick_pct
        self.windows = windows or DEFAULT_WINDOWS
        # PDF: "may be best to not trade the 2nd hour of NY trade windows
        # (optimization)" -- off by default since it's flagged as untested.
        self.restrict_to_first_hour = restrict_to_first_hour

        self._atr = RollingATR(period=atr_period)
        self._swings = SwingTracker(strength=swing_strength)
        self._current_date = None
        self._window_fv: dict = {}

    def reset(self) -> None:
        """Clear per-day state (swing structure, fair-value anchors). ATR keeps rolling across days."""
        self._swings = SwingTracker(strength=self.swing_strength)
        self._window_fv = {w.name: None for w in self.windows}

    def _roll_session(self, ts) -> None:
        d = ts.date()
        if self._current_date is None:
            self._current_date = d
            self.reset()
        elif d != self._current_date:
            self._current_date = d
            self.reset()

    def _active_window(self, ts) -> Optional[SessionWindow]:
        t = ts.time()
        for w in self.windows:
            if w.start <= t < w.end:
                return w
        return None

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)
        atr = self._atr.update(bar)
        self._swings.update(bar)

        window = self._active_window(bar.timestamp)
        if window is None:
            return None

        t = bar.timestamp.time()

        # Fair value = the open price of the window's first bar.
        if self._window_fv.get(window.name) is None:
            self._window_fv[window.name] = bar.open
        fv_price = self._window_fv[window.name]

        if t < window.entry_start or atr is None:
            return None

        if self.restrict_to_first_hour:
            minutes_into_window = _minutes_between(window.start, t)
            if minutes_into_window >= 60:
                return None

        if bar.close == fv_price:
            return None
        price_above_fv = bar.close > fv_price

        minutes_into_window = _minutes_between(window.start, t)
        phase: Literal["continuation", "reversion"]
        if minutes_into_window <= window.continuation_minutes:
            phase = "continuation"
            direction: Literal["long", "short"] = "long" if price_above_fv else "short"
        else:
            phase = "reversion"
            direction = "short" if price_above_fv else "long"

        swing = self._swings.last_swing_high if direction == "long" else self._swings.last_swing_low
        if not detect_mss(bar, swing, direction, self.min_mss_body_pct):
            return None
        if not is_displacement_candle(bar, direction, self.max_counter_wick_pct):
            return None

        sl_points, tp_points = atr_to_stop_target(atr)
        entry_price = bar.close
        if direction == "long":
            stop_price = entry_price - sl_points
            target_price = entry_price + tp_points
        else:
            stop_price = entry_price + sl_points
            target_price = entry_price - tp_points

        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            atr=atr,
            window=window.name,
            phase=phase,
            reason=(
                f"{window.name}/{phase} fv={fv_price} atr={atr:.2f} "
                f"swing={swing.price} sl={sl_points} tp={tp_points}"
            ),
        )


def _minutes_between(start: time, t: time) -> float:
    return (t.hour * 60 + t.minute + t.second / 60) - (start.hour * 60 + start.minute + start.second / 60)
