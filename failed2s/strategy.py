"""
Failed-2s strategy engine.

Cascade (per Trader Mike's "Failed 2s" system):
  1. A Failed-2 (F2U/F2D) completes on the *bias* timeframe -> sets a
     directional bias and the swept liquidity level.
  2. While that bias is active (and we're inside the intraday entry
     window), we wait for a Market Structure Shift (MSS) on the *entry*
     timeframe, in the same direction, with a strong-bodied close and an
     optional Fair Value Gap -> that's the entry trigger.

Timeframe pairs (entry_tf, bias_tf) used: 1m-15m, 5m-1h, 15m-4h.

Strictly intraday: bias/structure state resets every session day, and no
signal fires outside the configured entry window.
"""

from collections import deque
from dataclasses import dataclass, field
from datetime import time
from typing import Deque, Literal, Optional

from .bars import Bar, detect_failed_2
from .structure import SwingTracker, detect_fvg, detect_mss


@dataclass(frozen=True)
class TimeframePair:
    name: str
    entry_tf: str  # pandas resample rule / Timedelta string, e.g. "1min"
    bias_tf: str  # pandas resample rule / Timedelta string, e.g. "15min"


PAIRS = {
    "1m-15m": TimeframePair("1m-15m", "1min", "15min"),
    "5m-1h": TimeframePair("5m-1h", "5min", "1h"),
    "15m-4h": TimeframePair("15m-4h", "15min", "4h"),
}


@dataclass
class SessionConfig:
    session_start: time = time(9, 30)
    session_end: time = time(16, 0)
    no_entry_after: time = time(15, 45)
    flatten_at: time = time(15, 55)
    tz: str = "America/New_York"


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


@dataclass
class _PendingBias:
    direction: Literal["long", "short"]
    swept_level: float
    created_at: object
    bars_elapsed: int = 0


class Failed2sStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        stop_buffer_ticks: int = 2,
        target_r: float = 1.0,
        min_mss_body_pct: float = 0.5,
        require_fvg: bool = True,
        max_bias_age_bars: int = 3,
        swing_strength: int = 2,
        session: Optional[SessionConfig] = None,
    ):
        self.tick_size = tick_size
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.target_r = target_r
        self.min_mss_body_pct = min_mss_body_pct
        self.require_fvg = require_fvg
        self.max_bias_age_bars = max_bias_age_bars
        self.swing_strength = swing_strength
        self.session = session or SessionConfig()

        self._bias_history: Deque[Bar] = deque(maxlen=2)
        self._entry_history: Deque[Bar] = deque(maxlen=3)
        self._swings = SwingTracker(strength=swing_strength)
        self._pending_bias: Optional[_PendingBias] = None
        self._current_date = None

    def reset(self) -> None:
        """Clear all state -- called automatically on a new session day."""
        self._bias_history.clear()
        self._entry_history.clear()
        self._swings = SwingTracker(strength=self.swing_strength)
        self._pending_bias = None

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

        if self._pending_bias is not None:
            self._pending_bias.bars_elapsed += 1
            if self._pending_bias.bars_elapsed > self.max_bias_age_bars:
                self._pending_bias = None

        self._bias_history.append(bar)
        if len(self._bias_history) == 2:
            prev, curr = self._bias_history
            f2 = detect_failed_2(prev, curr)
            if f2 == "F2D":
                self._pending_bias = _PendingBias("long", curr.low, curr.timestamp)
            elif f2 == "F2U":
                self._pending_bias = _PendingBias("short", curr.high, curr.timestamp)

    def on_entry_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)

        self._entry_history.append(bar)
        self._swings.update(bar)

        if self._pending_bias is None or not self._in_entry_window(bar.timestamp):
            return None

        direction = self._pending_bias.direction
        swing = self._swings.last_swing_high if direction == "long" else self._swings.last_swing_low
        if not detect_mss(bar, swing, direction, self.min_mss_body_pct):
            return None

        if self.require_fvg:
            if len(self._entry_history) < 3:
                return None
            b1, b2, b3 = self._entry_history
            fvg = detect_fvg(b1, b2, b3)
            expected = "bullish" if direction == "long" else "bearish"
            if fvg != expected:
                return None

        entry_price = bar.close
        if direction == "long":
            stop_price = min(swing.price, self._pending_bias.swept_level) - self.stop_buffer
            risk = entry_price - stop_price
            target_price = entry_price + risk * self.target_r
        else:
            stop_price = max(swing.price, self._pending_bias.swept_level) + self.stop_buffer
            risk = stop_price - entry_price
            target_price = entry_price - risk * self.target_r

        if risk <= 0:
            return None

        signal = Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=(
                f"bias={direction} swept_level={self._pending_bias.swept_level} "
                f"swing={swing.price} mss+{'fvg' if self.require_fvg else 'no-fvg'}"
            ),
        )
        self._pending_bias = None  # consume the bias once it fires
        return signal
