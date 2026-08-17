"""
VWAP overextension mean-reversion scalp.

Trades short-term overextensions on a single lower timeframe (1m or 5m --
unlike failed2s/structure_scalp there's no separate bias timeframe here,
since "distance from fair value" is evaluated on the same chart you trade
off of). Thesis: on a high-beta name (e.g. NQ/MNQ), short bursts that push
price several standard deviations away from the session's volume-weighted
fair value tend to snap back toward it before the session resets.

Session VWAP is the fair-value anchor, and a cumulative volume-weighted
standard deviation band around it turns "distance from VWAP" into a
z-score, so the overextension threshold is defined in standard deviations
rather than raw points -- portable across symbols/instruments and across a
given symbol's own quiet/volatile days. Both VWAP and its band reset at the
start of every session day (see failed2s.strategy.SessionConfig).

State machine (long fade case; short is the exact mirror):
  1. Every bar, update the session VWAP/stdev and compute
     z = (close - vwap) / stdev.
  2. When z drops to/below -entry_z, the market is "extended" below fair
     value -- start tracking an extension episode: the extreme low reached
     and the most negative z reached.
  3. While the episode is active, keep updating the extreme on every new
     low and the extreme z on every new low reading -- deliberately not
     firing yet. Firing the instant the threshold is crossed means fading
     into the move while it's still accelerating (catching the knife).
  4. Fire the entry on the first bar that shows exhaustion: the bar is
     bullish (close > open) AND z has recovered by at least confirm_z off
     the episode's most negative reading. Cheap proxy for "the selling
     pressure that caused the extension has paused" -- never a first-tick
     fill, but it filters out entries into bars still pushing further away
     from fair value.
  5. Stop = the episode's extreme low, minus a tick buffer. Target = entry
     + reversion_target_pct * (current VWAP - entry) -- a (by default full)
     partial reversion back to fair value, not a fixed R multiple, since the
     thesis is literally "reverts to VWAP." Set reversion_target_pct below
     1.0 to book a partial fade -- a full round-trip back to VWAP is a
     lower-frequency outcome than a partial one snapping back.
  6. An episode is abandoned (no signal) if: price fully round-trips back
     through VWAP without ever confirming (z crosses the sign of the
     episode's direction -- the market mean-reverted on its own, nothing
     left to fade), it runs longer than max_extension_bars (stale -- no
     longer "short-term"), its extreme z ever exceeds max_z (a likely trend
     day / real news move, not a range-bound overextension -- don't fade
     it), or a fresh extension in the *opposite* direction starts first (it
     supersedes the stale one, same pattern as structure_scalp's pullback
     handling).

Data note: this needs real intrabar volume to be meaningful (the VWAP/stdev
math is volume-weighted) -- synthetic/zero-volume bars fall back to
equal-weighting each bar, which still exercises the pipeline but isn't a
real edge signal.
"""

import math
from dataclasses import dataclass, field
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig


def _typical_price(bar: Bar) -> float:
    return (bar.high + bar.low + bar.close) / 3.0


class SessionVWAP:
    """Cumulative session VWAP plus a volume-weighted stdev band, reset daily."""

    def __init__(self):
        self.sum_v = 0.0
        self.sum_pv = 0.0
        self.sum_pv2 = 0.0

    def reset(self) -> None:
        self.sum_v = 0.0
        self.sum_pv = 0.0
        self.sum_pv2 = 0.0

    def update(self, bar: Bar) -> None:
        tp = _typical_price(bar)
        vol = bar.volume if bar.volume > 0 else 1.0  # equal-weight fallback for zero-volume bars
        self.sum_v += vol
        self.sum_pv += tp * vol
        self.sum_pv2 += tp * tp * vol

    @property
    def vwap(self) -> Optional[float]:
        return self.sum_pv / self.sum_v if self.sum_v else None

    @property
    def stdev(self) -> Optional[float]:
        if not self.sum_v:
            return None
        vwap = self.sum_pv / self.sum_v
        variance = self.sum_pv2 / self.sum_v - vwap * vwap
        return math.sqrt(variance) if variance > 0 else 0.0


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


@dataclass
class _Extension:
    direction: Literal["long", "short"]  # direction of the FADE trade
    extreme: float  # lowest low (long fade) or highest high (short fade) reached
    extreme_z: float  # most negative (long) / most positive (short) z reached
    bars: int = 1


class OverextensionStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        entry_z: float = 2.0,
        confirm_z: float = 0.5,
        max_z: float = 4.0,
        reversion_target_pct: float = 1.0,
        warmup_bars: int = 20,
        min_stdev_ticks: float = 2.0,
        stop_buffer_ticks: int = 2,
        max_extension_bars: int = 30,
        session: Optional[SessionConfig] = None,
    ):
        self.tick_size = tick_size
        self.entry_z = entry_z
        self.confirm_z = confirm_z
        self.max_z = max_z
        self.reversion_target_pct = reversion_target_pct
        self.warmup_bars = warmup_bars
        self.min_stdev_ticks = min_stdev_ticks
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.max_extension_bars = max_extension_bars
        self.session = session or SessionConfig()

        self.vwap = SessionVWAP()
        self._extension: Optional[_Extension] = None
        self._bar_count = 0
        self._current_date = None

    def reset(self) -> None:
        """Clear all state -- called automatically on a new session day."""
        self.vwap.reset()
        self._extension = None
        self._bar_count = 0

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

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)
        self.vwap.update(bar)
        self._bar_count += 1

        vwap = self.vwap.vwap
        sd = self.vwap.stdev
        if vwap is None or sd is None or self._bar_count < self.warmup_bars:
            return None
        sd = max(sd, self.min_stdev_ticks * self.tick_size)
        z = (bar.close - vwap) / sd

        ext = self._extension

        # A fresh extension (first ever, or the opposite direction) supersedes.
        if z <= -self.entry_z and (ext is None or ext.direction != "long"):
            self._extension = _Extension("long", extreme=bar.low, extreme_z=z)
            return None
        if z >= self.entry_z and (ext is None or ext.direction != "short"):
            self._extension = _Extension("short", extreme=bar.high, extreme_z=z)
            return None

        if ext is None:
            return None

        ext.bars += 1
        if ext.bars > self.max_extension_bars:
            self._extension = None
            return None

        in_window = self._in_entry_window(bar.timestamp)

        if ext.direction == "long":
            ext.extreme = min(ext.extreme, bar.low)
            ext.extreme_z = min(ext.extreme_z, z)
            if abs(ext.extreme_z) > self.max_z:
                self._extension = None  # blow-off move -- don't fade a trend day
                return None
            if z > 0:
                self._extension = None  # already round-tripped past fair value on its own
                return None
            recovered = z - ext.extreme_z
            if in_window and recovered >= self.confirm_z and bar.is_bullish:
                return self._fire(bar, ext, vwap)
        else:
            ext.extreme = max(ext.extreme, bar.high)
            ext.extreme_z = max(ext.extreme_z, z)
            if abs(ext.extreme_z) > self.max_z:
                self._extension = None
                return None
            if z < 0:
                self._extension = None
                return None
            recovered = ext.extreme_z - z
            if in_window and recovered >= self.confirm_z and bar.is_bearish:
                return self._fire(bar, ext, vwap)

        return None

    def _fire(self, bar: Bar, ext: _Extension, vwap: float) -> Optional[Signal]:
        self._extension = None  # consumed -- a fresh episode is required for the next trade
        entry_price = bar.close

        if ext.direction == "long":
            stop_price = ext.extreme - self.stop_buffer
            target_price = entry_price + self.reversion_target_pct * (vwap - entry_price)
            direction = "long"
            degenerate = target_price <= entry_price
            risk = entry_price - stop_price
        else:
            stop_price = ext.extreme + self.stop_buffer
            target_price = entry_price - self.reversion_target_pct * (entry_price - vwap)
            direction = "short"
            degenerate = target_price >= entry_price
            risk = stop_price - entry_price

        if risk <= 0 or degenerate:
            return None

        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=f"z_extreme={ext.extreme_z:.2f}, vwap={vwap:.2f}, reversion_target_pct={self.reversion_target_pct}",
        )
