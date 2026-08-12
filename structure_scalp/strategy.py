"""
Structure pullback + anchored-VWAP-retest scalping strategy.

Two tiers, reusing the swing/MSS primitives from failed2s/structure.py:

  - 1-minute structure direction: a persisting state (StructureTracker) that
    flips "long" on a bullish Market Structure Shift (a strong-bodied close
    through the last confirmed swing high) and flips "short" on the bearish
    mirror. Stays that way until reversed. This is the trade bias.

  - 5-second sequence, only while the 1m bias is active (long case described,
    short is the exact mirror):
      1. Wait for the 5s's own structure to flip bearish (a pullback against
         the 1m uptrend). Track the lowest low reached while it stays bearish.
      2. Wait for the 5s to flip back bullish (an MSS through a swing high) --
         this is "the bullish leg." Its start is the lowest-low bar just
         tracked, not the breakout bar itself.
      3. Anchor a VWAP to that leg's start (retroactively summing every bar
         from the low through the breakout bar, then accumulating forward).
      4. Entry fires the first time price trades back down and touches that
         anchored VWAP. Stop = the leg's low. Target = target_r * risk.
      5. Invalidated if price makes a new low below the leg's low before ever
         touching the VWAP (the level that would become the stop is already
         broken), or if the 1m bias flips away, or if the 5s flips bearish
         again first (a fresh pullback supersedes the pending one).

This supersedes an earlier version of this strategy that entered immediately
on every 5s MSS while 15m+1m aligned -- that traded far more often but with
much lower quality (5s "structure" alone is mostly noise; this version adds
the pullback-into-reversal sequencing and a VWAP mean-reversion filter on
top of the structural trigger, at the cost of lower frequency than that
first version -- still meant to be well above failed2s.strategy's rate).

Data note: backtesting the 5-second entry timeframe needs real 5-second (or
finer) historical bars, which no free source available to this repo
provides (Yahoo Finance's finest granularity is 1-minute).
"""

from dataclasses import dataclass, field
from typing import List, Literal, Optional

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


def _typical_price(bar: Bar) -> float:
    return (bar.high + bar.low + bar.close) / 3.0


@dataclass
class _PendingLeg:
    direction: Literal["long", "short"]
    extreme: float  # the leg low (long) or leg high (short) -- becomes the stop
    sum_pv: float
    sum_v: float

    def avwap(self) -> Optional[float]:
        return self.sum_pv / self.sum_v if self.sum_v else None

    def accumulate(self, bar: Bar) -> None:
        self.sum_pv += _typical_price(bar) * bar.volume
        self.sum_v += bar.volume


class StructureScalpStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        target_r: float = 1.0,
        min_mss_body_pct: float = 0.5,
        swing_strength_1m: int = 2,
        swing_strength_5s: int = 2,
        session: Optional[SessionConfig] = None,
    ):
        self.tick_size = tick_size
        self.target_r = target_r
        self.min_mss_body_pct = min_mss_body_pct
        self.session = session or SessionConfig()

        self.struct_1m = StructureTracker(swing_strength_1m, min_mss_body_pct)
        self.entry_5s = StructureTracker(swing_strength_5s, min_mss_body_pct)
        self._entry_swing_strength = swing_strength_5s

        self._pullback_bars: List[Bar] = []
        self._pending: Optional[_PendingLeg] = None
        self._current_date = None

    def reset(self) -> None:
        """Clear all state -- called automatically on a new session day."""
        self.struct_1m.reset()
        self.entry_5s = StructureTracker(self._entry_swing_strength, self.min_mss_body_pct)
        self._pullback_bars = []
        self._pending = None

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

    def on_1m_bar(self, bar: Bar) -> None:
        self._roll_session(bar.timestamp)
        self.struct_1m.update(bar)

    def on_5s_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)
        prev_direction = self.entry_5s.direction
        self.entry_5s.update(bar)
        new_direction = self.entry_5s.direction

        bias = self.struct_1m.direction
        if bias is None:
            self._pending = None
            self._pullback_bars = []
            return None

        if self._pending is not None and self._pending.direction != bias:
            self._pending = None  # stale: bias moved on since this leg was armed

        pullback_dir = "short" if bias == "long" else "long"

        # 1. Track the pullback (against the 1m bias) while it's in progress.
        if new_direction == pullback_dir:
            if prev_direction != pullback_dir:
                self._pullback_bars = [bar]
                self._pending = None  # a fresh pullback supersedes any pending leg
            else:
                self._pullback_bars.append(bar)
            return None

        # 2. The pullback just resolved back in the bias direction -- arm a new leg.
        if new_direction == bias and prev_direction == pullback_dir and self._pullback_bars:
            if bias == "long":
                extreme_bar = min(self._pullback_bars, key=lambda b: b.low)
                extreme = extreme_bar.low
            else:
                extreme_bar = max(self._pullback_bars, key=lambda b: b.high)
                extreme = extreme_bar.high

            idx = self._pullback_bars.index(extreme_bar)
            vwap_bars = self._pullback_bars[idx:] + [bar]
            sum_pv = sum(_typical_price(b) * b.volume for b in vwap_bars)
            sum_v = sum(b.volume for b in vwap_bars)

            self._pending = _PendingLeg(direction=bias, extreme=extreme, sum_pv=sum_pv, sum_v=sum_v)
            self._pullback_bars = []
            return None

        # 3. Armed and waiting: update the anchored VWAP, check invalidation/touch.
        if self._pending is not None and self._pending.direction == bias:
            leg = self._pending
            leg.accumulate(bar)
            avwap = leg.avwap()

            if bias == "long":
                if bar.low < leg.extreme:
                    self._pending = None
                    return None
                if avwap is not None and bar.low <= avwap and self._in_entry_window(bar.timestamp):
                    return self._fire(bar.timestamp, "long", entry_price=avwap, stop_price=leg.extreme)
            else:
                if bar.high > leg.extreme:
                    self._pending = None
                    return None
                if avwap is not None and bar.high >= avwap and self._in_entry_window(bar.timestamp):
                    return self._fire(bar.timestamp, "short", entry_price=avwap, stop_price=leg.extreme)

        return None

    def _fire(self, ts, direction: Literal["long", "short"], entry_price: float, stop_price: float) -> Optional[Signal]:
        self._pending = None  # consumed -- a fresh pullback/leg is required for the next trade
        if direction == "long":
            risk = entry_price - stop_price
            target_price = entry_price + risk * self.target_r
        else:
            risk = stop_price - entry_price
            target_price = entry_price - risk * self.target_r

        if risk <= 0:
            return None

        return Signal(
            timestamp=ts,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=f"1m bias={direction}, 5s leg from {stop_price}, avwap retest at {entry_price}",
        )
