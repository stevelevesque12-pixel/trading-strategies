"""
NQ Asian-range liquidity-sweep strategy -- mechanical, prop-firm (Tradeify)
oriented. Single bar stream (15-minute bars recommended), driven entirely by
time-of-day session windows rather than a second higher timeframe.

Cascade:
  1. **Box**: map the Asian box high/low from bars between `box_start` and
     `box_end` (20:00-00:00 ET by default).
  2. **Sweep/break**: during the London/pre-market window (`box_end` to
     `sweep_end`, 00:00-09:30 ET by default), watch for a bar to *close*
     beyond the box high or low -- a break, not just a wick -- which sets a
     directional bias and remembers the broken boundary (the swept level).
  3. **Confirmation**: don't trade the break itself. Wait for a Market
     Structure Shift (MSS) in the break direction on the same bar stream.
     This reads the source rule ("MSS or FVG confirms") as MSS-required,
     with an accompanying 3-bar Fair Value Gap (FVG) -- if the same window
     that produced the MSS also qualifies as one -- used only to sharpen the
     retest level, since in practice the displacement leg that causes an MSS
     is usually the same leg that leaves the gap. Set `require_fvg=True` for
     a stricter mode that refuses to arm without one (mirrors failed2s).
  4. **Retest entry**: arms a pending retest at the FVG's near edge (the
     edge closest to price when it formed) if one qualified, else at the
     broken box boundary itself. Entry fires the first time a later bar's
     range touches that level. Invalidated if price closes back across the
     broken boundary before that touch ever happens -- the premise
     (continuation past the sweep) is already wrong.
  5. **Stop**: `stop_mode="structural"` (default) places it beyond the
     further of the MSS's confirming swing point or the swept boundary,
     plus a tick buffer -- tracks price closely. `stop_mode="box_extreme"`
     instead anchors it beyond the *opposite* side of the whole Asian box
     (wider, but never invalidated by intraday noise inside the range).
  6. **Target**: a fixed R multiple (default 2R). If `target_mode`
     is "liquidity", uses the previous trading day's high/low instead,
     whenever that's at least as far out as the fixed-R target would be
     (falls back to fixed R otherwise). "Previous day" here is the full
     prior trading day's range including its own overnight box, tracked
     from the same bar stream -- not RTH-only.
  7. **Intraday only**: all state (box, bias, structure) resets every
     trading day. No new retests armed after `no_entry_after`; an open
     position is force-flattened at `flatten_at` by the backtest engine /
     live runner (both Tradeify-style same-day-flat discipline).

Reuses the swing/MSS/FVG primitives from failed2s/structure.py.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional

from failed2s.bars import Bar
from failed2s.structure import SwingTracker, detect_fvg, detect_mss

from .session import AsianBox, SessionConfig, trading_date


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


@dataclass
class _PendingBreak:
    direction: Literal["long", "short"]
    swept_level: float
    bars_elapsed: int = 0


@dataclass
class _PendingRetest:
    direction: Literal["long", "short"]
    retest_level: float
    stop_price: float
    bars_elapsed: int = 0


class AsianSweepStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        stop_buffer_ticks: int = 4,
        target_r: float = 2.0,
        target_mode: Literal["fixed_r", "liquidity"] = "fixed_r",
        min_mss_body_pct: float = 0.5,
        require_fvg: bool = False,
        stop_mode: Literal["structural", "box_extreme"] = "structural",
        swing_strength: int = 2,
        max_break_age_bars: int = 12,
        max_retest_age_bars: int = 12,
        session: Optional[SessionConfig] = None,
    ):
        self.tick_size = tick_size
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.target_r = target_r
        self.target_mode = target_mode
        self.min_mss_body_pct = min_mss_body_pct
        self.require_fvg = require_fvg
        self.stop_mode = stop_mode
        self.swing_strength = swing_strength
        self.max_break_age_bars = max_break_age_bars
        self.max_retest_age_bars = max_retest_age_bars
        self.session = session or SessionConfig()

        self._swings = SwingTracker(strength=swing_strength)
        self._bar_history: Deque[Bar] = deque(maxlen=3)
        self._pending_break: Optional[_PendingBreak] = None
        self._pending_retest: Optional[_PendingRetest] = None

        self._box: Optional[AsianBox] = None
        self._box_high: Optional[float] = None
        self._box_low: Optional[float] = None

        self._day_high: Optional[float] = None
        self._day_low: Optional[float] = None
        self._prev_day_high: Optional[float] = None
        self._prev_day_low: Optional[float] = None

        self._current_trading_date = None

    def reset(self) -> None:
        """Clear all intraday state -- called automatically on a new trading day."""
        self._swings = SwingTracker(strength=self.swing_strength)
        self._bar_history.clear()
        self._pending_break = None
        self._pending_retest = None
        self._box = None
        self._box_high = None
        self._box_low = None
        self._day_high = None
        self._day_low = None

    def _roll_session(self, ts) -> None:
        d = trading_date(ts, self.session)
        if self._current_trading_date is None:
            self._current_trading_date = d
        elif d != self._current_trading_date:
            self._prev_day_high = self._day_high
            self._prev_day_low = self._day_low
            self._current_trading_date = d
            self.reset()

    def _in_box_window(self, ts) -> bool:
        return ts.time() >= self.session.box_start

    def _in_sweep_window(self, ts) -> bool:
        t = ts.time()
        return self.session.box_end <= t < self.session.sweep_end

    def _in_entry_window(self, ts) -> bool:
        t = ts.time()
        return self.session.box_end <= t < self.session.no_entry_after

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)

        self._day_high = bar.high if self._day_high is None else max(self._day_high, bar.high)
        self._day_low = bar.low if self._day_low is None else min(self._day_low, bar.low)

        if self._in_box_window(bar.timestamp):
            self._box_high = bar.high if self._box_high is None else max(self._box_high, bar.high)
            self._box_low = bar.low if self._box_low is None else min(self._box_low, bar.low)
            return None

        if self._box is None:
            if self._box_high is None:
                return None  # no overnight data seen this trading day -- nothing to trade off
            self._box = AsianBox(self._current_trading_date, self._box_high, self._box_low)

        self._swings.update(bar)
        self._bar_history.append(bar)

        if self._pending_break is not None:
            self._pending_break.bars_elapsed += 1
            if self._pending_break.bars_elapsed > self.max_break_age_bars:
                self._pending_break = None

        if self._pending_retest is not None:
            self._pending_retest.bars_elapsed += 1
            if self._pending_retest.bars_elapsed > self.max_retest_age_bars:
                self._pending_retest = None

        # 1. Sweep window: watch for a closing break of the box extreme.
        if (
            self._pending_break is None
            and self._pending_retest is None
            and self._in_sweep_window(bar.timestamp)
        ):
            if bar.close > self._box.high:
                self._pending_break = _PendingBreak("long", self._box.high)
            elif bar.close < self._box.low:
                self._pending_break = _PendingBreak("short", self._box.low)
            return None

        # 2. Confirmation: an MSS in the break direction arms the retest.
        if self._pending_break is not None:
            direction = self._pending_break.direction
            swing = self._swings.last_swing_high if direction == "long" else self._swings.last_swing_low
            if detect_mss(bar, swing, direction, self.min_mss_body_pct):
                fvg = None
                if len(self._bar_history) == 3:
                    b1, b2, b3 = self._bar_history
                    fvg = detect_fvg(b1, b2, b3)
                expected_fvg = "bullish" if direction == "long" else "bearish"

                if self.require_fvg and fvg != expected_fvg:
                    return None  # MSS without a qualifying FVG yet -- keep waiting

                if fvg == expected_fvg:
                    b1, b2, b3 = self._bar_history
                    retest_level = b3.low if direction == "long" else b3.high
                else:
                    retest_level = self._pending_break.swept_level

                if self.stop_mode == "box_extreme":
                    stop_price = (
                        self._box.low - self.stop_buffer
                        if direction == "long"
                        else self._box.high + self.stop_buffer
                    )
                elif direction == "long":
                    stop_price = min(swing.price, self._pending_break.swept_level) - self.stop_buffer
                else:
                    stop_price = max(swing.price, self._pending_break.swept_level) + self.stop_buffer

                self._pending_retest = _PendingRetest(
                    direction=direction, retest_level=retest_level, stop_price=stop_price
                )
                self._pending_break = None
            return None

        # 3. Armed: check retest touch (entry) or boundary give-back (invalidation).
        if self._pending_retest is not None:
            pr = self._pending_retest
            touched = pr.retest_level <= bar.high and pr.retest_level >= bar.low

            if touched and self._in_entry_window(bar.timestamp):
                return self._fire(bar.timestamp, pr)

            gave_back = (
                (pr.direction == "long" and bar.close < self._box.high)
                or (pr.direction == "short" and bar.close > self._box.low)
            )
            if gave_back:
                self._pending_retest = None

        return None

    def _fire(self, ts, pr: "_PendingRetest") -> Optional[Signal]:
        self._pending_retest = None
        entry_price = pr.retest_level
        direction = pr.direction
        stop_price = pr.stop_price

        if direction == "long":
            risk = entry_price - stop_price
            target_price = entry_price + risk * self.target_r
            if self.target_mode == "liquidity" and self._prev_day_high is not None:
                if self._prev_day_high - entry_price >= risk * self.target_r:
                    target_price = self._prev_day_high
        else:
            risk = stop_price - entry_price
            target_price = entry_price - risk * self.target_r
            if self.target_mode == "liquidity" and self._prev_day_low is not None:
                if entry_price - self._prev_day_low >= risk * self.target_r:
                    target_price = self._prev_day_low

        if risk <= 0:
            return None

        return Signal(
            timestamp=ts,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=(
                f"box={self._box.low}-{self._box.high} break={direction} "
                f"retest={entry_price} stop={stop_price}"
            ),
        )
