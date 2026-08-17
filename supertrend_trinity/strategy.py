"""
Trinity Trend: three independently-configured SuperTrend indicators
(typically the chart's own timeframe plus two higher timeframes, e.g.
1h/4h), trading only when they align. Ported from a user-supplied Pine v5
script ("Trinity Trend Triple ATR SuperTrend Pro Strategy").

**This is a swing/trend-following system, not a scalp** -- unlike every
other strategy in this repo, it is NOT force-flattened at end of day by
default. Confirming entries against 1h/4h trend direction only makes sense
if a position can actually ride a multi-day trend; flattening daily would
defeat the entire premise. An optional `session` (SessionConfig) still
lets you force an intraday flatten if your account requires day-trading
only -- pass one and flattening is handled the same way as the rest of
this repo; leave it `None` (the default) for genuine swing behavior.

1. **SuperTrend indicator** (`SuperTrendTracker`) -- the standard
   algorithm: `up = hl2 - mult*ATR`, `dn = hl2 + mult*ATR`, each ratcheted
   in the trend's favor using the *prior* bar's close vs. the *prior* bar's
   final band (never against this bar's own high/low, so it can't peek at
   its own outcome), direction flips when price closes through the
   opposite band. Uses Wilder-smoothed ATR (`ta.atr()`'s default in Pine),
   deliberately different from the simple-moving-average True Range used
   by the ORB/ATR-breakout strategies elsewhere in this repo -- SuperTrend
   is conventionally defined with Wilder smoothing, and this port matches
   that rather than reusing the simpler calculator for consistency's sake.
2. **Three tiers** -- ST1 (the entry timeframe -- whatever bars you feed
   `on_entry_bar`), ST2 and ST3 (higher timeframes, fed independently via
   `on_tf2_bar`/`on_tf3_bar`; only their current *direction* is needed, not
   their own flip events -- alignment, not timing, is what the entry modes
   check on ST2/ST3).
3. **Entry modes** (`entry_mode`: `"single"`, `"double"`, or `"triple"`,
   matching the Pine version's mutually-exclusive enable flags):
   - `single`: ST1 flips (either direction) -> enter immediately.
   - `double`: ST1 flips AND ST2's current direction agrees.
   - `triple`: ST1 flips AND ST2 AND ST3 both currently agree (the
     default entry mode in the original script, and the default here).
4. **Exit**:
   - Stop -- `entry_price -/+ sl_mult * ATR_risk` (fixed), where
     `ATR_risk` is ST2's ATR if `use_htf_atr` (default True) else ST1's.
     If `trailing=True` (default False, matching the original), the stop
     instead ratchets every bar toward price using its own independent
     ATR/percent trail (`trail_atr_period`/`trail_atr_mult`, or
     `trail_pct` if `trail_source="percent"`) -- monotonic, only ever
     tightens, never loosens.
   - Target -- always fixed: `entry_price +/- tp_mult * ATR_risk`. The
     original script never trails the target, only the stop.
5. **No lookahead**: like the rest of this repo, a bar's own high/low
   never influences the SuperTrend value computed for that same bar's
   entry decision -- the ratchet and direction-flip both reference the
   *prior* bar's already-finalized band values.
"""

from dataclasses import dataclass
from typing import Literal, Optional, Tuple

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig


class _WilderATR:
    """Wilder-smoothed ATR (ta.atr()'s default in Pine) -- NOT a simple
    moving average of True Range (see module docstring for why this
    differs from the ORB/ATR-breakout strategies' calculator). Does not
    reset per session."""

    def __init__(self, period: int):
        self.period = period
        self._prev_close: Optional[float] = None
        self._seed_sum = 0.0
        self._seed_count = 0
        self._value: Optional[float] = None

    def update(self, bar: Bar) -> None:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(bar.high - bar.low, abs(bar.high - self._prev_close), abs(bar.low - self._prev_close))
        self._prev_close = bar.close

        if self._value is None:
            self._seed_sum += tr
            self._seed_count += 1
            if self._seed_count == self.period:
                self._value = self._seed_sum / self.period
        else:
            self._value = (self._value * (self.period - 1) + tr) / self.period

    @property
    def value(self) -> Optional[float]:
        return self._value


class SuperTrendTracker:
    """Incremental SuperTrend, matching the Pine `f_supertrend` function
    bar-for-bar (ratchet + direction flip both reference the PRIOR bar's
    finalized up/dn/close -- see module docstring)."""

    def __init__(self, atr_period: int, mult: float):
        self.mult = mult
        self._atr = _WilderATR(atr_period)
        self._prev_up: Optional[float] = None
        self._prev_dn: Optional[float] = None
        self._prev_close: Optional[float] = None
        self.direction: int = 1  # 1 = bearish/down, -1 = bullish/up (matches the Pine convention)

    def update(self, bar: Bar) -> Tuple[Optional[float], int]:
        """Returns (supertrend_value, direction). Value is None until the ATR warms up."""
        self._atr.update(bar)
        atr = self._atr.value
        if atr is None:
            return None, self.direction

        hl2 = (bar.high + bar.low) / 2.0
        raw_up = hl2 - self.mult * atr
        raw_dn = hl2 + self.mult * atr

        if self._prev_up is not None and self._prev_close is not None and self._prev_close > self._prev_up:
            up = max(raw_up, self._prev_up)
        else:
            up = raw_up

        if self._prev_dn is not None and self._prev_close is not None and self._prev_close < self._prev_dn:
            dn = min(raw_dn, self._prev_dn)
        else:
            dn = raw_dn

        if self._prev_dn is not None and bar.close > self._prev_dn:
            self.direction = -1
        elif self._prev_up is not None and bar.close < self._prev_up:
            self.direction = 1
        # else: direction persists unchanged

        st = up if self.direction == -1 else dn

        self._prev_up = up
        self._prev_dn = dn
        self._prev_close = bar.close

        return st, self.direction


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


class TrinitySuperTrendStrategy:
    def __init__(
        self,
        atr_period1: int = 10,
        mult1: float = 1.0,
        atr_period2: int = 10,
        mult2: float = 1.0,
        atr_period3: int = 10,
        mult3: float = 1.0,
        entry_mode: Literal["single", "double", "triple"] = "triple",
        sl_mult: float = 1.0,
        tp_mult: float = 1.0,
        use_htf_atr: bool = True,
        trailing: bool = False,
        trail_source: Literal["atr", "percent"] = "atr",
        trail_atr_period: int = 14,
        trail_atr_mult: float = 2.0,
        trail_pct: float = 1.5,
        session: Optional[SessionConfig] = None,
    ):
        if entry_mode not in ("single", "double", "triple"):
            raise ValueError(f"entry_mode must be 'single', 'double', or 'triple', got {entry_mode!r}")

        self.entry_mode = entry_mode
        self.sl_mult = sl_mult
        self.tp_mult = tp_mult
        self.use_htf_atr = use_htf_atr
        self.trailing = trailing
        self.trail_source = trail_source
        self.trail_atr_mult = trail_atr_mult
        self.trail_pct = trail_pct
        self.session = session  # None (default) => no forced flatten; see module docstring

        self._st1 = SuperTrendTracker(atr_period1, mult1)
        self._st2 = SuperTrendTracker(atr_period2, mult2)
        self._st3 = SuperTrendTracker(atr_period3, mult3)
        self._trail_atr = _WilderATR(trail_atr_period)
        self._prev_dir1: Optional[int] = None
        self._atr2_value: Optional[float] = None  # last known ST2 ATR, for use_htf_atr risk sizing

    def on_tf2_bar(self, bar: Bar) -> None:
        """Higher-timeframe (ST2) bar -- updates direction/ATR only, never fires a signal."""
        self._st2.update(bar)
        self._atr2_value = self._st2._atr.value

    def on_tf3_bar(self, bar: Bar) -> None:
        """Higher-timeframe (ST3) bar -- updates direction only, never fires a signal."""
        self._st3.update(bar)

    def on_entry_bar(self, bar: Bar) -> Optional[Signal]:
        """Entry-timeframe (ST1) bar. Feed ST2/ST3 bars that have already
        closed by this point BEFORE calling this, same convention as the
        rest of this repo's multi-timeframe engines (no lookahead)."""
        prev_dir1 = self._prev_dir1
        st1_val, dir1 = self._st1.update(bar)
        self._prev_dir1 = dir1
        if st1_val is None:
            return None

        flipped_bull = prev_dir1 is not None and dir1 == -1 and prev_dir1 == 1
        flipped_bear = prev_dir1 is not None and dir1 == 1 and prev_dir1 == -1
        if not flipped_bull and not flipped_bear:
            return None

        if self.entry_mode == "double" and self._st2._atr.value is None:
            return None
        if self.entry_mode == "triple" and (self._st2._atr.value is None or self._st3._atr.value is None):
            return None

        direction: Optional[Literal["long", "short"]] = None
        if flipped_bull:
            aligned = (
                self.entry_mode == "single"
                or (self.entry_mode == "double" and self._st2.direction == -1)
                or (self.entry_mode == "triple" and self._st2.direction == -1 and self._st3.direction == -1)
            )
            if aligned:
                direction = "long"
        else:
            aligned = (
                self.entry_mode == "single"
                or (self.entry_mode == "double" and self._st2.direction == 1)
                or (self.entry_mode == "triple" and self._st2.direction == 1 and self._st3.direction == 1)
            )
            if aligned:
                direction = "short"

        if direction is None:
            return None

        atr_risk = self._atr2_value if self.use_htf_atr else self._st1._atr.value
        if atr_risk is None:
            return None

        entry_price = bar.close
        if direction == "long":
            stop_price = entry_price - self.sl_mult * atr_risk
            target_price = entry_price + self.tp_mult * atr_risk
            if stop_price >= entry_price or target_price <= entry_price:
                return None
        else:
            stop_price = entry_price + self.sl_mult * atr_risk
            target_price = entry_price - self.tp_mult * atr_risk
            if stop_price <= entry_price or target_price >= entry_price:
                return None

        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=f"{self.entry_mode} mode, ST1 flipped {'bull' if flipped_bull else 'bear'}, atr_risk={atr_risk:.4f}",
        )

    def trail_stop(self, bar: Bar, direction: str, current_stop: float) -> float:
        """Called by the engine every bar a position is open, only if
        `self.trailing` is True. Returns the new (never looser) stop."""
        self._trail_atr.update(bar)
        if self.trail_source == "atr":
            atr = self._trail_atr.value
            offset = atr * self.trail_atr_mult if atr is not None else None
        else:
            offset = bar.close * self.trail_pct / 100.0

        if offset is None:
            return current_stop

        if direction == "long":
            candidate = bar.close - offset
            return max(current_stop, candidate)
        else:
            candidate = bar.close + offset
            return min(current_stop, candidate)
