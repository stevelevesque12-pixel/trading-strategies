"""
Opening Range Breakout (ORB) with an ATR velocity filter.

1. **Opening Range (OR)** — the high/low of the first `or_minutes` of the
   session (default 15: 9:30-9:45 ET). Reset every session day, like the
   VWAP/structure state in this repo's other strategies.
2. **Trailing ATR** — a rolling simple-moving-average of True Range over
   `atr_period` bars (default 14), used as the breakout's "velocity" filter.
   Unlike the OR itself, the ATR window does **not** reset at session
   start: it's a continuously-updating volatility measure carried over from
   the prior session's trailing bars, same as it would behave as an
   indicator on a real chart. Resetting it daily would mean it isn't warmed
   up again until well past the default 10:15 breakout cutoff (14 bars *
   5min = 70 minutes into a session that opens at 9:30) -- carrying it
   across the session boundary is both more realistic and avoids that
   warm-up/cutoff conflict. (This is a simple MA of True Range, not
   Wilder's smoothed ATR -- kept as a simple O(1) rolling calculator,
   consistent with the rest of this repo's indicators.)
3. **Entry triggers** (both are resting-stop-style: they fire the instant a
   bar's range crosses the level, at that exact level, not at the bar's
   close -- unlike failed2s/structure_scalp/overextension's market-on-
   signal-bar-close entries, this strategy's entries are genuinely
   stop-order breakouts):
   - Long: `bar.high >= OR_high + atr_mult * ATR`
   - Short: `bar.low <= OR_low - atr_mult * ATR`
   The `atr_mult * ATR` buffer is the "velocity" filter -- a breakout has
   to clear the range by a volatility-scaled amount, not just tick through
   it, to count as a real move rather than noise. If a single bar's range
   improbably crosses both triggers, long is checked first (documented
   tie-break, mirrors the conservative "stop fills before target" tie-break
   the backtest engines already use elsewhere in this repo).
4. **One trade per day** — once a breakout fires (either side), the other
   side's resting trigger is implicitly cancelled (this strategy only ever
   watches for one entry per session) and no further entries are taken that
   day, win or lose.
5. **Time filter** — no entries once `session.no_entry_after` passes
   (default 10:15 ET) even if the OR breakout never triggered ("cancel all
   open orders"); any open position is force-flattened at
   `session.flatten_at` (default 15:45 ET, "square all positions").
6. **Stop / target** — stop at the opposite side of the opening range (the
   classic ORB placement: a breakout that fully round-trips back through
   the range has invalidated its own thesis), plus an optional tick buffer
   (default 0). Target = `target_r * risk` (default 1:1, tune upward --
   ORB breakouts with real velocity are commonly traded for >1R since the
   range itself is often a tight stop).
"""

from collections import deque
from dataclasses import dataclass
from datetime import time
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig


class _TrueRangeATR:
    """Rolling simple-moving-average of True Range. Does NOT reset per session."""

    def __init__(self, period: int = 14):
        self.period = period
        self._trs: deque = deque(maxlen=period)
        self._sum = 0.0
        self._prev_close: Optional[float] = None

    def update(self, bar: Bar) -> None:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(bar.high - bar.low, abs(bar.high - self._prev_close), abs(bar.low - self._prev_close))
        self._prev_close = bar.close

        if len(self._trs) == self.period:
            self._sum -= self._trs[0]
        self._trs.append(tr)
        self._sum += tr

    @property
    def value(self) -> Optional[float]:
        if len(self._trs) < self.period:
            return None
        return self._sum / self.period


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


class ORBStrategy:
    def __init__(
        self,
        or_minutes: int = 15,
        atr_period: int = 14,
        atr_mult: float = 0.2,
        target_r: float = 1.0,
        stop_buffer_ticks: int = 0,
        tick_size: float = 0.25,
        session: Optional[SessionConfig] = None,
    ):
        self.or_minutes = or_minutes
        self.atr_mult = atr_mult
        self.target_r = target_r
        self.tick_size = tick_size
        self.stop_buffer = stop_buffer_ticks * tick_size
        # Defaults encode the spec's own cutoffs: cancel unfilled breakout
        # orders at 10:15 ET, square all positions by 3:45 PM ET.
        self.session = session or SessionConfig(no_entry_after=time(10, 15), flatten_at=time(15, 45))

        start = self.session.session_start
        total_minutes = start.hour * 60 + start.minute + or_minutes
        self._or_end_time = time(total_minutes // 60, total_minutes % 60)
        if self._or_end_time >= self.session.no_entry_after:
            raise ValueError(
                f"or_minutes={or_minutes} makes the Opening Range end at {self._or_end_time}, "
                f"at/after session.no_entry_after={self.session.no_entry_after} -- there would be no "
                "time window left to ever take a breakout. Shorten or_minutes or push no_entry_after later."
            )

        self._atr = _TrueRangeATR(atr_period)
        self._or_high: Optional[float] = None
        self._or_low: Optional[float] = None
        self._traded_today = False
        self._current_date = None

    def reset(self) -> None:
        """Clear the day's opening-range state -- called automatically on a
        new session day. Deliberately does NOT reset the trailing ATR."""
        self._or_high = None
        self._or_low = None
        self._traded_today = False

    def _roll_session(self, ts) -> None:
        d = ts.date()
        if self._current_date is None:
            self._current_date = d
        elif d != self._current_date:
            self._current_date = d
            self.reset()

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)
        self._atr.update(bar)

        t = bar.timestamp.time()

        if t < self._or_end_time:
            if self._or_high is None:
                self._or_high, self._or_low = bar.high, bar.low
            else:
                self._or_high = max(self._or_high, bar.high)
                self._or_low = min(self._or_low, bar.low)
            return None

        if self._or_high is None:
            return None  # no bars during the OR window this session (data gap) -- can't trade

        if self._traded_today or t >= self.session.no_entry_after:
            return None

        atr = self._atr.value
        if atr is None:
            return None  # not enough trailing history yet to trust the velocity filter

        buy_trigger = self._or_high + self.atr_mult * atr
        sell_trigger = self._or_low - self.atr_mult * atr

        if bar.high >= buy_trigger:
            return self._fire("long", bar, buy_trigger, atr)
        if bar.low <= sell_trigger:
            return self._fire("short", bar, sell_trigger, atr)
        return None

    def _fire(self, direction: str, bar: Bar, entry_price: float, atr: float) -> Optional[Signal]:
        if direction == "long":
            stop_price = self._or_low - self.stop_buffer
            risk = entry_price - stop_price
            target_price = entry_price + risk * self.target_r
        else:
            stop_price = self._or_high + self.stop_buffer
            risk = stop_price - entry_price
            target_price = entry_price - risk * self.target_r

        if risk <= 0:
            return None

        self._traded_today = True  # one breakout attempt per day, win or lose
        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=f"OR=[{self._or_low:.2f},{self._or_high:.2f}] atr={atr:.2f} trigger={entry_price:.2f}",
        )
