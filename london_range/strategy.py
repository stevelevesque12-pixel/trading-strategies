"""
London range breakout -> pullback strategy (5-minute bars, New York time).

Rules (long case; short is the exact mirror):
  1. Mark the range from 20:00 (prior evening) to 02:00 New York time:
     range_high / range_low over every bar inside that window.
  2. After 02:00, wait for a 5-minute candle to CLOSE above range_high.
     That close must be aligned with both EMAs: close > EMA50 and
     close > EMA200 (both computed on 5m closes, carried across sessions).
     Optionally (require_ema_stack) also EMA50 > EMA200.
     No EMA alignment, no entry.
  3. On that breakout, place a limit buy 12.5% of the range back inside it:
     entry = range_high - pullback_pct * (range_high - range_low).
  4. Stop = the opposite range boundary (range_low). Target = entry +
     target_r * risk (default 0.5R).

One setup per day. By default only the FIRST 5m close beyond the range
counts as the breakout -- if it isn't EMA-aligned the day is skipped
(first_breakout_only=False instead lets a later, aligned close beyond either
side arm the order). The pending order is cancelled at no_entry_after; an
open position is flattened at flatten_at. The source setup doesn't say when
the day's trading window ends, so those two times are assumptions.
"""

from dataclasses import dataclass
from datetime import time, timedelta
from typing import Literal, Optional

from failed2s.bars import Bar


@dataclass
class LondonRangeConfig:
    range_start: time = time(20, 0)
    range_end: time = time(2, 0)
    no_entry_after: time = time(12, 0)
    flatten_at: time = time(15, 55)
    pullback_pct: float = 0.125
    target_r: float = 0.5
    ema_fast: int = 50
    ema_slow: int = 200
    require_ema_stack: bool = False
    first_breakout_only: bool = True


@dataclass
class Order:
    """A resting limit order armed by a breakout close."""

    created_at: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    range_high: float
    range_low: float
    reason: str


class _Ema:
    def __init__(self, length: int):
        self.alpha = 2.0 / (length + 1)
        self.length = length
        self.value: Optional[float] = None
        self.count = 0

    def update(self, x: float) -> None:
        self.value = x if self.value is None else self.value + self.alpha * (x - self.value)
        self.count += 1

    @property
    def ready(self) -> bool:
        return self.count >= self.length


def session_date(ts, range_start: time = time(20, 0)):
    """The trading day a bar belongs to: the range opens the evening before."""
    shift = timedelta(hours=24 - range_start.hour, minutes=-range_start.minute)
    return (ts + shift).date()


class LondonRangeStrategy:
    def __init__(self, config: Optional[LondonRangeConfig] = None):
        self.cfg = config or LondonRangeConfig()
        self.ema_fast = _Ema(self.cfg.ema_fast)
        self.ema_slow = _Ema(self.cfg.ema_slow)
        self._day = None
        self._reset_day()

    def _reset_day(self) -> None:
        self.range_high: Optional[float] = None
        self.range_low: Optional[float] = None
        self._range_complete = False
        self._done = False  # setup consumed (order placed or day skipped)

    def _in_range_window(self, t: time) -> bool:
        return t >= self.cfg.range_start or t < self.cfg.range_end

    def on_5m_bar(self, bar: Bar) -> Optional[Order]:
        """Feed one closed 5m bar (timestamp = bar open). Returns an Order at most once per day."""
        day = session_date(bar.timestamp, self.cfg.range_start)
        if day != self._day:
            self._day = day
            self._reset_day()

        self.ema_fast.update(bar.close)
        self.ema_slow.update(bar.close)

        t = bar.timestamp.time()
        if self._in_range_window(t):
            self.range_high = bar.high if self.range_high is None else max(self.range_high, bar.high)
            self.range_low = bar.low if self.range_low is None else min(self.range_low, bar.low)
            return None

        self._range_complete = self.range_high is not None
        if self._done or not self._range_complete or t >= self.cfg.no_entry_after:
            return None
        if not (self.ema_fast.ready and self.ema_slow.ready):
            return None

        if bar.close > self.range_high:
            direction = "long"
        elif bar.close < self.range_low:
            direction = "short"
        else:
            return None

        fast, slow = self.ema_fast.value, self.ema_slow.value
        if direction == "long":
            aligned = bar.close > fast and bar.close > slow and (not self.cfg.require_ema_stack or fast > slow)
        else:
            aligned = bar.close < fast and bar.close < slow and (not self.cfg.require_ema_stack or fast < slow)

        if not aligned:
            if self.cfg.first_breakout_only:
                self._done = True
            return None

        self._done = True
        return self._build_order(bar, direction)

    def _build_order(self, bar: Bar, direction: Literal["long", "short"]) -> Optional[Order]:
        hi, lo = self.range_high, self.range_low
        width = hi - lo
        if width <= 0:
            return None
        if direction == "long":
            entry = hi - self.cfg.pullback_pct * width
            stop = lo
            target = entry + self.cfg.target_r * (entry - stop)
        else:
            entry = lo + self.cfg.pullback_pct * width
            stop = hi
            target = entry - self.cfg.target_r * (stop - entry)
        return Order(
            created_at=bar.timestamp,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            range_high=hi,
            range_low=lo,
            reason=f"5m close {bar.close} beyond range [{lo}, {hi}], EMA-aligned",
        )
