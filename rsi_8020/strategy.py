"""
RSI 80-20 mean-reversion strategy.

A single-timeframe RSI reversal system using the wider 80/20 extremes
(instead of the textbook 70/30) so only genuinely stretched moves qualify.
Long case described; short is the exact mirror:

  1. RSI drops below `oversold` (default 20) -- an oversold excursion begins.
     Track the lowest low reached while RSI stays below the level.
  2. Entry fires when RSI crosses back up through `oversold` (the close that
     takes RSI from < 20 to >= 20). Waiting for the cross back out, rather
     than buying the moment RSI enters the zone, avoids catching the knife
     while the move is still accelerating.
  3. Stop = the excursion's lowest low minus `stop_buffer_ticks`.
  4. Target = entry + `target_r` * risk.

Short: RSI rises above `overbought` (default 80), stop above the excursion's
highest high, entry on the cross back down through 80.

RSI is Wilder's (the same RMA smoothing as TradingView's `ta.rsi`, seeded
with a simple average of the first `rsi_length` changes), so values line up
with the Pine port in tradingview/rsi_8020.pine once both are warmed up.

RSI itself runs continuously across sessions (futures trade overnight and a
14-bar daily re-warmup would waste the open), but the pending excursion is
cleared on each new session day so a stop is never anchored to a prior
day's extreme. Signals only fire inside the session entry window; the
backtest engine / Pine script handle the end-of-day flatten.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig, Signal


class WilderRSI:
    """Incremental Wilder RSI over closes. `value` is None until warmed up."""

    def __init__(self, length: int = 14):
        if length < 1:
            raise ValueError("RSI length must be >= 1")
        self.length = length
        self._prev_close: Optional[float] = None
        self._seed_gains = 0.0
        self._seed_losses = 0.0
        self._seed_count = 0
        self._avg_gain: Optional[float] = None
        self._avg_loss: Optional[float] = None
        self.value: Optional[float] = None

    def update(self, close: float) -> Optional[float]:
        if self._prev_close is None:
            self._prev_close = close
            return None

        change = close - self._prev_close
        self._prev_close = close
        gain = max(change, 0.0)
        loss = max(-change, 0.0)

        if self._avg_gain is None:
            self._seed_gains += gain
            self._seed_losses += loss
            self._seed_count += 1
            if self._seed_count < self.length:
                return None
            self._avg_gain = self._seed_gains / self.length
            self._avg_loss = self._seed_losses / self.length
        else:
            n = self.length
            self._avg_gain = (self._avg_gain * (n - 1) + gain) / n
            self._avg_loss = (self._avg_loss * (n - 1) + loss) / n

        if self._avg_loss == 0:
            self.value = 100.0
        elif self._avg_gain == 0:
            self.value = 0.0
        else:
            rs = self._avg_gain / self._avg_loss
            self.value = 100.0 - 100.0 / (1.0 + rs)
        return self.value


@dataclass
class _Excursion:
    direction: Literal["long", "short"]  # the trade direction this excursion sets up
    extreme: float  # lowest low (long) / highest high (short) while RSI was beyond the level


class RSI8020Strategy:
    """
    Exposes the same `on_bias_bar` / `on_entry_bar` interface as
    Failed2sStrategy so it runs unmodified in backtest.engine.BacktestEngine
    (use a TimeframePair whose bias_tf == entry_tf; bias bars are ignored).
    """

    def __init__(
        self,
        tick_size: float = 0.25,
        rsi_length: int = 14,
        overbought: float = 80.0,
        oversold: float = 20.0,
        stop_buffer_ticks: int = 2,
        target_r: float = 1.0,
        session: Optional[SessionConfig] = None,
    ):
        if not 0 < oversold < overbought < 100:
            raise ValueError("need 0 < oversold < overbought < 100")
        self.tick_size = tick_size
        self.rsi_length = rsi_length
        self.overbought = overbought
        self.oversold = oversold
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.target_r = target_r
        self.session = session or SessionConfig()

        self.rsi = WilderRSI(rsi_length)
        self._prev_rsi: Optional[float] = None
        self._excursion: Optional[_Excursion] = None
        self._current_date = None

    def reset(self) -> None:
        """Clear the pending setup -- called automatically on a new session day.

        RSI state is deliberately kept (see module docstring).
        """
        self._excursion = None

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
        """No higher-timeframe bias -- present only for engine compatibility."""

    def on_entry_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)
        prev = self._prev_rsi
        curr = self.rsi.update(bar.close)
        self._prev_rsi = curr
        if curr is None:
            return None

        # 1. Inside an extreme: start or extend the excursion.
        if curr < self.oversold:
            if self._excursion is None or self._excursion.direction != "long":
                self._excursion = _Excursion("long", bar.low)
            else:
                self._excursion.extreme = min(self._excursion.extreme, bar.low)
            return None
        if curr > self.overbought:
            if self._excursion is None or self._excursion.direction != "short":
                self._excursion = _Excursion("short", bar.high)
            else:
                self._excursion.extreme = max(self._excursion.extreme, bar.high)
            return None

        # 2. Back inside the 20-80 band: did RSI just cross out of an extreme?
        exc = self._excursion
        self._excursion = None
        if exc is None or prev is None:
            return None
        crossed_up = exc.direction == "long" and prev < self.oversold
        crossed_down = exc.direction == "short" and prev > self.overbought
        if not (crossed_up or crossed_down) or not self._in_entry_window(bar.timestamp):
            return None

        # The cross bar itself can print a fresh extreme (long wick, strong close).
        if exc.direction == "long":
            stop = min(exc.extreme, bar.low) - self.stop_buffer
        else:
            stop = max(exc.extreme, bar.high) + self.stop_buffer
        return self._build_signal(bar, exc.direction, stop, prev, curr)

    def _build_signal(self, bar: Bar, direction, stop_price: float, prev_rsi: float, rsi: float) -> Optional[Signal]:
        entry = bar.close
        risk = entry - stop_price if direction == "long" else stop_price - entry
        if risk <= 0:
            return None
        target = entry + risk * self.target_r if direction == "long" else entry - risk * self.target_r
        level = self.oversold if direction == "long" else self.overbought
        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry,
            stop_price=stop_price,
            target_price=target,
            reason=f"RSI({self.rsi_length}) crossed {level:g} ({prev_rsi:.1f} -> {rsi:.1f})",
        )
