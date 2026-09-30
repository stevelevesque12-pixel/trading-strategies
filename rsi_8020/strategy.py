"""
RSI 80-20 divergence strategy.

Long case described; short is the exact mirror (50-candle high, RSI > 80,
higher high with lower RSI, close below the first candle's low):

  1. First low: a candle makes the lowest low of the last `lookback` (50)
     candles while RSI is below `oversold` (20). Remember that candle
     (its low, its high, and its RSI).
  2. Second low: a later candle closes with a low *below* the first low, but
     with a *higher* RSI than the first low had -- bullish divergence.
     A lower low whose RSI is not higher is just a deeper sell-off: it
     replaces the first low instead (its RSI is necessarily still < 20).
     Further lower lows that keep the divergence extend the second low.
  3. Entry: on a later candle that closes above the first low candle's high.
  4. Stop: below the lowest low made after the first low (the second low),
     minus `stop_buffer_ticks`.
  5. Target: entry + `target_r` * risk (default 3R, per the 1:3 example).

A setup that hasn't triggered within `max_setup_bars` of its first low is
dropped. Swing state is not reset at session boundaries (futures trade
overnight and 50 candles routinely span sessions); signals only fire inside
the session entry window, and the backtest engine / Pine script still
flatten intraday.

RSI is Wilder's (the same RMA smoothing as TradingView's `ta.rsi`, seeded
with a simple average of the first `rsi_length` changes), so values line up
with the Pine port in tradingview/rsi_8020.pine once both are warmed up.
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, Literal, Optional

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
class _Setup:
    direction: Literal["long", "short"]
    first_extreme: float  # the 50-candle low (long) / high (short)
    first_rsi: float
    trigger: float  # first candle's high (long) / low (short) -- close beyond it enters
    bars_since_first: int = 0
    second_extreme: Optional[float] = None  # lowest low (long) / highest high (short) after the first
    second_confirmed: bool = False  # True once a divergent second extreme candle has closed


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
        lookback: int = 50,
        max_setup_bars: int = 50,
        stop_buffer_ticks: int = 2,
        target_r: float = 3.0,
        session: Optional[SessionConfig] = None,
    ):
        if not 0 < oversold < overbought < 100:
            raise ValueError("need 0 < oversold < overbought < 100")
        if lookback < 2:
            raise ValueError("lookback must be >= 2")
        self.tick_size = tick_size
        self.rsi_length = rsi_length
        self.overbought = overbought
        self.oversold = oversold
        self.lookback = lookback
        self.max_setup_bars = max_setup_bars
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.target_r = target_r
        self.session = session or SessionConfig()

        self.rsi = WilderRSI(rsi_length)
        # Prior (lookback - 1) candles; with the current one that's the 50-candle window.
        self._prior_lows: Deque[float] = deque(maxlen=lookback - 1)
        self._prior_highs: Deque[float] = deque(maxlen=lookback - 1)
        self._setup: Optional[_Setup] = None
        self._triggered = False  # the current candle consumed a setup (entered, or would have)

    def reset(self) -> None:
        self.rsi = WilderRSI(self.rsi_length)
        self._prior_lows.clear()
        self._prior_highs.clear()
        self._setup = None

    def _in_entry_window(self, ts) -> bool:
        t = ts.time()
        return self.session.session_start <= t < self.session.no_entry_after

    def on_bias_bar(self, bar: Bar) -> None:
        """No higher-timeframe bias -- present only for engine compatibility."""

    def on_entry_bar(self, bar: Bar) -> Optional[Signal]:
        rsi = self.rsi.update(bar.close)
        window_full = len(self._prior_lows) == self._prior_lows.maxlen
        is_50_low = window_full and bar.low <= min(self._prior_lows)
        is_50_high = window_full and bar.high >= max(self._prior_highs)
        self._prior_lows.append(bar.low)
        self._prior_highs.append(bar.high)

        if rsi is None:
            return None

        self._triggered = False
        signal = self._advance_setup(bar, rsi)
        if signal is not None:
            return signal

        # No setup in progress (or it just expired): look for a fresh first low/high.
        if self._setup is None and not self._triggered:
            if is_50_low and rsi < self.oversold:
                self._setup = _Setup("long", bar.low, rsi, trigger=bar.high)
            elif is_50_high and rsi > self.overbought:
                self._setup = _Setup("short", bar.high, rsi, trigger=bar.low)
        return None

    def _advance_setup(self, bar: Bar, rsi: float) -> Optional[Signal]:
        s = self._setup
        if s is None:
            return None
        s.bars_since_first += 1
        long = s.direction == "long"

        # Entry check first: uses only state from *earlier* candles, so the
        # second low must already have closed before a later candle triggers.
        if s.second_confirmed:
            beyond = bar.close > s.trigger if long else bar.close < s.trigger
            if beyond:
                self._setup = None
                self._triggered = True
                if not self._in_entry_window(bar.timestamp):
                    return None
                return self._build_signal(bar, s)

        new_extreme = bar.low < s.first_extreme if long else bar.high > s.first_extreme
        if new_extreme:
            extreme = bar.low if long else bar.high
            diverges = rsi > s.first_rsi if long else rsi < s.first_rsi
            if diverges:
                if s.second_extreme is None:
                    s.second_extreme = extreme
                else:
                    s.second_extreme = min(s.second_extreme, extreme) if long else max(s.second_extreme, extreme)
            else:
                # Deeper extreme with weaker (or equal) RSI: no divergence --
                # this candle becomes the new first low/high.
                trigger = bar.high if long else bar.low
                self._setup = _Setup(s.direction, extreme, rsi, trigger=trigger)
                return None

        if s.second_extreme is not None:
            s.second_confirmed = True

        if s.bars_since_first >= self.max_setup_bars:
            self._setup = None
        return None

    def _build_signal(self, bar: Bar, s: _Setup) -> Optional[Signal]:
        entry = bar.close
        if s.direction == "long":
            stop = min(s.second_extreme, bar.low) - self.stop_buffer  # entry candle may wick lower
            risk = entry - stop
            target = entry + risk * self.target_r
            level = f"{self.lookback}-bar low, RSI<{self.oversold:g}"
        else:
            stop = max(s.second_extreme, bar.high) + self.stop_buffer
            risk = stop - entry
            target = entry - risk * self.target_r
            level = f"{self.lookback}-bar high, RSI>{self.overbought:g}"
        if risk <= 0:
            return None
        return Signal(
            timestamp=bar.timestamp,
            direction=s.direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            reason=f"{level} at {s.first_extreme} (RSI {s.first_rsi:.1f}), divergent 2nd extreme {s.second_extreme}",
        )
