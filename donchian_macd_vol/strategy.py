"""
Donchian Channel breakout, confirmed by MACD momentum and volume --
three indicators that each measure something different (pure price-action
range, momentum, participation), none of them reused from any other
strategy in this repo.

1. **Donchian Channel** -- the core of the original "Turtle Trading"
   system: a rolling `donchian_period`-bar high/low channel, computed from
   the PRIOR `donchian_period` bars only (never including the current bar
   -- otherwise a breakout could never be detected, since the current
   bar's own extreme would always be part of its own channel). Does NOT
   reset per session -- same reasoning as ORB's/ATR-breakout's trailing
   ATR: a rolling window reset at session open wouldn't be usable again
   until well into the session, every single day.
2. **MACD** (12/26/9 EMA-based, standard formula) -- momentum
   confirmation. Uses the histogram as of the END of the PRIOR bar (before
   this bar's own close is folded into the EMAs) as the filter -- same
   no-lookahead discipline as ATR-breakout's ATR: you can only know a
   close-based indicator's confirmed value one bar behind the one you're
   using it to filter.
3. **Volume** -- requires the breakout bar's own volume to be at least
   `volume_mult` times the rolling `volume_period`-bar average volume
   (computed from PRIOR bars), filtering out low-conviction breaks with no
   real participation behind them.
4. **Entry** -- resting-stop-style, same fill convention as ORB/ATR-
   breakout: fires the instant a bar's range reaches the channel level, at
   that exact level, not at the bar's close. Long: `bar.high >=
   donchian_upper` AND prior-bar MACD histogram > 0 AND volume filter
   passes. Short: mirrors it. If a bar's range improbably crosses both
   channel edges, long is checked first (documented tie-break, consistent
   with the rest of this repo).
5. **Stop / target** -- stop at the opposite edge of the SAME entry-time
   Donchian channel (a breakout that fully round-trips back through its
   own channel has invalidated its own thesis), plus a small tick buffer.
   Target = `target_r * risk` (default 1.5:1).
6. **Multiple trades per day allowed**, same as overextension/ATR-
   breakout -- no artificial per-day cap; rely on `RiskManager` at the
   engine level for daily-loss-limit/max-daily-trades.
7. **Intraday only** -- same entry-window/flatten-cutoff convention as the
   rest of this repo (`SessionConfig`, shared). Like ATR-breakout, no
   other session-scoped state resets daily (the channel/MACD/volume
   average all persist continuously) -- so there's no `reset()` beyond the
   stateless time-of-day entry-window check.

`donchian_period` defaults to 10 (not the classic Turtle system's 20) --
tuned against real 2019 NAS100 data (backtest.run_donchian_macd_vol) on
the default 5-minute timeframe to land around 2-3 trades per active day, a
deliberate frequency target rather than a generic textbook default. 20 on
5min (~2.0/day) and 20 on 15min (~1.4/day) also produced a real edge in
that backtest (profit factor 1.3-1.4 across the board) but landed further
from the target frequency and/or with a worse expectancy per trade -- see
the CLI's own trades_per_active_day output if you retune this.
"""

from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig


class _DonchianChannel:
    """Rolling N-bar high/low channel. Does NOT reset per session."""

    def __init__(self, period: int):
        self.period = period
        self._highs: deque = deque(maxlen=period)
        self._lows: deque = deque(maxlen=period)

    def update(self, bar: Bar) -> None:
        self._highs.append(bar.high)
        self._lows.append(bar.low)

    @property
    def upper(self) -> Optional[float]:
        return max(self._highs) if len(self._highs) == self.period else None

    @property
    def lower(self) -> Optional[float]:
        return min(self._lows) if len(self._lows) == self.period else None


class _EMA:
    """Standard EMA (ta.ema()'s Pine default: seeds directly from the
    first sample, not an SMA-seeded warmup like Wilder's RMA)."""

    def __init__(self, period: int):
        self.alpha = 2.0 / (period + 1)
        self.value: Optional[float] = None

    def update(self, x: float) -> None:
        self.value = x if self.value is None else self.alpha * x + (1 - self.alpha) * self.value


class _MACD:
    """Standard MACD (fast EMA - slow EMA, plus an EMA of that difference
    as the signal line). Does NOT reset per session."""

    def __init__(self, fast: int = 12, slow: int = 26, signal: int = 9):
        self._fast = _EMA(fast)
        self._slow = _EMA(slow)
        self._signal = _EMA(signal)
        self.histogram: Optional[float] = None

    def update(self, close: float) -> None:
        self._fast.update(close)
        self._slow.update(close)
        if self._fast.value is None or self._slow.value is None:
            return
        macd = self._fast.value - self._slow.value
        self._signal.update(macd)
        if self._signal.value is not None:
            self.histogram = macd - self._signal.value


class _RollingAverage:
    """Rolling N-bar simple average. Does NOT reset per session."""

    def __init__(self, period: int):
        self.period = period
        self._values: deque = deque(maxlen=period)
        self._sum = 0.0

    def update(self, x: float) -> None:
        if len(self._values) == self.period:
            self._sum -= self._values[0]
        self._values.append(x)
        self._sum += x

    @property
    def value(self) -> Optional[float]:
        return self._sum / self.period if len(self._values) == self.period else None


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


class DonchianMacdVolumeStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        donchian_period: int = 10,
        macd_fast: int = 12,
        macd_slow: int = 26,
        macd_signal: int = 9,
        volume_period: int = 20,
        volume_mult: float = 1.5,
        target_r: float = 1.5,
        stop_buffer_ticks: int = 2,
        session: Optional[SessionConfig] = None,
    ):
        self.tick_size = tick_size
        self.target_r = target_r
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.session = session or SessionConfig()

        self._donchian = _DonchianChannel(donchian_period)
        self._macd = _MACD(macd_fast, macd_slow, macd_signal)
        self._volume_avg = _RollingAverage(volume_period)
        self.volume_mult = volume_mult

    def _in_entry_window(self, ts) -> bool:
        t = ts.time()
        return self.session.session_start <= t < self.session.no_entry_after

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        upper = self._donchian.upper
        lower = self._donchian.lower
        prior_hist = self._macd.histogram
        prior_avg_vol = self._volume_avg.value

        signal = None
        if (
            upper is not None
            and lower is not None
            and prior_hist is not None
            and prior_avg_vol is not None
            and self._in_entry_window(bar.timestamp)
        ):
            volume_ok = bar.volume >= self.volume_mult * prior_avg_vol

            if bar.high >= upper and prior_hist > 0 and volume_ok:
                signal = self._fire(bar, "long", upper, lower, prior_hist, prior_avg_vol)
            elif bar.low <= lower and prior_hist < 0 and volume_ok:
                signal = self._fire(bar, "short", lower, upper, prior_hist, prior_avg_vol)

        # Update AFTER computing this bar's trigger/stop -- next bar's
        # channel/MACD/volume-average must not include this bar until now.
        self._donchian.update(bar)
        self._macd.update(bar.close)
        self._volume_avg.update(bar.volume)

        return signal

    def _fire(
        self, bar: Bar, direction: str, entry_price: float, opposite_edge: float, hist: float, avg_vol: float
    ) -> Optional[Signal]:
        if direction == "long":
            stop_price = opposite_edge - self.stop_buffer
            risk = entry_price - stop_price
            if risk <= 0:
                return None
            target_price = entry_price + risk * self.target_r
        else:
            stop_price = opposite_edge + self.stop_buffer
            risk = stop_price - entry_price
            if risk <= 0:
                return None
            target_price = entry_price - risk * self.target_r

        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=f"donchian={entry_price:.2f}, macd_hist={hist:.4f}, vol={bar.volume:.0f} (avg={avg_vol:.0f})",
        )
