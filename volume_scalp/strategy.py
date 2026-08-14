"""
Volume-confirmed scalping strategy for MES/MNQ.

Two independent, volume-driven entry setups, checked in this order (one
position at a time -- the caller/engine is responsible for that flat-
position gate, same as failed2s.strategy.Failed2sStrategy and
structure_scalp.strategy.StructureScalpStrategy):

1. **Volume breakout** (momentum/continuation) -- price closes through a
   recent N-bar range on a bar with relative volume (RVOL) above
   threshold, same-direction net volume delta (the CLV-based proxy from
   volume_scalp/indicators.py) over the trailing window, and, optionally,
   on the correct side of session VWAP. This is the "real participation
   confirms the breakout" setup: a lot of retail breakout scalps fail
   because the level breaks on thin volume and snaps right back --
   requiring RVOL + delta confirmation filters those out.

2. **Volume climax fade** (exhaustion/reversal) -- a bar with RVOL far
   above normal (a volume "climax") that also shows a rejection wick
   against its own directional push (most of the bar's range is a wick,
   and the close is back near the middle/opposite side) fades the move, on
   the theory that a volume spike with no follow-through close is
   absorption/exhaustion, not real continuation -- classic Wyckoff
   upthrust/spring behavior. Set enable_climax_fade=False to run
   breakout-only.

Both setups are OHLCV-only (see volume_scalp/indicators.py for the
volume-delta proxy's caveats) and operate on a single timeframe -- default
1-minute, matching the resolution this repo's real historical data
(sample_data/fetch_real_data.py) and free/live retail feeds actually
provide. Unlike structure_scalp's 5-second entries, that means this CAN be
backtested end-to-end against real historical data (see
backtest/volume_engine.py) rather than only logic-tested against synthetic
bars.

Strictly intraday: all rolling state resets every session day, and no
signal fires outside the configured entry window. Session flatten is
enforced by the caller (BacktestEngine / a live runner), same pattern as
the other two strategies in this repo.
"""

from dataclasses import dataclass
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig

from .indicators import RollingChannel, RollingDelta, RollingVolume, SessionVWAP, bar_volume_delta


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


class VolumeScalpStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        stop_buffer_ticks: int = 2,
        target_r: float = 1.5,
        volume_window: int = 20,
        breakout_window: int = 10,
        delta_window: int = 5,
        breakout_rvol_threshold: float = 1.5,
        min_delta_confirmation: float = 0.0,
        require_vwap_alignment: bool = True,
        enable_breakout: bool = True,
        enable_climax_fade: bool = True,
        climax_rvol_threshold: float = 3.0,
        climax_wick_pct: float = 0.5,
        session: Optional[SessionConfig] = None,
    ):
        if breakout_rvol_threshold <= 0 or climax_rvol_threshold <= 0:
            raise ValueError("RVOL thresholds must be positive")
        if not 0.0 <= climax_wick_pct <= 1.0:
            raise ValueError("climax_wick_pct must be in [0, 1]")

        self.tick_size = tick_size
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.target_r = target_r
        self.breakout_rvol_threshold = breakout_rvol_threshold
        self.min_delta_confirmation = min_delta_confirmation
        self.require_vwap_alignment = require_vwap_alignment
        self.enable_breakout = enable_breakout
        self.enable_climax_fade = enable_climax_fade
        self.climax_rvol_threshold = climax_rvol_threshold
        self.climax_wick_pct = climax_wick_pct
        self.session = session or SessionConfig()

        self.volume_window = volume_window
        self.breakout_window = breakout_window
        self.delta_window = delta_window

        self.vwap = SessionVWAP()
        self.rvol_tracker = RollingVolume(window=volume_window)
        self.delta_tracker = RollingDelta(window=delta_window)
        self.channel = RollingChannel(window=breakout_window)

        self._current_date = None

    def reset(self) -> None:
        """Clear all rolling state -- called automatically on a new session day."""
        self.vwap.reset()
        self.rvol_tracker = RollingVolume(window=self.volume_window)
        self.delta_tracker = RollingDelta(window=self.delta_window)
        self.channel = RollingChannel(window=self.breakout_window)

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

        # Baselines the current bar is judged against -- snapshotted from
        # strictly-prior bars before any state is touched, so a bar can
        # never influence the threshold it's being compared to.
        channel_high, channel_low = self.channel.high, self.channel.low
        rvol = self.rvol_tracker.relvol(bar.volume)

        # VWAP and cumulative delta legitimately include the current bar
        # (they're confirmation signals computed from this bar's own known
        # OHLCV, not baselines) -- update-then-read.
        self.vwap.update(bar)
        self.delta_tracker.update(bar_volume_delta(bar))
        cum_delta = self.delta_tracker.value

        signal = None
        if self._in_entry_window(bar.timestamp):
            if self.enable_breakout:
                signal = self._check_breakout(bar, channel_high, channel_low, rvol, cum_delta)
            if signal is None and self.enable_climax_fade:
                signal = self._check_climax_fade(bar, rvol)

        # Roll the baseline trackers forward for the *next* bar.
        self.rvol_tracker.update(bar.volume)
        self.channel.update(bar)

        return signal

    def _check_breakout(
        self, bar: Bar, channel_high: Optional[float], channel_low: Optional[float], rvol: float, cum_delta: float
    ) -> Optional[Signal]:
        if channel_high is None or channel_low is None or rvol < self.breakout_rvol_threshold:
            return None

        vwap_value = self.vwap.value

        if bar.close > channel_high and bar.is_bullish and cum_delta > self.min_delta_confirmation:
            if self.require_vwap_alignment and vwap_value is not None and bar.close < vwap_value:
                return None
            entry_price = bar.close
            stop_price = min(bar.low, channel_low) - self.stop_buffer
            risk = entry_price - stop_price
            if risk <= 0:
                return None
            return Signal(
                timestamp=bar.timestamp,
                direction="long",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=entry_price + risk * self.target_r,
                reason=(
                    f"breakout long: close {entry_price} > {self.breakout_window}-bar high {channel_high}, "
                    f"rvol={rvol:.2f}x, cum_delta={cum_delta:.1f}"
                ),
            )

        if bar.close < channel_low and bar.is_bearish and cum_delta < -self.min_delta_confirmation:
            if self.require_vwap_alignment and vwap_value is not None and bar.close > vwap_value:
                return None
            entry_price = bar.close
            stop_price = max(bar.high, channel_high) + self.stop_buffer
            risk = stop_price - entry_price
            if risk <= 0:
                return None
            return Signal(
                timestamp=bar.timestamp,
                direction="short",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=entry_price - risk * self.target_r,
                reason=(
                    f"breakout short: close {entry_price} < {self.breakout_window}-bar low {channel_low}, "
                    f"rvol={rvol:.2f}x, cum_delta={cum_delta:.1f}"
                ),
            )

        return None

    def _check_climax_fade(self, bar: Bar, rvol: float) -> Optional[Signal]:
        rng = bar.range
        if rng <= 0 or rvol < self.climax_rvol_threshold:
            return None

        upper_wick = bar.high - max(bar.open, bar.close)
        lower_wick = min(bar.open, bar.close) - bar.low
        midpoint = (bar.high + bar.low) / 2.0

        # Buying-climax blow-off: a big up-push on huge volume, rejected back down -> fade short.
        if upper_wick / rng >= self.climax_wick_pct and bar.close <= midpoint:
            entry_price = bar.close
            stop_price = bar.high + self.stop_buffer
            risk = stop_price - entry_price
            if risk <= 0:
                return None
            return Signal(
                timestamp=bar.timestamp,
                direction="short",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=entry_price - risk * self.target_r,
                reason=f"climax fade short: rvol={rvol:.2f}x, upper_wick={upper_wick / rng:.0%} of range",
            )

        # Selling-climax spring: a big down-push on huge volume, rejected back up -> fade long.
        if lower_wick / rng >= self.climax_wick_pct and bar.close >= midpoint:
            entry_price = bar.close
            stop_price = bar.low - self.stop_buffer
            risk = entry_price - stop_price
            if risk <= 0:
                return None
            return Signal(
                timestamp=bar.timestamp,
                direction="long",
                entry_price=entry_price,
                stop_price=stop_price,
                target_price=entry_price + risk * self.target_r,
                reason=f"climax fade long: rvol={rvol:.2f}x, lower_wick={lower_wick / rng:.0%} of range",
            )

        return None
