"""
Initial Balance (IB) breakout strategy.

The Initial Balance is the high/low of the first `ib_minutes` of the regular
session (default 60 min: 09:30-10:30 ET). Once it's complete:

  1. Long trigger: an entry-timeframe bar *closes* above IB high (+ an
     optional tick buffer). Short trigger: the mirror below IB low. Entry is
     that bar's close -- the backtest engine then manages the position from
     the next bar on, so there's no same-bar lookahead.
  2. Stop, one of:
       - "mid":      the IB midpoint (default)
       - "opposite": the other side of the IB (full-range stop)
       - "ib_frac":  entry -/+ stop_ib_frac * IB range
  3. Target, one of:
       - "r":      target_r * risk (default)
       - "ib_ext": breakout level +/- target_ib_ext * IB range (classic
                   "IB extension" targets, e.g. 1.0 = a full-range extension)
  4. One trade per session by default. With `allow_reverse=True` a stopped-
     out breakout can be followed by one trade the other way (a failed break
     often runs to the opposite IB side), so at most one per direction.
  5. Optional IB-range filter (min/max points): skip days whose IB is too
     narrow (chop) or too wide (the move already happened).
  6. No new entries at/after `entry_cutoff`; the engine flattens anything
     still open at SessionConfig.flatten_at.
  7. Optional directional bias from how the IB formed (`bias_filter`):
       - "mid":   IB closed above its midpoint -> longs only; below -> shorts
       - "order": IB high printed last -> longs only; low printed last -> shorts
       - "both":  both must agree, else no trade that day
     On 10y of NQ these lift the breach rate of the favoured side from ~47%
     to ~74% by noon (see ib_break/stats.py) -- but a breach isn't a winning
     trade; the backtest is what says whether the filter pays.
  8. `entry_mode`: "close" (default) enters on a bar close beyond the level;
     "touch" models a resting stop order at level +/- buffer, filled at that
     price (or the bar open if it gaps through) on the first bar to trade
     there. Touch fills ignore a same-bar stop-out, so they're slightly
     optimistic on coarse entry timeframes.

Plugs into backtest.engine.BacktestEngine: on_entry_bar() returns a Signal,
on_bias_bar() is a no-op (the IB is built from the entry bars themselves).
"""

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig, Signal

StopMode = Literal["mid", "opposite", "ib_frac"]
BiasFilter = Literal["none", "mid", "order", "both"]
EntryMode = Literal["close", "touch"]
TargetMode = Literal["r", "ib_ext"]


@dataclass
class InitialBalance:
    high: float
    low: float
    close: float
    order: Literal["high_last", "low_last", "same_bar"]

    @property
    def range(self) -> float:
        return self.high - self.low

    @property
    def mid(self) -> float:
        return (self.high + self.low) / 2.0


class IBBreakStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        ib_minutes: int = 60,
        entry_tf_minutes: int = 1,
        breakout_buffer_ticks: int = 0,
        stop_mode: StopMode = "mid",
        stop_ib_frac: float = 0.5,
        target_mode: TargetMode = "r",
        target_r: float = 1.0,
        target_ib_ext: float = 1.0,
        allow_reverse: bool = False,
        min_ib_range: float = 0.0,
        max_ib_range: float = float("inf"),
        entry_cutoff: time = time(12, 0),
        bias_filter: BiasFilter = "none",
        entry_mode: EntryMode = "close",
        session: Optional[SessionConfig] = None,
    ):
        self.tick_size = tick_size
        self.breakout_buffer = breakout_buffer_ticks * tick_size
        self.stop_mode = stop_mode
        self.stop_ib_frac = stop_ib_frac
        self.target_mode = target_mode
        self.target_r = target_r
        self.target_ib_ext = target_ib_ext
        self.allow_reverse = allow_reverse
        self.min_ib_range = min_ib_range
        self.max_ib_range = max_ib_range
        self.entry_cutoff = entry_cutoff
        self.bias_filter = bias_filter
        self.entry_mode = entry_mode
        self.session = session or SessionConfig()

        start = datetime.combine(datetime.min, self.session.session_start)
        self.ib_end: time = (start + timedelta(minutes=ib_minutes)).time()
        self._entry_tf = timedelta(minutes=entry_tf_minutes)

        self._current_date = None
        self.reset()

    def reset(self) -> None:
        """Clear all state -- called automatically on a new session day."""
        self._ib_high: Optional[float] = None
        self._ib_low: Optional[float] = None
        self._ib_high_bar = self._ib_low_bar = self._ib_bars = 0
        self.ib: Optional[InitialBalance] = None
        self._taken: set = set()

    def _roll_session(self, ts) -> None:
        d = ts.date()
        if self._current_date is None:
            self._current_date = d
        elif d != self._current_date:
            self._current_date = d
            self.reset()

    def on_bias_bar(self, bar: Bar) -> None:
        pass

    def _build_ib(self, bar: Bar) -> None:
        t = bar.timestamp.time()
        if self.session.session_start <= t < self.ib_end:
            # Bar index of the first print of each extreme, for HL order.
            if self._ib_high is None or bar.high > self._ib_high:
                self._ib_high, self._ib_high_bar = bar.high, self._ib_bars
            if self._ib_low is None or bar.low < self._ib_low:
                self._ib_low, self._ib_low_bar = bar.low, self._ib_bars
            self._ib_bars += 1
            # The IB is final once the bar that closes at ib_end has been seen.
            bar_close = (datetime.combine(datetime.min, t) + self._entry_tf).time()
            if bar_close >= self.ib_end:
                if self._ib_high_bar > self._ib_low_bar:
                    order = "high_last"
                elif self._ib_low_bar > self._ib_high_bar:
                    order = "low_last"
                else:
                    order = "same_bar"
                self.ib = InitialBalance(self._ib_high, self._ib_low, bar.close, order)

    def on_entry_bar(self, bar: Bar) -> Optional[Signal]:
        self._roll_session(bar.timestamp)

        if self.ib is None:
            self._build_ib(bar)
            return None

        t = bar.timestamp.time()
        if not (self.ib_end <= t < min(self.entry_cutoff, self.session.no_entry_after)):
            return None

        ib = self.ib
        if not (self.min_ib_range <= ib.range <= self.max_ib_range) or ib.range <= 0:
            return None

        max_trades = 2 if self.allow_reverse else 1
        if len(self._taken) >= max_trades:
            return None

        allowed = self._allowed_directions(ib) - self._taken
        long_level = ib.high + self.breakout_buffer
        short_level = ib.low - self.breakout_buffer
        if self.entry_mode == "touch":
            long_hit, short_hit = bar.high >= long_level, bar.low <= short_level
        else:
            long_hit, short_hit = bar.close > long_level, bar.close < short_level

        direction: Optional[Literal["long", "short"]] = None
        if long_hit and "long" in allowed:
            direction = "long"
        elif short_hit and "short" in allowed:
            direction = "short"
        if direction is None:
            return None

        if self.entry_mode == "touch":
            entry = max(long_level, bar.open) if direction == "long" else min(short_level, bar.open)
        else:
            entry = bar.close
        signal = self._make_signal(bar, direction, ib, entry)
        if signal is not None:
            self._taken.add(direction)
        return signal

    def _allowed_directions(self, ib: InitialBalance) -> set:
        if self.bias_filter == "none":
            return {"long", "short"}
        by_mid = {"long"} if ib.close > ib.mid else {"short"} if ib.close < ib.mid else set()
        by_order = {"long"} if ib.order == "high_last" else {"short"} if ib.order == "low_last" else set()
        if self.bias_filter == "mid":
            return by_mid
        if self.bias_filter == "order":
            return by_order
        if self.bias_filter == "both":
            return by_mid & by_order
        raise ValueError(f"unknown bias_filter {self.bias_filter!r}")

    def _make_signal(self, bar: Bar, direction: str, ib: InitialBalance, entry: float) -> Optional[Signal]:
        sign = 1 if direction == "long" else -1

        if self.stop_mode == "mid":
            stop = ib.mid
        elif self.stop_mode == "opposite":
            stop = ib.low if direction == "long" else ib.high
        elif self.stop_mode == "ib_frac":
            stop = entry - sign * self.stop_ib_frac * ib.range
        else:
            raise ValueError(f"unknown stop_mode {self.stop_mode!r}")

        risk = (entry - stop) * sign
        if risk <= 0:
            return None

        if self.target_mode == "r":
            target = entry + sign * risk * self.target_r
        elif self.target_mode == "ib_ext":
            level = ib.high if direction == "long" else ib.low
            target = level + sign * self.target_ib_ext * ib.range
            if (target - entry) * sign <= 0:
                return None
        else:
            raise ValueError(f"unknown target_mode {self.target_mode!r}")

        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry,
            stop_price=stop,
            target_price=target,
            reason=(
                f"ib_break ib_high={ib.high} ib_low={ib.low} range={ib.range:.2f} "
                f"close={ib.close} order={ib.order} stop={self.stop_mode} target={self.target_mode}"
            ),
        )


# Best-supported config from the NQ sweeps (see README "IB Break"): trade only
# the side both IB confluences point to, stop order at the IB extreme, stop at
# the opposite IB extreme, and in practice hold to the session flatten (the
# 2R target is hit on <5% of trades).
CONFLUENCE_PRESET = dict(
    bias_filter="both",
    entry_mode="touch",
    stop_mode="opposite",
    target_mode="r",
    target_r=2.0,
    entry_cutoff=time(15, 45),
)
