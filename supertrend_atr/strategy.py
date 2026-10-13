"""
SuperTrend / ATR trend-following strategy with pending STOP-order entries.

Rules (long side; short is the exact mirror):

  1. Entry logic -- SuperTrend directional structure. On the bar the
     entry-timeframe SuperTrend flips to an uptrend, arm a BUY STOP at that
     bar's high + `entry_offset_ticks`. The order is pending, not a market
     order: price has to trade through the flip bar's high to confirm.
  2. Market-structure filter -- SuperTrend trend confirmation. The higher-
     timeframe SuperTrend (`htf_multiple` x the entry timeframe) must also
     be in an uptrend when the order is armed. `htf_multiple=0` disables it.
  3. Volatility filter -- ATR-based market behavior. ATR / SMA(ATR,
     `vol_lookback`) must be within [`vol_min`, `vol_max`]: skips dead,
     compressed tape where SuperTrend whipsaws, and (optionally) blow-off
     volatility.
  4. The pending stop expires after `order_expiry_bars` bars, or is
     cancelled as soon as the entry SuperTrend flips back.
  5. Stop-loss -- ATR-based: fill - `sl_atr_mult` (4.2) x ATR at the signal.
  6. Trailing stop -- ATR-based: once the trade is `trail_activation_mult`
     x ATR in profit, the stop trails the best price since entry by
     `trail_atr_mult` (0.6) x the current ATR. Only ever tightens. With the
     default activation equal to the trail distance this is the classic
     MetaTrader trailing stop (nothing moves until profit >= trail distance),
     which is what keeps the 4.2 ATR initial stop meaningful.
  7. No fixed profit target -- exits are the stop, the trail, or (in
     intraday mode) the session flatten.

Everything is evaluated on closed bars only; the backtester / broker side
decides fills (see `supertrend_atr/backtest.py`).
"""

from dataclasses import dataclass
from typing import Literal, Optional

from failed2s.bars import Bar

from .indicators import ATR, SMA, SuperTrend

Direction = Literal["long", "short"]


@dataclass
class StrategyParams:
    st_atr_period: int = 10
    st_factor: float = 3.0
    atr_period: int = 14  # ATR used for the stop, trail and volatility filter
    sl_atr_mult: float = 4.2
    trail_atr_mult: float = 0.6
    trail_activation_mult: Optional[float] = None  # None -> same as trail_atr_mult
    entry_offset_ticks: int = 1
    order_expiry_bars: int = 3
    htf_multiple: int = 4  # 0 disables the higher-timeframe confirmation
    vol_lookback: int = 100
    vol_min: float = 0.8
    vol_max: float = float("inf")

    @property
    def activation_mult(self) -> float:
        return self.trail_atr_mult if self.trail_activation_mult is None else self.trail_activation_mult


@dataclass
class PendingStop:
    direction: Direction
    price: float
    atr_at_signal: float
    placed_at: object
    bars_left: int

    def initial_stop(self, fill_price: float, sl_atr_mult: float) -> float:
        dist = sl_atr_mult * self.atr_at_signal
        return fill_price - dist if self.direction == "long" else fill_price + dist


@dataclass
class Position:
    direction: Direction
    entry_price: float
    entry_time: object
    initial_stop: float
    stop: float
    best_price: float  # highest high (long) / lowest low (short) since entry

    def trail(self, bar: Bar, atr: float, trail_mult: float, activation_mult: float) -> float:
        """Update best price + trailing stop from a closed bar. Returns the new stop."""
        dist = trail_mult * atr
        if self.direction == "long":
            self.best_price = max(self.best_price, bar.high)
            if self.best_price - self.entry_price >= activation_mult * atr:
                self.stop = max(self.stop, self.best_price - dist)
        else:
            self.best_price = min(self.best_price, bar.low)
            if self.entry_price - self.best_price >= activation_mult * atr:
                self.stop = min(self.stop, self.best_price + dist)
        return self.stop


class SuperTrendATRStrategy:
    def __init__(self, params: Optional[StrategyParams] = None, tick_size: float = 0.25):
        self.p = params or StrategyParams()
        self.tick_size = tick_size
        self.st = SuperTrend(self.p.st_atr_period, self.p.st_factor)
        self.htf_st = SuperTrend(self.p.st_atr_period, self.p.st_factor)
        self.atr = ATR(self.p.atr_period)
        self.atr_sma = SMA(self.p.vol_lookback)

    # -- indicator feeds -------------------------------------------------

    def on_htf_bar(self, bar: Bar) -> None:
        self.htf_st.update(bar)

    def on_bar(self, bar: Bar) -> None:
        self.st.update(bar)
        atr = self.atr.update(bar)
        if atr is not None:
            self.atr_sma.update(atr)

    # -- decisions (call after on_bar for the same closed bar) -----------

    @property
    def trend(self) -> Optional[Direction]:
        if self.st.direction is None:
            return None
        return "long" if self.st.direction == 1 else "short"

    def htf_confirms(self, direction: Direction) -> bool:
        if self.p.htf_multiple <= 0:
            return True
        want = 1 if direction == "long" else -1
        return self.htf_st.direction == want

    def volatility_ok(self) -> bool:
        atr, avg = self.atr.value, self.atr_sma.value
        if atr is None or avg is None or avg <= 0:
            return False
        ratio = atr / avg
        return self.p.vol_min <= ratio <= self.p.vol_max

    def entry_signal(self, bar: Bar) -> Optional[PendingStop]:
        """A new pending stop order if this closed bar is a qualifying SuperTrend flip."""
        if not self.st.flipped or self.atr.value is None:
            return None
        direction = self.trend
        if not self.htf_confirms(direction) or not self.volatility_ok():
            return None
        offset = self.p.entry_offset_ticks * self.tick_size
        price = bar.high + offset if direction == "long" else bar.low - offset
        return PendingStop(direction, price, self.atr.value, bar.timestamp, self.p.order_expiry_bars)

    def should_cancel(self, order: PendingStop) -> bool:
        return self.trend != order.direction

    def update_trailing(self, pos: Position, bar: Bar) -> float:
        return pos.trail(bar, self.atr.value, self.p.trail_atr_mult, self.p.activation_mult)
