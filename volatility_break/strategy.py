"""
Volatility Break -- a 15-minute Bollinger-squeeze breakout using pending STOP orders.

Spec (as given):
  - Timeframe:            M15
  - Entry logic:          Bollinger Band volatility condition + pending STOP confirmation
  - Volatility filter:    Bollinger Band Width Ratio
  - Order type:           pending STOP orders
  - Stop-loss:            3.5 x ATR
  - Profit target:        5.2 x ATR
  - Trailing stop:        60 points
  - Trailing activation:  2.1 x ATR (open profit)
  - Pending expiration:   121 bars

The spec doesn't pin down the indicator periods, the exact volatility
condition, or where the stop orders sit, so these are the choices made here
(all are constructor parameters):

  - Bollinger Bands(20, 2.0) on close. Width Ratio = (upper - lower) / middle
    (the normalized band width, as in StrategyQuant's BBWidthRatio).
  - Volatility condition ("squeeze"): Width Ratio < its own 50-bar SMA, i.e.
    volatility is currently compressed relative to its recent norm.
  - On a squeeze bar's close, with no position and no live orders, place an
    OCO bracket: BUY STOP at the upper band, SELL STOP at the lower band.
    Price trading through a band is the "confirmation" -- the order only
    fills if the expansion actually happens. When one side fills the other
    is cancelled. Unfilled brackets expire after `expiry_bars` bars.
  - ATR(14), Wilder smoothing, snapshotted when the bracket is placed. SL/TP
    are measured from the actual fill price.
  - Trailing: once open profit >= 2.1 x ATR, the stop trails `trail_points`
    behind the best price reached (only ever tightens). "60 points" is taken
    as 60.0 in price units (60 NQ index points) -- change `trail_points` if
    your platform's "point" means a tick.
  - Trades 24h (the 15m futures data is the full Globex session) and holds
    overnight -- there is no intraday flatten, matching how a generic M15 EA
    runs. One position or one bracket at a time.

Fill model (bars are processed only after they close, so everything below
is decided from that bar's OHLC with no lookahead):
  - A stop order placed on bar N's close is live from bar N+1. It fills at its
    level, or at the bar's open if the market gapped through it.
  - If both sides of the bracket trade in the same bar, the side nearer the
    open is assumed to have filled first.
  - Exits on later bars: stop first, then target (pessimistic when both are
    in range); gaps through either fill at the open.
  - On the fill bar itself the stop is checked against the rest of that bar's
    range (pessimistic); the target only counts if the stop wasn't touched.
  - The trailing stop is recomputed from each bar's high/low *after* that
    bar's exit checks, so it takes effect from the next bar.
"""

from collections import deque
from dataclasses import dataclass
from math import sqrt
from typing import Deque, List, Literal, Optional

from failed2s.bars import Bar


@dataclass
class Fill:
    """A completed round trip."""

    entry_time: object
    exit_time: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float  # initial stop (defines 1R)
    target_price: float
    exit_price: float
    exit_reason: str  # stop | trail_stop | target | end_of_data


@dataclass
class _Bracket:
    buy_stop: float
    sell_stop: float
    atr: float
    placed_index: int


@dataclass
class _Position:
    direction: Literal["long", "short"]
    entry_time: object
    entry_price: float
    initial_stop: float
    stop: float
    target: float
    atr: float
    best: float  # highest high (long) / lowest low (short) since entry
    trailing: bool = False


class _RollingMean:
    def __init__(self, n: int):
        self.n = n
        self.buf: Deque[float] = deque(maxlen=n)
        self.total = 0.0
        self.total_sq = 0.0

    def push(self, x: float) -> None:
        if len(self.buf) == self.n:
            old = self.buf[0]
            self.total -= old
            self.total_sq -= old * old
        self.buf.append(x)
        self.total += x
        self.total_sq += x * x

    @property
    def ready(self) -> bool:
        return len(self.buf) == self.n

    def mean(self) -> float:
        return self.total / len(self.buf)

    def stdev(self) -> float:
        # Population stdev, as Bollinger Bands (and Pine's ta.stdev) use.
        m = self.mean()
        return sqrt(max(self.total_sq / len(self.buf) - m * m, 0.0))


class _WilderATR:
    def __init__(self, n: int):
        self.n = n
        self.prev_close: Optional[float] = None
        self.seed: List[float] = []
        self.value: Optional[float] = None

    def push(self, bar: Bar) -> None:
        if self.prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(bar.high, self.prev_close) - min(bar.low, self.prev_close)
        self.prev_close = bar.close
        if self.value is None:
            self.seed.append(tr)
            if len(self.seed) == self.n:
                self.value = sum(self.seed) / self.n
        else:
            self.value = (self.value * (self.n - 1) + tr) / self.n


class VolatilityBreakStrategy:
    def __init__(
        self,
        bb_period: int = 20,
        bb_dev: float = 2.0,
        ratio_avg_period: int = 50,
        squeeze_threshold: float = 1.0,
        atr_period: int = 14,
        sl_atr: float = 3.5,
        tp_atr: float = 5.2,
        trail_points: float = 60.0,
        trail_activation_atr: float = 2.1,
        expiry_bars: int = 121,
    ):
        self.bb_dev = bb_dev
        self.squeeze_threshold = squeeze_threshold
        self.sl_atr = sl_atr
        self.tp_atr = tp_atr
        self.trail_points = trail_points
        self.trail_activation_atr = trail_activation_atr
        self.expiry_bars = expiry_bars

        self._closes = _RollingMean(bb_period)
        self._ratio_avg = _RollingMean(ratio_avg_period)
        self._atr = _WilderATR(atr_period)

        self.bracket: Optional[_Bracket] = None
        self.position: Optional[_Position] = None
        self.fills: List[Fill] = []
        self._index = -1
        self._last_bar: Optional[Bar] = None

    # ---- public API ----

    def on_bar(self, bar: Bar) -> List[Fill]:
        """Process one closed M15 bar. Returns any trades closed on it."""
        self._index += 1
        closed: List[Fill] = []

        if self.position is not None:
            self._manage_position(bar, closed)
        elif self.bracket is not None:
            self._check_bracket(bar, closed)

        self._update_indicators(bar)

        if self.position is None and self.bracket is None and self._squeeze():
            self._place_bracket()

        self._last_bar = bar
        self.fills.extend(closed)
        return closed

    def finish(self) -> List[Fill]:
        """Close any open position at the last close (end of data)."""
        if self.position is None or self._last_bar is None:
            return []
        fill = self._close(self._last_bar.close, self._last_bar.timestamp, "end_of_data")
        self.fills.append(fill)
        return [fill]

    # ---- indicators ----

    def _update_indicators(self, bar: Bar) -> None:
        self._atr.push(bar)
        self._closes.push(bar.close)
        self.upper = self.lower = self.ratio = None
        if not self._closes.ready:
            return
        mid = self._closes.mean()
        sd = self._closes.stdev()
        self.upper = mid + self.bb_dev * sd
        self.lower = mid - self.bb_dev * sd
        self.ratio = (self.upper - self.lower) / mid if mid else 0.0
        self._ratio_avg.push(self.ratio)

    def _squeeze(self) -> bool:
        if self.ratio is None or not self._ratio_avg.ready or self._atr.value is None:
            return False
        if self.upper <= self.lower:  # flat-lined data, no band to break
            return False
        return self.ratio < self._ratio_avg.mean() * self.squeeze_threshold

    # ---- orders ----

    def _place_bracket(self) -> None:
        self.bracket = _Bracket(
            buy_stop=self.upper,
            sell_stop=self.lower,
            atr=self._atr.value,
            placed_index=self._index,
        )

    def _check_bracket(self, bar: Bar, closed: List[Fill]) -> None:
        b = self.bracket
        if self._index - b.placed_index > self.expiry_bars:
            self.bracket = None
            return

        buy_hit = bar.high >= b.buy_stop
        sell_hit = bar.low <= b.sell_stop
        if not (buy_hit or sell_hit):
            return
        if buy_hit and sell_hit:
            # Whichever level is nearer the open is assumed to trade first.
            buy_hit = abs(b.buy_stop - bar.open) <= abs(bar.open - b.sell_stop)

        self.bracket = None  # OCO: the other side is cancelled
        if buy_hit:
            price = max(b.buy_stop, bar.open)
            self.position = _Position(
                "long", bar.timestamp, price,
                initial_stop=price - self.sl_atr * b.atr, stop=price - self.sl_atr * b.atr,
                target=price + self.tp_atr * b.atr, atr=b.atr, best=price,
            )
        else:
            price = min(b.sell_stop, bar.open)
            self.position = _Position(
                "short", bar.timestamp, price,
                initial_stop=price + self.sl_atr * b.atr, stop=price + self.sl_atr * b.atr,
                target=price - self.tp_atr * b.atr, atr=b.atr, best=price,
            )
        self._manage_position(bar, closed, fill_bar=True)

    def _manage_position(self, bar: Bar, closed: List[Fill], fill_bar: bool = False) -> None:
        p = self.position
        reason_stop = "trail_stop" if p.trailing else "stop"
        if p.direction == "long":
            if not fill_bar and bar.open <= p.stop:
                closed.append(self._close(bar.open, bar.timestamp, reason_stop))
            elif bar.low <= p.stop:
                closed.append(self._close(p.stop, bar.timestamp, reason_stop))
            elif not fill_bar and bar.open >= p.target:
                closed.append(self._close(bar.open, bar.timestamp, "target"))
            elif bar.high >= p.target:
                closed.append(self._close(p.target, bar.timestamp, "target"))
            else:
                p.best = max(p.best, bar.high)
                if p.best - p.entry_price >= self.trail_activation_atr * p.atr:
                    p.trailing = True
                if p.trailing:
                    p.stop = max(p.stop, p.best - self.trail_points)
        else:
            if not fill_bar and bar.open >= p.stop:
                closed.append(self._close(bar.open, bar.timestamp, reason_stop))
            elif bar.high >= p.stop:
                closed.append(self._close(p.stop, bar.timestamp, reason_stop))
            elif not fill_bar and bar.open <= p.target:
                closed.append(self._close(bar.open, bar.timestamp, "target"))
            elif bar.low <= p.target:
                closed.append(self._close(p.target, bar.timestamp, "target"))
            else:
                p.best = min(p.best, bar.low)
                if p.entry_price - p.best >= self.trail_activation_atr * p.atr:
                    p.trailing = True
                if p.trailing:
                    p.stop = min(p.stop, p.best + self.trail_points)

    def _close(self, price: float, ts, reason: str) -> Fill:
        p = self.position
        self.position = None
        return Fill(
            entry_time=p.entry_time,
            exit_time=ts,
            direction=p.direction,
            entry_price=p.entry_price,
            stop_price=p.initial_stop,
            target_price=p.target,
            exit_price=price,
            exit_reason=reason,
        )
