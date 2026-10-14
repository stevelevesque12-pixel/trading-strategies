"""
Intraday NQ Trend-Filtered Momentum Confluence -- a port of the user's
QuantConnect algorithm (`SimpleNQExample`) to a plain-Python, bar-by-bar
engine that runs against this repo's 1-minute data.

Indicators (5-minute bars consolidated from 1-minute, plus one on 1-minute):

  - EMA(10), EMA(500), SMA(RSI(12), 5), EMA(Momentum(9), 3),
    EMA(UltimateOscillator(7, 14, 28), 3) on 5-minute bars
  - EMA(3) on 1-minute bars

Each is reduced to its *slope*: the change since its previous update
(`indicator - Delay(1).of(indicator)` in LEAN).

Signals, evaluated on every 1-minute bar close:

  - all_bullish / all_bearish: every slope except EMA(500) is > 0 / < 0
  - bullish_confirmed / bearish_confirmed: the *previous* 5-minute value of
    every 5-minute slope except EMA(500) was > 0 / < 0 too (two 5m bars in
    a row agreeing)
  - trend_up: EMA(500) slope > 0

  Inside the trading window (default 08:00-19:30 ET, never on Sunday):
    trend up:   all_bullish                          -> go long (reversing a short)
                all_bearish and bearish_confirmed    -> go short (counter-trend)
    trend down: all_bearish                          -> go short
                all_bullish and bullish_confirmed    -> go long (counter-trend)

  After the window (19:30 to midnight): no entries, but a position is
  closed when the same signal that would reverse it appears.

  "First trade" filter: after the daily 23:00 reset the first signal (either
  direction) is skipped, and only signals in the *opposite* direction to that
  skipped one can trade until one has. In practice: ignore the first signal
  of each day, then trade from the first opposite signal onward.

  Everything is flattened Fridays at 16:58 ET.

Faithful-port notes (behaviour inherited from the QC code, kept on purpose):

  - Positions are held overnight (it's always-in-the-market-ish, reversing
    on signals), so it's not strictly intraday despite the name. Set
    `flatten_daily_at` (see `intraday_rth_config()`) for a flat-every-day
    variant.
  - After the Friday 16:58 flatten, 16:58-17:00 is still inside the trading
    window, so a fresh signal on those last bars can re-open a position that
    is then held over the weekend. `weekend_reentries` counts these.

Not modelled: contract rolls (the data is a back-adjusted continuous
contract), margin, fills other than at the signal bar's close.
"""

from dataclasses import dataclass, field
from datetime import datetime, time, timedelta
from typing import List, Literal, Optional

from .indicators import EMA, RSI, SMA, Momentum, Slope, UltimateOscillator

FIVE = timedelta(minutes=5)
ONE = timedelta(minutes=1)


@dataclass
class ConfluenceConfig:
    trade_start: time = time(8, 0)
    trade_end: time = time(19, 30)
    no_trading_sunday: bool = True
    reset_time: time = time(23, 0)
    weekly_flatten_weekday: Optional[int] = 4  # Friday
    weekly_flatten_time: time = time(16, 58)
    # None = faithful port (hold overnight). Set a time to flatten daily at it.
    flatten_daily_at: Optional[time] = None
    skip_first_signal: bool = True


def intraday_rth_config() -> ConfluenceConfig:
    """Variant that's actually intraday: trade only 09:35-15:50 ET and be
    flat by 15:55 every day."""
    return ConfluenceConfig(
        trade_start=time(9, 35),
        trade_end=time(15, 50),
        flatten_daily_at=time(15, 55),
    )


@dataclass
class Fill:
    time: datetime  # bar-close time the order filled at
    position_before: int
    position_after: int
    price: float
    reason: str


@dataclass
class _Pending5m:
    start: datetime
    high: float
    low: float
    close: float


def _last_occurrence(t: datetime, at: time, weekday: Optional[int] = None) -> datetime:
    """Most recent datetime <= t whose wall-clock time is `at` (and, if given,
    whose weekday is `weekday`)."""
    cand = datetime.combine(t.date(), at)
    if cand > t:
        cand -= timedelta(days=1)
    if weekday is not None:
        cand -= timedelta(days=(cand.weekday() - weekday) % 7)
    return cand


class MomentumConfluenceStrategy:
    def __init__(self, config: Optional[ConfluenceConfig] = None):
        self.cfg = config or ConfluenceConfig()

        self.ema3 = EMA(3)
        self.ema10 = EMA(10)
        self.ema500 = EMA(500)
        self.rsi = RSI(12)
        self.rsi_sma = SMA(5)
        self.mom = Momentum(9)
        self.mom_ema = EMA(3)
        self.uo = UltimateOscillator(7, 14, 28)
        self.uo_ema = EMA(3)

        self.slopes = {
            "ema10": Slope(self.ema10),
            "ema500": Slope(self.ema500),
            "rsi_sma5": Slope(self.rsi_sma),
            "mom_ema_3": Slope(self.mom_ema),
            "ou_ema_3": Slope(self.uo_ema),
            "ema": Slope(self.ema3),
        }

        self.position = 0
        self.first_trade_long = 0
        self.first_trade_short = 0
        self.fills: List[Fill] = []
        self.weekend_reentries = 0

        self._pending: Optional[_Pending5m] = None
        self._prev_end: Optional[datetime] = None
        self._prev_close: Optional[float] = None
        self._flattened_this_week_at: Optional[datetime] = None

    # ---- indicator plumbing -------------------------------------------------

    def _on_5m_bar(self, b: _Pending5m) -> None:
        self.ema10.update(b.close)
        self.ema500.update(b.close)
        self.rsi.update(b.close)
        if self.rsi.is_ready:
            self.rsi_sma.update(self.rsi.value)
        self.mom.update(b.close)
        if self.mom.is_ready:
            self.mom_ema.update(self.mom.value)
        self.uo.update(b.high, b.low, b.close)
        if self.uo.is_ready:
            self.uo_ema.update(self.uo.value)
        for name, s in self.slopes.items():
            if name != "ema":
                s.update()

    def _consolidate(self, start: datetime, high: float, low: float, close: float) -> None:
        window = start - timedelta(minutes=start.minute % 5, seconds=start.second, microseconds=start.microsecond)
        p = self._pending
        if p is not None and p.start != window:
            # a gap skipped the last minute(s) of the previous window
            self._on_5m_bar(p)
            p = None
        if p is None:
            p = _Pending5m(window, high, low, close)
        else:
            p.high, p.low, p.close = max(p.high, high), min(p.low, low), close
        self._pending = p
        if start + ONE >= window + FIVE:
            self._on_5m_bar(p)
            self._pending = None

    # ---- orders -------------------------------------------------------------

    def _set_position(self, t: datetime, target: int, price: float, reason: str) -> None:
        if target == self.position:
            return
        self.fills.append(Fill(t, self.position, target, price, reason))
        self.position = target

    def _execute_long(self, t: datetime, price: float) -> None:
        if not self.cfg.skip_first_signal:
            self._set_position(t, 1, price, "long")
            return
        if self.first_trade_long == 0:
            self.first_trade_long = 1
            self.first_trade_short = 2
        if self.first_trade_long == 2:
            self._set_position(t, 1, price, "long")
            self.first_trade_short = 2

    def _execute_short(self, t: datetime, price: float) -> None:
        if not self.cfg.skip_first_signal:
            self._set_position(t, -1, price, "short")
            return
        if self.first_trade_short == 0:
            self.first_trade_short = 1
            self.first_trade_long = 2
        if self.first_trade_short == 2:
            self._set_position(t, -1, price, "short")
            self.first_trade_long = 2

    # ---- main loop ----------------------------------------------------------

    def _scheduled_events(self, end: datetime, close: float) -> None:
        prev = self._prev_end
        if prev is None:
            return

        def fired(ev: datetime) -> bool:
            return prev < ev <= end

        def price_at(ev: datetime) -> float:
            return close if ev == end else self._prev_close

        cfg = self.cfg
        ev = _last_occurrence(end, cfg.reset_time)
        if fired(ev):
            self.first_trade_long = 0
            self.first_trade_short = 0
        if cfg.weekly_flatten_weekday is not None:
            ev = _last_occurrence(end, cfg.weekly_flatten_time, cfg.weekly_flatten_weekday)
            if fired(ev):
                self._set_position(ev, 0, price_at(ev), "weekly_flatten")
                self._flattened_this_week_at = ev
        if cfg.flatten_daily_at is not None:
            ev = _last_occurrence(end, cfg.flatten_daily_at)
            if fired(ev):
                self._set_position(ev, 0, price_at(ev), "daily_flatten")

    def on_bar(self, start: datetime, open_: float, high: float, low: float, close: float) -> None:
        """Feed one 1-minute bar (`start` = bar-open wall-clock time, naive
        America/New_York). Decisions are taken at the bar's close."""
        end = start + ONE
        self._scheduled_events(end, close)

        self.ema3.update(close)
        self.slopes["ema"].update()
        self._consolidate(start, high, low, close)

        self._decide(end, close)
        self._prev_end = end
        self._prev_close = close

    def _decide(self, t: datetime, price: float) -> None:
        slopes = self.slopes
        if not all(s.is_ready for s in slopes.values()):
            return

        cfg = self.cfg
        tod = t.time()
        trading_allowed = cfg.trade_start <= tod <= cfg.trade_end and not (
            cfg.no_trading_sunday and t.weekday() == 6
        )
        after_hours = tod > cfg.trade_end

        momentum = [s.value for n, s in slopes.items() if n != "ema500"]
        all_bullish = all(v > 0 for v in momentum)
        all_bearish = all(v < 0 for v in momentum)
        prev5 = [s.previous for n, s in slopes.items() if n not in ("ema500", "ema")]
        bullish_confirmed = all(v is not None and v > 0 for v in prev5)
        bearish_confirmed = all(v is not None and v < 0 for v in prev5)
        trend_up = slopes["ema500"].value > 0

        is_long, is_short = self.position > 0, self.position < 0
        before = self.position
        if trading_allowed:
            if trend_up:
                if all_bullish and not is_long:
                    self._execute_long(t, price)
                elif all_bearish and not is_short and bearish_confirmed:
                    self._execute_short(t, price)
            else:
                if all_bearish and not is_short:
                    self._execute_short(t, price)
                elif all_bullish and not is_long and bullish_confirmed:
                    self._execute_long(t, price)
        elif after_hours:
            if trend_up:
                exit_ = (all_bullish and is_short) or (all_bearish and is_long and bearish_confirmed)
            else:
                exit_ = (all_bearish and is_long) or (all_bullish and is_short and bullish_confirmed)
            if exit_:
                self._set_position(t, 0, price, "after_hours_exit")

        if (
            before == 0
            and self.position != 0
            and self._flattened_this_week_at is not None
            and t - self._flattened_this_week_at < timedelta(hours=1)
        ):
            self.weekend_reentries += 1
