"""
ATR-everything breakout: ATR sets the entry trigger, the entry filter, and
the exit -- no other indicator involved.

1. **Entry** -- every bar computes its own breakout levels from that same
   bar's own open: `long_trigger = open + entry_atr_mult * ATR_long`,
   `short_trigger = open - entry_atr_mult * ATR_long` (spec default
   entry_atr_mult=2.5, ATR_long = ATR(atr_period_long), default 20). Fires
   the instant that same bar's range reaches the level (`bar.high >=
   long_trigger` / `bar.low <= short_trigger`), filled at the exact trigger
   price -- a resting-stop-style entry, same fill convention as ORB's
   breakout triggers. If a bar's range improbably crosses both levels, long
   is checked first (documented tie-break, consistent with the rest of
   this repo). **No lookahead**: ATR_long (and ATR_short, below) is always
   the value as of the END of the PREVIOUS bar -- computed before this
   bar's own high/low/close are folded into the rolling True Range window,
   exactly matching how this would work live (you can only know the ATR as
   of the bar's open, not using data from the bar that hasn't happened
   yet).
2. **Filter** -- compares a short-term ATR (`ATR_short`,
   `atr_period_short`, default 5) to the same long-term ATR used for the
   entry level. `ATR_short < ATR_long` (volatility contraction -- "the
   market is coiling") is required for an entry to be allowed at all;
   `ATR_short >= ATR_long` (already expanding) sits out entirely, no
   entries fire that bar regardless of whether a breakout level was
   reached.
3. **Exit** -- "flip the entry math": stop = `entry_price -/+
   stop_atr_mult * ATR_long` (same ATR_long basis as the entry, default
   stop_atr_mult=0.5, i.e. a genuine fraction of a full ATR -- a
   deliberately tight, volatility-scaled stop). The spec as given only
   describes an ATR-based stop, not a profit target, so the default here
   still has **no fixed target** -- a position rides until the ATR stop is
   hit or the session flattens, a real architectural difference from every
   other strategy in this repo. Pass `target_atr_mult` (a float, e.g. 2.0)
   to opt into an ATR-scaled target instead -- `entry_price +/-
   target_atr_mult * ATR_long`, "flipping the entry math" a second time.
   This is an explicit, backtestable A/B toggle (`target_atr_mult=None`
   default vs. a chosen value), not a replacement of the original design.
4. **ATR does NOT reset per session** -- both ATR_long and ATR_short carry
   over from the prior session's trailing bars, same reasoning as ORB's
   trailing ATR (a 20-bar window reset at session open wouldn't be usable
   again until well into the session, every single day).
5. **Multiple trades per day allowed** -- a fresh breakout level exists on
   every bar (recomputed from that bar's own open), so unlike ORB's
   one-shot opening-range breakout, this can fire repeatedly through the
   session whenever flat.
6. **Intraday only** -- same entry-window/flatten-cutoff convention as the
   rest of this repo (`SessionConfig`, shared). Unlike the VWAP-based
   strategies, there's no other session-scoped state to reset daily (no
   VWAP, no opening range) -- so this strategy has no `reset()`/daily
   rollover logic at all beyond the stateless time-of-day entry-window
   check.
"""

from collections import deque
from dataclasses import dataclass
from typing import Literal, Optional

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig


class _TrueRangeATR:
    """Rolling simple-moving-average of True Range. Does NOT reset per session."""

    def __init__(self, period: int):
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
    reason: str
    target_price: Optional[float] = None  # None unless target_atr_mult is set -- see module docstring


class ATRBreakoutStrategy:
    def __init__(
        self,
        atr_period_long: int = 20,
        atr_period_short: int = 5,
        entry_atr_mult: float = 2.5,
        stop_atr_mult: float = 0.5,
        target_atr_mult: Optional[float] = None,
        session: Optional[SessionConfig] = None,
    ):
        self.atr_period_long = atr_period_long
        self.atr_period_short = atr_period_short
        self.entry_atr_mult = entry_atr_mult
        self.stop_atr_mult = stop_atr_mult
        self.target_atr_mult = target_atr_mult
        self.session = session or SessionConfig()

        self._atr_long = _TrueRangeATR(atr_period_long)
        self._atr_short = _TrueRangeATR(atr_period_short)

    def _in_entry_window(self, ts) -> bool:
        t = ts.time()
        return self.session.session_start <= t < self.session.no_entry_after

    def on_bar(self, bar: Bar) -> Optional[Signal]:
        atr_long = self._atr_long.value
        atr_short = self._atr_short.value

        signal = None
        if atr_long is not None and atr_short is not None and self._in_entry_window(bar.timestamp):
            if atr_short < atr_long:  # volatility contraction -- coiling, higher-quality breakout
                long_trigger = bar.open + self.entry_atr_mult * atr_long
                short_trigger = bar.open - self.entry_atr_mult * atr_long

                if bar.high >= long_trigger:
                    signal = self._fire(bar, "long", long_trigger, atr_long, atr_short)
                elif bar.low <= short_trigger:
                    signal = self._fire(bar, "short", short_trigger, atr_long, atr_short)

        # Update AFTER computing this bar's trigger/stop -- next bar's ATR
        # must not include this bar's own high/low/close until now.
        self._atr_long.update(bar)
        self._atr_short.update(bar)

        return signal

    def _fire(self, bar: Bar, direction: str, entry_price: float, atr_long: float, atr_short: float) -> Optional[Signal]:
        if direction == "long":
            stop_price = entry_price - self.stop_atr_mult * atr_long
            if stop_price >= entry_price:
                return None
            target_price = entry_price + self.target_atr_mult * atr_long if self.target_atr_mult is not None else None
        else:
            stop_price = entry_price + self.stop_atr_mult * atr_long
            if stop_price <= entry_price:
                return None
            target_price = entry_price - self.target_atr_mult * atr_long if self.target_atr_mult is not None else None

        return Signal(
            timestamp=bar.timestamp,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=target_price,
            reason=f"atr_short={atr_short:.4f} < atr_long={atr_long:.4f}, trigger={entry_price:.4f}",
        )
