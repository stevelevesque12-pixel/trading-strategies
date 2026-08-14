"""Streaming Average True Range (Wilder's smoothing), no lookahead."""

from collections import deque
from typing import Optional

from failed2s.bars import Bar


class RollingATR:
    """
    ATR over the same bars the strategy trades (1-minute, per the PDF's
    stated general parameters), seeded with a simple average of the first
    `period` true ranges and Wilder-smoothed after that. `update()` folds in
    the bar passed to it and returns the ATR *including* that bar -- safe to
    call at a signal bar's close since by then its full OHLC is known.
    """

    def __init__(self, period: int = 14):
        self.period = period
        self._trs: deque = deque(maxlen=period)
        self._prev_close: Optional[float] = None
        self._atr: Optional[float] = None

    def update(self, bar: Bar) -> Optional[float]:
        if self._prev_close is None:
            tr = bar.high - bar.low
        else:
            tr = max(
                bar.high - bar.low,
                abs(bar.high - self._prev_close),
                abs(bar.low - self._prev_close),
            )
        self._prev_close = bar.close
        self._trs.append(tr)

        if len(self._trs) < self.period:
            self._atr = None
        elif self._atr is None:
            self._atr = sum(self._trs) / self.period
        else:
            self._atr = (self._atr * (self.period - 1) + tr) / self.period

        return self._atr
