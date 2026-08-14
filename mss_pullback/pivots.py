"""
Valid-pivot tracking: a reconstruction of the "Valid Pullbacks Lite"
(promuckaj, pullback type 2) indicator's rule for marking valid highs/lows.

A candidate high/low is the running most-extreme-since-last-resolution
value. It only becomes a CONFIRMED valid high/low once a later bar's
*close* clears the candidate candle's own opposite wick -- a full-body
sweep, not just a touch. Ties keep whichever candle is harder to
invalidate (the lower low, for a tied high; the higher high, for a tied
low). Valid highs and valid lows are tracked independently and don't have
to alternate.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from failed2s.bars import Bar


@dataclass
class Pivot:
    timestamp: object
    price: float


class PivotTracker:
    def __init__(self):
        self._cand_high: Optional[Bar] = None
        self._cand_low: Optional[Bar] = None
        self.valid_highs: List[Pivot] = []
        self.valid_lows: List[Pivot] = []

    def update(self, bar: Bar) -> Tuple[Optional[Pivot], Optional[Pivot]]:
        """Feed one bar. Returns (confirmed_high, confirmed_low) for this bar, if any."""
        if (
            self._cand_high is None
            or bar.high > self._cand_high.high
            or (bar.high == self._cand_high.high and bar.low < self._cand_high.low)
        ):
            self._cand_high = bar

        if (
            self._cand_low is None
            or bar.low < self._cand_low.low
            or (bar.low == self._cand_low.low and bar.high > self._cand_low.high)
        ):
            self._cand_low = bar

        confirmed_high = None
        # A candidate can't confirm on the very bar that just set/updated it.
        if self._cand_high is not None and bar is not self._cand_high and bar.close < self._cand_high.low:
            confirmed_high = Pivot(self._cand_high.timestamp, self._cand_high.high)
            self.valid_highs.append(confirmed_high)
            self._cand_high = None

        confirmed_low = None
        if self._cand_low is not None and bar is not self._cand_low and bar.close > self._cand_low.high:
            confirmed_low = Pivot(self._cand_low.timestamp, self._cand_low.low)
            self.valid_lows.append(confirmed_low)
            self._cand_low = None

        return confirmed_high, confirmed_low

    @property
    def candidate_high(self) -> Optional[float]:
        """The current running (unconfirmed) candidate high -- the trailing leg peak post-MSS."""
        return self._cand_high.high if self._cand_high is not None else None

    @property
    def candidate_low(self) -> Optional[float]:
        return self._cand_low.low if self._cand_low is not None else None


def window_extreme(anchor: List[Pivot], target: List[Pivot], want_max: bool) -> Optional[float]:
    """
    Highest/lowest `target` value whose pivot sits between the two most
    recent `anchor` pivots. want_max=True -> bullish MSS level (highest
    valid high between the last two valid lows); want_max=False -> bearish
    MSS level (lowest valid low between the last two valid highs).
    """
    if len(anchor) < 2:
        return None
    t1, t2 = anchor[-2].timestamp, anchor[-1].timestamp
    candidates = [p.price for p in target if t1 < p.timestamp < t2]
    if not candidates:
        return None
    return max(candidates) if want_max else min(candidates)


def nearest_above(pivots: List[Pivot], ref: float) -> Optional[float]:
    above = [p.price for p in pivots if p.price > ref]
    return min(above) if above else None


def nearest_below(pivots: List[Pivot], ref: float) -> Optional[float]:
    below = [p.price for p in pivots if p.price < ref]
    return max(below) if below else None
