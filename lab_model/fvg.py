"""
Fair Value Gap (FVG) and Inverse FVG (iFVG) tracking.

An FVG is the classic 3-candle imbalance (reusing `failed2s.structure.detect_fvg`).
An "inverse FVG" per the deck is "FVG with candle closure through it": a
previously-formed FVG that price later closes fully through, flipping it
from a continuation zone into a reversal zone --

  - a bullish FVG (support) that gets closed *below* inverts into a
    bearish (short) signal
  - a bearish FVG (resistance) that gets closed *above* inverts into a
    bullish (long) signal
"""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Literal, Optional

from failed2s.bars import Bar
from failed2s.structure import detect_fvg


@dataclass
class FVGZone:
    kind: Literal["bullish", "bearish"]
    low: float
    high: float
    formed_at: object
    inverted: bool = False

    @property
    def inverted_direction(self) -> Literal["long", "short"]:
        return "short" if self.kind == "bullish" else "long"


class InverseFVGTracker:
    """Feed bars in order; `update` returns the FVGZone that just inverted on this bar, if any."""

    def __init__(self, max_open: int = 20):
        self._buf: Deque[Bar] = deque(maxlen=3)
        self.open_fvgs: List[FVGZone] = []
        self.max_open = max_open

    def update(self, bar: Bar) -> Optional[FVGZone]:
        inverted = None
        for z in self.open_fvgs:
            if z.inverted:
                continue
            if z.kind == "bullish" and bar.close < z.low:
                z.inverted = True
                inverted = z
                break
            if z.kind == "bearish" and bar.close > z.high:
                z.inverted = True
                inverted = z
                break

        self._buf.append(bar)
        if len(self._buf) == 3:
            b1, b2, b3 = self._buf
            fvg = detect_fvg(b1, b2, b3)
            if fvg == "bullish":
                self.open_fvgs.append(FVGZone("bullish", low=b1.high, high=b3.low, formed_at=b3.timestamp))
            elif fvg == "bearish":
                self.open_fvgs.append(FVGZone("bearish", low=b3.high, high=b1.low, formed_at=b3.timestamp))
            if len(self.open_fvgs) > self.max_open:
                self.open_fvgs = self.open_fvgs[-self.max_open:]

        return inverted
