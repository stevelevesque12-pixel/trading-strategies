"""
3m-15s market structure pullback strategy.

Python port of tradingview/mss_pullback.pine -- see that file's header for
the full rule description. Summary (long case; short is the exact mirror):

  1. 3m bias flips bullish on a close through the highest confirmed valid
     high sitting between the two most recent confirmed valid lows (a
     Market Structure Shift). A fresh valid point that just extends the
     existing trend only needs a wick, not a close.
  2. While 3m bias is bullish, Entry 1 is a resting buy stop at the current
     15s MSS level, repriced every bar the window changes. When price
     trades through it, Entry 1 fills and that is the 15s bullish MSS.
  3. The leg's 0% is the most recently confirmed 15s valid low at that
     moment. Its 100% is the running (unconfirmed) candidate high post-MSS,
     trailing upward as price prints fresh peaks. Entry 2 is a buy limit at
     50% of that leg, repricing upward with the peak, adding to the same
     position if it fills.
  4. Entry 2 is cancelled only if a candidate high actually CONFIRMS before
     it fills -- not merely because price ticks to a new high.
  5. Both entries share one stop (the leg low) and one target (the nearest
     3m valid high still ahead of price, floored at 1:1 against the stop
     distance).
  6. Once the trade closes, a fresh 15s MSS is a new tradeable setup as
     long as 3m bias hasn't flipped away.

Position sizing is risk-based in USD, with combined risk across BOTH
entries capped at the configured budget (split via `entry1_risk_share`,
default 50/50 -- an assumption, not something pinned down precisely;
tune it if a different split is wanted).

Not backtested against real data -- no free source provides 15-second
history, same gap as this repo's structure_scalp strategy. Validated here
against synthetic bar sequences only (see tests/test_mss_pullback.py).
"""

from dataclasses import dataclass
from typing import List, Literal, Optional, Union

from failed2s.bars import Bar
from failed2s.instruments import Instrument

from .pivots import PivotTracker, nearest_above, nearest_below, window_extreme


@dataclass
class Fill:
    timestamp: object
    order: Literal["entry1", "entry2"]
    direction: Literal["long", "short"]
    price: float
    qty: int
    stop_price: float
    target_price: float


@dataclass
class Exit:
    timestamp: object
    direction: Literal["long", "short"]
    price: float
    reason: Literal["stop", "target"]
    qty: int


Event = Union[Fill, Exit]


class StructureTracker:
    """Persisting bias for one timeframe, driven by the valid-pivot MSS rule."""

    def __init__(self):
        self.pivots = PivotTracker()
        self.direction: Optional[Literal["long", "short"]] = None

    def update(self, bar: Bar) -> None:
        self.pivots.update(bar)
        bull_level = window_extreme(self.pivots.valid_lows, self.pivots.valid_highs, want_max=True)
        bear_level = window_extreme(self.pivots.valid_highs, self.pivots.valid_lows, want_max=False)
        if bull_level is not None and bar.close > bull_level:
            self.direction = "long"
        if bear_level is not None and bar.close < bear_level:
            self.direction = "short"


def _qty_for_risk(risk_share_usd: float, entry_price: float, stop_price: float, point_value: float, max_contracts: int) -> int:
    """Like webhook.sizing.contracts_for_risk, but returns 0 (not an error) rather than raising
    when the distance isn't priceable yet -- callers use 0 to mean "don't place this order"."""
    dist = abs(entry_price - stop_price)
    if dist <= 0:
        return 0
    risk_per_contract = dist * point_value
    if risk_per_contract <= 0:
        return 0
    return max(1, min(int(risk_share_usd // risk_per_contract), max_contracts))


class MSSPullbackStrategy:
    def __init__(
        self,
        instrument: Instrument,
        risk_usd: float = 200.0,
        entry1_risk_share: float = 0.5,
        max_contracts: int = 10,
    ):
        self.instrument = instrument
        self.risk_usd = risk_usd
        self.entry1_risk_share = entry1_risk_share
        self.max_contracts = max_contracts

        self.struct_3m = StructureTracker()
        self.pivots_15s = PivotTracker()

        self.direction: Optional[Literal["long", "short"]] = None  # None while flat
        self.stop_price: Optional[float] = None
        self.target_price: Optional[float] = None
        self._entry1_qty = 0
        self._entry2_qty = 0
        self._entry2_price: Optional[float] = None
        self._entry2_cancelled = False
        self._entry2_filled = False

    def on_3m_bar(self, bar: Bar) -> None:
        self.struct_3m.update(bar)

    @property
    def entry1_price(self) -> Optional[float]:
        """The resting Entry 1 level for the current 3m bias, if computable, while flat."""
        if self.struct_3m.direction == "long":
            return window_extreme(self.pivots_15s.valid_lows, self.pivots_15s.valid_highs, want_max=True)
        if self.struct_3m.direction == "short":
            return window_extreme(self.pivots_15s.valid_highs, self.pivots_15s.valid_lows, want_max=False)
        return None

    @property
    def entry2_price(self) -> Optional[float]:
        """The resting Entry 2 level while a trade is open and it hasn't filled/been cancelled."""
        return self._entry2_price

    def on_15s_bar(self, bar: Bar) -> List[Event]:
        confirmed_high, confirmed_low = self.pivots_15s.update(bar)

        if self.direction is None:
            return self._try_entry1(bar)
        return self._manage_open_trade(bar, confirmed_high, confirmed_low)

    def _try_entry1(self, bar: Bar) -> List[Event]:
        bias = self.struct_3m.direction
        if bias == "long":
            level = window_extreme(self.pivots_15s.valid_lows, self.pivots_15s.valid_highs, want_max=True)
            if level is not None and self.pivots_15s.valid_lows and bar.high >= level:
                leg_low = self.pivots_15s.valid_lows[-1].price
                qty = _qty_for_risk(self.risk_usd * self.entry1_risk_share, level, leg_low, self.instrument.point_value, self.max_contracts)
                if qty > 0:
                    return self._open_trade(bar, "long", level, qty, leg_low)
        elif bias == "short":
            level = window_extreme(self.pivots_15s.valid_highs, self.pivots_15s.valid_lows, want_max=False)
            if level is not None and self.pivots_15s.valid_highs and bar.low <= level:
                leg_high = self.pivots_15s.valid_highs[-1].price
                qty = _qty_for_risk(self.risk_usd * self.entry1_risk_share, level, leg_high, self.instrument.point_value, self.max_contracts)
                if qty > 0:
                    return self._open_trade(bar, "short", level, qty, leg_high)
        return []

    def _open_trade(self, bar: Bar, direction: Literal["long", "short"], fill_price: float, qty: int, stop_anchor: float) -> List[Event]:
        self.direction = direction
        self.stop_price = stop_anchor
        self._entry1_qty = qty
        self._entry2_qty = 0
        self._entry2_price = None
        self._entry2_cancelled = False
        self._entry2_filled = False

        if direction == "long":
            raw_tp = nearest_above(self.struct_3m.pivots.valid_highs, fill_price)
            floor_tp = fill_price + (fill_price - stop_anchor)
            self.target_price = floor_tp if raw_tp is None else max(raw_tp, floor_tp)
        else:
            raw_tp = nearest_below(self.struct_3m.pivots.valid_lows, fill_price)
            floor_tp = fill_price - (stop_anchor - fill_price)
            self.target_price = floor_tp if raw_tp is None else min(raw_tp, floor_tp)

        return [Fill(bar.timestamp, "entry1", direction, fill_price, qty, self.stop_price, self.target_price)]

    def _manage_open_trade(self, bar: Bar, confirmed_high, confirmed_low) -> List[Event]:
        events: List[Event] = []

        # Entry 2 cancellation takes priority over a same-bar fill -- a
        # genuine confirmed reversal kills the shallow-pullback thesis
        # outright, even if price also touched 50% this same bar.
        if not self._entry2_filled and not self._entry2_cancelled:
            if self.direction == "long" and confirmed_high is not None:
                self._entry2_cancelled = True
            elif self.direction == "short" and confirmed_low is not None:
                self._entry2_cancelled = True

        if not self._entry2_filled and not self._entry2_cancelled:
            fill = self._try_entry2(bar)
            if fill is not None:
                events.append(fill)

        exit_event = self._check_exit(bar)
        if exit_event is not None:
            events.append(exit_event)

        return events

    def _try_entry2(self, bar: Bar) -> Optional[Fill]:
        if self.direction == "long":
            peak = self.pivots_15s.candidate_high
            if peak is None or peak <= self.stop_price:
                return None
            price = self.stop_price + 0.5 * (peak - self.stop_price)
            self._entry2_price = price
            qty = _qty_for_risk(self.risk_usd * (1 - self.entry1_risk_share), price, self.stop_price, self.instrument.point_value, self.max_contracts)
            if qty > 0 and bar.low <= price:
                self._entry2_qty = qty
                self._entry2_filled = True
                return Fill(bar.timestamp, "entry2", "long", price, qty, self.stop_price, self.target_price)
        else:
            trough = self.pivots_15s.candidate_low
            if trough is None or trough >= self.stop_price:
                return None
            price = self.stop_price - 0.5 * (self.stop_price - trough)
            self._entry2_price = price
            qty = _qty_for_risk(self.risk_usd * (1 - self.entry1_risk_share), price, self.stop_price, self.instrument.point_value, self.max_contracts)
            if qty > 0 and bar.high >= price:
                self._entry2_qty = qty
                self._entry2_filled = True
                return Fill(bar.timestamp, "entry2", "short", price, qty, self.stop_price, self.target_price)
        return None

    def _check_exit(self, bar: Bar) -> Optional[Exit]:
        # Conservative, matching this repo's backtest engine convention: if
        # a bar's range could hit both stop and target, the stop wins.
        if self.direction == "long":
            if bar.low <= self.stop_price:
                return self._close(bar, "stop")
            if bar.high >= self.target_price:
                return self._close(bar, "target")
        else:
            if bar.high >= self.stop_price:
                return self._close(bar, "stop")
            if bar.low <= self.target_price:
                return self._close(bar, "target")
        return None

    def _close(self, bar: Bar, reason: Literal["stop", "target"]) -> Exit:
        price = self.stop_price if reason == "stop" else self.target_price
        event = Exit(bar.timestamp, self.direction, price, reason, self._entry1_qty + self._entry2_qty)
        self.direction = None
        self.stop_price = None
        self.target_price = None
        self._entry1_qty = 0
        self._entry2_qty = 0
        self._entry2_price = None
        self._entry2_cancelled = False
        self._entry2_filled = False
        return event
