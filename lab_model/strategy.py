"""
Lab Model strategy engine -- ties together HTF premium/discount zones
(4h/1h/5m), SMT divergence (NQ vs ES), and inverse-FVG entries into the two
entry triggers described in the deck: Reversal (Entry Trigger #1) and
Continuation (Entry Trigger #2).

This is a deterministic reconstruction of a manual, visual, discretionary
trading system -- see README.md for the specific judgment calls made to
turn "identify zones that need to re-balance" and similar discretionary
language into concrete rules.
"""

from dataclasses import dataclass
from datetime import time
from typing import Literal, Optional

from failed2s.bars import Bar

from .fvg import InverseFVGTracker
from .smt import detect_smt
from .zones import ZoneTracker


@dataclass
class LabModelSession:
    entry_window_start: time = time(10, 0)
    entry_window_end: time = time(13, 0)
    flatten_at: time = time(15, 0)
    tz: str = "America/New_York"


@dataclass
class Signal:
    timestamp: object
    direction: Literal["long", "short"]
    entry_price: float
    stop_price: float
    target_price: float
    reason: str


@dataclass
class _Pending:
    direction: Literal["long", "short"]
    stop_price: float
    armed_at: object
    kind: Literal["reversal", "continuation"]
    smt_seen: bool = False
    ifvg_seen: bool = False
    bars_elapsed: int = 0


class LabModelStrategy:
    def __init__(
        self,
        tick_size: float = 0.25,
        stop_buffer_ticks: int = 2,
        swing_strength: int = 2,
        exec_swing_strength: int = 5,
        pending_expiry_bars: int = 30,
        session: Optional[LabModelSession] = None,
    ):
        self.tick_size = tick_size
        self.stop_buffer = stop_buffer_ticks * tick_size
        self.pending_expiry_bars = pending_expiry_bars
        self.session = session or LabModelSession()

        # HTF/LTF context zones, all on NQ.
        self.zone_4h = ZoneTracker(swing_strength=swing_strength)
        self.zone_1h = ZoneTracker(swing_strength=swing_strength)
        self.zone_5m = ZoneTracker(swing_strength=swing_strength)
        # Execution-timeframe zone (drives LLT targeting) and swing history for SMT.
        self.zone_exec = ZoneTracker(swing_strength=exec_swing_strength)
        # ES tracked only for its execution-timeframe swing history (SMT comparison).
        self.smt_es = ZoneTracker(swing_strength=exec_swing_strength)
        self.ifvg = InverseFVGTracker()

        # (high, low) of the 4h candle closing at 10:00, one per symbol -- NQ and ES trade on
        # different price scales, so each must sweep its *own* reference candle, not the other's.
        self._reference_candle_nq: Optional[tuple] = None
        self._reference_candle_es: Optional[tuple] = None
        self._pending: Optional[_Pending] = None
        self._current_date = None

    def reset_daily_state(self) -> None:
        """Clears per-day state (today's reference candles, any pending setup). HTF/LTF swing
        structure is NOT reset -- it's genuine continuous multi-day price structure."""
        self._reference_candle_nq = None
        self._reference_candle_es = None
        self._pending = None

    def _roll_session(self, ts) -> None:
        d = ts.date()
        if self._current_date is None:
            self._current_date = d
        elif d != self._current_date:
            self._current_date = d
            self.reset_daily_state()

    def _in_entry_window(self, ts) -> bool:
        t = ts.time()
        return self.session.entry_window_start <= t < self.session.entry_window_end

    # ---- higher-timeframe feed -----------------------------------------

    def on_htf_bar(self, tf: Literal["4h", "1h", "5m"], bar: Bar) -> None:
        self._roll_session(bar.timestamp)

        if tf == "4h":
            self.zone_4h.update(bar)
            if bar.timestamp.time() == time(6, 0):  # this bar closes at 10:00 local
                self._reference_candle_nq = (bar.high, bar.low)
        elif tf == "1h":
            self.zone_1h.update(bar)
        elif tf == "5m":
            was_balanced = self.zone_5m.is_balanced
            self.zone_5m.update(bar)
            if self.zone_5m.is_balanced and not was_balanced:
                self._arm_continuation(bar.timestamp)
        else:
            raise ValueError(f"unknown timeframe: {tf}")

    def on_es_4h_bar(self, bar: Bar) -> None:
        """ES's own 10am-closing 4h candle, tracked only as a sweep reference (no ES zone structure needed)."""
        self._roll_session(bar.timestamp)
        if bar.timestamp.time() == time(6, 0):
            self._reference_candle_es = (bar.high, bar.low)

    def _htf_continuation_direction(self) -> Optional[Literal["long", "short"]]:
        """Direction that would continue price toward a still-unfilled HTF (4h/1h) imbalance."""
        for zone in (self.zone_1h, self.zone_4h):
            leg = zone.current_leg
            if leg is None or zone.is_balanced:
                continue
            return "short" if leg.direction == "up" else "long"
        return None

    def _arm_continuation(self, ts) -> None:
        if self._pending is not None:
            return
        direction = self._htf_continuation_direction()
        if direction is None:
            return
        leg = self.zone_exec.current_leg
        if leg is None:
            return
        stop_price = (leg.high + self.stop_buffer) if direction == "short" else (leg.low - self.stop_buffer)
        self._pending = _Pending(direction=direction, stop_price=stop_price, armed_at=ts, kind="continuation")

    # ---- execution-timeframe feed --------------------------------------

    def on_execution_bars(self, nq_bar: Bar, es_bar: Bar) -> Optional[Signal]:
        """Feed one synchronized (NQ, ES) bar pair on the chosen execution timeframe (1/3/5m)."""
        self._roll_session(nq_bar.timestamp)

        self.zone_exec.update(nq_bar)
        self.smt_es.update(es_bar)
        inverted = self.ifvg.update(nq_bar)

        self._try_arm_reversal(nq_bar, es_bar)

        if self._pending is None:
            return None

        self._pending.bars_elapsed += 1
        if self._pending.bars_elapsed > self.pending_expiry_bars:
            self._pending = None
            return None

        kind = "low" if self._pending.direction == "long" else "high"
        if detect_smt(self.zone_exec.history, self.smt_es.history, kind):
            latest = [s for s in self.zone_exec.history if s.kind == kind][-1]
            if latest.timestamp >= self._pending.armed_at:
                self._pending.smt_seen = True
        if inverted is not None and inverted.inverted_direction == self._pending.direction:
            self._pending.ifvg_seen = True

        if not (self._pending.smt_seen and self._pending.ifvg_seen):
            return None
        if not self._in_entry_window(nq_bar.timestamp):
            return None

        return self._fire(nq_bar)

    def _try_arm_reversal(self, nq_bar: Bar, es_bar: Bar) -> None:
        if self._pending is not None or self._reference_candle_nq is None:
            return
        nq_high, nq_low = self._reference_candle_nq

        swept_low = nq_bar.low < nq_low
        swept_high = nq_bar.high > nq_high
        if self._reference_candle_es is not None:
            es_high, es_low = self._reference_candle_es
            swept_low = swept_low or es_bar.low < es_low
            swept_high = swept_high or es_bar.high > es_high

        # The stop always sits beyond NQ's own reference level -- that's the instrument traded.
        if swept_low and not swept_high:
            self._pending = _Pending(
                direction="long", stop_price=nq_low - self.stop_buffer, armed_at=nq_bar.timestamp, kind="reversal"
            )
        elif swept_high and not swept_low:
            self._pending = _Pending(
                direction="short", stop_price=nq_high + self.stop_buffer, armed_at=nq_bar.timestamp, kind="reversal"
            )

    def _fire(self, bar: Bar) -> Optional[Signal]:
        pending = self._pending
        self._pending = None  # consumed either way

        entry_price = bar.close
        target = self.zone_exec.llt
        if target is None:
            return None

        if pending.direction == "long":
            valid = pending.stop_price < entry_price < target
        else:
            valid = target < entry_price < pending.stop_price

        if not valid:
            return None

        return Signal(
            timestamp=bar.timestamp,
            direction=pending.direction,
            entry_price=entry_price,
            stop_price=pending.stop_price,
            target_price=target,
            reason=f"{pending.kind} direction={pending.direction} smt+ifvg stop={pending.stop_price} llt={target}",
        )
