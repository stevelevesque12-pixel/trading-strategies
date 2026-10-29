"""Backtest engine for structure_scalp.stop_entry (bias TF structure + stop-order entries)."""

from typing import List, Optional, Tuple

import pandas as pd

from failed2s.bars import Bar
from failed2s.instruments import Instrument
from failed2s.risk import RiskManager
from failed2s.strategy import SessionConfig
from structure_scalp.stop_entry import StopEntryScalpStrategy, StopOrder

from .data import load_1m_csv, resample_ohlc
from .engine import Trade


class StopEntryBacktest:
    """
    Walks entry-timeframe bars. Bias bars are fed only once fully closed.
    The strategy's order is computed on a bar's close and can only fill on
    a *later* bar (no lookahead). Fill = max(open, trigger) for longs (a gap
    through the trigger fills at the open), plus `slippage_ticks` against
    you. Stop exits also take `slippage_ticks`; targets are limit fills.

    Same-bar ambiguity is resolved conservatively: on the fill bar, if the
    bar also reaches the stop the trade is a loss; the target counts on the
    fill bar only if the stop wasn't touched (the bar's extreme in the trade
    direction must come at/after the fill, so that's safe). On later bars,
    stop is checked before target.
    """

    def __init__(
        self,
        instrument: Instrument,
        entry_tf: str = "1min",
        bias_tf: str = "12min",
        strategy: Optional[StopEntryScalpStrategy] = None,
        risk: Optional[RiskManager] = None,
        session: Optional[SessionConfig] = None,
        contracts: int = 1,
        slippage_ticks: float = 0.0,
        commission_per_side: float = 0.0,
    ):
        self.instrument = instrument
        self.entry_tf = entry_tf
        self.bias_tf = bias_tf
        self.session = session or SessionConfig()
        self.strategy = strategy or StopEntryScalpStrategy(tick_size=instrument.tick_size, session=self.session)
        self.risk = risk or RiskManager(daily_loss_limit=float("inf"), max_daily_trades=10_000)
        self.contracts = contracts
        self.slip = slippage_ticks * instrument.tick_size
        self.commission_per_side = commission_per_side
        self.trades: List[Trade] = []

    def run(self, path: str) -> List[Trade]:
        base = load_1m_csv(path, tz=self.session.tz)
        return self.run_frame(base)

    def run_frame(self, base: pd.DataFrame) -> List[Trade]:
        self.trades = []
        entry_df = base if self.entry_tf in ("1min", None) else resample_ohlc(base, self.entry_tf)
        bias_df = resample_ohlc(base, self.bias_tf)
        bias_close = (bias_df.index + pd.Timedelta(self.bias_tf)).to_numpy()
        bias_rows = list(bias_df.itertuples(index=True))
        entry_len = pd.Timedelta(self.entry_tf)
        bptr, nb = 0, len(bias_rows)

        order: Optional[StopOrder] = None
        order_date = None
        pos: Optional[dict] = None
        last = None

        for row in entry_df.itertuples(index=True):
            ts = row.Index
            bar = Bar(ts, row.open, row.high, row.low, row.close, row.volume)
            date = ts.date()

            if pos is not None and date != pos["entry_time"].date():
                self._close(pos, bar.open, ts, "session_flatten")
                pos = None

            # 1. Manage an open position, or let the resting order (placed on
            #    the previous bar's close, same day only) fill on this bar.
            if pos is not None:
                pos = self._check_exit(pos, bar)
            elif order is not None and order_date == date and self.risk.can_enter(date):
                filled, pos = self._try_fill(order, bar)  # also resolves fill-bar exits
                if filled:
                    self.strategy.mark_filled(order)
            order = None

            if pos is not None and ts.time() >= self.session.flatten_at:
                self._close(pos, bar.close, ts, "session_flatten")
                pos = None

            # 3. Feed closed bias bars (closing at or before this entry bar's close), then the entry bar.
            bar_close = ts + entry_len
            while bptr < nb and bias_close[bptr] <= bar_close:
                b = bias_rows[bptr]
                self.strategy.on_bias_bar(Bar(b.Index, b.open, b.high, b.low, b.close, b.volume))
                bptr += 1
            new_order = self.strategy.on_entry_bar(bar)
            if pos is None:
                order, order_date = new_order, date

            last = (ts, row.close)

        if pos is not None and last is not None:
            self._close(pos, last[1], last[0], "end_of_data")
        return self.trades

    def _try_fill(self, order: StopOrder, bar: Bar) -> Tuple[bool, Optional[dict]]:
        """Returns (filled, open_position); the position is None if it also exited on the fill bar."""
        if order.direction == "long":
            if bar.high < order.trigger:
                return False, None
            fill = max(bar.open, order.trigger) + self.slip
        else:
            if bar.low > order.trigger:
                return False, None
            fill = min(bar.open, order.trigger) - self.slip

        pos = {
            "direction": order.direction,
            "entry_price": fill,
            "stop_price": order.stop_price,
            "target_price": order.target_for(fill),
            "entry_time": bar.timestamp,
        }
        long = order.direction == "long"
        hit_stop = bar.low <= pos["stop_price"] if long else bar.high >= pos["stop_price"]
        hit_target = bar.high >= pos["target_price"] if long else bar.low <= pos["target_price"]
        if hit_stop:
            self._close(pos, self._stop_fill(pos), bar.timestamp, "stop")
            return True, None
        if hit_target:
            self._close(pos, pos["target_price"], bar.timestamp, "target")
            return True, None
        return True, pos

    def _stop_fill(self, pos: dict) -> float:
        return pos["stop_price"] - self.slip if pos["direction"] == "long" else pos["stop_price"] + self.slip

    def _check_exit(self, pos: dict, bar: Bar) -> Optional[dict]:
        long = pos["direction"] == "long"
        if (bar.low <= pos["stop_price"]) if long else (bar.high >= pos["stop_price"]):
            # Gap through the stop fills at the open.
            px = min(bar.open, self._stop_fill(pos)) if long else max(bar.open, self._stop_fill(pos))
            self._close(pos, px, bar.timestamp, "stop")
            return None
        if (bar.high >= pos["target_price"]) if long else (bar.low <= pos["target_price"]):
            self._close(pos, pos["target_price"], bar.timestamp, "target")
            return None
        return pos

    def _close(self, pos: dict, exit_price: float, exit_time, reason: str) -> None:
        sign = 1 if pos["direction"] == "long" else -1
        pnl_points = (exit_price - pos["entry_price"]) * sign
        pnl_dollars = pnl_points * self.instrument.point_value * self.contracts - 2 * self.commission_per_side * self.contracts
        risk_points = abs(pos["entry_price"] - pos["stop_price"])
        self.trades.append(
            Trade(
                entry_time=pos["entry_time"],
                exit_time=exit_time,
                direction=pos["direction"],
                entry_price=pos["entry_price"],
                stop_price=pos["stop_price"],
                target_price=pos["target_price"],
                exit_price=exit_price,
                exit_reason=reason,
                contracts=self.contracts,
                pnl_points=pnl_points,
                pnl_dollars=pnl_dollars,
                r_multiple=pnl_points / risk_points if risk_points else 0.0,
            )
        )
        self.risk.record_trade(exit_time.date(), pnl_dollars)
