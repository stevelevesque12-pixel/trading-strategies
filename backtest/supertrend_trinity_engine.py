"""Event-driven backtest engine for the Trinity SuperTrend strategy.

Three-tier multi-timeframe engine (the entry timeframe driving ST1, plus
two higher timeframes for ST2/ST3), generalizing the two-tier bias/entry
merge pattern in backtest/engine.py to three tiers via two independent
advancing pointers. Feeds ST2/ST3 bars that have fully closed by each
entry-timeframe bar's close time -- same no-lookahead convention as the
rest of this repo.

Unlike every other engine here, session flattening is OPTIONAL: this
strategy is a swing system by design (see strategy.py's module docstring)
and is not force-flattened unless the strategy was given a `session`
(SessionConfig) -- in which case flatten/gap handling works exactly like
the other engines.

Position management also differs in supporting a TRAILING stop
(`strategy.trail_stop()`, called every entry-timeframe bar a position is
open when `strategy.trailing` is True) alongside the usual fixed
stop/target.
"""

from typing import List, Optional

import pandas as pd

from failed2s.bars import Bar
from failed2s.instruments import Instrument
from failed2s.risk import RiskManager
from supertrend_trinity.strategy import TrinitySuperTrendStrategy

from .data import load_1m_csv, resample_ohlc
from .engine import Trade


class TrinitySuperTrendBacktestEngine:
    def __init__(
        self,
        tf1: str,
        tf2: str,
        tf3: str,
        instrument: Instrument,
        strategy: Optional[TrinitySuperTrendStrategy] = None,
        risk: Optional[RiskManager] = None,
        contracts: int = 1,
        tz: str = "America/New_York",
    ):
        self.tf1 = tf1
        self.tf2 = tf2
        self.tf3 = tf3
        self.instrument = instrument
        self.strategy = strategy or TrinitySuperTrendStrategy()
        self.risk = risk or RiskManager(daily_loss_limit=float("inf"), max_daily_trades=10_000)
        self.contracts = contracts
        self.tz = tz

        self.trades: List[Trade] = []
        self.position: Optional[dict] = None

    def run(self, csv_path: str) -> List[Trade]:
        self.trades = []
        self.position = None

        base = load_1m_csv(csv_path, tz=self.tz)
        df1 = resample_ohlc(base, self.tf1)
        df2 = resample_ohlc(base, self.tf2)
        df3 = resample_ohlc(base, self.tf3)

        period1 = pd.Timedelta(self.tf1)
        period2 = pd.Timedelta(self.tf2)
        period3 = pd.Timedelta(self.tf3)

        close_times2 = (df2.index + period2).to_numpy()
        close_times3 = (df3.index + period3).to_numpy()
        rows2 = list(df2.itertuples(index=True))
        rows3 = list(df3.itertuples(index=True))
        ptr2 = 0
        ptr3 = 0
        n2, n3 = len(rows2), len(rows3)

        session = self.strategy.session  # None => no forced flatten (see module docstring)

        last_ts = None
        last_close = None

        for row in df1.itertuples(index=True):
            ts = row.Index
            date = ts.date()
            entry_bar_close = ts + period1

            while ptr2 < n2 and close_times2[ptr2] <= entry_bar_close:
                r = rows2[ptr2]
                self.strategy.on_tf2_bar(Bar(r.Index, r.open, r.high, r.low, r.close, r.volume))
                ptr2 += 1
            while ptr3 < n3 and close_times3[ptr3] <= entry_bar_close:
                r = rows3[ptr3]
                self.strategy.on_tf3_bar(Bar(r.Index, r.open, r.high, r.low, r.close, r.volume))
                ptr3 += 1

            bar = Bar(ts, row.open, row.high, row.low, row.close, row.volume)

            if session is not None and self.position is not None and date != self.position["entry_time"].date():
                self._close_position(bar.open, ts, "session_flatten")

            if self.position is not None:
                self._check_exit(bar)

            if session is not None and self.position is not None and ts.time() >= session.flatten_at:
                self._close_position(bar.close, ts, "session_flatten")

            signal = self.strategy.on_entry_bar(bar)

            if signal is not None and self.position is None and self.risk.can_enter(date):
                self.position = {
                    "direction": signal.direction,
                    "entry_price": signal.entry_price,
                    "stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "entry_time": signal.timestamp,
                }

            last_ts, last_close = ts, row.close

        if self.position is not None and last_ts is not None:
            self._close_position(last_close, last_ts, "end_of_data")

        return self.trades

    def _check_exit(self, bar: Bar) -> None:
        pos = self.position
        if self.strategy.trailing:
            pos["stop_price"] = self.strategy.trail_stop(bar, pos["direction"], pos["stop_price"])

        if pos["direction"] == "long":
            if bar.low <= pos["stop_price"]:
                self._close_position(pos["stop_price"], bar.timestamp, "stop")
            elif bar.high >= pos["target_price"]:
                self._close_position(pos["target_price"], bar.timestamp, "target")
        else:
            if bar.high >= pos["stop_price"]:
                self._close_position(pos["stop_price"], bar.timestamp, "stop")
            elif bar.low <= pos["target_price"]:
                self._close_position(pos["target_price"], bar.timestamp, "target")

    def _close_position(self, exit_price: float, exit_time, reason: str) -> None:
        pos = self.position
        direction = pos["direction"]
        entry_price = pos["entry_price"]
        stop_price = pos["stop_price"]

        sign = 1 if direction == "long" else -1
        pnl_points = (exit_price - entry_price) * sign
        pnl_dollars = pnl_points * self.instrument.point_value * self.contracts
        risk_points = abs(entry_price - stop_price)
        r_multiple = pnl_points / risk_points if risk_points else 0.0

        trade = Trade(
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            stop_price=stop_price,
            target_price=pos["target_price"],
            exit_price=exit_price,
            exit_reason=reason,
            contracts=self.contracts,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=r_multiple,
        )
        self.trades.append(trade)
        self.risk.record_trade(exit_time.date(), pnl_dollars)
        self.position = None
