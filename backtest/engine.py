"""Event-driven backtest engine for the Failed-2s strategy."""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

from failed2s.bars import Bar
from failed2s.instruments import Instrument
from failed2s.risk import RiskManager
from failed2s.strategy import Failed2sStrategy, SessionConfig, TimeframePair

from .data import load_1m_csv, resample_ohlc


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    direction: str
    entry_price: float
    stop_price: float
    target_price: float
    exit_price: float
    exit_reason: str
    contracts: int
    pnl_points: float
    pnl_dollars: float
    r_multiple: float


class BacktestEngine:
    """
    Walks the entry-timeframe bars in order. Before processing each entry
    bar, feeds the strategy any bias-timeframe bars that have *fully closed*
    by that point (no lookahead). Position management (stop/target/session
    flatten) is checked using each subsequent entry-timeframe bar's
    high/low -- i.e. one entry-timeframe bar of latency after a signal,
    same as a real fill would have.
    """

    def __init__(
        self,
        pair: TimeframePair,
        instrument: Instrument,
        strategy: Optional[Failed2sStrategy] = None,
        risk: Optional[RiskManager] = None,
        session: Optional[SessionConfig] = None,
        contracts: int = 1,
    ):
        self.pair = pair
        self.instrument = instrument
        self.session = session or SessionConfig()
        self.strategy = strategy or Failed2sStrategy(tick_size=instrument.tick_size, session=self.session)
        self.risk = risk or RiskManager(daily_loss_limit=float("inf"), max_daily_trades=10_000)
        self.contracts = contracts

        self.trades: List[Trade] = []
        self.position: Optional[dict] = None

    def run(self, csv_path: str) -> List[Trade]:
        self.trades = []
        self.position = None

        base = load_1m_csv(csv_path, tz=self.session.tz)
        entry_df = resample_ohlc(base, self.pair.entry_tf)
        bias_df = resample_ohlc(base, self.pair.bias_tf)
        bias_period_len = pd.Timedelta(self.pair.bias_tf)
        entry_period_len = pd.Timedelta(self.pair.entry_tf)

        # Both frames are time-sorted and entry_bar_close_time only moves
        # forward, so a single advancing pointer (O(n+m) total) replaces
        # re-filtering the whole bias frame on every entry-bar iteration.
        bias_close_times = (bias_df.index + bias_period_len).to_numpy()
        bias_rows = list(bias_df.itertuples(index=True))
        bias_ptr = 0
        n_bias = len(bias_rows)

        last_ts = None
        last_close = None

        for erow in entry_df.itertuples(index=True):
            ts = erow.Index
            date = ts.date()
            entry_bar_close_time = ts + entry_period_len

            while bias_ptr < n_bias and bias_close_times[bias_ptr] <= entry_bar_close_time:
                brow = bias_rows[bias_ptr]
                self.strategy.on_bias_bar(
                    Bar(brow.Index, brow.open, brow.high, brow.low, brow.close, brow.volume)
                )
                bias_ptr += 1

            bar = Bar(ts, erow.open, erow.high, erow.low, erow.close, erow.volume)

            if self.position is not None and date != self.position["entry_time"].date():
                # A data gap spanned the flatten cutoff (e.g. a thin holiday
                # session with no bar exactly at/after flatten_at) -- force
                # closeout on the gap-open price rather than ever holding
                # into a new session day.
                self._close_position(bar.open, ts, "session_flatten")

            if self.position is not None:
                self._check_exit(bar, date)

            if self.position is not None and ts.time() >= self.session.flatten_at:
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

            last_ts, last_close = ts, erow.close

        if self.position is not None and last_ts is not None:
            self._close_position(last_close, last_ts, "end_of_data")

        return self.trades

    def _check_exit(self, bar: Bar, date) -> None:
        pos = self.position
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
