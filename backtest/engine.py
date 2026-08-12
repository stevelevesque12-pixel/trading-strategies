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
        self._bias_fed = 0

    def run(self, csv_path: str) -> List[Trade]:
        self.trades = []
        self.position = None
        self._bias_fed = 0

        base = load_1m_csv(csv_path, tz=self.session.tz)
        entry_df = resample_ohlc(base, self.pair.entry_tf)
        bias_df = resample_ohlc(base, self.pair.bias_tf)
        bias_period_len = pd.Timedelta(self.pair.bias_tf)
        entry_period_len = pd.Timedelta(self.pair.entry_tf)

        for ts, erow in entry_df.iterrows():
            date = ts.date()
            entry_bar_close_time = ts + entry_period_len

            # Feed every bias-tf bar that has fully closed as of this entry bar's close.
            closed_bias = bias_df[bias_df.index + bias_period_len <= entry_bar_close_time]
            while self._bias_fed < len(closed_bias):
                bts = closed_bias.index[self._bias_fed]
                brow = closed_bias.iloc[self._bias_fed]
                self.strategy.on_bias_bar(
                    Bar(bts, brow.open, brow.high, brow.low, brow.close, brow.volume)
                )
                self._bias_fed += 1

            bar = Bar(ts, erow.open, erow.high, erow.low, erow.close, erow.volume)

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

        if self.position is not None:
            last_ts = entry_df.index[-1]
            self._close_position(entry_df.iloc[-1].close, last_ts, "end_of_data")

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
