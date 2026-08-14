"""Event-driven backtester for the Fair Value Theory (FVT) strategy."""

from typing import List, Optional

from failed2s.bars import Bar
from failed2s.instruments import Instrument
from failed2s.risk import RiskManager

from fair_value.risk import contracts_for_target_risk
from fair_value.strategy import FairValueStrategy

from .data import load_1m_csv
from .engine import Trade


class FairValueBacktestEngine:
    """
    Walks 1-minute bars in order (this strategy trades a single timeframe --
    no bias/entry pair). Position management (stop/target/window flatten) is
    checked against each subsequent bar's high/low, one bar of latency after
    a signal, same convention as the Failed-2s engine.
    """

    def __init__(
        self,
        instrument: Instrument,
        strategy: Optional[FairValueStrategy] = None,
        risk: Optional[RiskManager] = None,
        target_risk: float = 1000.0,
        min_contracts: int = 1,
        max_contracts: int = 3,
        fixed_contracts: Optional[int] = None,
        tz: str = "America/New_York",
    ):
        self.instrument = instrument
        self.strategy = strategy or FairValueStrategy(tick_size=instrument.tick_size)
        self.risk = risk or RiskManager(daily_loss_limit=float("inf"), max_daily_trades=10_000)
        self.target_risk = target_risk
        self.min_contracts = min_contracts
        self.max_contracts = max_contracts
        self.fixed_contracts = fixed_contracts
        self.tz = tz

        self.trades: List[Trade] = []
        self.position: Optional[dict] = None

    def run(self, csv_path: str) -> List[Trade]:
        self.trades = []
        self.position = None

        df = load_1m_csv(csv_path, tz=self.tz)

        last_ts = None
        last_close = None

        for row in df.itertuples(index=True):
            ts = row.Index
            date = ts.date()
            bar = Bar(ts, row.open, row.high, row.low, row.close, row.volume)

            if self.position is not None and date != self.position["entry_time"].date():
                self._close_position(bar.open, ts, "session_flatten")

            if self.position is not None:
                self._check_exit(bar)

            if self.position is not None:
                window = self.position["window"]
                if any(w.name == window and ts.time() >= w.end for w in self.strategy.windows):
                    self._close_position(bar.close, ts, "window_flatten")

            signal = self.strategy.on_bar(bar)

            if signal is not None and self.position is None and self.risk.can_enter(date):
                stop_points = abs(signal.entry_price - signal.stop_price)
                contracts = self.fixed_contracts or contracts_for_target_risk(
                    stop_points,
                    self.instrument.point_value,
                    target_risk=self.target_risk,
                    min_contracts=self.min_contracts,
                    max_contracts=self.max_contracts,
                )
                self.position = {
                    "direction": signal.direction,
                    "entry_price": signal.entry_price,
                    "stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "entry_time": signal.timestamp,
                    "window": signal.window,
                    "phase": signal.phase,
                    "contracts": contracts,
                }

            last_ts, last_close = ts, row.close

        if self.position is not None and last_ts is not None:
            self._close_position(last_close, last_ts, "end_of_data")

        return self.trades

    def _check_exit(self, bar: Bar) -> None:
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
        contracts = pos["contracts"]

        sign = 1 if direction == "long" else -1
        pnl_points = (exit_price - entry_price) * sign
        pnl_dollars = pnl_points * self.instrument.point_value * contracts
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
            contracts=contracts,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=r_multiple,
            window=pos["window"],
            phase=pos["phase"],
        )
        self.trades.append(trade)
        self.risk.record_trade(exit_time.date(), pnl_dollars)
        self.position = None
