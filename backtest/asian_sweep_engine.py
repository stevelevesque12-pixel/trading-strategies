"""Event-driven backtest engine for the Asian-range liquidity-sweep strategy.

Unlike `engine.py` (Failed-2s's two-timeframe bias/entry split), this walks a
single resampled bar stream -- the box, sweep/break, MSS confirmation, and
retest entry are all evaluated off the same bars via `AsianSweepStrategy.on_bar`,
gated by time-of-day session windows instead of a second higher timeframe.
Position management (stop/target/session flatten) follows the same one-bar-
latency, stop-fills-first-on-overlap convention as `engine.py`.
"""

from typing import List, Optional

from asian_sweep.session import SessionConfig
from asian_sweep.strategy import AsianSweepStrategy
from failed2s.bars import Bar
from failed2s.instruments import Instrument
from failed2s.risk import RiskManager

from .data import load_1m_csv, resample_ohlc
from .engine import Trade


class AsianSweepEngine:
    def __init__(
        self,
        instrument: Instrument,
        strategy: Optional[AsianSweepStrategy] = None,
        risk: Optional[RiskManager] = None,
        session: Optional[SessionConfig] = None,
        entry_tf: str = "15min",
        contracts: int = 1,
    ):
        self.instrument = instrument
        self.session = session or SessionConfig()
        self.strategy = strategy or AsianSweepStrategy(tick_size=instrument.tick_size, session=self.session)
        self.risk = risk or RiskManager(daily_loss_limit=float("inf"), max_daily_trades=10_000)
        self.entry_tf = entry_tf
        self.contracts = contracts

        self.trades: List[Trade] = []
        self.position: Optional[dict] = None

    def run(self, csv_path: str) -> List[Trade]:
        self.trades = []
        self.position = None

        base = load_1m_csv(csv_path, tz=self.session.tz)
        entry_df = resample_ohlc(base, self.entry_tf)

        last_ts = None
        last_close = None

        for erow in entry_df.itertuples(index=True):
            ts = erow.Index
            date = ts.date()
            bar = Bar(ts, erow.open, erow.high, erow.low, erow.close, erow.volume)

            if self.position is not None and date != self.position["entry_time"].date():
                # A data gap spanned the flatten cutoff -- force closeout on
                # the gap-open price rather than ever holding into a new day.
                self._close_position(bar.open, ts, "session_flatten")

            if self.position is not None:
                self._check_exit(bar, date)

            if self.position is not None and ts.time() >= self.session.flatten_at:
                self._close_position(bar.close, ts, "session_flatten")

            signal = self.strategy.on_bar(bar)

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
