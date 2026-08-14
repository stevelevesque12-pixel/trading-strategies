"""Two-instrument (NQ + ES), multi-timeframe backtest engine for the Lab Model strategy."""

from typing import List, Optional

import pandas as pd

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS, Instrument

from lab_model.strategy import LabModelSession, LabModelStrategy
from lab_model.zones import four_hour_origin

from .data import load_1m_csv, resample_ohlc
from .engine import Trade

# (source dataframe rule label, strategy timeframe label, resample rule)
_HTF_CONFIG = [("4h", "4h"), ("1h", "1h"), ("5m", "5min")]


class LabModelEngine:
    """
    Walks the execution-timeframe bars (NQ+ES, synchronized) in order. Before
    each execution bar, feeds the strategy any 4h/1h/5m NQ bars that have
    fully closed by that point (no lookahead) -- same pattern as
    `backtest.engine.BacktestEngine`. Position management (stop/target/
    breakeven/session flatten) is checked on subsequent execution bars, one
    bar of latency after a signal, same as a real fill would have.
    """

    def __init__(
        self,
        strategy: Optional[LabModelStrategy] = None,
        instrument: Optional[Instrument] = None,
        session: Optional[LabModelSession] = None,
        execution_tf: str = "1min",
        contracts: int = 1,
        breakeven_at_r: Optional[float] = 0.5,
    ):
        self.session = session or LabModelSession()
        self.strategy = strategy or LabModelStrategy(session=self.session)
        self.instrument = instrument or INSTRUMENTS["NQ"]
        self.execution_tf = execution_tf
        self.contracts = contracts
        self.breakeven_at_r = breakeven_at_r

        self.trades: List[Trade] = []
        self.position: Optional[dict] = None

    def run(self, nq_csv: str, es_csv: str) -> List[Trade]:
        self.trades = []
        self.position = None

        nq_1m = load_1m_csv(nq_csv, tz=self.session.tz)
        es_1m = load_1m_csv(es_csv, tz=self.session.tz)

        origin = four_hour_origin(nq_1m)
        nq_htf = {
            "4h": resample_ohlc(nq_1m, "4h", origin=origin),
            "1h": resample_ohlc(nq_1m, "1h"),
            "5m": resample_ohlc(nq_1m, "5min"),
        }
        es_4h = resample_ohlc(es_1m, "4h", origin=four_hour_origin(es_1m))
        nq_exec = resample_ohlc(nq_1m, self.execution_tf)
        es_exec = resample_ohlc(es_1m, self.execution_tf)

        common_idx = nq_exec.index.intersection(es_exec.index).sort_values()
        nq_exec = nq_exec.loc[common_idx]
        es_exec = es_exec.loc[common_idx]
        exec_period_len = pd.Timedelta(self.execution_tf)

        # Each entry feeds NQ's own on_htf_bar(tf, bar), except the ES 4h stream, which feeds
        # the dedicated on_es_4h_bar (ES needs only its own reference candle, not full zones).
        htf_streams = []
        stream_config = [(tf_label, rule, nq_htf[tf_label], self.strategy.on_htf_bar, tf_label) for tf_label, rule in _HTF_CONFIG]
        stream_config.append(("es_4h", "4h", es_4h, lambda _tf, bar: self.strategy.on_es_4h_bar(bar), None))
        for name, rule, df, feed_fn, tf_arg in stream_config:
            period_len = pd.Timedelta(rule)
            close_times = (df.index + period_len).to_numpy()
            rows = list(df.itertuples(index=True))
            htf_streams.append(
                {"feed_fn": feed_fn, "tf_arg": tf_arg, "rows": rows, "close_times": close_times, "ptr": 0, "n": len(rows)}
            )

        last_ts = None
        last_close = None

        for erow, es_row in zip(nq_exec.itertuples(index=True), es_exec.itertuples(index=True)):
            ts = erow.Index
            date = ts.date()
            exec_bar_close_time = ts + exec_period_len

            for stream in htf_streams:
                while stream["ptr"] < stream["n"] and stream["close_times"][stream["ptr"]] <= exec_bar_close_time:
                    brow = stream["rows"][stream["ptr"]]
                    stream["feed_fn"](
                        stream["tf_arg"], Bar(brow.Index, brow.open, brow.high, brow.low, brow.close, brow.volume)
                    )
                    stream["ptr"] += 1

            nq_bar = Bar(ts, erow.open, erow.high, erow.low, erow.close, erow.volume)
            es_bar = Bar(ts, es_row.open, es_row.high, es_row.low, es_row.close, es_row.volume)

            if self.position is not None and date != self.position["entry_time"].date():
                self._close_position(nq_bar.open, ts, "session_flatten")

            if self.position is not None:
                self._check_exit(nq_bar)

            if self.position is not None and ts.time() >= self.session.flatten_at:
                self._close_position(nq_bar.close, ts, "session_flatten")

            signal = self.strategy.on_execution_bars(nq_bar, es_bar)

            if signal is not None and self.position is None:
                self.position = {
                    "direction": signal.direction,
                    "entry_price": signal.entry_price,
                    "stop_price": signal.stop_price,
                    "initial_stop_price": signal.stop_price,
                    "target_price": signal.target_price,
                    "entry_time": signal.timestamp,
                    "breakeven_done": False,
                }

            last_ts, last_close = ts, erow.close

        if self.position is not None and last_ts is not None:
            self._close_position(last_close, last_ts, "end_of_data")

        return self.trades

    def _check_exit(self, bar: Bar) -> None:
        pos = self.position
        if pos["direction"] == "long":
            if bar.low <= pos["stop_price"]:
                self._close_position(pos["stop_price"], bar.timestamp, "stop")
                return
            if bar.high >= pos["target_price"]:
                self._close_position(pos["target_price"], bar.timestamp, "target")
                return
        else:
            if bar.high >= pos["stop_price"]:
                self._close_position(pos["stop_price"], bar.timestamp, "stop")
                return
            if bar.low <= pos["target_price"]:
                self._close_position(pos["target_price"], bar.timestamp, "target")
                return

        self._maybe_breakeven(bar, pos)

    def _maybe_breakeven(self, bar: Bar, pos: dict) -> None:
        if pos["breakeven_done"] or self.breakeven_at_r is None:
            return
        entry, target = pos["entry_price"], pos["target_price"]
        halfway = entry + (target - entry) * self.breakeven_at_r
        if pos["direction"] == "long":
            if bar.high >= halfway:
                pos["stop_price"] = max(pos["stop_price"], entry)
                pos["breakeven_done"] = True
        else:
            if bar.low <= halfway:
                pos["stop_price"] = min(pos["stop_price"], entry)
                pos["breakeven_done"] = True

    def _close_position(self, exit_price: float, exit_time, reason: str) -> None:
        pos = self.position
        direction = pos["direction"]
        entry_price = pos["entry_price"]

        sign = 1 if direction == "long" else -1
        pnl_points = (exit_price - entry_price) * sign
        pnl_dollars = pnl_points * self.instrument.point_value * self.contracts
        risk_points = abs(entry_price - pos["initial_stop_price"])
        r_multiple = pnl_points / risk_points if risk_points else 0.0

        trade = Trade(
            entry_time=pos["entry_time"],
            exit_time=exit_time,
            direction=direction,
            entry_price=entry_price,
            stop_price=pos["stop_price"],
            target_price=pos["target_price"],
            exit_price=exit_price,
            exit_reason=reason,
            contracts=self.contracts,
            pnl_points=pnl_points,
            pnl_dollars=pnl_dollars,
            r_multiple=r_multiple,
        )
        self.trades.append(trade)
        self.position = None
