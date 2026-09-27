"""
Fill simulator + CLI for the Morning Dip Limit strategy.

Walks the raw base bars (1-second bars as published; 1-minute bars work as
a coarser approximation) one at a time, builds candles on the fly, and
models a resting buy limit and its bracket at base-bar resolution:

  - An order can fill on a base bar that ends after its arm time (for
    1-second bars: from the arm second on) and before it expires. It fills
    at the limit price only if the bar's low trades `trade_through_ticks`
    through it.
  - On the fill bar the position can stop out but can't reach its target
    (inside a bar the adverse move is assumed to come first).
  - After that, on every bar, in this order: time stop (exit at that bar's
    open), stop (exit at the stop, or the open if it gapped through), then
    target (exit at the target, no slippage -- it's a resting limit).
  - Stops and market exits (time stop, 11:00 flatten) slip
    `slippage_ticks`. Commission is charged per side per contract.
  - One order or position at a time: a signal that arrives while an order
    is resting or a position is open is dropped.

Usage:
    python -m morning_dip.backtest --data nq_1s.parquet --symbol NQ
    python -m morning_dip.backtest --data nq_1s.parquet --offsets 0,18,36,54,72,90,108,126,144,162

With several offsets it prints each one's metrics plus their average,
which is how the source reports its headline numbers.
"""

import argparse
from dataclasses import replace
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd

from backtest.data import load_1m_csv
from backtest.engine import Trade
from backtest.metrics import compute_metrics
from backtest.report import write_trades_csv
from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS, Instrument

from .strategy import LimitOrder, MorningDipConfig, MorningDipStrategy

NS = 1_000_000_000


def _seconds_of_day(t) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def infer_bar_seconds(index: pd.DatetimeIndex) -> int:
    """The base resolution: the most common gap between consecutive bars."""
    if len(index) < 2:
        return 1
    diffs = np.diff(index.as_unit("ns").asi8[: min(len(index), 10_000)]) // NS
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1
    values, counts = np.unique(diffs, return_counts=True)
    return int(values[np.argmax(counts)])


class MorningDipBacktest:
    def __init__(
        self,
        instrument: Instrument,
        config: Optional[MorningDipConfig] = None,
        contracts: int = 1,
        bar_seconds: Optional[int] = None,
    ):
        self.instrument = instrument
        self.config = config or MorningDipConfig()
        self.contracts = contracts
        self.bar_seconds = bar_seconds
        self.strategy = MorningDipStrategy(tick_size=instrument.tick_size, config=self.config)
        self.trades: List[Trade] = []

    def run_file(self, path: str) -> List[Trade]:
        return self.run(load_1m_csv(path, tz=self.config.tz))

    def run(self, df: pd.DataFrame) -> List[Trade]:
        """`df`: tz-aware OHLC(V) bars indexed by bar start time."""
        cfg = self.config
        tick = self.instrument.tick_size
        slip = cfg.slippage_ticks * tick
        through = cfg.trade_through_ticks * tick

        self.trades = []
        self.strategy.reset()
        if df.empty:
            return self.trades

        df = df.sort_index()
        if df.index.tz is None:
            df = df.tz_localize(cfg.tz)
        else:
            df = df.tz_convert(cfg.tz)

        df.index = df.index.as_unit("ns")  # the grid maths below assumes nanosecond timestamps

        bar_ns = (self.bar_seconds or infer_bar_seconds(df.index)) * NS
        step = cfg.candle_seconds * NS
        offset = cfg.candle_offset_seconds * NS

        idx = df.index
        ts_ns = idx.asi8
        bucket_ns = (ts_ns - offset) // step * step + offset  # UTC-grid candle starts
        tod = (idx.hour * 3600 + idx.minute * 60 + idx.second).to_numpy()
        day_ns = idx.normalize().asi8
        o, h, l, c = (df[k].to_numpy(dtype=float) for k in ("open", "high", "low", "close"))
        v = df["volume"].to_numpy(dtype=float) if "volume" in df.columns else np.zeros(len(df))

        feed_start = _seconds_of_day(cfg.feed_start)
        flatten_at = _seconds_of_day(cfg.flatten_at)
        tz = idx.tz

        def stamp(ns: int) -> pd.Timestamp:
            return pd.Timestamp(ns, unit="ns", tz="UTC").tz_convert(tz)

        day = None
        order: Optional[LimitOrder] = None
        order_arm_ns = order_expire_ns = 0
        pos: Optional[dict] = None
        candle = None  # [start_ns, open, high, low, close, volume]
        last_i = None
        day_done = False

        for i in range(len(ts_ns)):
            if day_ns[i] != day:
                if pos is not None:
                    # No bar at/after the flatten time on the previous day
                    # (a data gap) -- close on that day's last price.
                    self._close(pos, c[last_i] - slip, stamp(ts_ns[last_i]), "session_flatten")
                    pos = None
                day = day_ns[i]
                order = None
                candle = None
                day_done = False
                self.strategy.reset()

            if day_done or tod[i] < feed_start:
                continue
            last_i = i
            t_ns = ts_ns[i]

            if tod[i] >= flatten_at:
                if pos is not None:
                    self._close(pos, o[i] - slip, stamp(t_ns), "session_flatten")
                    pos = None
                order = None
                day_done = True
                continue

            # -- candle bookkeeping: a new bucket closes the previous candle
            if candle is not None and bucket_ns[i] != candle[0]:
                closed = Bar(stamp(candle[0]), candle[1], candle[2], candle[3], candle[4], candle[5])
                new_order = self.strategy.on_candle(closed, stamp(candle[0] + step))
                if new_order is not None and order is None and pos is None:
                    order = new_order
                    order_arm_ns = new_order.arm_time.value
                    order_expire_ns = new_order.expire_time.value
                candle = None
            if candle is None:
                candle = [bucket_ns[i], o[i], h[i], l[i], c[i], v[i]]
            else:
                candle[2] = max(candle[2], h[i])
                candle[3] = min(candle[3], l[i])
                candle[4] = c[i]
                candle[5] += v[i]

            # -- open position: time stop, then stop, then target
            if pos is not None:
                if t_ns >= pos["time_stop_ns"]:
                    self._close(pos, o[i] - slip, stamp(t_ns), "time_stop")
                    pos = None
                elif l[i] <= pos["stop_price"]:
                    self._close(pos, min(pos["stop_price"], o[i]) - slip, stamp(t_ns), "stop")
                    pos = None
                elif h[i] >= pos["target_price"]:
                    self._close(pos, pos["target_price"], stamp(t_ns), "target")
                    pos = None
                continue

            # -- resting order: expiry, then fill (the fill bar can stop, not target)
            if order is not None:
                if t_ns >= order_expire_ns:
                    order = None
                elif t_ns + bar_ns > order_arm_ns and l[i] <= order.limit_price - through:
                    pos = {
                        "entry_time": stamp(t_ns),
                        "entry_price": order.limit_price,
                        "stop_price": order.stop_price,
                        "target_price": order.target_price,
                        "time_stop_ns": t_ns + int(cfg.time_stop.total_seconds() * NS),
                    }
                    order = None
                    if l[i] <= pos["stop_price"]:
                        self._close(pos, min(pos["stop_price"], o[i]) - slip, stamp(t_ns), "stop")
                        pos = None

        if pos is not None and last_i is not None:
            self._close(pos, c[last_i] - slip, stamp(ts_ns[last_i]), "end_of_data")

        return self.trades

    def _close(self, pos: dict, exit_price: float, exit_time, reason: str) -> None:
        entry = pos["entry_price"]
        pnl_points = exit_price - entry
        commissions = 2 * self.config.commission_per_side * self.contracts
        pnl_dollars = pnl_points * self.instrument.point_value * self.contracts - commissions
        risk_points = entry - pos["stop_price"]
        self.trades.append(
            Trade(
                entry_time=pos["entry_time"],
                exit_time=exit_time,
                direction="long",
                entry_price=entry,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Morning Dip Limit strategy")
    parser.add_argument("--data", required=True, help="1-second (or 1-minute) OHLCV CSV/parquet")
    parser.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--candle-seconds", type=int, default=180)
    parser.add_argument("--offsets", default="0", help="comma-separated candle start offsets in seconds, e.g. 0,18,36,...,162")
    parser.add_argument("--er-max", type=float, default=0.35)
    parser.add_argument("--dip-atr", type=float, default=1.0)
    parser.add_argument("--stop-atr", type=float, default=1.5)
    parser.add_argument("--commission", type=float, default=2.25, help="per side per contract")
    parser.add_argument("--out-dir", default=".", help="where trades_<offset>s.csv files go")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    base = MorningDipConfig(
        candle_seconds=args.candle_seconds,
        er_max=args.er_max,
        dip_atr=args.dip_atr,
        stop_atr=args.stop_atr,
        commission_per_side=args.commission,
    )
    df = load_1m_csv(args.data, tz=base.tz)
    bar_seconds = infer_bar_seconds(df.index)
    offsets = [int(x) for x in args.offsets.split(",") if x.strip()]
    if bar_seconds > 1:
        print(f"Note: base bars are {bar_seconds}s, not 1s -- fills are approximate.")
        if any(off % bar_seconds for off in offsets):
            print("Note: offsets that aren't a multiple of the bar size can't be honoured exactly.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for off in offsets:
        cfg = replace(base, candle_offset_seconds=off)
        trades = MorningDipBacktest(instrument, cfg, contracts=args.contracts, bar_seconds=bar_seconds).run(df)
        write_trades_csv(trades, str(out_dir / f"trades_{off}s.csv"))
        m = compute_metrics(trades)
        rows.append(m)
        print(f"start {off:>3}s: " + "  ".join(f"{k}={val}" for k, val in m.items()))

    if len(rows) > 1:
        traded = [r for r in rows if r.get("num_trades")]
        if traded:
            avg_pnl = sum(r["total_pnl"] for r in traded) / len(rows)
            avg_n = sum(r["num_trades"] for r in traded) / len(rows)
            positive = sum(1 for r in traded if r["total_pnl"] > 0)
            print(f"average of {len(rows)} start times: net={avg_pnl:.2f}  trades={avg_n:.1f}  positive={positive}/{len(rows)}")


if __name__ == "__main__":
    main()
