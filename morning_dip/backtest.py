"""
Fill simulator + CLI for the Morning Dip Limit strategy, matched to the
author's reference implementation.

Per day: build candles from the day's base bars (08:25-15:00 CT feed), then
walk the candles in order. Each qualifying candle's order is simulated on
the base bars before the next candle is considered; a candle closing while
the previous order is still resting or its position still open is skipped.

Order (from the first bar starting at/after the arm time, until expiry):
  - fills at the limit if the bar's low trades `entry_through_ticks` through
    it (even when the bar opened below -- a pessimistic resting fill);
  - on 1-second data a feed gap over 30 s cancels it.

Position, on every bar from the fill bar on (the adverse move comes first):
  1. opened at/below the stop -> exit at the open, less slippage;
  2. (not the fill bar) opened `target_through_ticks` above the target ->
     exit at the target;
  3. (not the fill bar) at/after 11:00 CT, the 15-minute time stop, or
     after a feed gap over 30 s -> exit at the open, less slippage;
  4. low at/below the stop -> exit at the stop, less slippage;
  5. (not the fill bar) high `target_through_ticks` above the target ->
     exit at the target.
A position still open when the data ends is left unresolved (no trade) and
blocks the rest of that day.

Usage:
    python -m morning_dip.backtest --data nq_1s.parquet --symbol NQ
    python -m morning_dip.backtest --data day_files/ --offsets 0,18,36,54,72,90,108,126,144,162

`--data` is a CSV/parquet file (see backtest.data.load_1m_csv) or a
directory of per-day files with columns t (UTC epoch seconds, bar start),
o, h, l, c -- the reference implementation's format. With several offsets
it prints each one's metrics plus their average, which is how the source
reports its headline numbers.
"""

import argparse
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import List, Optional
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from backtest.data import load_1m_csv
from backtest.engine import Trade
from backtest.metrics import compute_metrics
from backtest.report import write_trades_csv
from failed2s.instruments import INSTRUMENTS, Instrument

from .strategy import EPS, LimitOrder, MorningDipConfig, build_candles, order_for


@dataclass
class _Bars:
    t: np.ndarray  # bar start, UTC epoch seconds
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray


def infer_bar_seconds(t: np.ndarray) -> int:
    """The base resolution: the most common gap between consecutive bars."""
    diffs = np.diff(t[:10_000])
    diffs = diffs[diffs > 0]
    if len(diffs) == 0:
        return 1
    values, counts = np.unique(diffs, return_counts=True)
    return int(values[np.argmax(counts)])


def load_bars(path: str, tz: str = "America/Chicago") -> pd.DataFrame:
    """A single CSV/parquet file, or a directory of per-day t/o/h/l/c files."""
    p = Path(path)
    if not p.is_dir():
        return load_1m_csv(path, tz=tz)
    frames = []
    for f in sorted(p.iterdir()):
        if f.suffix not in (".csv", ".parquet"):
            continue
        frames.append(pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f))
    raw = pd.concat(frames).sort_values("t")
    idx = pd.to_datetime(raw["t"].astype("int64"), unit="s", utc=True).dt.tz_convert(tz)
    df = pd.DataFrame(
        {"open": raw["o"].values, "high": raw["h"].values, "low": raw["l"].values, "close": raw["c"].values},
        index=pd.DatetimeIndex(idx),
    )
    df["volume"] = 0.0
    return df.astype(float)


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
        self.trades: List[Trade] = []

    def run_file(self, path: str) -> List[Trade]:
        return self.run(load_bars(path, tz=self.config.tz))

    def run(self, df: pd.DataFrame) -> List[Trade]:
        """`df`: OHLC bars indexed by bar start time (tz-aware, or naive in config.tz)."""
        cfg = self.config
        self.trades = []
        if df.empty:
            return self.trades

        df = df.sort_index()
        df = df.tz_localize(cfg.tz) if df.index.tz is None else df.tz_convert(cfg.tz)
        idx = df.index.as_unit("s")
        t_all = idx.asi8
        resolution = self.bar_seconds or infer_bar_seconds(t_all)

        tod = idx.hour * 3600 + idx.minute * 60 + idx.second
        in_feed = (tod >= _sod(cfg.feed_start)) & (tod < _sod(cfg.feed_end))
        df, idx, t_all = df[in_feed], idx[in_feed], t_all[in_feed]

        dates = idx.date
        for date in pd.unique(dates):
            m = dates == date
            day = df[m]
            bars = _Bars(
                t_all[m],
                day["open"].to_numpy(float),
                day["high"].to_numpy(float),
                day["low"].to_numpy(float),
                day["close"].to_numpy(float),
            )
            self._run_day(bars, date, resolution)
        return self.trades

    def _run_day(self, bars: _Bars, date, resolution: int) -> None:
        cfg = self.config
        tick = self.instrument.tick_size
        flat = int(datetime.combine(date, cfg.flatten_at, ZoneInfo(cfg.tz)).timestamp())
        free = -np.inf
        for candle in build_candles(bars.t, bars.o, bars.h, bars.l, bars.c, cfg, resolution):
            order = order_for(candle, cfg, tick, flat, resolution)
            if order is None or candle.end < free:
                continue
            free = self._simulate(order, bars, flat, resolution)

    def _simulate(self, order: LimitOrder, bars: _Bars, flat: int, resolution: int) -> float:
        """Run one order to its end; returns when the next order may start."""
        cfg = self.config
        tick = self.instrument.tick_size
        slip = cfg.slippage_ticks * tick
        through_entry = cfg.entry_through_ticks * tick
        through_target = cfg.target_through_ticks * tick - EPS
        max_gap = cfg.max_gap_seconds
        t, o, h, l = bars.t, bars.o, bars.h, bars.l
        limit, stop, target = order.limit_price, order.stop_price, order.target_price

        begin = int(np.searchsorted(t, order.arm))
        entry = -1
        for j in range(begin, len(t)):
            now = t[j]
            if now >= min(order.expire, flat):
                break
            if resolution == 1 and ((j > 0 and now - t[j - 1] > max_gap) or (j == begin and now - order.arm > max_gap)):
                return min(now, order.expire)  # feed gap: cancel
            if l[j] <= limit - through_entry:
                entry = j
                break
        if entry < 0:
            return order.expire

        due = t[entry] + int(cfg.time_stop.total_seconds())
        for j in range(entry, len(t)):
            now, first = t[j], j == entry
            if o[j] <= stop:
                px, why = o[j] - slip, "stop_gap"
            elif not first and o[j] - target >= through_target:
                px, why = target, "target_gap"
            elif not first and (now >= flat or now >= due or (resolution == 1 and now - t[j - 1] > max_gap)):
                px = o[j] - slip
                why = "session_flatten" if now >= flat else ("time_stop" if now >= due else "feed_gap_exit")
            elif l[j] <= stop:
                px, why = stop - slip, "stop"
            elif not first and h[j] >= target + through_target:
                px, why = target, "target"
            else:
                continue
            self._record(order, t[entry], t[j], px, why)
            return t[j] + resolution
        return np.inf  # unresolved at the end of the data

    def _record(self, order: LimitOrder, entry_t: int, exit_t: int, exit_price: float, reason: str) -> None:
        entry = order.limit_price
        pnl_points = exit_price - entry
        commissions = 2 * self.config.commission_per_side * self.contracts
        risk_points = entry - order.stop_price
        self.trades.append(
            Trade(
                entry_time=_stamp(entry_t, self.config.tz),
                exit_time=_stamp(exit_t, self.config.tz),
                direction="long",
                entry_price=entry,
                stop_price=order.stop_price,
                target_price=order.target_price,
                exit_price=exit_price,
                exit_reason=reason,
                contracts=self.contracts,
                pnl_points=pnl_points,
                pnl_dollars=pnl_points * self.instrument.point_value * self.contracts - commissions,
                r_multiple=pnl_points / risk_points if risk_points else 0.0,
            )
        )


def _sod(t) -> int:
    return t.hour * 3600 + t.minute * 60 + t.second


def _stamp(epoch: int, tz: str) -> pd.Timestamp:
    return pd.Timestamp(int(epoch), unit="s", tz="UTC").tz_convert(tz)


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Morning Dip Limit strategy")
    parser.add_argument("--data", required=True, help="1-second (or 1-minute) OHLCV CSV/parquet, or a directory of per-day t/o/h/l/c files")
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
    df = load_bars(args.data, tz=base.tz)
    bar_seconds = infer_bar_seconds(df.index.as_unit("s").asi8)
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
        avg_pnl = sum(r.get("total_pnl", 0.0) for r in rows) / len(rows)
        avg_n = sum(r["num_trades"] for r in rows) / len(rows)
        positive = sum(1 for r in rows if r.get("total_pnl", 0.0) > 0)
        print(f"average of {len(rows)} start times: net={avg_pnl:.2f}  trades={avg_n:.1f}  positive={positive}/{len(rows)}")


if __name__ == "__main__":
    main()
