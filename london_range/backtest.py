"""Backtest the London range strategy.

Signals come from 5-minute bars (resampled from the input). Fills and exits
are simulated on the input's own bars -- pass 1-minute data where you have it
so intrabar stop-vs-target ordering is resolved at 1m rather than 5m. Each
5m bar is handed to the strategy only once it has fully closed (no lookahead);
the resting limit order can fill from the next execution bar on.

Conservative fill conventions:
  - limit fills when price touches the entry (or trades through it, with
    --fill-through-ticks); a bar that opens beyond the limit fills at the open.
  - stop is checked before target on the same bar, and the fill bar itself
    can only stop out (we don't know whether its high came before the fill).
  - cost_points is a round-trip cost (slippage + commission) in points,
    charged on every trade.

Usage:
    python -m london_range.backtest --data sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet --symbol NQ
"""

import argparse
from collections import defaultdict
from dataclasses import replace
from datetime import time
from typing import List, Optional

import pandas as pd

from backtest.data import load_1m_csv, resample_ohlc
from backtest.engine import Trade
from backtest.metrics import compute_metrics
from backtest.report import write_trades_csv
from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS, Instrument

from .strategy import LondonRangeConfig, LondonRangeStrategy, Order, session_date

INSTRUMENTS = {**INSTRUMENTS, "MYM": Instrument("MYM", 1.0, 0.5), "M2K": Instrument("M2K", 0.1, 5.0)}


def run_backtest(
    df: pd.DataFrame,
    instrument: Instrument,
    config: Optional[LondonRangeConfig] = None,
    cost_points: float = 0.0,
    fill_through_ticks: int = 0,
    contracts: int = 1,
) -> List[Trade]:
    cfg = config or LondonRangeConfig()
    strategy = LondonRangeStrategy(cfg)
    five = resample_ohlc(df, "5min")
    five_close = (five.index + pd.Timedelta("5min")).to_numpy()
    five_rows = list(five.itertuples(index=True))
    ptr, n5 = 0, len(five_rows)
    through = fill_through_ticks * instrument.tick_size

    trades: List[Trade] = []
    order: Optional[Order] = None
    pos: Optional[dict] = None

    def close(exit_price: float, exit_time, reason: str) -> None:
        nonlocal pos
        sign = 1 if pos["direction"] == "long" else -1
        pnl_points = (exit_price - pos["entry_price"]) * sign - cost_points
        risk_points = abs(pos["entry_price"] - pos["stop_price"])
        trades.append(
            Trade(
                entry_time=pos["entry_time"],
                exit_time=exit_time,
                direction=pos["direction"],
                entry_price=pos["entry_price"],
                stop_price=pos["stop_price"],
                target_price=pos["target_price"],
                exit_price=exit_price,
                exit_reason=reason,
                contracts=contracts,
                pnl_points=pnl_points,
                pnl_dollars=pnl_points * instrument.point_value * contracts,
                r_multiple=pnl_points / risk_points if risk_points else 0.0,
            )
        )
        pos = None

    for row in df.itertuples(index=True):
        ts = row.Index
        t = ts.time()
        day = session_date(ts, cfg.range_start)

        # Hand the strategy every 5m bar that has closed by this bar's open.
        while ptr < n5 and five_close[ptr] <= ts:
            r = five_rows[ptr]
            new = strategy.on_5m_bar(Bar(r.Index, r.open, r.high, r.low, r.close, r.volume))
            if new is not None and pos is None:
                order = new
            ptr += 1

        if pos is not None and (day != pos["day"] or t >= cfg.flatten_at):
            close(row.open, ts, "session_flatten")

        if order is not None and (
            session_date(order.created_at, cfg.range_start) != day or t >= cfg.no_entry_after
        ):
            order = None  # never pulled back in time

        if order is not None:
            long = order.direction == "long"
            touched = row.low <= order.entry_price - through if long else row.high >= order.entry_price + through
            if touched:
                gap = row.open < order.entry_price if long else row.open > order.entry_price
                fill = row.open if gap else order.entry_price
                pos = {
                    "direction": order.direction,
                    "entry_price": fill,
                    "stop_price": order.stop_price,
                    "target_price": order.target_price,
                    "entry_time": ts,
                    "day": day,
                }
                order = None
                if long and row.low <= pos["stop_price"]:
                    close(min(fill, pos["stop_price"]), ts, "stop")
                elif not long and row.high >= pos["stop_price"]:
                    close(max(fill, pos["stop_price"]), ts, "stop")
                continue

        if pos is not None:
            long = pos["direction"] == "long"
            if long:
                if row.low <= pos["stop_price"]:
                    close(min(row.open, pos["stop_price"]), ts, "stop")
                elif row.high >= pos["target_price"]:
                    close(max(row.open, pos["target_price"]), ts, "target")
            else:
                if row.high >= pos["stop_price"]:
                    close(max(row.open, pos["stop_price"]), ts, "stop")
                elif row.low <= pos["target_price"]:
                    close(min(row.open, pos["target_price"]), ts, "target")

    if pos is not None:
        last = df.iloc[-1]
        close(last.close, df.index[-1], "end_of_data")
    return trades


def summarize(trades: List[Trade]) -> dict:
    m = compute_metrics(trades)
    if trades:
        m["net_r"] = round(sum(t.r_multiple for t in trades), 2)
        eq = peak = dd = 0.0
        for t in trades:
            eq += t.r_multiple
            peak = max(peak, eq)
            dd = max(dd, peak - eq)
        m["max_drawdown_r"] = round(dd, 2)
        reasons = defaultdict(int)
        for t in trades:
            reasons[t.exit_reason] += 1
        m["exits"] = dict(reasons)
    return m


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest the London range breakout/pullback strategy")
    p.add_argument("--data", required=True, help="1m or 5m OHLCV CSV/parquet")
    p.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    p.add_argument("--start", help="first date to include (YYYY-MM-DD, NY time)")
    p.add_argument("--end", help="last date to include (YYYY-MM-DD, NY time)")
    p.add_argument("--target-r", type=float, default=0.5)
    p.add_argument("--pullback-pct", type=float, default=0.125)
    p.add_argument("--no-entry-after", default="12:00")
    p.add_argument("--flatten-at", default="15:55")
    p.add_argument("--require-ema-stack", action="store_true", help="also require EMA50 above/below EMA200")
    p.add_argument("--any-breakout", action="store_true", help="a later aligned breakout can arm the order if the first wasn't aligned")
    p.add_argument("--cost-points", type=float, default=0.0, help="round-trip slippage+commission in points")
    p.add_argument("--fill-through-ticks", type=int, default=0)
    p.add_argument("--out", default="london_range_trades.csv")
    args = p.parse_args()

    df = load_1m_csv(args.data)
    if args.start:
        # keep 3 days of warmup so EMA200 on 5m is settled by the first range
        df = df[df.index >= pd.Timestamp(args.start, tz=df.index.tz) - pd.Timedelta(days=3)]
    if args.end:
        df = df[df.index < pd.Timestamp(args.end, tz=df.index.tz) + pd.Timedelta(days=1)]

    cfg = LondonRangeConfig(
        target_r=args.target_r,
        pullback_pct=args.pullback_pct,
        no_entry_after=time.fromisoformat(args.no_entry_after),
        flatten_at=time.fromisoformat(args.flatten_at),
        require_ema_stack=args.require_ema_stack,
        first_breakout_only=not args.any_breakout,
    )
    trades = run_backtest(df, INSTRUMENTS[args.symbol], cfg, args.cost_points, args.fill_through_ticks)
    if args.start:
        start = pd.Timestamp(args.start).date()
        trades = [t for t in trades if t.entry_time.date() >= start]

    write_trades_csv(trades, args.out)
    for k, v in summarize(trades).items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
