"""
Backtest the Step Range Breakout port on futures bars.

Fill model (the indicator itself assumes you trade at the signal bar's close):
- Entry: market order at the NEXT bar's open after the breakout close.
- Exit, `--exit close` (faithful to the indicator): market order at the
  next bar's open after a close through the trail.
- Exit, `--exit stop`: a resting stop order at the trail (the value as of
  the previous bar's close); fills at the stop, or at the open if the bar
  gaps through it. If the close crosses a trail that just ratcheted
  without the resting stop being touched, exit at the next open as above.
- Costs: `--slip-ticks` of slippage per side on every fill (stops
  included), plus `--commission` round-trip per contract.

Usage:
    python -m step_range.backtest --symbols ES,NQ,GC,CL --tf 15min,1h
"""

import argparse
import glob
from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd

from backtest.data import load_1m_csv, resample_ohlc
from backtest.metrics import compute_metrics
from failed2s.instruments import INSTRUMENTS, Instrument

from .strategy import StepRangeParams, find_breakouts

DATA_DIR = "sample_data/real_multi_instrument"
# CL has no full-size file in the repo; MCL trades at the identical price,
# so its bars are used with CL's $1000/pt point value.
DATA_SYMBOL = {"ES": "es", "NQ": "nq", "GC": "gc", "CL": "mcl", "SI": "si"}


@dataclass
class Trade:
    entry_time: object
    exit_time: object
    direction: str
    entry_price: float
    exit_price: float
    initial_stop: float
    bars_held: int
    exit_reason: str
    pnl_points: float
    pnl_dollars: float
    r_multiple: float
    indicator_win: bool  # the indicator's own definition: exit-signal close beyond entry-signal close


def load_bars(symbol: str, tf: str) -> pd.DataFrame:
    matches = glob.glob(f"{DATA_DIR}/real_{DATA_SYMBOL[symbol]}_15m_*.parquet")
    if not matches:
        raise FileNotFoundError(f"No 15m data for {symbol} in {DATA_DIR}")
    df = load_1m_csv(matches[0])
    return df if tf == "15min" else resample_ohlc(df, tf)


def simulate(
    df: pd.DataFrame,
    inst: Instrument,
    params: StepRangeParams = StepRangeParams(),
    exit_mode: str = "close",
    slip_ticks: float = 1.0,
    commission: float = 5.0,
    contracts: int = 1,
) -> List[Trade]:
    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    lo = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    idx = df.index
    n = len(df)
    slip = slip_ticks * inst.tick_size

    trades: List[Trade] = []
    for b in find_breakouts(df, params):
        s = b.signal_idx
        if s + 1 >= n:
            break
        sign = 1 if b.direction == "long" else -1
        entry = o[s + 1] + sign * slip
        last = b.exit_signal_idx if b.exit_signal_idx >= 0 else n - 1

        exit_px: Optional[float] = None
        exit_i = last
        reason = "trail_close"
        if exit_mode == "stop":
            for j in range(s + 1, last + 1):
                stop = b.trail[j - 1 - s]  # trail in force from the prior bar's close
                if sign == 1 and lo[j] <= stop:
                    exit_px, exit_i, reason = min(o[j], stop) - slip, j, "trail_stop"
                    break
                if sign == -1 and h[j] >= stop:
                    exit_px, exit_i, reason = max(o[j], stop) + slip, j, "trail_stop"
                    break
        if exit_px is None:
            if b.exit_signal_idx < 0 or last + 1 >= n:
                exit_px, exit_i, reason = c[n - 1], n - 1, "end_of_data"
            else:
                exit_px, exit_i = o[last + 1] - sign * slip, last + 1

        pts = (exit_px - entry) * sign
        risk_pts = abs(entry - b.initial_stop)
        trades.append(Trade(
            entry_time=idx[s + 1],
            exit_time=idx[exit_i],
            direction=b.direction,
            entry_price=entry,
            exit_price=exit_px,
            initial_stop=b.initial_stop,
            bars_held=exit_i - s,
            exit_reason=reason,
            pnl_points=pts,
            pnl_dollars=pts * inst.point_value * contracts - commission * contracts,
            r_multiple=pts / risk_pts if risk_pts else 0.0,
            indicator_win=bool(b.exit_signal_idx >= 0 and (c[b.exit_signal_idx] - c[s]) * sign > 0),
        ))
    return trades


def summarize(trades: List[Trade], inst: Instrument) -> dict:
    m = compute_metrics(trades)
    if not trades:
        return m
    t = pd.DataFrame([vars(x) for x in trades])
    yearly = t.groupby(pd.DatetimeIndex(t["exit_time"]).year)["pnl_dollars"].sum()
    risk_dollars = (t["entry_price"] - t["initial_stop"]).abs() * inst.point_value
    m.update(
        indicator_win_rate_pct=round(t["indicator_win"].mean() * 100, 1),
        avg_initial_risk=round(risk_dollars.mean(), 0),
        avg_bars_held=round(t["bars_held"].mean(), 1),
        long_pnl=round(t.loc[t.direction == "long", "pnl_dollars"].sum(), 0),
        short_pnl=round(t.loc[t.direction == "short", "pnl_dollars"].sum(), 0),
        years_positive=f"{int((yearly > 0).sum())}/{len(yearly)}",
        yearly=yearly.round(0).to_dict(),
    )
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="ES,NQ,GC,CL")
    ap.add_argument("--tf", default="15min,1h", help="Comma-separated bar sizes (15min native; larger ones resampled)")
    ap.add_argument("--exit", default="close,stop", help="Exit model(s): close, stop")
    ap.add_argument("--start", default=None, help="Only trade from this date (indicator still warms up on earlier bars)")
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--consolidation-bars", type=int, default=5)
    ap.add_argument("--atr-length", type=int, default=14)
    ap.add_argument("--atr-mult", type=float, default=3.0)
    ap.add_argument("--slip-ticks", type=float, default=1.0)
    ap.add_argument("--commission", type=float, default=5.0, help="Round trip, per contract")
    ap.add_argument("--yearly", action="store_true", help="Print per-year net P&L")
    ap.add_argument("--trades-out", default=None, help="Write every trade to this CSV")
    args = ap.parse_args()

    params = StepRangeParams(args.length, args.consolidation_bars, args.atr_length, args.atr_mult)
    rows, all_trades = [], []
    for sym in [s.strip().upper() for s in args.symbols.split(",")]:
        inst = INSTRUMENTS[sym]
        for tf in args.tf.split(","):
            df = load_bars(sym, tf)
            for mode in args.exit.split(","):
                trades = simulate(df, inst, params, mode, args.slip_ticks, args.commission)
                if args.start:
                    start = pd.Timestamp(args.start, tz=df.index.tz)
                    trades = [t for t in trades if t.entry_time >= start]
                m = summarize(trades, inst)
                span = f"{(trades[0].entry_time if trades else df.index[0]):%Y-%m}..{df.index[-1]:%Y-%m}"
                rows.append({"symbol": sym, "tf": tf, "exit": mode, "span": span, **m})
                all_trades += [{"symbol": sym, "tf": tf, "exit": mode, **vars(t)} for t in trades]

    cols = ["symbol", "tf", "exit", "span", "num_trades", "indicator_win_rate_pct", "win_rate_pct",
            "profit_factor", "total_pnl", "expectancy_per_trade", "avg_r", "max_drawdown",
            "avg_initial_risk", "avg_bars_held", "long_pnl", "short_pnl", "years_positive"]
    table = pd.DataFrame(rows)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(table[[c for c in cols if c in table]].to_string(index=False))
        if args.yearly:
            print()
            for r in rows:
                print(f"{r['symbol']:>3} {r['tf']:>5} {r['exit']:>5}  " +
                      "  ".join(f"{y}:{v:+,.0f}" for y, v in r.get("yearly", {}).items()))
    if args.trades_out:
        pd.DataFrame(all_trades).to_csv(args.trades_out, index=False)
        print(f"\nTrades: {args.trades_out}")


if __name__ == "__main__":
    main()
