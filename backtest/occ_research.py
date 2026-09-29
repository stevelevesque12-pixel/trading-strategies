"""Honest (non-repainting) backtest of the Open Close Cross strategy.

Companion to `tradingview/occ_nonrepainting.pine`. The signal is computed
on CLOSED higher-timeframe bars and filled at the next bar's open with
commission + slippage -- no lookahead. Sweeps a parameter grid per symbol
and reports how many configs are profitable, including a 60/40
in-sample/out-of-sample split so a lucky parameter pick doesn't pass as
an edge.

Usage:
    python -m backtest.occ_research                   # default sweep
    python -m backtest.occ_research --symbols MCL MNQ
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

from failed2s.instruments import INSTRUMENTS

from .data import load_1m_csv, resample_ohlc

DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "real_multi_instrument"

# Full-size 10-year datasets are sized as the micro contract, so dollar
# results are comparable with the micro-only datasets.
MICRO_OF = {"ES": "MES", "NQ": "MNQ", "GC": "MGC", "CL": "MCL"}
EXTRA_SPECS = {"SI": (0.005, 1000.0)}  # sized as 1000oz micro silver

COMMISSION_RT = 2.0  # $ per contract round trip
SLIPPAGE_TICKS = 2   # per side, matching the R5.1 trade list


def _smma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = (out[i - 1] * (n - 1) + x[i]) / n
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    return pd.Series(x).ewm(span=n, adjust=False).mean().values


def moving_average(kind: str, x: np.ndarray, n: int) -> np.ndarray:
    if kind == "SMMA":
        return _smma(x, n)
    if kind == "EMA":
        return _ema(x, n)
    if kind == "DEMA":
        e = _ema(x, n)
        return 2 * e - _ema(e, n)
    raise ValueError(kind)


def run_occ(bars: pd.DataFrame, tick: float, point_value: float, kind: str = "SMMA", length: int = 8,
            session: tuple[int, int] | None = None, trend: int = 0,
            slippage_ticks: int = SLIPPAGE_TICKS, commission_rt: float = COMMISSION_RT) -> tuple[pd.Series, int]:
    """Per-bar net P&L (1 contract) and number of position changes.

    `bars` are the strategy-resolution bars (America/New_York index). The
    position is decided at bar i's close and held from bar i+1's open.
    Flat from 16:55 ET to midnight (end-of-day flatten, no entries), and
    outside `session` (minutes after midnight ET) when given.
    """
    o, c = bars.open.values, bars.close.values
    up = moving_average(kind, c, length) > moving_average(kind, o, length)
    prev = np.r_[False, up[:-1]]
    xl, xs = up & ~prev, ~up & prev
    xl[0] = xs[0] = False

    idx = bars.index
    step = idx[1] - idx[0] if len(idx) > 1 else pd.Timedelta(minutes=15)
    start_min = idx.hour * 60 + idx.minute
    close_t = idx + step
    close_min = close_t.hour * 60 + close_t.minute
    past_eod = (close_min >= 16 * 60 + 55) | (close_min < start_min)
    in_sess = np.ones(len(bars), bool) if session is None else (start_min >= session[0]) & (start_min < session[1])
    if trend:
        t = _ema(c, trend)
        ok_long, ok_short = c > t, c < t
    else:
        ok_long = ok_short = np.ones(len(bars), bool)

    pos = np.zeros(len(bars))
    p = 0
    for i in range(len(bars)):
        if past_eod[i] or not in_sess[i]:
            p = 0
        elif xl[i]:
            p = 1 if ok_long[i] else 0
        elif xs[i]:
            p = -1 if ok_short[i] else 0
        pos[i] = p

    held = np.r_[0, pos[:-1]]
    next_open = np.r_[o[1:], c[-1]]
    gross = held * (next_open - o) * point_value
    changes = np.abs(np.diff(np.r_[0, held]))
    cost = changes * (commission_rt / 2 + slippage_ticks * tick * point_value)
    return pd.Series(gross - cost, index=idx), int((changes > 0).sum())


def sweep(symbol: str, path: Path) -> pd.DataFrame:
    if symbol in EXTRA_SPECS:
        tick, pv = EXTRA_SPECS[symbol]
    else:
        inst = INSTRUMENTS[MICRO_OF.get(symbol, symbol)]
        tick, pv = inst.tick_size, inst.point_value
    base = load_1m_csv(str(path))
    rows = []
    grid = itertools.product(["15min", "30min", "60min"], ["SMMA", "EMA", "DEMA"], [4, 8, 14, 21],
                             [None, (9 * 60 + 30, 15 * 60 + 45)], [0, 200])
    frames = {}
    for tf, kind, n, sess, trend in grid:
        bars = frames.setdefault(tf, resample_ohlc(base, tf))
        pnl, trades = run_occ(bars, tick, pv, kind, n, sess, trend)
        cut = bars.index[int(len(bars) * 0.6)]
        rows.append(dict(tf=tf, ma=f"{kind}{n}", rth=sess is not None, trend=trend, trades=trades,
                         net=round(pnl.sum()), in_sample=round(pnl[pnl.index < cut].sum()),
                         out_sample=round(pnl[pnl.index >= cut].sum())))
    return pd.DataFrame(rows).sort_values("net", ascending=False)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbols", nargs="*", default=["MCL", "MES", "MNQ", "MGC", "ES", "NQ", "GC", "SI"])
    args = ap.parse_args()
    for sym in args.symbols:
        files = sorted(DATA_DIR.glob(f"real_{sym.lower()}_15m_*.parquet"))
        if not files:
            print(f"{sym}: no 15m dataset in {DATA_DIR}")
            continue
        r = sweep(sym, files[0])
        both = ((r.in_sample > 0) & (r.out_sample > 0)).sum()
        print(f"\n{sym} ({files[0].name}): {len(r)} configs | net>0: {(r.net > 0).sum()} | "
              f"profitable in AND out of sample: {both} | median net ${r.net.median():.0f}")
        print(r.head(3).to_string(index=False))


if __name__ == "__main__":
    main()
