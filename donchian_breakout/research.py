"""
Walk-forward research for the Donchian breakout strategy, across chart
timeframes and higher-timeframe (HTF) trend filters.

For every (timeframe, HTF filter) pair, a small grid of entry/exit/stop/VWAP
settings is tuned on NQ 2023-2024 (best in-sample profit factor with at least
MIN_TRADES trades). The winner is then run unchanged on data never used for
selection:
  - NQ 2025 (held out from the same dataset)
  - MNQ and MES, May-Aug 2026 (timeframes that are multiples of 5m only)
  - NQ and ES 2016-2022 (timeframes that are multiples of 15m only)

Usage:
    python -m donchian_breakout.research [--timeframes 5min 15min] [--htf none 60min] [--out results.csv]
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from functools import lru_cache
from pathlib import Path

import pandas as pd

from backtest.data import load_1m_csv
from backtest.metrics import compute_metrics
from failed2s.instruments import INSTRUMENTS

from .strategy import Costs, Params, simulate, to_rth_bars

DATA = Path(__file__).resolve().parent.parent / "sample_data" / "real_multi_instrument"
NQ_1M = DATA / "real_nq_1m_2022-12-26_2025-12-11.parquet"
MICRO_5M = {s: DATA / f"real_{s.lower()}_5m_2026-05-10_2026-08-25.parquet" for s in ("MNQ", "MES")}
LONG_15M = {"NQ": DATA / "real_nq_15m_2016-05-29_2026-08-25.parquet",
            "ES": DATA / "real_es_15m_2016-05-29_2026-08-25.parquet"}

TIMEFRAMES = ["1min", "3min", "5min", "10min", "15min", "30min"]
HTFS = ["none", "60min", "240min", "1D"]
GRID = {
    "entry_len": [10, 20, 30],
    "exit_len": [5, 10],
    "stop_atr": [1.5, 2.5],
    "vwap_filter": [False, True],
}
MIN_TRADES = 150
COSTS = {"NQ": Costs(1.0, 4.5), "ES": Costs(1.0, 4.5), "MNQ": Costs(1.0, 1.5), "MES": Costs(1.0, 1.5)}


@lru_cache(maxsize=None)
def _raw(path: Path) -> pd.DataFrame:
    return load_1m_csv(str(path))


def _bars(path: Path, tf: str, htf: str, start=None, end=None) -> pd.DataFrame:
    p = replace(Params(), timeframe=tf, htf=None if htf == "none" else htf)
    bars = to_rth_bars(_raw(path), p)
    if start is not None:
        bars = bars[(bars.index.year >= start) & (bars.index.year <= end)]
    return bars


def _stats(trades) -> dict:
    m = compute_metrics(trades)
    n = m["num_trades"]
    return {"trades": n, "win_pct": m.get("win_rate_pct"), "pf": m.get("profit_factor"),
            "avg_pts": round(sum(t.pnl_points for t in trades) / n, 2) if n else None,
            "max_dd_usd": m.get("max_drawdown")}


def _params(tf: str, htf: str, combo: dict) -> Params:
    return replace(Params(), timeframe=tf, htf=None if htf == "none" else htf, **combo)


def evaluate(tf: str, htf: str) -> dict:
    """Tune on NQ 2023-24, then report the winner on every out-of-sample set it fits."""
    train = _bars(NQ_1M, tf, htf, 2023, 2024)
    best = None
    for values in itertools.product(*GRID.values()):
        combo = dict(zip(GRID, values))
        s = _stats(simulate(train, _params(tf, htf, combo), INSTRUMENTS["NQ"], COSTS["NQ"]))
        if s["trades"] >= MIN_TRADES and (best is None or s["pf"] > best[1]["pf"]):
            best = (combo, s)
    if best is None:
        return {"timeframe": tf, "htf": htf}
    combo, train_s = best
    p = _params(tf, htf, combo)
    row = {"timeframe": tf, "htf": htf, **combo, **{f"train_{k}": v for k, v in train_s.items()}}

    oos = {"NQ_2025": (_bars(NQ_1M, tf, htf, 2025, 2025), "NQ")}
    minutes = pd.Timedelta(tf).total_seconds() / 60
    if minutes % 5 == 0:
        oos.update({f"{s}_2026": (_bars(path, tf, htf), s) for s, path in MICRO_5M.items()})
    if minutes % 15 == 0:
        oos.update({f"{s}_2016_22": (_bars(path, tf, htf, 2016, 2022), s) for s, path in LONG_15M.items()})
    for name, (bars, sym) in oos.items():
        s = _stats(simulate(bars, p, INSTRUMENTS[sym], COSTS[sym]))
        row[f"{name}_trades"], row[f"{name}_pf"], row[f"{name}_avg_pts"] = s["trades"], s["pf"], s["avg_pts"]
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--timeframes", nargs="+", default=TIMEFRAMES)
    parser.add_argument("--htf", nargs="+", default=HTFS, help="HTF filters to compare; 'none' = no filter")
    parser.add_argument("--out", default=None, help="Optional CSV of the full results table")
    parser.add_argument("--workers", type=int, default=None)
    args = parser.parse_args()

    jobs = list(itertools.product(args.timeframes, args.htf))
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        rows = list(pool.map(evaluate, *zip(*jobs)))
    df = pd.DataFrame(rows)

    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)
    setup = ["timeframe", "htf"] + list(GRID)
    print("Selected settings per (timeframe, HTF filter), tuned on NQ 2023-24:")
    print(df[setup + ["train_trades", "train_pf", "train_avg_pts", "train_max_dd_usd"]].to_string(index=False))
    pf_cols = [c for c in df.columns if c.endswith("_pf") and not c.startswith("train")]
    print("\nProfit factor, in-sample vs out-of-sample (same settings, never re-tuned):")
    print(df[["timeframe", "htf", "train_pf"] + pf_cols].to_string(index=False))
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
