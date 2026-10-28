"""
Walk-forward research for the Donchian breakout strategy.

Tunes a small parameter grid on NQ 1m (resampled to 5m) for 2023-2024,
picks the best combination by in-sample profit factor (min 200 trades),
then reports it unchanged on:
  - NQ 2025 (held out, never used for selection)
  - MNQ / MES 5m, May-Aug 2026 (a later period and a second market)

Usage:
    python -m donchian_breakout.research [--out results.csv]
"""

import argparse
import itertools
from dataclasses import replace
from pathlib import Path

import pandas as pd

from backtest.data import load_1m_csv
from backtest.metrics import compute_metrics
from failed2s.instruments import INSTRUMENTS

from .strategy import Costs, Params, simulate, to_rth_bars

DATA = Path(__file__).resolve().parent.parent / "sample_data" / "real_multi_instrument"
NQ_1M = DATA / "real_nq_1m_2022-12-26_2025-12-11.parquet"

GRID = {
    "entry_len": [10, 20, 30],
    "exit_len": [5, 10],
    "stop_atr": [1.5, 2.5],
    "vwap_filter": [False, True],
}
MIN_TRADES = 200
COSTS = {"NQ": Costs(1.0, 4.5), "MNQ": Costs(1.0, 1.5), "MES": Costs(1.0, 1.5)}


def summarize(trades, label: str, params: Params) -> dict:
    m = compute_metrics(trades)
    pts = sum(t.pnl_points for t in trades)
    row = {"label": label, **{k: getattr(params, k) for k in GRID}, **m, "total_points": round(pts, 1)}
    if trades:
        row["avg_points"] = round(pts / len(trades), 2)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None, help="Optional CSV of every grid row + out-of-sample rows")
    args = parser.parse_args()

    base = Params()
    nq = to_rth_bars(load_1m_csv(str(NQ_1M)), base)
    train = nq[(nq.index.year >= 2023) & (nq.index.year <= 2024)]
    test = nq[nq.index.year == 2025]

    rows = []
    for combo in itertools.product(*GRID.values()):
        p = replace(base, **dict(zip(GRID, combo)))
        rows.append(summarize(simulate(train, p, INSTRUMENTS["NQ"], COSTS["NQ"]), "NQ train 2023-24", p))
    grid = pd.DataFrame(rows)
    eligible = grid[grid["num_trades"] >= MIN_TRADES].sort_values("profit_factor", ascending=False)

    cols = list(GRID) + ["num_trades", "win_rate_pct", "profit_factor", "avg_points", "total_points", "max_drawdown"]
    print("In-sample grid (NQ 5m, 2023-2024), best first:")
    print(eligible[cols].to_string(index=False))

    best = replace(base, **{k: eligible.iloc[0][k] for k in GRID})
    best = replace(best, entry_len=int(best.entry_len), exit_len=int(best.exit_len),
                   stop_atr=float(best.stop_atr), vwap_filter=bool(best.vwap_filter))
    print(f"\nSelected: {({k: getattr(best, k) for k in GRID})}")

    oos = [summarize(simulate(test, best, INSTRUMENTS["NQ"], COSTS["NQ"]), "NQ test 2025", best)]
    for sym in ("MNQ", "MES"):
        bars = to_rth_bars(load_1m_csv(str(DATA / f"real_{sym.lower()}_5m_2026-05-10_2026-08-25.parquet")), best)
        oos.append(summarize(simulate(bars, best, INSTRUMENTS[sym], COSTS[sym]), f"{sym} May-Aug 2026", best))
    oos_df = pd.DataFrame(oos)
    print("\nOut-of-sample, same parameters:")
    print(oos_df[["label"] + cols[len(GRID):]].to_string(index=False))

    if args.out:
        pd.concat([grid, oos_df]).to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
