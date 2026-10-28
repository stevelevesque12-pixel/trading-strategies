"""
Sweep lookback lengths, stop sizes and reward:risk targets for the Donchian
breakout on the 5m chart, with and without the 4h trend filter.

Exit types:
  trail       -- trailing exit channel only (exit_len=10), no target
  RR x        -- fixed target at x times the stop distance, no trailing exit
  RR x+trail  -- both: whichever comes first

Every setup is scored in R (profit / initial risk, net of costs), so NQ and
micro results can be pooled: in-sample = NQ 2023-24, out-of-sample = NQ 2025
+ MNQ + MES May-Aug 2026.

Usage:
    python -m donchian_breakout.sweep_rr [--out sweep.csv]
"""

import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
import pandas as pd

from failed2s.instruments import INSTRUMENTS

from .research import COSTS, MICRO_5M, NQ_1M, _bars
from .strategy import Params, simulate

ENTRY_LENS = [3, 5, 10, 15, 20, 30, 40]
STOP_ATRS = [1.0, 1.5, 2.0, 3.0]
TARGETS = [1.0, 1.5, 2.0, 3.0, 4.0]
EXITS = [("trail", None, True)] + [(f"RR {r:g}", r, False) for r in TARGETS] + [(f"RR {r:g}+trail", r, True) for r in TARGETS]
HTFS = ["none", "240min"]


def _score(trades) -> dict:
    r = np.array([t.r_multiple for t in trades])
    if not len(r):
        return {"n": 0, "avg_r": np.nan, "win_pct": np.nan, "pf": np.nan}
    loss = -r[r < 0].sum()
    return {"n": len(r), "avg_r": round(r.mean(), 3), "win_pct": round((r > 0).mean() * 100, 1),
            "pf": round(r[r > 0].sum() / loss, 2) if loss else np.inf}


def run_block(htf: str, entry_len: int) -> list:
    sets = {"train": (_bars(NQ_1M, "5min", htf, 2023, 2024), "NQ"),
            "nq25": (_bars(NQ_1M, "5min", htf, 2025, 2025), "NQ"),
            **{s.lower(): (_bars(path, "5min", htf), s) for s, path in MICRO_5M.items()}}
    rows = []
    for stop_atr, (exit_name, target_r, trail) in itertools.product(STOP_ATRS, EXITS):
        p = replace(Params(), htf=None if htf == "none" else htf, entry_len=entry_len, exit_len=10,
                    stop_atr=stop_atr, target_r=target_r, use_trail=trail, vwap_filter=False)
        trades = {k: simulate(b, p, INSTRUMENTS[sym], COSTS[sym]) for k, (b, sym) in sets.items()}
        oos = trades["nq25"] + trades["mnq"] + trades["mes"]
        row = {"htf": htf, "entry_len": entry_len, "stop_atr": stop_atr, "exit": exit_name}
        row.update({f"is_{k}": v for k, v in _score(trades["train"]).items()})
        row.update({f"oos_{k}": v for k, v in _score(oos).items()})
        for k in ("nq25", "mnq", "mes"):
            row[f"{k}_avg_r"] = _score(trades[k])["avg_r"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    # One process per (htf, entry_len) block keeps all cores busy.
    with ProcessPoolExecutor() as pool:
        jobs = list(itertools.product(HTFS, ENTRY_LENS))
        rows = [r for block in pool.map(run_block, *zip(*jobs)) for r in block]
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 50)

    order = [e[0] for e in EXITS]
    for metric in ("is_avg_r", "oos_avg_r"):
        label = "In-sample NQ 2023-24" if metric.startswith("is") else "Out-of-sample (NQ 2025 + MNQ/MES 2026)"
        print(f"\n{label}: avg R per trade, averaged over the 4 stop sizes and both HTF settings")
        print(df.pivot_table(index="entry_len", columns="exit", values=metric, aggfunc="mean")[order].round(3).to_string())

    print("\nAvg R by stop size (all lookbacks/exits/HTF):")
    print(df.groupby("stop_atr")[["is_avg_r", "oos_avg_r"]].mean().round(3).to_string())
    print("\nAvg R by HTF filter:")
    print(df.groupby("htf")[["is_avg_r", "oos_avg_r"]].mean().round(3).to_string())

    cols = ["htf", "entry_len", "stop_atr", "exit", "is_n", "is_avg_r", "is_pf", "oos_n", "oos_avg_r", "oos_pf",
            "nq25_avg_r", "mnq_avg_r", "mes_avg_r"]
    top = df[df["is_n"] >= 300].sort_values("is_avg_r", ascending=False).head(15)
    print("\nTop 15 by in-sample avg R (>= 300 trades), with their out-of-sample results:")
    print(top[cols].to_string(index=False))
    print(f"\nShare of {len(df)} setups with positive OOS avg R: {(df['oos_avg_r'] > 0).mean():.0%}")
    print(f"Correlation of in-sample vs OOS avg R across setups: {df['is_avg_r'].corr(df['oos_avg_r']):.2f}")
    if args.out:
        df.to_csv(args.out, index=False)
        print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
