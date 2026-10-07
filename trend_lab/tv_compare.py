"""
Compare a TradingView "List of trades" CSV export of tradingview/mcl_trend_dip.pine
against the Pine-parity simulator (trend_lab/combined.py), trade by trade.

  python -m trend_lab.tv_compare path/to/export.csv [--no-scale]

Prints: entries matched / only-TV / only-sim, P&L of each group, entry-price agreement,
and the notional of sim-only trades (the first TradingView run lost every entry above
$50k notional to the default 100% margin setting).
"""

import argparse

import pandas as pd

from .combined import run
from .data import load_bars


def load_tv(path):
    tv = pd.read_csv(path)
    tv.columns = [c.strip().lstrip("﻿") for c in tv.columns]
    en = tv[tv["Type"].str.startswith("Entry")].copy()
    ex = tv[tv["Type"].str.startswith("Exit")].copy()
    en["et"] = pd.to_datetime(en["Date and time"])
    pnl = ex.groupby("Trade number")["Net PnL USD"].sum()
    en = en.drop_duplicates("Trade number").set_index("Trade number")
    en["pnl"] = pnl
    # partial exits show up as separate trade numbers with the same entry time: one position each
    pos = en.groupby("et").agg(pnl=("pnl", "sum"), px=("Price USD", "first"), sig=("Signal", "first"),
                               notional=("Size (value)", "sum"))
    return pos, ex["Signal"].value_counts().to_dict()


def pf(p):
    w, l = p[p > 0].sum(), -p[p <= 0].sum()
    return f"n={len(p)} pf={(w / l if l else float('inf')):.2f} net={p.sum():.0f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--no-scale", action="store_true", help="simulate without the Engine A scale-out")
    args = ap.parse_args()
    tv, exits = load_tv(args.csv)
    df = load_bars("15min")
    last = df.index.max().tz_localize(None)
    trades, _ = run(df, a_scale_r=None if args.no_scale else 1.0)
    sim = pd.DataFrame([(t.entry_time.tz_localize(None), t.pnl, t.entry, t.contracts * t.entry * 100)
                        for t in trades], columns=["et", "pnl", "px", "notional"]).set_index("et")
    lo = max(tv.index.min(), sim.index.min())
    tv_in, tv_new = tv[(tv.index >= lo) & (tv.index <= last)], tv[tv.index > last]
    sim = sim[sim.index >= lo]
    m = tv_in.join(sim, how="outer", lsuffix="_tv", rsuffix="_sim")
    both = m.dropna(subset=["pnl_tv", "pnl_sim"])
    only_tv, only_sim = m[m.pnl_sim.isna()], m[m.pnl_tv.isna()]
    print(f"TradingView exits by signal: {exits}")
    print(f"overlap window {lo} .. {last}")
    print(f"  TradingView: {pf(tv_in.pnl)}   simulator: {pf(sim.pnl)}")
    print(f"  matched entries {len(both)} | only TradingView {len(only_tv)} | only simulator {len(only_sim)}")
    if len(both):
        d = (both.px_tv - both.px_sim).abs()
        print(f"  entry price |diff| on matched: median {d.median():.4f}, max {d.max():.4f}")
        print(f"  matched P&L  TV {pf(both.pnl_tv)}  sim {pf(both.pnl_sim)}")
    if len(only_sim):
        big = (only_sim.notional_sim > 50000).sum()
        print(f"  only-simulator trades with notional > $50k: {big}/{len(only_sim)} (margin setting?)")
    if len(tv_new):
        print(f"after the simulator's data ends ({last}): TradingView {pf(tv_new.pnl)}")


if __name__ == "__main__":
    main()
