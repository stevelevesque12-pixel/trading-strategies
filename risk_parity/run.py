"""
Backtest the micro-only prop version on the repo's real datasets.

    python -m risk_parity.run                      # both datasets, default config
    python -m risk_parity.run --sweep              # also sweep risk / trend / weekly stop

Two datasets, because the micros only have ~11 months of history here:
  - "micro": real MES, MNQ, MCL, MGC 15m bars, Sep 2025 - Aug 2026.
  - "proxy": ES, NQ, GC 15m bars 2016-2026 priced at micro point values
    (same underlying price moves). No long crude history exists in
    sample_data/, so MCL is absent from this run.
"""

import argparse
import glob
from dataclasses import replace

import pandas as pd

from backtest.data import load_1m_csv
from risk_parity.strategy import (PropRules, RiskParityConfig, eval_stats,
                                  run_backtest, summary)

DATA_DIR = "sample_data/real_multi_instrument"


def _load(sym_file: str) -> pd.DataFrame:
    (path,) = glob.glob(f"{DATA_DIR}/real_{sym_file}_15m_*.parquet")
    return load_1m_csv(path)


def load_micro():
    return {s: _load(s.lower()) for s in ("MES", "MNQ", "MCL", "MGC")}


def load_proxy():
    return {"MES": _load("es"), "MNQ": _load("nq"), "MGC": _load("gc")}


def report(name, data, cfg, rules):
    res = run_backtest(data, cfg)
    invested = res.rebalances.drop(columns="time").sum(axis=1) > 0
    first = res.rebalances.loc[invested, "time"].min() if invested.any() else None
    first_day = pd.Timestamp(first.date()) if first is not None else None
    s = summary(res, first_day)
    start = res.eod_pnl.index.searchsorted(first_day) if first_day is not None else 0
    e = eval_stats(res.eod_pnl, rules, first=max(start - 1, 0))
    return res, {"run": name, **{k: round(v, 2) if isinstance(v, float) else v for k, v in {**s, **e}.items()}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--target", type=float, default=4000.0)
    ap.add_argument("--trailing-dd", type=float, default=2500.0)
    args = ap.parse_args()
    rules = PropRules(args.target, args.trailing_dd)
    pd.set_option("display.width", 250, "display.max_columns", 30)

    datasets = {"micro 2025-26": load_micro(), "proxy 2016-26": load_proxy()}
    base = RiskParityConfig()
    rows = []
    for name, data in datasets.items():
        res, row = report(name, data, base, rules)
        rows.append(row)
        yearly = res.eod_pnl.resample("YE").last().diff()
        yearly.iloc[0] = res.eod_pnl.resample("YE").last().iloc[0]
        print(f"\n{name}: $ P&L by year\n{yearly.round(0).to_string()}")
        held = res.rebalances.drop(columns="time")
        print(f"{name}: share of weeks holding each symbol\n{(held > 0).mean().round(2).to_string()}")
        print(f"{name}: avg contracts when held\n{held[held > 0].mean().round(1).to_string()}")
    print("\nDefault config:", base)
    print(pd.DataFrame(rows).set_index("run").T.to_string())

    if args.sweep:
        variants = {
            "risk $600/mkt, $1200 total": replace(base, risk_per_market_usd=600, max_total_risk_usd=1200),
            "risk $1500/mkt, $3000 total": replace(base, risk_per_market_usd=1500, max_total_risk_usd=3000),
            "stop 1.5 ATR": replace(base, stop_atr=1.5),
            "no trend filter": replace(base, trend_len=0),
            "trend 50d": replace(base, trend_len=50),
            "no weekly stop": replace(base, weekly_stop_usd=None),
            "stop 3 ATR": replace(base, stop_atr=3.0),
            "both equities": replace(base, one_equity_slot=False, max_positions=4),
        }
        rows = []
        for name, data in datasets.items():
            for vname, cfg in variants.items():
                rows.append(report(f"{name} | {vname}", data, cfg, rules)[1])
        cols = ["pnl_per_year", "sharpe", "max_eod_dd", "worst_week", "pct_weeks_invested",
                "pass", "fail", "open", "median_sessions_to_pass"]
        print("\nSweep:\n" + pd.DataFrame(rows).set_index("run")[cols].to_string())


if __name__ == "__main__":
    main()
