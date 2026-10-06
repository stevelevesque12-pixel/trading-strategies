"""Compare the Python engine against a TradingView "List of trades" export, trade by trade.

    python -m research.compare_tv --tv path/to/tv_export.csv [--start 2025-10-05 --end 2026-08-25]

Without --tv it just writes the Python trade list for the live candidate on real MES 15m
(research/compare/python_trades_mes15.csv) and prints monthly P&L, which you can line up
against TradingView's tester over the same custom date range.

TradingView's export has two rows per trade (entry + exit); columns are matched loosely by
name ("Type", "Date/Time" or "Date and time", "Price", "Contracts"/"Quantity", "Profit"/"P&L").
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from research.candidate import CANDIDATES
from trend import engine
from trend.data import load
from trend.strategy import LucidRules, backtest

OUT = Path(__file__).resolve().parent / "compare"


def python_trades(start, end, which="live"):
    spec = CANDIDATES[which][0]
    # run on the whole file so indicators are warmed up like TradingView's, then keep the
    # trades entered inside [start, end)
    m = load("mes_15m")
    tr = backtest(m, spec, LucidRules())
    t_in = m.index[tr[:, engine.T_ENTRY_I].astype(int)]
    keep = (t_in >= pd.Timestamp(start, tz=m.index.tz)) & (t_in < pd.Timestamp(end, tz=m.index.tz))
    tr = tr[np.asarray(keep)]
    df = pd.DataFrame({
        "entry_time": m.index[tr[:, engine.T_ENTRY_I].astype(int)],
        "exit_time": m.index[tr[:, engine.T_EXIT_I].astype(int)] + pd.Timedelta(minutes=m.tf_min),
        "dir": np.where(tr[:, engine.T_DIR] > 0, "long", "short"),
        "qty": tr[:, engine.T_QTY].astype(int),
        "entry_px": tr[:, engine.T_ENTRY_PX], "exit_px": tr[:, engine.T_EXIT_PX],
        "pnl": tr[:, engine.T_PNL].round(2),
        "reason": [engine.REASONS[int(r)] for r in tr[:, engine.T_REASON]],
    })
    return df


def _col(df, *names):
    for c in df.columns:
        lc = c.lower()
        if any(n in lc for n in names):
            return c
    raise KeyError(names)


def tv_trades(path):
    raw = pd.read_csv(path)
    typ, dt = _col(raw, "type"), _col(raw, "date")
    px, qty, pnl = _col(raw, "price"), _col(raw, "contracts", "quantity", "size"), _col(raw, "profit", "p&l", "net p")
    num = _col(raw, "trade #", "trade")
    rows = []
    for _, g in raw.groupby(num):
        e = g[g[typ].str.lower().str.startswith("entry")]
        x = g[g[typ].str.lower().str.startswith("exit")]
        if e.empty or x.empty:
            continue
        e, x = e.iloc[0], x.iloc[0]
        rows.append({
            "entry_time": pd.Timestamp(e[dt]).tz_localize("America/New_York", nonexistent="shift_forward", ambiguous="NaT")
            if pd.Timestamp(e[dt]).tzinfo is None else pd.Timestamp(e[dt]).tz_convert("America/New_York"),
            "dir": "long" if "long" in str(e[typ]).lower() else "short",
            "qty": int(float(str(e[qty]).replace(",", ""))),
            "entry_px": float(str(e[px]).replace(",", "")), "exit_px": float(str(x[px]).replace(",", "")),
            "pnl": float(str(x[pnl]).replace(",", "").replace("−", "-")),
            "signal": x.get(_col(raw, "signal"), ""),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tv")
    ap.add_argument("--start", default="2025-10-05")
    ap.add_argument("--end", default="2026-08-25")
    ap.add_argument("--strategy", default="live", choices=["live", "orb", "orb_vwap"])
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    py = python_trades(a.start, a.end, a.strategy)
    py.to_csv(OUT / f"python_trades_mes15_{a.strategy}.csv", index=False)
    pf = py.pnl[py.pnl > 0].sum() / -py.pnl[py.pnl <= 0].sum()
    print(f"Python {a.start}..{a.end}: {len(py)} trades, win {np.mean(py.pnl > 0):.1%}, PF {pf:.2f}, net ${py.pnl.sum():,.0f}")
    print(py.groupby(py.entry_time.dt.strftime("%Y-%m")).pnl.agg(["count", "sum"]).round(0).to_string())
    if not a.tv:
        return
    tv = tv_trades(a.tv)
    tv = tv[(tv.entry_time >= pd.Timestamp(a.start, tz="America/New_York")) & (tv.entry_time < pd.Timestamp(a.end, tz="America/New_York"))]
    pf_tv = tv.pnl[tv.pnl > 0].sum() / -tv.pnl[tv.pnl <= 0].sum()
    print(f"\nTradingView same range: {len(tv)} trades, win {np.mean(tv.pnl > 0):.1%}, PF {pf_tv:.2f}, net ${tv.pnl.sum():,.0f}")
    py = py.assign(entry_time=py.entry_time.astype("datetime64[ns, America/New_York]"))
    tv = tv.assign(entry_time=tv.entry_time.astype("datetime64[ns, America/New_York]"))
    m = pd.merge_asof(py.sort_values("entry_time"), tv.sort_values("entry_time"), on="entry_time",
                      tolerance=pd.Timedelta("16min"), direction="nearest", suffixes=("_py", "_tv"))
    matched = m.dropna(subset=["pnl_tv"])
    print(f"matched entries (within one bar): {len(matched)} of {len(py)} Python / {len(tv)} TV")
    matched = matched.assign(diff=matched.pnl_tv - matched.pnl_py)
    print("biggest P&L disagreements on matched trades:")
    print(matched.reindex(matched["diff"].abs().sort_values(ascending=False).index).head(12)[
        ["entry_time", "dir_py", "dir_tv", "qty_py", "qty_tv", "entry_px_py", "entry_px_tv", "exit_px_py", "exit_px_tv",
         "pnl_py", "pnl_tv", "reason"]].to_string(index=False))
    only_py = py[~py.entry_time.isin(matched.entry_time)]
    print(f"\nPython-only trades: {len(only_py)}; TV-only: {len(tv) - len(matched)}")
    m.to_csv(OUT / "py_vs_tv.csv", index=False)


if __name__ == "__main__":
    main()
