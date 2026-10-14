"""Robustness search on NQ 15m 2016-2026 (chart TF = 15m, approximate port).

The 1m file only covers Dec 2022 onward, so it never sees the 2018 Q4, 2020
or 2022 selloffs. Here the chart is 15m: EMA1 is a plain 15m EMA and the 1m
EMA2 length is converted to 15m bars (len/15). Coarser fills than the 1m
test, so use it to rank risk, not to predict exact P&L.

    python -m ppo_dd.stress15
"""

import itertools
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace

import numpy as np
import pandas as pd

from . import data
from .optimize import (COMMISSION, MULTS, POINT_VALUE, SLIP, Params, csv, evaluate, summarize)
from .sim import R_NET, simulate

NQ_15M = "sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet"
_c = {}


def _df():
    if "df" not in _c:
        df = pd.read_parquet(NQ_15M).tz_convert("America/New_York")
        df.columns = [c.lower() for c in df.columns]
        _c["df"] = df[["open", "high", "low", "close"]].astype(float)
        _c["arr"] = data.arrays(_c["df"])
    return _c["df"]


def _ema(n):
    key = ("e", n)
    if key not in _c:
        _c[key] = _df()["close"].ewm(span=n, adjust=False).mean().to_numpy()
    return _c[key]


def run15(p: Params, start, end=None):
    df = _df()
    o, h, l, c, t, day = _c["arr"]
    idx = df.index
    s = int(idx.searchsorted(pd.Timestamp(start, tz=idx.tz)))
    e = len(idx) if end is None else int(idx.searchsorted(pd.Timestamp(end, tz=idx.tz)))
    return simulate(o, h, l, c, t, day, _ema(p.len1), _ema(max(2, round(p.len2 / 15))),
                    p.direction, p.thr1, p.thr2, p.entry_qty, p.base_tp / 100, p.max_dd_times - 1,
                    p.sl_perc / 100, p.sl_at_max, p.tp_catch,
                    csv(p.dd_csv, 10) / 100 * p.dd_scale, csv(p.ddtp_csv, 10) / 100, csv(p.mf_csv, 10),
                    np.full(11, p.decay_base), np.full(11, p.min_tp / 100), p.one_day,
                    POINT_VALUE, COMMISSION, SLIP, s, e, np.zeros((0, 6)))


def evaluate15(p: Params):
    a = summarize(run15(p, "2016-06-01", "2023-01-01"))
    b = summarize(run15(p, "2023-01-01"))
    worst_year = min(run15(p, f"{y}-01-01", f"{y + 1}-01-01")[R_NET] for y in range(2016, 2023))
    return {**asdict(p), **{f"IS_{k}": v for k, v in a.items()}, "IS_worstYear": round(worst_year),
            **{f"OOS_{k}": v for k, v in b.items()}}


def _many(ps):
    return [evaluate15(p) for p in ps]


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    space = {
        "thr1": [1.0075, 1.01, 1.015],
        "thr2": [1.0025, 1.005],
        "len2": [100, 300],
        "base_tp": [0.135, 0.2, 0.3, 0.5],
        "dd_scale": [0.75, 1.0, 1.5, 2.0],
        "max_dd_times": [4, 5, 6, 7],
        "mf_csv": list(MULTS.values()),
        "sl_perc": [3.0, 5.0, 8.0],
    }
    base = Params(len1=300)
    ps = [replace(base, **dict(zip(space, combo))) for combo in itertools.product(*space.values())]
    w = os.cpu_count() or 1
    with ProcessPoolExecutor(w) as ex:
        rows = [r for part in ex.map(_many, [ps[i::w * 4] for i in range(w * 4)]) for r in part]
    df = pd.DataFrame(rows)
    df["mf_csv"] = df["mf_csv"].map({v: k for k, v in MULTS.items()})
    df.to_csv("ppo_dd/results_stress15.csv", index=False)
    show = ["mf_csv", "thr1", "thr2", "len2", "base_tp", "dd_scale", "max_dd_times", "sl_perc",
            "IS_net", "IS_maxDD", "IS_ret/dd", "IS_worstYear", "IS_trades", "IS_SLs", "IS_maxQty",
            "OOS_net", "OOS_maxDD", "OOS_ret/dd", "OOS_SLs"]
    ok = df[df["IS_trades"] >= 100].sort_values("IS_ret/dd", ascending=False)
    print(ok[show].head(30).to_string(index=False))
    print("\nBest per ladder:")
    print(ok.groupby("mf_csv").head(3)[show].to_string(index=False))
