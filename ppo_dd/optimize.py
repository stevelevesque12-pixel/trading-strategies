"""Parameter search for the PPO - DD strategy on NQ 1m.

    python -m ppo_dd.optimize            # baseline + grid search + OOS check

In-sample: 2023-01-01 .. 2024-12-31. Out-of-sample: 2025-01-01 .. end.
Ranking uses in-sample only; the OOS columns are there to catch overfitting.
"""

import itertools
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, replace

import numpy as np
import pandas as pd

from . import data
from .sim import (N_RESULTS, R_GROSS_L, R_GROSS_W, R_MAXDD, R_MAXDD_COUNT, R_MAXQTY, R_NET, R_SL,
                  R_TRADES, R_WINS, simulate)

POINT_VALUE = 20.0      # NQ
COMMISSION = 2.50       # $ per contract per side (incl. exchange fees)
SLIP = 0.25             # points on market/stop fills (1 tick)

IS_START, IS_END = "2023-01-01", "2025-01-01"
OOS_START = "2025-01-01"


def csv(s, n):
    v = [float(x) for x in s.split(",")]
    return np.array((v + [0.0] * n)[:n])


@dataclass(frozen=True)
class Params:
    direction: int = 1
    thr1: float = 1.05
    len1: int = 150
    tf1: int = 15            # minutes; 1 = chart TF
    thr2: float = 1.05
    len2: int = 300
    tf2: int = 1
    entry_qty: int = 1
    base_tp: float = 0.135   # %
    max_dd_times: int = 7    # Pine input (DDs = this - 1)
    sl_perc: float = 4.0     # %
    sl_at_max: bool = True
    tp_catch: bool = True
    dd_csv: str = "0.618, 1.05, 1.35, 1.75, 2.2, 3.2, 0, 0, 0, 0"
    dd_scale: float = 1.0    # multiplies every DD distance
    ddtp_csv: str = "0.02, 0.05, 0.05, 0.05, 0.05, 0.05, 0, 0, 0, 0"
    mf_csv: str = "2, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2, 2"
    decay_base: float = 0.0  # minutes, applied to every slot (0 = off)
    min_tp: float = 0.01     # %
    one_day: bool = False


_cache = {}

# Named multiplier ladders. Each DD adds round(mult * contracts-so-far), so
# "script" grows ~x2.5-3 per level, "double" x2, "x1.5" x1.5, "linear" +1.
MULTS = {
    "script": "2, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2, 2",
    "double": "1, 1, 1, 1, 1, 1, 1, 1, 1, 1",
    "x1.5": "0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5",
    "linear": "0, 0, 0, 0, 0, 0, 0, 0, 0, 0",
}


def _ema(df, tf, length):
    key = (tf, length)
    if key not in _cache:
        _cache[key] = data.ema_chart(df, length) if tf == 1 else data.ema_htf(df, tf, length)
    return _cache[key]


def run(p: Params, start: str, end: str | None = None, log_rows: int = 0):
    df = data.load()
    if "arr" not in _cache:
        _cache["arr"] = data.arrays(df)
    o, h, l, c, t_ms, day = _cache["arr"]
    idx = df.index
    s = int(idx.searchsorted(pd.Timestamp(start, tz=idx.tz)))
    e = len(idx) if end is None else int(idx.searchsorted(pd.Timestamp(end, tz=idx.tz)))
    log = np.zeros((log_rows, 6))
    r = simulate(o, h, l, c, t_ms, day, _ema(df, p.tf1, p.len1), _ema(df, p.tf2, p.len2),
                 p.direction, p.thr1, p.thr2, p.entry_qty, p.base_tp / 100, p.max_dd_times - 1,
                 p.sl_perc / 100, p.sl_at_max, p.tp_catch,
                 csv(p.dd_csv, 10) / 100 * p.dd_scale, csv(p.ddtp_csv, 10) / 100, csv(p.mf_csv, 10),
                 np.full(11, p.decay_base), np.full(11, p.min_tp / 100), p.one_day,
                 POINT_VALUE, COMMISSION, SLIP, s, e, log)
    return r, log


def summarize(r):
    n = r[R_TRADES]
    return {
        "net": round(r[R_NET]),
        "maxDD": round(r[R_MAXDD]),
        "ret/dd": round(r[R_NET] / r[R_MAXDD], 2) if r[R_MAXDD] > 0 else np.nan,
        "trades": int(n),
        "win%": round(100 * r[R_WINS] / n, 1) if n else np.nan,
        "PF": round(r[R_GROSS_W] / r[R_GROSS_L], 2) if r[R_GROSS_L] > 0 else np.inf,
        "SLs": int(r[R_SL]),
        "maxQty": int(r[R_MAXQTY]),
        "deepestDD": int(r[R_MAXDD_COUNT]),
    }


def evaluate(p: Params):
    a = summarize(run(p, IS_START, IS_END)[0])
    b = summarize(run(p, OOS_START)[0])
    return {**asdict(p), **{f"IS_{k}": v for k, v in a.items()}, **{f"OOS_{k}": v for k, v in b.items()}}


def _eval_many(ps):
    return [evaluate(p) for p in ps]


def grid(base: Params, space: dict):
    keys = list(space)
    for combo in itertools.product(*(space[k] for k in keys)):
        yield replace(base, **dict(zip(keys, combo)))


def search(base: Params, space: dict, workers: int = os.cpu_count() or 1) -> pd.DataFrame:
    ps = list(grid(base, space))
    chunks = [ps[i::workers * 4] for i in range(workers * 4)]
    t0 = time.time()
    with ProcessPoolExecutor(workers) as ex:
        rows = [row for part in ex.map(_eval_many, chunks) for row in part]
    print(f"  {len(ps)} combos in {time.time() - t0:.0f}s", file=sys.stderr)
    return pd.DataFrame(rows)


SHOW = ["mf_csv", "thr1", "len1", "tf1", "thr2", "len2", "base_tp", "dd_scale", "max_dd_times", "sl_perc",
        "IS_net", "IS_maxDD", "IS_ret/dd", "IS_trades", "IS_win%", "IS_SLs", "IS_maxQty",
        "OOS_net", "OOS_maxDD", "OOS_ret/dd", "OOS_trades", "OOS_SLs"]


def rank(df: pd.DataFrame, min_trades: int = 40) -> pd.DataFrame:
    ok = df[(df["IS_trades"] >= min_trades) & (df["IS_net"] > 0)]
    return ok.sort_values("IS_ret/dd", ascending=False)


if __name__ == "__main__":
    pd.set_option("display.width", 250)
    pd.set_option("display.max_columns", 40)
    print("Baseline (script defaults):")
    print(pd.DataFrame([evaluate(Params())])[SHOW].to_string(index=False))

    # ---- Stage 1: entry signal, with a fixed modest exit/ladder ----
    print("\nStage 1: signal search (linear ladder, default DD distances)")
    s1_base = Params(mf_csv=MULTS["linear"])
    s1 = search(s1_base, {
        "thr1": [1.0025, 1.005, 1.0075, 1.01, 1.015, 1.02],
        "len1": [50, 150, 300],
        "thr2": [1.0, 1.0025, 1.005, 1.01],
        "len2": [100, 300, 600],
        "base_tp": [0.135, 0.25],
    })
    s1.to_csv("ppo_dd/results_stage1.csv", index=False)
    top1 = rank(s1)
    print(top1[SHOW].head(15).to_string(index=False))

    # ---- Stage 2: exits / ladder / sizing on the best distinct signals ----
    sig_cols = ["thr1", "len1", "thr2", "len2"]
    sigs = top1.drop_duplicates(sig_cols).head(5)[sig_cols].to_dict("records")
    print("\nStage 2: exit/ladder search on top signals")
    parts = []
    for sg in sigs:
        parts.append(search(replace(Params(), **sg), {
            "base_tp": [0.1, 0.135, 0.2, 0.3, 0.5],
            "dd_scale": [0.5, 0.75, 1.0, 1.5, 2.0],
            "max_dd_times": [3, 5, 7],
            "mf_csv": list(MULTS.values()),
            "sl_perc": [2.0, 4.0, 6.0],
        }))
    s2 = pd.concat(parts, ignore_index=True)
    s2["mf_csv"] = s2["mf_csv"].map({v: k for k, v in MULTS.items()})
    s2.to_csv("ppo_dd/results_stage2.csv", index=False)
    print(rank(s2)[SHOW].head(25).to_string(index=False))
