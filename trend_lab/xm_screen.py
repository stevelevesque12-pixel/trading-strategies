"""
Cross-market screener: find trend logic that is a real market behavior, not
an 11-month MCL curve-fit.

  train : 2016-01-01 .. 2022-12-31 on GC, ES, NQ, SI (10-year 15m histories)
  test  : 2023-01-01 .. 2026-08 on the same four, AND the full MCL year
          (MCL is never used for selection).

Everything is in R (equal risk per trade): $1,000 risk with uncapped
contracts, daily loss stop 2.25R, profit lock 4R, per-bar max stop = 2.1% of
price (the same filter MCL's 1.5-point cap applies at ~$70).

The train score is the median per-market t-stat of R (mean / sd * sqrt(n)),
and a config whose avg R is negative on any train market gets no credit:
we want logic that works everywhere, not great somewhere.

  python -m trend_lab.xm_screen --families trend_dip_atr,supertrend_adx --n 80 --procs 3
"""

import argparse
import json
import random
from concurrent.futures import ProcessPoolExecutor
from datetime import date, datetime, timezone

import numpy as np

from .cross_market import SPECS
from .data import load_bars
from .metrics import compute, equity_points
from .optimize import load_registry, save_registry
from .sim import SimConfig, simulate
from .strategies import ALL_FAMILIES

TRAIN_MKTS = ("gc", "es", "nq", "si")
SPLIT = date(2023, 1, 1)
RISK = 1000.0


def _sim(sym, fam, p):
    df = load_bars("15min", symbol=sym)
    pv, tick, comm = SPECS[sym]
    sig = fam.generate(df, p)
    sd = np.asarray(sig["stop_dist"], dtype=float)
    sig["stop_dist"] = np.where(sd > 0.021 * df["close"].to_numpy(), np.nan, sd)
    cfg = SimConfig(session=p.get("session", "us"), risk_usd=RISK, max_contracts=10**7, point_value=pv,
                    tick=tick, commission_rt=comm, max_stop_dist=float("inf"),
                    daily_loss_stop=2.25 * RISK, daily_profit_lock=4 * RISK)
    return simulate(df, sig, cfg)


def r_stats(trades):
    r = np.array([t.r for t in trades])
    if len(r) < 2:
        return {"n": int(len(r)), "avg_r": 0.0, "pf_r": 0.0, "t": 0.0, "win": 0.0}
    pos, neg = r[r > 0].sum(), -r[r <= 0].sum()
    return {"n": int(len(r)), "avg_r": round(float(r.mean()), 3), "pf_r": round(float(pos / neg), 2) if neg else 99.0,
            "t": round(float(r.mean() / (r.std() + 1e-9) * np.sqrt(len(r))), 2),
            "win": round(float(100 * (r > 0).mean()), 1)}


def eval_config(args):
    name, p = args
    fam = ALL_FAMILIES[name]
    out = {"params": p, "train": {}, "test": {}}
    for sym in TRAIN_MKTS:
        tr = _sim(sym, fam, p)
        out["train"][sym] = r_stats([t for t in tr if t.trade_day < SPLIT])
        out["test"][sym] = r_stats([t for t in tr if t.trade_day >= SPLIT])
    ts = [out["train"][s]["t"] for s in TRAIN_MKTS]
    neg = any(out["train"][s]["avg_r"] <= 0 for s in TRAIN_MKTS)
    few = any(out["train"][s]["n"] < 100 for s in TRAIN_MKTS)
    out["score"] = -99.0 if few else (float(np.median(ts)) - (5.0 if neg else 0.0))
    return out


def mcl_check(name, p):
    """The selected config on MCL, sized like the live strategy ($300 risk)."""
    fam = ALL_FAMILIES[name]
    df = load_bars("15min")
    days = sorted(set(df["trade_day"]))
    trades = simulate(df, fam.generate(df, p), SimConfig(session=p.get("session", "us"), risk_usd=300,
                                                          daily_loss_stop=675))
    m = compute(trades, days)
    m.update({"r_" + k: v for k, v in r_stats(trades).items()})
    return m, trades, days


def screen(name, n, procs, seed=0):
    fam = ALL_FAMILIES[name]
    rng = random.Random(seed)
    cands, seen = [], set()
    for _ in range(n * 5):
        if len(cands) >= n:
            break
        p = {k: rng.choice(v) for k, v in fam.space.items()}
        k = json.dumps(p, sort_keys=True, default=str)
        if k not in seen:
            seen.add(k)
            cands.append(p)
    with ProcessPoolExecutor(procs) as ex:
        res = list(ex.map(eval_config, [(name, p) for p in cands], chunksize=2))
    res.sort(key=lambda r: -r["score"])
    best = res[0]
    top = res[:10]
    test_t = {s: float(np.median([r["test"][s]["t"] for r in top])) for s in TRAIN_MKTS}
    m, trades, days = mcl_check(name, best["params"])
    return {
        "family": name, "description": fam.description, "configs": len(res), "params": best["params"],
        "train_score": round(best["score"], 2), "train": best["train"], "test": best["test"],
        "top10_test_t_median": {k: round(v, 2) for k, v in test_t.items()},
        "mcl": {k: m.get(k) for k in ("trades", "win_rate", "profit_factor", "net", "max_dd", "r_avg_r", "r_pf_r",
                                       "r_t", "lucid_pass_pct", "lucid_fail_pct", "lucid_attempts")},
        "mcl_equity": equity_points(trades), "mcl_start": str(days[0]), "mcl_end": str(days[-1]),
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--n", type=int, default=80)
    ap.add_argument("--procs", type=int, default=3)
    args = ap.parse_args()
    names = list(ALL_FAMILIES) if args.families == "all" else args.families.split(",")
    for name in names:
        r = screen(name, args.n, args.procs)
        reg = load_registry()
        reg.setdefault("xm_runs", {})[name] = r
        save_registry(reg)
        te = " ".join(f"{s}:{r['test'][s]['avg_r']:+.3f}/{r['test'][s]['pf_r']}" for s in TRAIN_MKTS)
        mc = r["mcl"]
        print(f"{name:<24} train {r['train_score']:>6} | test avgR/PF {te} | MCL pf {mc['profit_factor']} "
              f"R {mc['r_avg_r']} n {mc['trades']} lucid {mc['lucid_pass_pct']}", flush=True)


if __name__ == "__main__":
    main()
