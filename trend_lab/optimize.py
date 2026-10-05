"""
Random-search optimizer with a strict in-sample / out-of-sample split.

For each (family, timeframe): sample configs, score them on the first
IS_FRACTION of trading days only, then report how the top in-sample configs
did on the untouched remainder. The *selected* config is the best IN-SAMPLE
one -- OOS is never used to pick, so the OOS numbers on the dashboard are an
honest(ish) estimate. "oos_pf_top10_median" shows whether the edge is a
lucky single config or a robust region of parameter space.

Usage:
  python -m trend_lab.optimize --families all --tfs 5min,15min --n 300 --iteration 1
"""

import argparse
import json
import random
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from .data import load_bars
from .metrics import compute, equity_points
from .sim import SimConfig, simulate
from .strategies import ALL_FAMILIES

IS_FRACTION = 0.6
RESULTS = Path(__file__).resolve().parent / "results"
REGISTRY = RESULTS / "registry.json"

# timeframe -> native source file it is built from
TIMEFRAMES = {"1min": "1min", "3min": "1min", "5min": "5min", "10min": "5min", "15min": "15min"}
SIM_KEYS = {"session"}


def load_registry():
    if REGISTRY.exists():
        return json.loads(REGISTRY.read_text())
    return {"runs": [], "iterations": []}


def save_registry(reg):
    RESULTS.mkdir(exist_ok=True)
    REGISTRY.write_text(json.dumps(reg, indent=1, default=str))


def score(m):
    """In-sample objective: return/drawdown, penalized for thin samples and sub-1 PF."""
    if m["trades"] < 30 or m["profit_factor"] <= 1.0:
        return -1e9 + m.get("net", 0)
    return m["net"] / max(m["max_dd"], 300.0) * min(1.0, m["trades"] / 80) * min(m["profit_factor"], 3.0)


def sample(space, rng):
    return {k: rng.choice(v) for k, v in space.items()}


def neighbors(space, p):
    """Every config that differs from p by one step in one parameter."""
    for k, choices in space.items():
        if len(choices) < 2 or p[k] not in choices:
            continue
        i = choices.index(p[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(choices):
                yield dict(p, **{k: choices[j]})


def robust_pick(cands, score_fn, space, top=20):
    """
    Among the `top` best configs by score_fn, pick the one whose one-step
    neighborhood has the best MEDIAN score (itself included): favors a
    plateau over an isolated spike. score_fn must use in-sample data only.
    """
    memo = {}

    def sc(p):
        k = json.dumps(p, sort_keys=True, default=str)
        if k not in memo:
            memo[k] = score_fn(p)
        return memo[k]

    ranked = sorted(cands, key=lambda p: -sc(p))[:top]
    best, best_r = ranked[0], -1e18
    for p in ranked:
        r = float(np.median([sc(p)] + [sc(q) for q in neighbors(space, p)]))
        if r > best_r:
            best, best_r = p, r
    return best, best_r


def run_config(df, fam, p, split_day, days):
    cfg = SimConfig(session=p["session"])
    trades = simulate(df, fam.generate(df, p), cfg)
    is_t = [t for t in trades if t.trade_day < split_day]
    oos_t = [t for t in trades if t.trade_day >= split_day]
    is_days = [d for d in days if d < split_day]
    oos_days = [d for d in days if d >= split_day]
    return trades, compute(is_t, is_days), compute(oos_t, oos_days)


def optimize(fam, tf, n, seed=0, top=10, select="best"):
    df = load_bars(tf, TIMEFRAMES[tf])
    days = sorted(set(df["trade_day"]))
    split_day = days[int(len(days) * IS_FRACTION)]
    rng = random.Random(seed)
    seen, results = set(), []
    for _ in range(n * 3):
        if len(results) >= n:
            break
        p = sample(fam.space, rng)
        key = json.dumps(p, sort_keys=True, default=str)
        if key in seen:
            continue
        seen.add(key)
        _, ism, _ = run_config(df, fam, p, split_day, days)
        results.append((score(ism), p))
    results.sort(key=lambda x: -x[0])
    if select == "robust":
        is_score = lambda p: score(run_config(df, fam, p, split_day, days)[1])
        pick, _ = robust_pick([p for _, p in results], is_score, fam.space)
        results = [(None, pick)] + [r for r in results if r[1] is not pick]
    best = []
    for sc, p in results[:top]:
        trades, ism, oosm = run_config(df, fam, p, split_day, days)
        best.append({"score": sc, "params": p, "is": ism, "oos": oosm, "trades": trades})
    sel = best[0]
    oos_pfs = [b["oos"].get("profit_factor", 0) for b in best if b["is"]["trades"] >= 30]
    full = compute(sel["trades"], days)
    return {
        "family": fam.name,
        "description": fam.description,
        "tf": tf,
        "selection": select,
        "configs_tested": len(results),
        "split_day": str(split_day),
        "data_start": str(days[0]),
        "data_end": str(days[-1]),
        "params": sel["params"],
        "is": sel["is"],
        "oos": sel["oos"],
        "full": full,
        "oos_pf_top10_median": round(float(np.median(oos_pfs)), 2) if oos_pfs else None,
        "top10": [{"params": b["params"], "is_pf": b["is"].get("profit_factor"), "oos_pf": b["oos"].get("profit_factor"),
                   "is_net": b["is"].get("net"), "oos_net": b["oos"].get("net"), "oos_trades": b["oos"]["trades"]}
                  for b in best],
        "equity": equity_points(sel["trades"]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="all")
    ap.add_argument("--tfs", default="5min,15min")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iteration", type=int, default=0)
    ap.add_argument("--note", default="")
    ap.add_argument("--select", choices=["best", "robust"], default="best",
                    help="best = top in-sample score; robust = best median score over one-step neighbors")
    args = ap.parse_args()

    fams = list(ALL_FAMILIES) if args.families == "all" else args.families.split(",")
    started = time.time()
    for name in fams:
        for tf in args.tfs.split(","):
            t0 = time.time()
            run = optimize(ALL_FAMILIES[name], tf, args.n, seed=args.seed, select=args.select)
            run["id"] = f"{name}@{tf}#it{args.iteration}" + ("r" if args.select == "robust" else "")
            run["iteration"] = args.iteration
            run["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
            reg = load_registry()  # re-read: parallel optimizer processes may have written since
            reg["runs"] = [r for r in reg["runs"] if r["id"] != run["id"]] + [run]
            save_registry(reg)
            o = run["oos"]
            print(f"{run['id']:<34} IS pf={run['is'].get('profit_factor'):>5} net={run['is'].get('net'):>7} | "
                  f"OOS pf={o.get('profit_factor'):>5} wr={o.get('win_rate'):>5} net={o.get('net'):>7} "
                  f"dd={o.get('max_dd'):>6} n={o['trades']:>4} lucid={o.get('lucid_pass_pct')} "
                  f"top10med={run['oos_pf_top10_median']} ({time.time() - t0:.0f}s)", flush=True)
    reg = load_registry()
    reg["iterations"].append({"iteration": args.iteration, "note": args.note, "families": fams, "tfs": args.tfs,
                              "n": args.n, "finished": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                              "seconds": round(time.time() - started)})
    save_registry(reg)


if __name__ == "__main__":
    main()
