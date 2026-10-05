"""
Deeper validation for a promising family on 15m data:

  walk-forward  Re-optimize on a rolling 120-trading-day window, trade the next
                20 days with the winner, roll forward. The stitched test segments
                are a fully out-of-sample equity curve of the *process*, not of
                one lucky parameter set.
  neighborhood  Perturb each parameter of a chosen config to its adjacent values
                and report how PF moves: a real edge is a plateau, not a spike.
  risk sweep    Replay the walk-forward trades at different $ risk per trade
                and run the Lucid 50K eval simulator on each.

  python -m trend_lab.validate --family trend_dip_atr --n 250
"""

import argparse
import json
import random
from datetime import datetime, timezone

import numpy as np

from .data import load_bars
from .metrics import compute, daily_pnl, equity_points, lucid_eval
from .optimize import load_registry, robust_pick, save_registry, score
from .sim import SimConfig, simulate
from .strategies import ALL_FAMILIES

TRAIN_DAYS, TEST_DAYS = 120, 20


def _run(df, fam, p, risk=200.0):
    return simulate(df, fam.generate(df, p), SimConfig(session=p["session"], risk_usd=risk,
                                                      **getattr(fam, "sim_overrides", {})))


def walk_forward(fam, n, seed=0, tf="15min", anchored=False, select="best"):
    """
    anchored=False: rolling TRAIN_DAYS window. anchored=True: train on every day
    from the start of the data up to the test window (expanding window).
    select="robust": pick per window by median neighbor score (plateau).
    """
    df = load_bars(tf)
    days = sorted(set(df["trade_day"]))
    rng = random.Random(seed)
    cands, seen = [], set()
    while len(cands) < n:
        p = {k: rng.choice(v) for k, v in fam.space.items()}
        key = json.dumps(p, sort_keys=True, default=str)
        if key not in seen:
            seen.add(key)
            cands.append(p)
    # one full-history sim per config, sliced per window (indicators are causal, so this is equivalent)
    sims = {}

    def trades_of(p):
        k = json.dumps(p, sort_keys=True, default=str)
        if k not in sims:
            sims[k] = _run(df, fam, p)
        return sims[k]

    oos_trades, windows = [], []
    start = 0
    while start + TRAIN_DAYS + 1 < len(days):
        lo = 0 if anchored else start
        train_days = days[lo:start + TRAIN_DAYS]
        tr0, tr1 = train_days[0], train_days[-1]
        te0 = days[start + TRAIN_DAYS]
        te1 = days[min(start + TRAIN_DAYS + TEST_DAYS - 1, len(days) - 1)]

        def sc(p):
            return score(compute([t for t in trades_of(p) if tr0 <= t.trade_day <= tr1], train_days))

        if select == "robust":
            pick, _ = robust_pick(cands, sc, fam.space)
        else:
            pick = max(cands, key=sc)
        test = [t for t in trades_of(pick) if te0 <= t.trade_day <= te1]
        oos_trades += test
        windows.append({"train": f"{tr0}..{tr1}", "test": f"{te0}..{te1}", "params": pick,
                        "test_pnl": round(sum(t.pnl for t in test), 0), "test_trades": len(test)})
        start += TEST_DAYS
    test_days = [d for d in days if d >= days[TRAIN_DAYS]]
    return df, oos_trades, test_days, windows


def neighborhood(fam, params, tf="15min"):
    df = load_bars(tf)
    days = sorted(set(df["trade_day"]))
    rows = []
    for k, choices in fam.space.items():
        if len(choices) < 2 or params[k] not in choices:
            continue
        i = choices.index(params[k])
        for j in (i - 1, i + 1):
            if 0 <= j < len(choices):
                q = dict(params, **{k: choices[j]})
                m = compute(_run(df, fam, q), days)
                rows.append({"param": k, "value": str(choices[j]), "pf": m.get("profit_factor"),
                             "net": m.get("net"), "trades": m["trades"]})
    base = compute(_run(df, fam, params), days)
    pfs = [r["pf"] for r in rows]
    return {"base_pf": base.get("profit_factor"), "base_net": base.get("net"), "rows": rows,
            "neighbors_pf_median": round(float(np.median(pfs)), 2) if pfs else None,
            "neighbors_pf_gt1_pct": round(100 * np.mean([x > 1 for x in pfs]), 1) if pfs else None}


def risk_sweep(trades, test_days, risks=(150, 200, 300, 400, 500, 600, 800)):
    """Scale walk-forward trades linearly by risk (contracts scale with risk; rounding ignored)."""
    out = []
    for r in risks:
        f = r / 200.0
        days = daily_pnl(trades, test_days) * f
        le = lucid_eval(days)
        deq = days.cumsum().to_numpy()
        dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], deq]))[1:] - deq)) if len(deq) else 0.0
        out.append({"risk": r, "net": round(float(days.sum()), 0), "eod_dd": round(dd, 0)} | le)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", required=True)
    ap.add_argument("--n", type=int, default=250)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--anchored", action="store_true")
    ap.add_argument("--select", choices=["best", "robust"], default="best")
    args = ap.parse_args()
    tag = f"{args.family}|{'anchored' if args.anchored else 'rolling'}|{args.select}"
    fam = ALL_FAMILIES[args.family]

    _, trades, test_days, windows = walk_forward(fam, args.n, args.seed, anchored=args.anchored, select=args.select)
    wf = compute(trades, test_days)
    print(f"walk-forward {tag}: {json.dumps({k: wf.get(k) for k in ('trades','win_rate','profit_factor','net','max_dd','lucid_pass_pct','lucid_fail_pct')})}")
    for w in windows:
        print(f"  test {w['test']}: {w['test_trades']:>3} trades  {w['test_pnl']:>7}")

    reg = load_registry()
    best = [r for r in reg["runs"] if r["family"] == fam.name and r["tf"] == "15min"]
    nb = neighborhood(fam, best[-1]["params"]) if best else None
    if nb:
        print(f"neighborhood: base pf {nb['base_pf']}, neighbors median pf {nb['neighbors_pf_median']}, "
              f"{nb['neighbors_pf_gt1_pct']}% of neighbors pf>1")
    rs = risk_sweep(trades, test_days)
    for r in rs:
        print(f"  risk ${r['risk']}: net {r['net']}, eod dd {r['eod_dd']}, lucid pass {r['lucid_pass_pct']}% "
              f"fail {r['lucid_fail_pct']}% median days {r['lucid_median_days']}")

    reg = load_registry()
    reg.setdefault("validations", {})[tag] = {
        "tag": tag, "family": fam.name, "anchored": args.anchored, "select": args.select, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "candidates_per_window": args.n, "train_days": TRAIN_DAYS, "test_days": TEST_DAYS,
        "walk_forward": wf, "equity": equity_points(trades), "start": str(test_days[0]), "end": str(test_days[-1]),
        "windows": windows, "neighborhood": nb, "risk_sweep": rs,
    }
    save_registry(reg)


if __name__ == "__main__":
    main()
