"""Stress-test a logged strategy before trusting it.

    python -m research.validate --id <result id>      # or --top 5 (best robust 15m results)

Checks, all on the 10-year 15m history:
  * year-by-year net P&L (is the edge spread out or one lucky year?)
  * cost stress: fees x1.5 and 2 ticks of slippage per market fill
  * parameter neighbourhood: every numeric parameter nudged one step either way in its
    search grid; reports the share of neighbours that stay profitable OOS (PF > 1)
  * Monte Carlo Lucid 50K: 2,000 evals built from 5-session blocks of the OOS daily P&L,
    resampled with replacement, giving pass / bust odds that don't hinge on one start date

Writes research/validation/<slug>.json and prints a summary.
"""

import argparse
import json
import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.search import ATR_N, RESULTS, SL_K, TP_K, TRAIL_K
from trend import components as comp
from trend.data import load
from trend.strategy import FEE_RT, INSTRUMENT, LucidRules, Spec, daily_series, metrics
from trend.strategy import backtest as _backtest

OUT = Path(__file__).resolve().parent / "validation"


def _neighbours(values, v):
    vals = sorted(set(values))
    if v not in vals:
        return []
    i = vals.index(v)
    return [vals[j] for j in (i - 1, i + 1) if 0 <= j < len(vals)]


def neighbours(s: Spec):
    out = []
    for attr, table, name in (("trend_p", comp.TREND, s.trend), ("conf1_p", comp.CONFIRM, s.conf1),
                              ("conf2_p", comp.CONFIRM, s.conf2), ("regime_p", comp.REGIME, s.regime)):
        space = table[name][1]
        for k, v in getattr(s, attr).items():
            for nv in _neighbours(space.get(k, []), v):
                p = dict(getattr(s, attr))
                p[k] = nv
                out.append((f"{attr[:-2]}.{k}={nv}", replace(s, **{attr: p})))
    for attr, grid in (("atr_n", ATR_N), ("sl_k", SL_K), ("tp_k", TP_K), ("trail_k", TRAIL_K)):
        for nv in _neighbours(grid, getattr(s, attr)):
            out.append((f"{attr}={nv}", replace(s, **{attr: nv})))
    return out


def mc_lucid(pnl, low, ntr, rules, cushion, n=2000, block=5, horizon=250, seed=0):
    from trend.strategy import lucid_sim
    rng = np.random.default_rng(seed)
    nb = len(pnl) // block
    if nb < 4:
        return None
    res = {"pass": 0, "bust": 0, "timeout": 0}
    for _ in range(n):
        picks = rng.integers(0, nb, horizon // block)
        idx = (picks[:, None] * block + np.arange(block)).ravel()
        r = lucid_sim(pnl[idx], low[idx], ntr[idx], rules, step=10**9, cushion_sizing=cushion)
        k = max(("pass_rate", "pass"), ("bust_rate", "bust"), ("timeout_rate", "timeout"), key=lambda t: r[t[0]])[1]
        res[k] += 1
    return {k: v / n for k, v in res.items()}


def validate(rec):
    s = Spec(**rec["spec"])
    full = load(rec["dataset"])
    pv, tick = INSTRUMENT.get(rec["dataset"].split("_")[0], (5.0, 0.25))

    def backtest(m, spec, rules, **kw):
        return _backtest(m, spec, rules, point_value=pv, tick=tick, **kw)

    split = rec["split"]
    oos = full.slice(start=split)
    rules = LucidRules()

    tr = backtest(full, s, rules)
    yearly = {}
    for t in tr:
        y = full.index[int(t[1])].year
        yearly[y] = yearly.get(y, 0.0) + t[6]

    stress = {}
    for label, fee, slip in (("base", FEE_RT, 1.0), ("fees_x1.5", FEE_RT * 1.5, 1.0), ("slip_2ticks", FEE_RT, 2.0),
                             ("both", FEE_RT * 1.5, 2.0)):
        r = metrics(oos, backtest(oos, s, rules, fee_rt=fee, slip_ticks=slip), rules, curve_points=0,
                    cushion_sizing=s.cushion_sizing)
        stress[label] = {"pf": r["profit_factor"], "net": r["net"], "pass": r["lucid"]["pass_rate"],
                         "bust": r["lucid"]["bust_rate"]}

    neigh = []
    for label, ns in neighbours(s):
        r = metrics(oos, backtest(oos, ns, rules), rules, curve_points=0, cushion_sizing=ns.cushion_sizing)
        neigh.append({"change": label, "pf": r["profit_factor"], "net": r["net"], "trades": r["trades"]})
    share_ok = float(np.mean([n["pf"] > 1.0 for n in neigh])) if neigh else None

    _, pnl, low, ntr = daily_series(oos, backtest(oos, s, rules))
    mc = mc_lucid(pnl, low, ntr, rules, s.cushion_sizing)

    return {
        "id": rec["id"], "name": rec["name"], "spec": rec["spec"],
        "yearly_net": {str(k): round(v, 2) for k, v in sorted(yearly.items())},
        "losing_years": sum(v < 0 for v in yearly.values()),
        "cost_stress_oos": stress,
        "neighbours_oos": neigh, "neighbours_profitable_share": share_ok,
        "mc_lucid_oos": mc,
    }


def save(rec):
    v = validate(rec)
    OUT.mkdir(exist_ok=True)
    slug = re.sub(r"[^A-Za-z0-9]+", "_", rec["id"]).strip("_")
    (OUT / f"{slug}.json").write_text(json.dumps(v, indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--id")
    ap.add_argument("--top", type=int, default=0)
    args = ap.parse_args()
    recs = [json.loads(l) for l in RESULTS.read_text().splitlines() if l.strip()]
    if args.id:
        chosen = [r for r in recs if r["id"] == args.id]
    else:
        vmax = max(r.get("v", 1) for r in recs)
        pool = [r for r in recs if r.get("v", 1) == vmax and r["track"] in ("15m_full", "15m_joint") and r["robust"]]
        pool.sort(key=lambda r: r["oos"]["lucid"]["pass_rate"] - r["oos"]["lucid"]["bust_rate"], reverse=True)
        chosen = pool[: args.top or 5]
    for rec in chosen:
        v = save(rec)
        mc = v["mc_lucid_oos"] or {}
        print(f"{rec['name']:<44} losing yrs {v['losing_years']}/{len(v['yearly_net'])}  "
              f"neighbours PF>1 {v['neighbours_profitable_share'] if v['neighbours_profitable_share'] is None else round(v['neighbours_profitable_share'], 2)}  "
              f"stress PF {v['cost_stress_oos']['both']['pf']:.2f}  MC pass {mc.get('pass', 0):.0%} bust {mc.get('bust', 0):.0%}")


if __name__ == "__main__":
    main()
