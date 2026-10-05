"""Combine several robust strategies into one account and test it as a Lucid 50K portfolio.

    python -m research.portfolio [--max-legs 4]

1. Every robust 15m result logged so far (any engine version) is re-run on the current engine.
2. Legs with in-sample PF >= 1.15 and >= 300 trades form the pool.
3. Greedy forward selection on the IN-SAMPLE daily P&L only: add the leg that most improves the
   portfolio's daily Sharpe, up to --max-legs; each leg trades its own position (P&L summed per
   session, which matches separate sub-accounts; netting in one account would only save fees).
4. The chosen mix is then scored once on 2023-2026 and on real MES 2025-26, and a single risk
   scale for the whole portfolio is picked on IN-SAMPLE Lucid pass-minus-bust.

Writes research/portfolio.json.
"""

import argparse
import json
from dataclasses import replace
from pathlib import Path

import numpy as np

from research.search import RESULTS
from research.validate import mc_lucid
from trend.data import load
from trend.strategy import LucidRules, Spec, backtest, daily_series, lucid_sim

OUT = Path(__file__).resolve().parent / "portfolio.json"


def daily(m, s, rules):
    _, p, lo, n = daily_series(m, backtest(m, s, rules))
    return p, lo, n


def sharpe(p):
    return p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0.0


def pf(p):
    w, l = p[p > 0].sum(), -p[p < 0].sum()
    return w / l if l else 99.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-legs", type=int, default=4)
    ap.add_argument("--all", action="store_true", help="equal-weight every qualifying leg (no selection)")
    ap.add_argument("--is-only", action="store_true",
                    help="build the pool from ALL logged 15m results using in-sample stats only (the 'robust' flag "
                         "looks at OOS, so pooling only robust results leaks OOS information into the selection)")
    ap.add_argument("--min-pf", type=float, default=1.15)
    ap.add_argument("--scale", type=float, default=None, help="fix the per-leg risk scale instead of choosing it")
    a = ap.parse_args()
    rules = LucidRules()
    full = load("es_15m")
    per = {"is": full.slice(end="2023-01-01"), "oos": full.slice(start="2023-01-01"), "mes": load("mes_15m")}

    seen, legs = set(), []
    for line in RESULTS.read_text().splitlines():
        d = json.loads(line)
        if d["track"] != "15m_full" or (not a.is_only and not d["robust"]) or d.get("candidate"):
            continue
        s = Spec(**{**d["spec"], "cushion_sizing": False})
        # normalise risk so legs are comparable; the portfolio scale is chosen later
        s = replace(s, risk_usd=200.0, daily_loss_limit=1e9, daily_profit_cap=1e9)
        k = s.key()
        if k in seen:
            continue
        seen.add(k)
        is_p = daily(per["is"], s, rules)
        tr_n = int(is_p[2].sum())
        if tr_n < 300 or pf(is_p[0]) < a.min_pf:
            continue
        legs.append({"name": d["name"], "spec": s, "is": is_p})
    print(f"{len(legs)} legs qualify (IS PF >= 1.15, >= 300 trades) out of {len(seen)} robust specs")
    if not legs:
        return

    chosen, cur = ([], None) if not a.all else (list(legs), None)
    for _ in range(0 if a.all else a.max_legs):
        best, best_sh = None, sharpe(cur) if cur is not None else -9
        for lg in legs:
            if lg in chosen:
                continue
            trial = lg["is"][0] if cur is None else cur + lg["is"][0]
            sh = sharpe(trial)
            if sh > best_sh + 0.02:
                best, best_sh = lg, sh
        if best is None:
            break
        chosen.append(best)
        cur = best["is"][0] if cur is None else cur + best["is"][0]
        print(f"  + {best['name']:<44} IS portfolio Sharpe {best_sh:.2f}")

    def combo(m, scale):
        ps, ls, ns = zip(*(daily(m, replace(c["spec"], risk_usd=200.0 * scale), rules) for c in chosen))
        return np.sum(ps, axis=0), np.sum(ls, axis=0), np.sum(ns, axis=0)

    # correlation of legs (IS daily P&L)
    mat = np.corrcoef([c["is"][0] for c in chosen]) if len(chosen) > 1 else np.ones((1, 1))
    print("IS daily P&L correlation between legs:\n", np.round(mat, 2))

    best_scale, best_sc = 1.0, -9
    for scale in ((a.scale,) if a.scale else (0.5, 0.75, 1.0, 1.25, 1.5, 2.0)):
        for cush in (False, True):
            p, lo, n = combo(per["is"], scale)
            r = lucid_sim(p, lo, n, rules, cushion_sizing=cush)
            sc = r["pass_rate"] - r["bust_rate"]
            if sc > best_sc:
                best_sc, best_scale, best_cush = sc, scale, cush
    if a.all:
        for c in chosen:
            print(f"  = {c['name']}")
    print(f"risk scale chosen in-sample: {best_scale}x $200 per leg, cushion sizing {best_cush}")

    res = {"legs": [{"name": c["name"], "spec": c["spec"].__dict__} for c in chosen], "scale": best_scale,
           "cushion": best_cush, "correlation_is": mat.round(3).tolist(), "periods": {}}
    for k, m in per.items():
        p, lo, n = combo(m, best_scale)
        roll = lucid_sim(p, lo, n, rules, cushion_sizing=best_cush)
        mc = mc_lucid(p, lo, n, rules, best_cush, n=1000) or {}
        eq = np.cumsum(p)
        dd = float((np.maximum.accumulate(eq) - eq).max())
        res["periods"][k] = {"net": float(p.sum()), "pf_daily": float(pf(p)), "sharpe": float(sharpe(p)),
                             "max_dd": dd, "trades": int(n.sum()), "roll": roll, "mc": mc}
        print(f"{k:<4} net {p.sum():>8.0f} daily-PF {pf(p):.2f} Sharpe {sharpe(p):.2f} DD {dd:>6.0f} trades {int(n.sum())} | "
              f"rolling pass {roll['pass_rate']:.0%} bust {roll['bust_rate']:.0%} days {roll['median_days_to_pass']} | "
              f"MC pass {mc.get('pass', 0):.0%} bust {mc.get('bust', 0):.0%}")
    OUT.write_text(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
