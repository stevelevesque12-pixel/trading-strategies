"""Evaluate the consensus ORB trend strategy (modal parameters across 12 independent joint-track runs).

    python -m research.experiments.orb_consensus
"""

import collections
import json
from dataclasses import replace

import numpy as np

from research.search import RESULTS
from research.validate import mc_lucid, neighbours
from trend.data import load
from trend.strategy import FEE_RT, LucidRules, Spec, backtest, daily_series, lucid_sim, metrics

import sys
FAMILY = sys.argv[1] if len(sys.argv) > 1 else "orb+daily_ema+fast_ema|er"
PV = {"es_15m": 5.0, "nq_15m": 2.0, "mes_15m": 5.0}


def consensus():
    runs = [json.loads(l)["spec"] for l in RESULTS.read_text().splitlines()
            if json.loads(l)["track"] == "15m_joint" and json.loads(l)["name"] == FAMILY]
    mode = {k: json.loads(collections.Counter(json.dumps(r[k]) for r in runs).most_common(1)[0][0]) for k in runs[0]}
    return Spec(**mode), len(runs)


def stats(m, s, rules, pv, **kw):
    t = backtest(m, s, rules, point_value=pv, **kw)
    return t, metrics(m, t, rules, curve_points=0, cushion_sizing=s.cushion_sizing)


def main():
    rules = LucidRules()
    s, n = consensus()
    print(f"consensus of {n} runs:", json.dumps(s.__dict__))
    es, nq, mes = load("es_15m"), load("nq_15m"), load("mes_15m")
    sets = {"ES 2016-22": (es.slice(end="2023-01-01"), 5.0), "ES 2023-26": (es.slice(start="2023-01-01"), 5.0),
            "NQ 2016-22": (nq.slice(end="2023-01-01"), 2.0), "NQ 2023-26": (nq.slice(start="2023-01-01"), 2.0),
            "real MES 25-26": (mes, 5.0)}
    for lab, (m, pv) in sets.items():
        t, r = stats(m, s, rules, pv)
        _, p, lo, nn = daily_series(m, t)
        mc = mc_lucid(p, lo, nn, rules, s.cushion_sizing, n=800) or {}
        print(f"{lab:<15} n{r['trades']:<4} wr {r['win_rate']:.2f} pf {r['profit_factor']:.2f} net {r['net']:>7.0f} dd {r['max_dd']:>5.0f} "
              f"| roll pass {r['lucid']['pass_rate']:.0%} bust {r['lucid']['bust_rate']:.0%} days {r['lucid']['median_days_to_pass']} "
              f"| MC pass {mc.get('pass', 0):.0%} bust {mc.get('bust', 0):.0%}")

    print("\nparameter neighbours (PF on ES OOS / NQ OOS):")
    ok = []
    for lab, ns in neighbours(s):
        a = stats(es.slice(start="2023-01-01"), ns, rules, 5.0)[1]["profit_factor"]
        b = stats(nq.slice(start="2023-01-01"), ns, rules, 2.0)[1]["profit_factor"]
        c = stats(es.slice(end="2023-01-01"), ns, rules, 5.0)[1]["profit_factor"]
        ok.append((a > 1) and (b > 1) and (c > 1))
        print(f"  {lab:<26} ES IS {c:.2f}  ES OOS {a:.2f}  NQ OOS {b:.2f}")
    print(f"  -> {np.mean(ok):.0%} of neighbours profitable on ES IS, ES OOS and NQ OOS")

    print("\ncost stress (fees x1.5 + 2 ticks slippage):")
    for lab in ("ES 2023-26", "NQ 2023-26"):
        m, pv = sets[lab]
        r = stats(m, s, rules, pv, fee_rt=FEE_RT * 1.5, slip_ticks=2.0)[1]
        print(f"  {lab}: PF {r['profit_factor']:.2f} net {r['net']:.0f}")

    print("\nMES + MNQ together in one account (half risk each), 2023-26 and 2016-22:")
    for per in ("2016-22", "2023-26"):
        half = replace(s, risk_usd=s.risk_usd / 2)
        dp, dl, dn = [], [], []
        for m, pv in ((sets[f"ES {per}"][0], 5.0), (sets[f"NQ {per}"][0], 2.0)):
            _, p, lo, nn = daily_series(m, backtest(m, half, rules, point_value=pv))
            dp.append((m.day_dates[np.unique(m.day_id)], p, lo, nn))
        # align sessions by date
        dates = sorted(set(dp[0][0]) & set(dp[1][0]))
        idx = [{d: i for i, d in enumerate(x[0])} for x in dp]
        P = np.array([sum(x[1][ix[d]] for x, ix in zip(dp, idx)) for d in dates])
        L = np.array([sum(x[2][ix[d]] for x, ix in zip(dp, idx)) for d in dates])
        N = np.array([sum(x[3][ix[d]] for x, ix in zip(dp, idx)) for d in dates])
        for scale in (1.0, 1.5, 2.0):
            roll = lucid_sim(P * scale, L * scale, N, rules, cushion_sizing=True)
            mc = mc_lucid(P * scale, L * scale, N, rules, True, n=800) or {}
            print(f"  {per} risk x{scale}: net {P.sum() * scale:>7.0f} trades {N.sum()} | roll pass {roll['pass_rate']:.0%} "
                  f"bust {roll['bust_rate']:.0%} days {roll['median_days_to_pass']} | MC pass {mc.get('pass', 0):.0%} bust {mc.get('bust', 0):.0%}")


if __name__ == "__main__":
    main()
