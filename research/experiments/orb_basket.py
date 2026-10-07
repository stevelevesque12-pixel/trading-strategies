"""Basket of robust opening-range variants on MES + MNQ: does diversifying within the ORB idea raise
Sharpe enough to pass Lucid faster?

    python -m research.experiments.orb_basket

Legs: the consensus (modal) spec of every 15m_joint family that (a) uses the 'orb' trend, (b) has >= 6 runs
and (c) >= 60% of its runs robust, each traded on MES and MNQ. Family robustness used OOS results, so the
OOS numbers here are optimistic; 2016-22 is the cleaner read.
"""

import collections
import json
from dataclasses import replace

import numpy as np

from research.experiments.fast_pass import sim
from research.search import RESULTS
from trend.data import load
from trend.strategy import LucidRules, Spec, backtest, daily_series

RULES = LucidRules()


def families():
    recs = [json.loads(l) for l in RESULTS.read_text().splitlines()]
    fam = collections.defaultdict(list)
    for r in recs:
        if r["track"] == "15m_joint" and r["spec"]["trend"] == "orb":
            fam[r["name"]].append(r)
    out = []
    for name, rs in fam.items():
        if len(rs) >= 6 and np.mean([r["robust"] for r in rs]) >= 0.6:
            mode = {k: json.loads(collections.Counter(json.dumps(r["spec"][k]) for r in rs).most_common(1)[0][0])
                    for k in rs[0]["spec"]}
            out.append((name, Spec(**mode)))
    return out


def daily(per, spec):
    a, b = per
    res = {}
    for ds, pv in (("es_15m", 5.0), ("nq_15m", 2.0)):
        m = load(ds).slice(start=a, end=b)
        days, p, lo, n = daily_series(m, backtest(m, spec, RULES, point_value=pv))
        for d, x, y, z in zip(m.day_dates[days], p, lo, n):
            q = res.setdefault(d, [0.0, 0.0, 0])
            q[0] += x
            q[1] += y
            q[2] += z
    keys = sorted(res)
    return keys, np.array([res[k] for k in keys])


def main():
    fams = families()
    print(f"{len(fams)} legs:", ", ".join(f for f, _ in fams))
    for lab, per in (("2016-22", (None, "2023-01-01")), ("2023-26", ("2023-01-01", None))):
        mats = []
        for _, s in fams:
            s = replace(s, risk_usd=200.0, cushion_sizing=False, daily_loss_limit=1e9, daily_profit_cap=1e9)
            keys, arr = daily(per, s)
            mats.append(dict(zip(keys, arr)))
        common = sorted(set.intersection(*[set(m) for m in mats]))
        A = np.array([[m[d] for d in common] for m in mats])  # legs x days x 3
        corr = np.corrcoef(A[:, :, 0])
        P, L, N = A[:, :, 0].sum(0), A[:, :, 1].sum(0), A[:, :, 2].sum(0)
        sh = P.mean() / P.std() * np.sqrt(252)
        single = np.mean([A[i, :, 0].mean() / A[i, :, 0].std() * np.sqrt(252) for i in range(len(A))])
        print(f"\n{lab}: basket daily Sharpe {sh:.2f} (avg single leg {single:.2f}), mean leg correlation "
              f"{(corr.sum() - len(corr)) / (len(corr) ** 2 - len(corr)):.2f}, trades/week {N.sum() / (len(N) / 5):.1f}")
        for scale in (0.5, 0.75, 1.0, 1.5):
            pr, br, md = sim(P * scale, L * scale, N, 42)
            pr2, br2, md2 = sim(P * scale, L * scale, N, 250)
            print(f"  risk ${200 * scale:.0f}/leg: pass<=42 sessions {pr:.0%} bust {br:.0%} (median {md}) | "
                  f"within ~1y: pass {pr2:.0%} bust {br2:.0%} (median {md2})")


if __name__ == "__main__":
    main()
