"""How fast can the ORB family pass a Lucid 50K eval? Scores every variant on the share of evals that pass
within --horizon sessions (default 42, ~2 months) vs bust, starting an eval every 5 sessions.

    python -m research.experiments.fast_pass [--horizon 42]

Legs: the two ORB consensus strategies on MES and MNQ. Speed levers tried: re-entry on every aligned
bar ('any' trigger), the whole RTH session as entry window, no daily profit cap, and a risk scale.
Variants are ranked on 2016-22 only; 2023-26 is reported alongside, never used to choose.
"""

import argparse
import itertools
from dataclasses import replace

import numpy as np

from research.candidate import ORB, ORB_VWAP
from trend.data import load
from trend.strategy import LucidRules, backtest, daily_series

RULES = LucidRules()


def sim(p, lo, n, horizon, cushion=True, step=5):
    passed = bust = 0
    days = []
    starts = range(0, len(p) - horizon, step)
    for s0 in starts:
        cum = peak = best = 0.0
        mll = -RULES.max_loss
        traded = 0
        for k in range(s0, s0 + horizon):
            f = max(0.25, min(1.0, np.floor((cum - mll) / RULES.max_loss * 4) / 4)) if cushion else 1.0
            if cum + f * lo[k] <= mll:
                bust += 1
                break
            cum += f * p[k]
            traded += n[k] > 0
            best = max(best, f * p[k])
            if cum > peak:
                peak = cum
                mll = min(peak - RULES.max_loss, RULES.lock_at_profit)
            if cum >= RULES.profit_target and best <= RULES.consistency * cum and traded >= RULES.min_days:
                passed += 1
                days.append(k - s0 + 1)
                break
    N = max(1, len(starts))
    return passed / N, bust / N, (float(np.median(days)) if days else None)


def legs_daily(per, spec_mods, scale):
    """Sum daily P&L of the 4 legs (2 strategies x MES/MNQ), each leg at its base risk x scale / 2."""
    a, b = per
    out = None
    for spec in (ORB, ORB_VWAP):
        s = replace(spec, **spec_mods)
        s = replace(s, risk_usd=s.risk_usd * scale / 2, cushion_sizing=False)
        for ds, pv in (("es_15m", 5.0), ("nq_15m", 2.0)):
            m = load(ds).slice(start=a, end=b)
            days, p, lo, n = daily_series(m, backtest(m, s, RULES, point_value=pv))
            d = dict(zip(m.day_dates[days], zip(p, lo, n)))
            if out is None:
                out = {k: list(v) for k, v in d.items()}
            else:
                for k, v in d.items():
                    if k in out:
                        out[k] = [out[k][0] + v[0], out[k][1] + v[1], out[k][2] + v[2]]
    keys = sorted(out)
    arr = np.array([out[k] for k in keys])
    return arr[:, 0], arr[:, 1], arr[:, 2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--horizon", type=int, default=42)
    a = ap.parse_args()
    IS, OOS = (None, "2023-01-01"), ("2023-01-01", None)
    rows = []
    for trig, win, cap, scale in itertools.product(["fresh", "any"], ["ny_am", "all_rth"], [900, 1e9], [1, 1.5, 2, 3]):
        mods = {"trigger": trig, "window": win, "daily_profit_cap": cap, "daily_loss_limit": 300 * scale}
        r = {}
        for lab, per in (("is", IS), ("oos", OOS)):
            p, lo, n = legs_daily(per, mods, scale)
            r[lab] = sim(p, lo, n, a.horizon) + (n.sum() / (len(n) / 5),)  # + trades per week
        rows.append((mods, scale, r))
        print(f"{trig:<5} {win:<7} cap {'none' if cap > 1e8 else int(cap):<4} risk x{scale:<3} | "
              f"IS pass<= {a.horizon}d {r['is'][0]:.0%} bust {r['is'][1]:.0%} med {r['is'][2]} tpw {r['is'][3]:.1f} | "
              f"OOS pass {r['oos'][0]:.0%} bust {r['oos'][1]:.0%} med {r['oos'][2]} tpw {r['oos'][3]:.1f}", flush=True)
    best = max(rows, key=lambda x: x[2]["is"][0] - x[2]["is"][1])
    print("\nbest on in-sample pass-minus-bust:", best[0], "risk x", best[1], best[2])


if __name__ == "__main__":
    main()
