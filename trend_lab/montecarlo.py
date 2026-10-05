"""
Monte Carlo of the Lucid 50K Flex evaluation and funded phase, from the
walk-forward (all out-of-sample) daily P&L of the exclusive portfolio.

Stationary block bootstrap (mean block 5 trading days, keeps short-run
streaks) -> 10,000 synthetic paths per risk level.

  eval   : pass = +$3,000 with largest day <= 50% of profit; fail = EOD balance
           <= trailing max-loss line ($2,000 below the EOD high-water mark,
           locking at the starting balance). Capped at 120 trading days.
  funded : from a fresh funded account, probability of reaching +$2,000
           (a proxy for a first payout buffer) before the trailing line.

  python -m trend_lab.montecarlo
"""

from datetime import datetime, timezone

import numpy as np

from .metrics import LUCID_50K
from .optimize import load_registry, save_registry

KEY = "portfolio:trend_dip_atr+trend_dip_rsi|exclusive"


def bootstrap_path(daily, n, rng, mean_block=5):
    out = np.empty(n)
    i = rng.integers(len(daily))
    for k in range(n):
        if rng.random() < 1.0 / mean_block:
            i = rng.integers(len(daily))
        out[k] = daily[i]
        i = (i + 1) % len(daily)
    return out


def run_eval(path, rules=LUCID_50K, consistency=True, target=None):
    target = target if target is not None else rules["profit_target"]
    bal = peak = best_day = 0.0
    for k, v in enumerate(path):
        bal += v
        best_day = max(best_day, v)
        if bal <= min(peak - rules["max_loss"], rules["lock_at"]):
            return "fail", k + 1
        peak = max(peak, bal)
        if bal >= target and (not consistency or best_day <= rules["consistency"] * bal):
            return "pass", k + 1
    return "timeout", len(path)


def run_eval_dynamic(path200, frac, lo, hi, rules=LUCID_50K, consistency=True, target=None):
    """
    Cushion-based sizing: each day risk = clamp(frac * (balance - max-loss line), lo, hi),
    applied to a $200-risk daily P&L path (P&L ~ linear in risk). Returns outcome, days.
    """
    target = target if target is not None else rules["profit_target"]
    bal = peak = best_day = 0.0
    for k, v in enumerate(path200):
        line = min(peak - rules["max_loss"], rules["lock_at"])
        risk = min(hi, max(lo, frac * (bal - line)))
        day = v * risk / 200.0
        bal += day
        best_day = max(best_day, day)
        if bal <= line:
            return "fail", k + 1
        peak = max(peak, bal)
        if bal >= target and (not consistency or best_day <= rules["consistency"] * bal):
            return "pass", k + 1
    return "timeout", len(path200)


def simulate_dynamic(daily200, frac, lo, hi, sims=10000, horizon=120, seed=1):
    rng = np.random.default_rng(seed)
    ev = [run_eval_dynamic(bootstrap_path(daily200, horizon, rng), frac, lo, hi) for _ in range(sims)]
    fu = [run_eval_dynamic(bootstrap_path(daily200, horizon, rng), frac, lo, hi, consistency=False, target=2000.0)
          for _ in range(sims)]
    pass_days = [d for o, d in ev if o == "pass"]
    return {
        "rule": f"risk = {frac:.0%} of cushion, ${lo}-${hi}",
        "eval_pass_pct": round(100 * np.mean([o == "pass" for o, _ in ev]), 1),
        "eval_fail_pct": round(100 * np.mean([o == "fail" for o, _ in ev]), 1),
        "eval_timeout_pct": round(100 * np.mean([o == "timeout" for o, _ in ev]), 1),
        "eval_pass_within_30d_pct": round(100 * np.mean([o == "pass" and d <= 30 for o, d in ev]), 1),
        "eval_median_days": float(np.median(pass_days)) if pass_days else None,
        "funded_reach_2k_pct": round(100 * np.mean([o == "pass" for o, _ in fu]), 1),
        "funded_blow_pct": round(100 * np.mean([o == "fail" for o, _ in fu]), 1),
    }


def simulate(daily, sims=10000, horizon=120, seed=0):
    rng = np.random.default_rng(seed)
    ev = [run_eval(bootstrap_path(daily, horizon, rng)) for _ in range(sims)]
    fu = [run_eval(bootstrap_path(daily, horizon, rng), consistency=False, target=2000.0) for _ in range(sims)]
    pass_days = [d for o, d in ev if o == "pass"]
    return {
        "eval_pass_pct": round(100 * np.mean([o == "pass" for o, _ in ev]), 1),
        "eval_fail_pct": round(100 * np.mean([o == "fail" for o, _ in ev]), 1),
        "eval_timeout_pct": round(100 * np.mean([o == "timeout" for o, _ in ev]), 1),
        "eval_pass_within_30d_pct": round(100 * np.mean([o == "pass" and d <= 30 for o, d in ev]), 1),
        "eval_median_days": float(np.median(pass_days)) if pass_days else None,
        "funded_reach_2k_pct": round(100 * np.mean([o == "pass" for o, _ in fu]), 1),
        "funded_blow_pct": round(100 * np.mean([o == "fail" for o, _ in fu]), 1),
    }


def main():
    reg = load_registry()
    v = reg["validations"][KEY]
    rows = []
    for r in v["risk_sweep"]:
        d = np.array(r["daily"])
        res = {"risk": r["risk"], "days": len(d), "mean_day": round(float(d.mean()), 1)} | simulate(d)
        rows.append(res)
        print(res, flush=True)
    d200 = np.array(next(r for r in v["risk_sweep"] if r["risk"] == 200)["daily"])
    dyn = []
    for frac, lo, hi in [(0.15, 150, 400), (0.20, 150, 500), (0.12, 200, 400), (0.25, 200, 600), (0.15, 200, 500)]:
        res = simulate_dynamic(d200, frac, lo, hi)
        dyn.append(res)
        print(res, flush=True)
    reg = load_registry()
    reg["montecarlo"] = {"source": KEY, "dynamic": dyn, "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                         "method": "stationary block bootstrap, mean block 5 days, 10k paths, 120-day cap",
                         "rows": rows}
    save_registry(reg)


if __name__ == "__main__":
    main()
