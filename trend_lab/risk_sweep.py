"""
Risk-per-trade sweep for maximizing yearly net dollars on Tradeify Select Flex 50K (rules in propsim.py),
accepting blown accounts (a new eval is bought the next day).

Unlike propsim.py (which scales one $200 path linearly), every risk level is re-simulated with the
Pine-parity simulator, so contract rounding, the max-contracts cap and the daily loss stop (2.25 x risk)
are exact. Bootstrapped years draw the SAME day sequence for every risk level, so eval risk and funded
risk can differ inside one simulated year.

  python -m trend_lab.risk_sweep [--sims 4000]
"""

import argparse

import numpy as np

from .combined import run
from .data import load_bars
from .metrics import daily_pnl
from .propsim import TRADEIFY_SELECT_FLEX_50K as R

LEVELS = [200, 300, 400, 500, 600, 750, 1000, 1250]


def boot_idx(n_days, n, rng, mean_block=5):
    out = np.empty(n, dtype=int)
    i = rng.integers(n_days)
    for k in range(n):
        if rng.random() < 1.0 / mean_block:
            i = rng.integers(n_days)
        out[k] = i
        i = (i + 1) % n_days
    return out


def year(idx, d_eval, d_fund, days=250):
    fees = paid = 0.0
    evals = funded = blown = payouts = 0
    mode = None
    for k in range(days):
        if mode is None:
            mode, bal, peak, best, nd, month_left = "eval", 0.0, 0.0, 0.0, 0, 0
            evals += 1
        line = min(peak - R["max_loss"], 0.0)
        if mode == "eval":
            if month_left == 0:
                fees += R["fee"]
                month_left = R["month_days"]
            day = d_eval[idx[k]]
            bal += day
            month_left -= 1
            nd += 1
            best = max(best, day)
            if bal <= line:
                mode = None
                continue
            peak = max(peak, bal)
            if bal >= R["target"] and nd >= R["min_days"] and best <= R["consistency"] * bal:
                funded += 1
                mode, bal, peak, wins = "funded", 0.0, 0.0, 0
            continue
        day = d_fund[idx[k]]
        bal += day
        if bal <= line:
            blown += 1
            mode = None
            continue
        peak = max(peak, bal)
        wins += day >= R["win_day"]
        if wins >= R["payout_days"] and bal > 0:
            amt = min(R["payout_cap"], R["payout_frac"] * bal)
            if amt >= R["min_payout"]:
                paid += amt * R["split"]
                payouts += 1
                bal -= amt
                peak = bal
                wins = 0
    return paid - fees, fees, evals, funded, blown, payouts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=4000)
    args = ap.parse_args()
    df = load_bars("15min")
    days = sorted(set(df["trade_day"]))
    paths = {}
    for ecf in (20, None):
        for lock in (1200.0, 1e9):
            for lv in LEVELS:
                _, live = run(df, risk_fixed=lv, ecf_len=ecf, a_scale_r=1.0, profit_lock=lock)
                d = daily_pnl(live, days).to_numpy()
                paths[(ecf, lock, lv)] = d
                print(f"ecf={ecf} lock={lock:.0f} risk={lv}: year net ${d.sum():,.0f}, worst day ${d.min():,.0f}",
                      flush=True)
    rng = np.random.default_rng(7)
    idxs = [boot_idx(len(days), 250, rng) for _ in range(args.sims)]
    for edge in (1.0, 0.5):
        print(f"\n=== edge x{edge} ===")
        adj = {k: v - (1 - edge) * v.mean() for k, v in paths.items()}
        rows = []
        for ecf in (20, None):
            for lock in (1200.0, 1e9):
                for re in LEVELS:
                    for rf in LEVELS:
                        res = np.array([year(ix, adj[(ecf, 1200.0, re)], adj[(ecf, lock, rf)]) for ix in idxs])
                        net = res[:, 0]
                        rows.append((net.mean(), np.median(net), np.percentile(net, 10), (net > 0).mean(),
                                     res[:, 2].mean(), res[:, 3].mean(), res[:, 4].mean(), res[:, 5].mean(),
                                     ecf, lock, re, rf))
        rows.sort(key=lambda r: -r[0])
        print("mean     median   p10      P>0  evals funded blown payouts  ks   funded-lock  eval$  funded$")
        show = rows[:15] + [r for r in rows if r[10] == r[11] and r[8] == 20 and r[9] == 1200.0]
        for r in show:
            print(f"{r[0]:8,.0f} {r[1]:8,.0f} {r[2]:8,.0f} {100 * r[3]:4.0f}% {r[4]:5.1f} {r[5]:6.2f} {r[6]:5.2f} "
                  f"{r[7]:7.2f}  {'on ' if r[8] else 'off'}  {'1200' if r[9] < 1e8 else 'none':>4}        "
                  f"{r[10]:5} {r[11]:6}")


if __name__ == "__main__":
    main()
