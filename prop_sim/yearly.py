"""One account slot run for a year under the proposed plan: eval at 5 MNQ,
Flex at 6 MNQ until the first payout, then 3 MNQ; on any failure buy a new
eval and start over. Counts payouts, evals bought and net cash per year.

Two views:
  * Monte Carlo: 252 days bootstrapped from --start onward.
  * Rolling: every 252-day window of real, in-order days from --start on.
  * Historical: each calendar year replayed day by day in real order,
    starting fresh on its first trading day.

Usage:
  python -m prop_sim.yearly [tradelist.csv] --cost 1.5 --fee 150
"""
import argparse

import numpy as np
import pandas as pd

from prop_sim.tradeify_flex import (
    CONSISTENCY, DD, DEFAULT_CSV, LOCK_AT, LOCK_FLOOR, MIN_DAYS, MIN_PAYOUT,
    PAYOUT_CAP, PAYOUT_PCT, SPLIT, TARGET, WIN_DAY, WIN_DAYS_PER_PAYOUT,
)


def load_dated_days(path, cost):
    d = pd.read_csv(path, encoding="utf-8-sig")
    e = d[d["Type"].str.startswith("Entry")].copy()
    e["day"] = pd.to_datetime(e["Date and time"]).dt.date
    return [(day, g["Net PnL USD"].to_numpy() - cost, g["Adverse excursion USD"].to_numpy())
            for day, g in e.groupby("day")]


def day_result(pnl, mae, n):
    run = worst = 0.0
    for p, m in zip(pnl, mae):
        worst = min(worst, run + m * n)
        run += p * n
    return run, worst


def run_slot(days, fee, n_eval=5, n_before=6, n_after=3):
    """Walk one slot through a sequence of (pnl, mae) days.
    Returns dict of payouts (gross list), evals bought, passes, funded failures."""
    out = {"payouts": [], "evals": 1, "passes": 0, "funded_fails": 0}
    phase = "eval"
    bal = hwm = best = 0.0
    k = wins = 0
    locked = paid_here = False
    for pnl, mae in days:
        if phase == "eval":
            n = n_eval
            floor = LOCK_FLOOR if hwm >= LOCK_AT else hwm - DD
        else:
            n = n_after if paid_here else n_before
            floor = LOCK_FLOOR if locked else hwm - DD
        run, worst = day_result(pnl, mae, n)
        if bal + worst <= floor:  # blown: buy a new eval, start tomorrow
            if phase == "funded":
                out["funded_fails"] += 1
            out["evals"] += 1
            phase, bal, hwm, best, k, wins, locked, paid_here = "eval", 0.0, 0.0, 0.0, 0, 0, False, False
            continue
        bal += run
        hwm = max(hwm, bal)
        k += 1
        if phase == "eval":
            best = max(best, run)
            if bal >= TARGET and best <= CONSISTENCY * bal and k >= MIN_DAYS:
                out["passes"] += 1
                phase, bal, hwm, wins, locked, paid_here = "funded", 0.0, 0.0, 0, False, False
        else:
            locked = locked or hwm >= LOCK_AT
            wins += run >= WIN_DAY
            if wins >= WIN_DAYS_PER_PAYOUT and bal > 0:
                amt = min(PAYOUT_PCT * bal, PAYOUT_CAP)
                if amt >= MIN_PAYOUT and bal - amt > LOCK_FLOOR:
                    out["payouts"].append(amt)
                    bal -= amt
                    locked, paid_here, wins = True, True, 0
    out["net"] = SPLIT * sum(out["payouts"]) - fee * out["evals"]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    ap.add_argument("--cost", type=float, default=1.5)
    ap.add_argument("--fee", type=float, default=150.0)
    ap.add_argument("--start", default="2023-01-01", help="Monte Carlo samples days from here on")
    ap.add_argument("--runs", type=int, default=5000)
    ap.add_argument("--days", type=int, default=252)
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    dated = load_dated_days(a.csv, a.cost)
    start = pd.Timestamp(a.start).date()
    pool = [(p, m) for d, p, m in dated if d >= start]
    rng = np.random.default_rng(a.seed)

    res = [run_slot([pool[i] for i in rng.integers(len(pool), size=a.days)], a.fee)
           for _ in range(a.runs)]
    k = np.array([len(r["payouts"]) for r in res])
    net = np.array([r["net"] for r in res])
    ev = np.array([r["evals"] for r in res])
    gross = np.array([sum(r["payouts"]) for r in res])
    q = lambda x: "  ".join(f"p{p}={np.percentile(x, p):,.0f}" for p in (10, 25, 50, 75, 90))
    print(f"MONTE CARLO: one slot, {a.days} days sampled from {a.start}, cost ${a.cost}/RT, fee ${a.fee}, {a.runs} runs")
    print(f"  payouts/yr   mean {k.mean():.1f}   {q(k)}")
    print(f"  evals bought mean {ev.mean():.1f}   {q(ev)}")
    print(f"  gross paid   mean ${gross.mean():,.0f}")
    print(f"  net to you   mean ${net.mean():,.0f}   {q(net)}")
    print(f"  P(net < 0) = {np.mean(net < 0) * 100:.0f}%   P(0 payouts) = {np.mean(k == 0) * 100:.0f}%")
    print("  payouts/yr distribution:", {int(v): f"{c / len(k) * 100:.0f}%" for v, c in zip(*np.unique(k, return_counts=True))})

    ordered = [(p, m) for _, p, m in dated]
    first = next(i for i, (d, _, _) in enumerate(dated) if d >= start)
    win = [run_slot(ordered[i:i + a.days], a.fee) for i in range(first, len(ordered) - a.days + 1)]
    wk = np.array([len(r["payouts"]) for r in win])
    wn = np.array([r["net"] for r in win])
    print(f"\nROLLING: {len(win)} real {a.days}-day windows starting from {a.start}")
    print(f"  payouts/yr   mean {wk.mean():.1f}   {q(wk)}")
    print(f"  net to you   mean ${wn.mean():,.0f}   {q(wn)}")
    print(f"  P(net < 0) = {np.mean(wn < 0) * 100:.0f}%   P(0 payouts) = {np.mean(wk == 0) * 100:.0f}%")

    print("\nHISTORICAL: each calendar year replayed in order, fresh start on day 1")
    print("  year  days  evals  passes  payouts  gross$   net$")
    by_year = {}
    for d, p, m in dated:
        by_year.setdefault(d.year, []).append((p, m))
    for y, days in sorted(by_year.items()):
        r = run_slot(days, a.fee)
        print(f"  {y}  {len(days):4d}  {r['evals']:5d}  {r['passes']:6d}  {len(r['payouts']):7d}"
              f"  {sum(r['payouts']):6,.0f}  {r['net']:6,.0f}")


if __name__ == "__main__":
    main()
