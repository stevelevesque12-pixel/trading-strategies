"""Monte Carlo of a Tradeify Select 50K eval + Select Flex funded account,
driven by a TradingView "List of trades" CSV traded at 1 micro contract.

Days are bootstrapped (iid) from the trade list; each day's trades replay in
order and the intraday breach check uses each trade's adverse excursion.

Rules modeled (verify against your dashboard -- they change):
  Eval:  $3,000 target, $2,000 EOD trailing drawdown (breached intraday),
         best day <= 40% of total profit, min 3 days.
  Flex:  $2,000 EOD trailing drawdown, locks at start+$100 once EOD balance
         reaches start+$2,100 or on the first payout request; payout every
         5 winning days (>= $150), up to 50% of profit capped at $3,000,
         min payout $250, 90/10 split.

Usage:
  python -m prop_sim.tradeify_flex [tradelist.csv] --cost 1.5 --start 2023-01-01

Defaults to the MNQ 5-min ORB trade list in sample_data/tradelists/.
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

TARGET, DD, LOCK_AT, LOCK_FLOOR = 3000.0, 2000.0, 2100.0, 100.0
CONSISTENCY, MIN_DAYS = 0.40, 3
WIN_DAY, WIN_DAYS_PER_PAYOUT = 150.0, 5
PAYOUT_PCT, PAYOUT_CAP, MIN_PAYOUT, SPLIT = 0.5, 3000.0, 250.0, 0.9
DEFAULT_CSV = (Path(__file__).resolve().parent.parent / "sample_data" / "tradelists"
               / "PropQuantX_MNQ_5min_ORB_2026-09-30.csv")


def load_days(path, cost, start):
    """Per-day arrays of (net pnl, adverse excursion) for 1 contract."""
    d = pd.read_csv(path, encoding="utf-8-sig")
    e = d[d["Type"].str.startswith("Entry")].copy()
    e["dt"] = pd.to_datetime(e["Date and time"])
    e = e[e["dt"] >= start]
    e["day"] = e["dt"].dt.date
    return [(g["Net PnL USD"].to_numpy() - cost, g["Adverse excursion USD"].to_numpy())
            for _, g in e.groupby("day")]


class Sim:
    def __init__(self, days, seed=1):
        self.days = days
        self.rng = np.random.default_rng(seed)

    def day(self, n):
        pnl, mae = self.days[self.rng.integers(len(self.days))]
        run = worst = 0.0
        for p, m in zip(pnl, mae):
            worst = min(worst, run + m * n)
            run += p * n
        return run, worst

    def eval(self, n, max_days=250):
        """Returns (passed, days_used)."""
        bal = hwm = best = 0.0
        for k in range(1, max_days + 1):
            floor = LOCK_FLOOR if hwm >= LOCK_AT else hwm - DD
            pnl, worst = self.day(n)
            if bal + worst <= floor:
                return False, k
            bal += pnl
            best = max(best, pnl)
            hwm = max(hwm, bal)
            if bal >= TARGET and best <= CONSISTENCY * bal and k >= MIN_DAYS:
                return True, k
        return False, max_days

    def funded(self, n_before, n_after, max_days=1500):
        """Size n_before until the first payout, n_after afterwards.
        Returns (list of gross payouts, days survived)."""
        bal = hwm = 0.0
        locked, wins, paid = False, 0, []
        for k in range(1, max_days + 1):
            n = n_after if paid else n_before
            floor = LOCK_FLOOR if locked else hwm - DD
            pnl, worst = self.day(n)
            if bal + worst <= floor:
                return paid, k
            bal += pnl
            hwm = max(hwm, bal)
            locked = locked or hwm >= LOCK_AT
            wins += pnl >= WIN_DAY
            if wins >= WIN_DAYS_PER_PAYOUT and bal > 0:
                amt = min(PAYOUT_PCT * bal, PAYOUT_CAP)
                if amt >= MIN_PAYOUT and bal - amt > LOCK_FLOOR:
                    paid.append(amt)
                    bal -= amt
                    locked, wins = True, 0
        return paid, max_days


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    ap.add_argument("--cost", type=float, default=1.5, help="$ per micro round trip (commission + slippage)")
    ap.add_argument("--start", default="2023-01-01", help="only sample days from this date on")
    ap.add_argument("--fee", type=float, default=150.0, help="eval fee per attempt")
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--eval-sizes", default="4,5,6,8")
    ap.add_argument("--eval-size", type=int, default=5, help="eval size used for per-attempt EV")
    ap.add_argument("--plans", default="6:3,6:4,6:6,4:3,8:4", help="funded before:after sizes")
    ap.add_argument("--seed", type=int, default=1)
    a = ap.parse_args()

    sim = Sim(load_days(a.csv, a.cost, a.start), a.seed)
    print(f"{len(sim.days)} days from {a.start}, cost ${a.cost}/RT, fee ${a.fee}, {a.runs} runs\n")

    print("EVAL  size  pass%  avg days")
    evals = {}
    for n in sorted({int(x) for x in a.eval_sizes.split(",")} | {a.eval_size}):
        r = [sim.eval(n) for _ in range(a.runs)]
        evals[n] = (np.mean([p for p, _ in r]), np.mean([d for _, d in r]))
        print(f"      {n:4d}  {evals[n][0] * 100:5.1f}  {evals[n][1]:8.0f}")

    ps, eval_days = evals[a.eval_size]
    print(f"\nFUNDED FLEX (per-attempt columns use eval size {a.eval_size})")
    print("plan   payouts  P(>=1)  gross$  avg days | payouts/attempt  net EV/attempt  net EV/month")
    for plan in a.plans.split(","):
        nb, na = (int(x) for x in plan.split(":"))
        r = [sim.funded(nb, na) for _ in range(a.runs)]
        k = np.array([len(p) for p, _ in r])
        g = np.array([sum(p) for p, _ in r])
        life = np.mean([d for _, d in r])
        ev = ps * SPLIT * g.mean() - a.fee
        days = eval_days + ps * life
        print(f"{plan:6s} {k.mean():7.2f}  {np.mean(k >= 1) * 100:5.1f}  {g.mean():6.0f}  {life:8.0f} |"
              f" {ps * k.mean():15.2f}  {ev:14.0f}  {ev / days * 21:12.0f}")


if __name__ == "__main__":
    main()
