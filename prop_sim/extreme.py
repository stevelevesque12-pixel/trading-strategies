"""The "Extreme" plan: Tradeify Growth 50K accounts at up to 5 firms x 5 funded.

Plan (see docs/EXTREME_PLAN.md):
  * size 10 / 10 / 6 MNQ: eval 10, funded 10 until the first payout, then 6;
  * firms open one after another: the next firm opens once the current one
    holds 5 funded accounts;
  * every open firm keeps one eval running per empty funded slot
    (5 - funded), so a breached eval or funded account is replaced at once;
  * every firm is modelled with the Tradeify Growth 50K rules below.

Growth 50K rules modelled (from public summaries, not confirmed with Tradeify):
  Eval:   +$3,000 target, $2,000 EOD trailing drawdown (an intraday touch
          fails; the floor locks at +$100 once EOD balance reaches +$2,100),
          $1,250 soft daily loss limit (trading stops for the day, the account
          survives), no consistency rule.
  Funded: same drawdown and daily loss limit. A payout needs balance
          >= +$3,000 ($53,000), 5 days of +$150 since the last payout, and the
          best day since the last payout <= 35% of the profit since then.
          ASSUMED: amount = min(cap, balance - $1,500), caps 1,500 / 2,000 /
          2,500 / 3,000 for payouts 1-4+, floor locks at +$100 on the first
          payout. 90% profit split.

Usage:
  python -m prop_sim.extreme [tradelist.csv] [--start 2023-01-01] [--fee 145]
      [--plan 10:10:6] [--firms 5] [--cap 5] [--one-eval] [--all-open]
"""
import argparse

import numpy as np
import pandas as pd

from prop_sim.tradeify_flex import DEFAULT_CSV
from prop_sim.yearly import load_dated_days

TARGET, DD, LOCK_AT, LOCK_FLOOR = 3000.0, 2000.0, 2100.0, 100.0
DAILY_LOSS_LIMIT = 1250.0
PAYOUT_MIN_BALANCE, PAYOUT_BUFFER = 3000.0, 1500.0
WIN_DAY, WIN_DAYS_PER_PAYOUT, CONSISTENCY = 150.0, 5, 0.35
CAPS = [1500.0, 2000.0, 2500.0, 3000.0]
MIN_PAYOUT, SPLIT = 250.0, 0.9


def day_result(pnl, mae, n):
    """(day pnl, worst intraday pnl) for n contracts, flattened at the daily loss limit."""
    run = worst = 0.0
    for p, m in zip(pnl, mae):
        low = run + m * n
        if low <= -DAILY_LOSS_LIMIT:
            return -DAILY_LOSS_LIMIT, min(worst, -DAILY_LOSS_LIMIT)
        worst = min(worst, low)
        run += p * n
    return run, worst


class Eval:
    def __init__(self):
        self.bal = self.hwm = 0.0

    def step(self, day):
        """'fail', 'pass' or None."""
        pnl, worst = day
        floor = LOCK_FLOOR if self.hwm >= LOCK_AT else self.hwm - DD
        if self.bal + worst <= floor:
            return "fail"
        self.bal += pnl
        self.hwm = max(self.hwm, self.bal)
        return "pass" if self.bal >= TARGET else None


class Funded:
    def __init__(self):
        self.bal = self.hwm = self.since = self.best = 0.0
        self.locked, self.wins, self.payouts = False, 0, 0

    def step(self, day):
        """(alive, payout amount)."""
        pnl, worst = day
        floor = LOCK_FLOOR if self.locked else self.hwm - DD
        if self.bal + worst <= floor:
            return False, 0.0
        self.bal += pnl
        self.hwm = max(self.hwm, self.bal)
        self.locked = self.locked or self.hwm >= LOCK_AT
        self.since += pnl
        self.best = max(self.best, pnl)
        self.wins += pnl >= WIN_DAY
        if (self.wins >= WIN_DAYS_PER_PAYOUT and self.bal >= PAYOUT_MIN_BALANCE
                and self.since > 0 and self.best <= CONSISTENCY * self.since):
            amt = min(CAPS[min(self.payouts, len(CAPS) - 1)], self.bal - PAYOUT_BUFFER)
            if amt >= MIN_PAYOUT:
                self.bal -= amt
                self.payouts += 1
                self.locked, self.wins, self.since, self.best = True, 0, 0.0, 0.0
                return True, amt
        return True, 0.0


def simulate(days, plan=(10, 10, 6), fee=145.0, firms=5, cap=5, parallel=True, all_open=False):
    """Run the multi-firm plan over a sequence of (pnl, mae) days, all accounts on the same signal."""
    ne, nb, na = plan
    state = [dict(open=(j == 0 or all_open), evals=[], funded=[]) for j in range(firms)]
    bought, paid, held, full_at = 0, [], [], None
    for t, (pnl, mae) in enumerate(days):
        r_eval, r_before, r_after = (day_result(pnl, mae, n) for n in (ne, nb, na))
        for j, f in enumerate(state):
            if not f["open"]:
                continue
            want = cap - len(f["funded"]) if parallel else int(len(f["funded"]) < cap)
            while len(f["evals"]) < want:
                f["evals"].append(Eval())
                bought += 1
            alive = []
            for a in f["funded"]:
                ok, amt = a.step(r_after if a.payouts else r_before)
                if amt:
                    paid.append((t, amt))
                if ok:
                    alive.append(a)
            f["funded"] = alive
            running = []
            for e in f["evals"]:
                res = e.step(r_eval)
                if res == "pass":
                    f["funded"].append(Funded())
                elif res is None:
                    running.append(e)
            f["evals"] = running
            if len(f["funded"]) >= cap and j + 1 < firms:
                state[j + 1]["open"] = True
        held.append(sum(len(f["funded"]) for f in state))
        if full_at is None and held[-1] >= firms * cap:
            full_at = t
    gross = sum(a for _, a in paid)
    return dict(payouts=len(paid), gross=gross, net=SPLIT * gross - fee * bought, evals=bought,
                avg_funded=float(np.mean(held)), max_funded=max(held),
                firms_open=sum(f["open"] for f in state), full_at=full_at)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    ap.add_argument("--cost", type=float, default=1.5, help="$ per micro round trip")
    ap.add_argument("--fee", type=float, default=145.0, help="eval price")
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--plan", default="10:10:6", help="eval:funded-before:funded-after-first-payout (MNQ)")
    ap.add_argument("--firms", type=int, default=5)
    ap.add_argument("--cap", type=int, default=5, help="funded accounts per firm")
    ap.add_argument("--one-eval", action="store_true", help="one eval at a time per firm instead of one per empty slot")
    ap.add_argument("--all-open", action="store_true", help="open every firm on day one")
    ap.add_argument("--step", type=int, default=5, help="days between one-year window starts")
    a = ap.parse_args()

    dated = load_dated_days(a.csv, a.cost)
    days = [(p, m) for _, p, m in dated]
    start = pd.Timestamp(a.start).date()
    first = next(i for i, (d, _, _) in enumerate(dated) if d >= start)
    kw = dict(plan=tuple(int(x) for x in a.plan.split(":")), fee=a.fee, firms=a.firms, cap=a.cap,
              parallel=not a.one_eval, all_open=a.all_open)

    runs = [simulate(days[i:i + 252], **kw) for i in range(first, len(days) - 252 + 1, a.step)]
    net = np.array([r["net"] for r in runs])
    k = np.array([r["payouts"] for r in runs])
    ev = np.array([r["evals"] for r in runs])
    print(f"EXTREME plan {a.plan}, {a.firms} firms x {a.cap}, "
          f"{'one eval at a time' if a.one_eval else 'one eval per empty slot'}"
          f"{', all firms open' if a.all_open else ', firms open in turn'}; fee ${a.fee:.0f}, cost ${a.cost}/RT")
    print(f"\nONE-YEAR WINDOWS from {a.start} ({len(runs)})")
    print(f"  payouts/yr     mean {k.mean():.0f}  median {np.median(k):.0f}")
    print(f"  net/yr         mean ${net.mean():,.0f}  median ${np.median(net):,.0f}  "
          f"p10 ${np.percentile(net, 10):,.0f}  p90 ${np.percentile(net, 90):,.0f}  worst ${net.min():,.0f}")
    print(f"  losing years   {np.mean(net < 0) * 100:.0f}%")
    print(f"  evals/yr       {ev.mean():.0f} (${ev.mean() * a.fee:,.0f})")
    print(f"  funded held    avg {np.mean([r['avg_funded'] for r in runs]):.1f}  "
          f"max {np.mean([r['max_funded'] for r in runs]):.1f}  firms opened {np.mean([r['firms_open'] for r in runs]):.1f}")

    r = simulate(days[first:], **kw)
    yrs = (len(days) - first) / 252
    print(f"\nCONTINUOUS {a.start} to {dated[-1][0]} ({yrs:.1f} yrs)")
    print(f"  payouts {r['payouts']}  gross ${r['gross']:,.0f}  net ${r['net']:,.0f}  evals {r['evals']}")
    print(f"  funded held avg {r['avg_funded']:.1f}  max {r['max_funded']}  firms opened {r['firms_open']}  "
          f"all {a.firms * a.cap} funded after: {r['full_at'] if r['full_at'] is not None else 'never'} trading days")


if __name__ == "__main__":
    main()
