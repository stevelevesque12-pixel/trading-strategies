"""Balanced single-account plan for a Tradeify Growth 50K account.

Plan:
  * size 10 / 10 / 6 MNQ: eval at 10, funded at 10 until the first payout, then 6;
  * regime filter: trade only when the ORB's own last 60 trading days are net
    positive at 1 contract after costs; otherwise stand aside that day;
  * eval brake: after 2 failed evals in a row, wait 20 trading days before
    buying the next one;
  * request every payout as soon as the Growth rules allow it.

Growth rules are the ones modelled in prop_sim.extreme (eval target, EOD
drawdown with lock, soft $1,250 daily loss limit, $53,000 payout balance,
5 days of $150+, 35% consistency, stepped caps). `run` keeps a detailed event
log for one account slot.

Usage:  python -m prop_sim.growth_balanced [--start 2023-01-01] [--plan 10:10:6]
            [--lookback 60] [--brake 2:20] [--fee 145]
"""
import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd

from prop_sim.extreme import (CONSISTENCY, DAILY_LOSS_LIMIT, PAYOUT_MIN_BALANCE, SPLIT, WIN_DAY,
                              WIN_DAYS_PER_PAYOUT, Eval, Funded, day_result)
from prop_sim.tradeify_flex import DEFAULT_CSV
from prop_sim.yearly import load_dated_days


@dataclass
class Plan:
    ne: int = 10
    nb: int = 10
    na: int = 6
    lookback: int | None = 60    # regime filter window in trading days; None = always trade
    brake_fails: int | None = 2  # consecutive failed evals that trigger a pause; None = never
    brake_days: int = 20
    fee: float = 145.0


class History:
    """Trade-list days plus 1-contract daily pnl, shared by every run."""

    def __init__(self, csv=str(DEFAULT_CSV), cost=1.5):
        dated = load_dated_days(csv, cost)
        self.dates = [d for d, _, _ in dated]
        self.days = [(p, m) for _, p, m in dated]
        self.one = np.array([float(sum(p)) for p, _ in self.days])
        self._cache = {}

    def index(self, date):
        d = pd.Timestamp(date).date()
        return next(i for i, x in enumerate(self.dates) if x >= d)

    def result(self, t, n):
        key = (t, n)
        if key not in self._cache:
            self._cache[key] = day_result(*self.days[t], n)
        return self._cache[key]

    def filter_on(self, t, lookback):
        return lookback is None or self.one[max(0, t - lookback):t].sum() > 0


def run(h: History, i0: int, i1: int, plan: Plan):
    """Run one account slot over absolute day indexes [i0, i1). Returns totals and an event log."""
    events = []
    ev, fu = Eval(), None
    evals, fails_row, wait = 1, 0, 0
    ev_start, fu_start, last_pay = i0, None, None
    events.append(dict(t=i0, kind="eval_start"))
    idle = 0
    for t in range(i0, i1):
        if ev is None and fu is None:          # eval brake: waiting before the next eval
            wait -= 1
            if wait <= 0:
                ev, evals, ev_start = Eval(), evals + 1, t + 1
                events.append(dict(t=t + 1, kind="eval_start"))
            continue
        if not h.filter_on(t, plan.lookback):
            idle += 1
            continue
        if fu is None:
            res = h.result(t, plan.ne)
            st = ev.step(res)
            if res[0] <= -DAILY_LOSS_LIMIT:
                events.append(dict(t=t, kind="dll", where="eval"))
            if st == "fail":
                fails_row += 1
                events.append(dict(t=t, kind="eval_fail", days=t - ev_start + 1))
                if plan.brake_fails and fails_row >= plan.brake_fails:
                    ev, wait, fails_row = None, plan.brake_days, 0
                    events.append(dict(t=t, kind="brake"))
                else:
                    ev, evals, ev_start = Eval(), evals + 1, t + 1
                    events.append(dict(t=t + 1, kind="eval_start"))
            elif st == "pass":
                fails_row = 0
                events.append(dict(t=t, kind="eval_pass", days=t - ev_start + 1))
                ev, fu, fu_start, last_pay = None, Funded(), t + 1, t + 1
        else:
            res = h.result(t, plan.na if fu.payouts else plan.nb)
            would_qualify = (fu.wins + (res[0] >= WIN_DAY) >= WIN_DAYS_PER_PAYOUT
                             and fu.bal + res[0] >= PAYOUT_MIN_BALANCE)
            blocked = would_qualify and max(fu.best, res[0]) > CONSISTENCY * (fu.since + res[0])
            ok, amt = fu.step(res)
            if res[0] <= -DAILY_LOSS_LIMIT:
                events.append(dict(t=t, kind="dll", where="funded"))
            if amt:
                events.append(dict(t=t, kind="payout", n=fu.payouts, amount=amt,
                                   days=t - last_pay + 1, since_funded=t - fu_start + 1))
                last_pay = t + 1
            elif ok and blocked:
                events.append(dict(t=t, kind="consistency_wait"))
            if not ok:
                events.append(dict(t=t, kind="funded_fail", payouts=fu.payouts, days=t - fu_start + 1))
                fu, ev, evals, ev_start = None, Eval(), evals + 1, t + 1
                events.append(dict(t=t + 1, kind="eval_start"))
    gross = sum(e["amount"] for e in events if e["kind"] == "payout")
    return dict(events=events, evals=evals, payouts=sum(e["kind"] == "payout" for e in events),
                gross=gross, net=SPLIT * gross - plan.fee * evals, idle=idle)


def windows(h: History, start, end, plan: Plan, length=252, step=4):
    i0, i1 = h.index(start), h.index(end) if end else len(h.days)
    return [run(h, i, i + length, plan) for i in range(i0, i1 - length + 1, step)]


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("csv", nargs="?", default=str(DEFAULT_CSV))
    ap.add_argument("--cost", type=float, default=1.5)
    ap.add_argument("--start", default="2023-01-01")
    ap.add_argument("--end", default=None, help="last window must end before this date")
    ap.add_argument("--plan", default="10:10:6")
    ap.add_argument("--lookback", type=int, default=60, help="0 disables the regime filter")
    ap.add_argument("--brake", default="2:20", help="fails:days, or 0 to disable")
    ap.add_argument("--fee", type=float, default=145.0)
    a = ap.parse_args()
    ne, nb, na = (int(x) for x in a.plan.split(":"))
    bf, bd = (int(x) for x in a.brake.split(":")) if a.brake != "0" else (None, 0)
    plan = Plan(ne, nb, na, a.lookback or None, bf, bd, a.fee)
    h = History(a.csv, a.cost)
    res = windows(h, a.start, a.end, plan)
    net = np.array([r["net"] for r in res])
    k = np.array([r["payouts"] for r in res])
    ev = np.array([r["evals"] for r in res])
    print(f"Plan {a.plan}, filter {plan.lookback}, brake {a.brake}, fee ${a.fee:.0f}, "
          f"from {a.start} ({len(res)} one-year windows)")
    print(f"  payouts/yr  mean {k.mean():.1f}  median {np.median(k):.0f}")
    print(f"  net/yr      mean ${net.mean():,.0f}  median ${np.median(net):,.0f}  "
          f"p10 ${np.percentile(net, 10):,.0f}  p90 ${np.percentile(net, 90):,.0f}  losing {np.mean(net < 0) * 100:.0f}%")
    print(f"  evals/yr    {ev.mean():.1f}   days standing aside {np.mean([r['idle'] for r in res]):.0f}")


if __name__ == "__main__":
    main()
