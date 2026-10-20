"""Tradeify 50K Select (Flex) account simulation for Morning Dip trade logs.

Rules modelled (from public summaries, Oct 2026; check Tradeify's help pages before relying on them):
  Evaluation: +$3,000 target, $2,000 end-of-day trailing drawdown (breached intraday too),
              best day <= 40% of total profit, no daily loss limit, fee per attempt.
  Funded Flex: $2,000 EOD trailing drawdown, locks at start+$100 once EOD balance reaches
              start+$2,100 or at the first payout. Payout after 5 winning days (>= $150) since the
              last payout: min(50% of profit, cap), at least $250, 90% to the trader; each later
              request needs a higher balance than the previous one.
A failed evaluation or funded account is replaced the next session by a new evaluation.

Trades are sized in MNQ ($2/point, round-trip cost per contract). Intraday breach uses each
trade's maximum adverse excursion, so the floor check is conservative within a trade.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Rules:
    start: float = 50_000.
    target: float = 3_000.
    drawdown: float = 2_000.
    consistency: float = .40
    eval_fee: float = 100.
    win_day: float = 150.
    win_days: int = 5
    cap: float = 2_500.
    min_payout: float = 250.
    share: float = .5
    split: float = .9
    lock_at: float = 2_100.
    lock_floor: float = 100.


def day_trades(trades, mnq, rt_cost=1.82, pv=2.):
    """trades: date, pts (gross points per contract), mae (points), in time order."""
    t = trades.assign(pnl=trades.pts * pv * mnq - rt_cost * mnq,
                      worst=-trades.mae * pv * mnq - rt_cost / 2 * mnq)
    return {d: g[['pnl', 'worst']].to_numpy() for d, g in t.groupby('date', sort=False)}


def simulate(sessions, by_day, r=Rules()):
    """Walk the sessions once; returns a summary dict and an event log."""
    log = []
    fees = paid = 0.
    evals = passes = funded_busts = 0
    phase = None

    def new_eval(day):
        nonlocal fees, evals
        fees += r.eval_fee; evals += 1
        log.append((day, 'buy eval'))
        return dict(kind='eval', bal=r.start, peak=r.start, floor=r.start - r.drawdown, days={})

    acct = None
    for day in sessions:
        if acct is None:
            acct = new_eval(day)
        open_bal = acct['bal']
        dead = False
        for pnl, worst in by_day.get(day, ()):
            if acct['bal'] + worst < acct['floor'] - 1e-9:
                dead = True; break
            acct['bal'] += pnl
            if acct['bal'] < acct['floor'] - 1e-9:
                dead = True; break
        day_pnl = acct['bal'] - open_bal
        if dead:
            if acct['kind'] == 'funded':
                funded_busts += 1
            log.append((day, f"{acct['kind']} breached"))
            acct = None
            continue
        # End of day
        if acct['bal'] > acct['peak']:
            acct['peak'] = acct['bal']
        if acct['kind'] == 'eval':
            acct['days'][day] = day_pnl
            acct['floor'] = max(acct['floor'], acct['peak'] - r.drawdown)
            profit = acct['bal'] - r.start
            best = max(acct['days'].values())
            if profit >= r.target and best <= r.consistency * profit and len(acct['days']) >= 3:
                passes += 1
                log.append((day, 'passed eval'))
                acct = dict(kind='funded', bal=r.start, peak=r.start, floor=r.start - r.drawdown,
                            locked=False, wins=0, last_req=-np.inf)
        else:
            if not acct['locked']:
                acct['floor'] = max(acct['floor'], acct['peak'] - r.drawdown)
                if acct['peak'] >= r.start + r.lock_at:
                    acct['floor'] = r.start + r.lock_floor; acct['locked'] = True
            if day_pnl >= r.win_day:
                acct['wins'] += 1
            profit = acct['bal'] - r.start
            if acct['wins'] >= r.win_days and acct['bal'] > acct['last_req']:
                amount = min(r.share * profit, r.cap)
                if amount >= r.min_payout:
                    acct['last_req'] = acct['bal']
                    acct['bal'] -= amount
                    acct['wins'] = 0
                    acct['floor'] = r.start + r.lock_floor; acct['locked'] = True
                    paid += amount * r.split
                    log.append((day, f'payout {amount:.0f}'))
    status = 'none' if acct is None else acct['kind']
    open_profit = 0. if acct is None or acct['kind'] == 'eval' else acct['bal'] - r.start
    return dict(evals=evals, passes=passes, funded_busts=funded_busts, payouts=sum(1 for _, e in log if e.startswith('payout')),
                paid=paid, fees=fees, net_cash=paid - fees, end=status, funded_open_profit=open_profit), log


def bootstrap(sessions, by_day, r=Rules(), n=2000, days=252, seed=0):
    """Resample whole sessions (keeping each day's trade sequence) into n synthetic years."""
    rng = np.random.default_rng(seed)
    sessions = list(sessions)
    rows = []
    for _ in range(n):
        pick = rng.integers(0, len(sessions), days)
        seq = [f'{k}#{sessions[i]}' for k, i in enumerate(pick)]
        bd = {f'{k}#{sessions[i]}': by_day[sessions[i]] for k, i in enumerate(pick) if sessions[i] in by_day}
        rows.append(simulate(seq, bd, r)[0])
    return pd.DataFrame(rows)
