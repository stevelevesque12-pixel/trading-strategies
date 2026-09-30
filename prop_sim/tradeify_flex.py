"""Replay a TradingView trade list through Tradeify Select 50K eval + Select Flex funded rules.

Rules modeled (from Tradeify's public summaries, Sept 2026 -- verify on your dashboard):
  Eval:   profit target (default $3,000), $2,000 EOD-trailing drawdown enforced intraday,
          40% consistency (best day <= 40% of total profit), min 3 trading days, no DLL.
  Funded (Flex): $2,000 EOD-trailing drawdown, locks at start+$100 once EOD balance reaches
          start+$2,100 OR at the first payout. No DLL, no consistency.
          Payout after 5 winning days (>= $150 net each) since the last payout,
          amount = min(50% of (balance - start), $3,000), minimum $250, 90% split to trader.
Per-trade intraday low is approximated with TradingView's adverse-excursion column.
"""
import argparse
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

START = 50_000.0
DD = 2_000.0
LOCK = START + 100


@dataclass
class Acct:
    """Select account size parameters. Defaults = 50K. Larger sizes: verify on Tradeify's site."""
    start: float = 50_000.0
    dd: float = 2_000.0
    target: float = 3_000.0
    win_day: float = 150.0
    cap: float = 3_000.0
    fee: float = 159.0
    monthly: bool = True  # False = one-time fee per eval attempt
    funded_dd: float = None  # funded drawdown if different from eval drawdown


A50 = Acct()
# 150K per third-party summaries (Sept 2026): $9k target, $4.5k DD, $250 winning day,
# $4.5k payout cap, $221 one-time. Funded Flex DD assumed equal to eval DD.
A150 = Acct(150_000.0, 4_500.0, 9_000.0, 250.0, 4_500.0, 221.0, False)


def load_days(path):
    d = pd.read_csv(path, encoding="utf-8-sig")
    e = d[d.Type.str.startswith("Entry")].set_index("Trade number")
    x = d[d.Type.str.startswith("Exit")].set_index("Trade number")
    t = pd.DataFrame({
        "entry": pd.to_datetime(e["Date and time"]),
        "pnl": x["Net PnL USD"], "mae": x["Adverse excursion USD"],
    }).sort_values("entry")
    t["day"] = t.entry.dt.normalize()
    days = [(k, list(zip(g.pnl, g.mae))) for k, g in t.groupby("day")]
    return days


def day_result(trades, n, bal):
    """Return (end_balance, intraday_low) trading n contracts."""
    low = bal
    for pnl, mae in trades:
        low = min(low, bal + mae * n)
        bal += pnl * n
    return bal, low


def run_eval(days, i, n, target, consistency=0.40, acct=A50):
    START, DD = acct.start, acct.dd
    bal, peak, best, tdays = START, START, 0.0, 0
    for j in range(i, len(days)):
        bal2, low = day_result(days[j][1], n, bal)
        if low <= peak - DD:
            return "fail", j
        best = max(best, bal2 - bal)
        bal, tdays = bal2, tdays + 1
        peak = max(peak, bal)
        prof = bal - START
        if prof >= target and best <= consistency * prof and tdays >= 3:
            return "pass", j
    return "open", len(days) - 1


def contracts_for(policy, bal, floor):
    if isinstance(policy, int):
        return policy
    # dynamic: size off cushion to floor; policy = ("dyn", $ risk budget per contract, min, max)
    _, per, lo, hi = policy
    return max(lo, min(hi, int((bal - floor) // per)))


def run_funded(days, i, policy, min_profit_for_payout=0.0, acct=A50):
    START, LOCK = acct.start, acct.start + 100
    DD = acct.funded_dd or acct.dd
    bal, peak, locked, win_days = START, START, False, 0
    payouts = []
    for j in range(i, len(days)):
        floor = LOCK if locked else peak - DD
        n = contracts_for(policy, bal, floor)
        bal2, low = day_result(days[j][1], n, bal)
        if low <= floor:
            return payouts, j, "blown"
        if bal2 - bal >= acct.win_day:
            win_days += 1
        bal = bal2
        if not locked:
            peak = max(peak, bal)
            if bal >= START + DD + 100:
                locked = True
        prof = bal - START
        if win_days >= 5 and prof >= max(500, min_profit_for_payout):
            amt = min(0.5 * prof, acct.cap)
            if amt >= 250:
                payouts.append((days[j][0], amt))
                bal -= amt
                locked, win_days = True, 0
    return payouts, len(days) - 1, "open"


def campaign(days, eval_n, funded_policy, target, fee=None, start_idx=0, min_payout_profit=0.0, acct=A50):
    """Sequential real-time replay: buy eval, pass/fail, trade funded until blown, repeat."""
    i, evals, months, pays, log = start_idx, 0, 0, [], []
    while i < len(days) - 1:
        evals += 1
        res, j = run_eval(days, i, eval_n, target, acct=acct)
        months += max(1, math.ceil((days[j][0] - days[i][0]).days / 30)) if acct.monthly else 1
        if res != "pass":
            log.append((days[i][0].date(), "eval " + res, days[j][0].date(), 0))
            i = j + 1
            continue
        p, k, st = run_funded(days, j + 1, funded_policy, min_payout_profit, acct)
        pays += p
        log.append((days[j][0].date(), "funded " + st, days[k][0].date(), len(p)))
        i = k + 1
    gross = sum(a for _, a in pays) * 0.9
    fee = acct.fee if fee is None else fee
    return dict(evals=evals, fees=months * fee, payouts=len(pays), trader_cash=gross,
                net=gross - months * fee, log=log)




def bootstrap_days(days, n_days, rng, block=20, bdays_per_trade_day=1.84):
    """Block-bootstrap a synthetic path of strategy trading days with fake, evenly spaced dates."""
    out = []
    base = pd.Timestamp("2030-01-01")
    while len(out) < n_days:
        s = rng.integers(0, len(days) - block)
        for _, tr in days[s:s + block]:
            d = base + pd.Timedelta(days=len(out) * bdays_per_trade_day * 7 / 5)
            out.append((d, tr))
    return out[:n_days]


def monte_carlo(days, eval_n, funded_policy, target, min_payout_profit=0.0, horizon_days=270,
                paths=500, seed=0):
    rng = np.random.default_rng(seed)
    res = [campaign(bootstrap_days(days, horizon_days, rng), eval_n, funded_policy, target,
                    min_payout_profit=min_payout_profit) for _ in range(paths)]
    net = np.array([r["net"] for r in res])
    return dict(evals=np.mean([r["evals"] for r in res]), payouts=np.mean([r["payouts"] for r in res]),
                cash=np.mean([r["trader_cash"] for r in res]), fees=np.mean([r["fees"] for r in res]),
                net_mean=net.mean(), net_p10=np.percentile(net, 10), net_p50=np.median(net),
                p_loss=(net < 0).mean())


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Monte Carlo a trade list through Tradeify Select 50K Flex rules")
    ap.add_argument("csv", help="TradingView 'List of trades' export for 1 contract")
    ap.add_argument("--target", type=float, default=3000)
    ap.add_argument("--eval-n", type=int, default=3, help="contracts during eval")
    ap.add_argument("--funded-n", type=int, default=2, help="contracts once funded")
    ap.add_argument("--min-payout-profit", type=float, default=3000,
                    help="wait until balance-start >= this before requesting a payout")
    ap.add_argument("--since", default="2022-01-01", help="only bootstrap from trades on/after this date")
    ap.add_argument("--days", type=int, default=270, help="horizon in strategy trading days (~135/yr)")
    ap.add_argument("--paths", type=int, default=1000)
    a = ap.parse_args()
    days = [d for d in load_days(a.csv) if d[0] >= pd.Timestamp(a.since)]
    r = monte_carlo(days, a.eval_n, a.funded_n, a.target, a.min_payout_profit, a.days, a.paths)
    for k, v in r.items():
        print(f"{k:>10}: {v:,.2f}")
