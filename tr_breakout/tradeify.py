"""
Monte Carlo of the TR1 breakout on a Tradeify Select Flex 50K: evaluation,
then the funded Flex account with payouts.

Rules modeled (from Tradeify's published Select/Flex terms as of Sep 2026 --
re-check before relying on them, they change):

Evaluation
  - +$3,000 profit target (older accounts: $2,500 -> --target 2500)
  - $2,000 end-of-day trailing max loss; floor locks at $50,100 once the EOD
    balance reaches $52,100
  - no daily loss limit
  - 40% consistency: best day <= 40% of total profit. Not a failure -- you
    just keep trading until it holds (with the target still met).
  - max 4 NQ / 40 MNQ (irrelevant at 1-3 MNQ)

Funded (Select Flex)
  - same $2,000 EOD trailing floor, locked at $50,100 at EOD $52,100 or the
    first payout, whichever comes first
  - no daily loss limit, no consistency rule
  - payout eligible every 5 winning days (day P&L >= $150) since the last
    payout; withdraw up to 50% of profit above $50,000, capped at $2,500;
    trader keeps 90%
  - contract scaling starts at 20 MNQ (irrelevant at 1-3 MNQ)

Breaches are checked on closed-trade P&L (worst point within the day), not
open P&L -- the floor is EOD-based, so this matches how Tradeify measures it
except for the intraday liquidation of the account when equity itself goes
below the floor, which this can't see.

    python -m tr_breakout.tradeify --data <parquet> --from 2022-01-01 --block 20
"""

import argparse

import numpy as np
import pandas as pd

from backtest.data import load_1m_csv
from tr_breakout.monte_carlo import day_table, sample_idx
from tr_breakout.strategy import Config

ACCOUNT = 50_000.0


def floor_of(peak, locked, max_loss=2_000.0, lock_floor=100.0):
    f = np.where(locked, lock_floor, peak - max_loss)
    return np.minimum(f, lock_floor)


def eval_mc(tot, wst, target, max_days, consistency=0.40, lock_trigger=2_100.0):
    n = tot.shape[0]
    bal = np.zeros(n); peak = np.zeros(n); best = np.zeros(n)
    locked = np.zeros(n, bool)
    status = np.zeros(n, int); used = np.full(n, max_days)
    for d in range(max_days):
        live = status == 0
        if not live.any():
            break
        fl = floor_of(peak, locked)
        fail = live & (bal + np.minimum(wst[:, d], tot[:, d]) <= fl)
        step = live & ~fail
        bal = np.where(step, bal + tot[:, d], bal)
        best = np.where(step, np.maximum(best, tot[:, d]), best)
        peak = np.maximum(peak, bal)
        locked |= peak >= lock_trigger
        ok = step & (bal >= target) & (best <= consistency * bal)
        status[fail], status[ok] = -1, 1
        used[fail | ok] = d + 1
    return status, used


def funded_mc(tot, wst, win_day=150.0, need_wins=5, pct=0.5, cap=2_500.0, split=0.9,
              lock_trigger=2_100.0, cushion=0.0):
    """`cushion`: your own policy -- only withdraw what leaves at least this much
    profit in the account (0 = always take the max Tradeify allows)."""
    n, days = tot.shape
    bal = np.zeros(n); peak = np.zeros(n)
    locked = np.zeros(n, bool); alive = np.ones(n, bool)
    wins = np.zeros(n, int); paid = np.zeros(n); n_pay = np.zeros(n, int)
    first_pay = np.full(n, -1); bust_day = np.full(n, -1)
    for d in range(days):
        fl = floor_of(peak, locked)
        bust = alive & (bal + np.minimum(wst[:, d], tot[:, d]) <= fl)
        alive &= ~bust
        bust_day[bust] = d + 1
        bal = np.where(alive, bal + tot[:, d], bal)
        peak = np.where(alive, np.maximum(peak, bal), peak)
        locked |= alive & (peak >= lock_trigger)
        wins += alive & (tot[:, d] >= win_day)
        elig = alive & (wins >= need_wins) & (bal > 0)
        amt = np.minimum(np.minimum(pct * bal, cap), bal - cushion)
        amt = np.floor(np.where(elig & (amt >= 250), amt, 0.0))  # skip tiny payouts
        bal -= amt
        paid += split * amt
        n_pay += amt > 0
        first_pay = np.where((amt > 0) & (first_pay < 0), d + 1, first_pay)
        wins = np.where(amt > 0, 0, wins)
        locked |= amt > 0
    return paid, n_pay, first_pay, bust_day, bal


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--block", type=int, default=20)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--risk", type=float, nargs="+", default=[200, 250, 300, 400])
    ap.add_argument("--target", type=float, default=3_000)
    ap.add_argument("--eval-days", type=int, default=120)
    ap.add_argument("--funded-days", type=int, default=252)
    ap.add_argument("--haircut", type=float, default=0.0)
    ap.add_argument("--cushion", type=float, default=0.0,
                    help="only withdraw what leaves at least this much profit in the funded account")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_1m_csv(args.data)
    ev_rows, fu_rows = [], []
    for risk in args.risk:
        cfg = Config(start_equity=ACCOUNT, risk_pct=risk / ACCOUNT, max_contracts=40)
        total, worst = day_table(df, cfg, args.start)
        shift = args.haircut * total.mean()
        total, worst = total - shift, worst - shift

        idx = sample_idx(rng, len(total), args.paths, args.eval_days, args.block)
        st, used = eval_mc(total[idx], worst[idx], args.target, args.eval_days)
        p = (st == 1).mean()
        ev_rows.append({
            "risk_$": int(risk), "pass_%": round(100 * p, 1), "fail_%": round(100 * (st == -1).mean(), 1),
            "timeout_%": round(100 * (st == 0).mean(), 1),
            "median_days": int(np.median(used[st == 1])) if p else None,
            "p90_days": int(np.percentile(used[st == 1], 90)) if p else None,
            "evals_per_pass": round(1 / p, 2) if p else None,
        })

        idx = sample_idx(rng, len(total), args.paths, args.funded_days, args.block)
        paid, n_pay, first, bust, end_bal = funded_mc(total[idx], worst[idx], cushion=args.cushion)
        fu_rows.append({
            "risk_$": int(risk),
            "busted_%": round(100 * (bust > 0).mean(), 1),
            "any_payout_%": round(100 * (n_pay > 0).mean(), 1),
            "median_days_to_1st": int(np.median(first[first > 0])) if (first > 0).any() else None,
            "payouts_median": int(np.median(n_pay)),
            "take_home_mean_$": int(paid.mean()),
            "take_home_p25_$": int(np.percentile(paid, 25)),
            "take_home_p50_$": int(np.percentile(paid, 50)),
            "take_home_p75_$": int(np.percentile(paid, 75)),
            "alive_balance_mean_$": int(np.where(bust > 0, 0, end_bal).mean()),
            "per_eval_bought_$": int(paid.mean() * p) if p else 0,
        })

    tag = f"{'iid days' if args.block <= 1 else f'{args.block}-day blocks'}" + (
        f", edge cut {100 * args.haircut:.0f}%" if args.haircut else "")
    print(f"== Tradeify Select 50K evaluation (+${args.target:,.0f}, $2k EOD trail, 40% consistency; "
          f"{args.paths:,} paths, {tag}) ==")
    print(pd.DataFrame(ev_rows).to_string(index=False))
    print(f"\n== Select Flex funded, first {args.funded_days} trading days "
          f"(payout every 5 days >= $150, 50% of profit capped $2,500, 90% split; "
          f"cushion ${args.cushion:,.0f}) ==")
    print(pd.DataFrame(fu_rows).to_string(index=False))
    print("\nper_eval_bought_$ = expected take-home per evaluation purchased (pass rate x mean take-home);"
          " subtract the eval fee (+ any activation fee) to get the net value of one attempt.")


if __name__ == "__main__":
    main()
