"""
One-year prop-firm simulation of MCL Trend Dip (final config: Engine A half off at +1R, equity-curve
kill switch, cushion sizing 25% of distance to the max-loss line, $200-$600 risk).

Default firm: Tradeify Select Flex 50K (rules from third-party summaries, Oct 2026 -- verify):
  eval   : $3,000 target, $2,000 EOD trailing drawdown (locks at the starting balance), 40% consistency
           (best day <= 40% of total profit), min 3 trading days, $159 per 21-trading-day month, no activation fee
  funded : $2,000 EOD trailing drawdown, no consistency; payout after every 5 winning days (>= $150),
           up to 50% of profit, max $3,000 per payout, 90% split. After a payout the trailing high-water mark
           resets to the new balance (assumption).
If an account fails, a new evaluation is bought the next day. One account at a time.

Daily P&L paths are block-bootstrapped (mean block 5 days) from the Pine-parity simulation's full-year daily
results at $200 risk and scaled by each day's cushion-based risk.

  python -m trend_lab.propsim [--target 3000] [--edge 1.0]
"""

import argparse

import numpy as np

from .combined import run
from .data import load_bars
from .metrics import daily_pnl
from .montecarlo import bootstrap_path

TRADEIFY_SELECT_FLEX_50K = dict(target=3000.0, max_loss=2000.0, consistency=0.40, min_days=3, fee=159.0,
                                month_days=21, win_day=150.0, payout_days=5, payout_frac=0.5, payout_cap=3000.0,
                                split=0.9, min_payout=250.0)


def risk_for(bal, peak, frac=0.25, lo=200.0, hi=600.0, max_loss=2000.0):
    line = min(peak - max_loss, 0.0)
    return min(hi, max(lo, frac * (bal - line))), line


def simulate_year(path200, r, days=250):
    """Returns dict for one simulated year."""
    fees = paid = 0.0
    evals = passes = funded_fails = payouts = 0
    mode = None
    k = 0
    while k < days:
        if mode is None:                     # buy a new evaluation
            mode, bal, peak, best, ndays, month_left = "eval", 0.0, 0.0, 0.0, 0, 0
            evals += 1
        if mode == "eval" and month_left == 0:
            fees += r["fee"]
            month_left = r["month_days"]
        risk, line = risk_for(bal, peak, max_loss=r["max_loss"])
        day = path200[k] * risk / 200.0
        k += 1
        bal += day
        if mode == "eval":
            month_left -= 1
            ndays += 1
            best = max(best, day)
            if bal <= line:
                mode = None
                continue
            peak = max(peak, bal)
            if bal >= r["target"] and ndays >= r["min_days"] and best <= r["consistency"] * bal:
                passes += 1
                mode, bal, peak, wins = "funded", 0.0, 0.0, 0
            continue
        # funded
        if bal <= line:
            funded_fails += 1
            mode = None
            continue
        peak = max(peak, bal)
        wins += day >= r["win_day"]
        if wins >= r["payout_days"] and bal > 0:
            amt = min(r["payout_cap"], r["payout_frac"] * bal)
            if amt >= r["min_payout"]:
                paid += amt * r["split"]
                payouts += 1
                bal -= amt
                peak = bal
                wins = 0
    return {"net": paid - fees, "paid": paid, "fees": fees, "evals": evals, "passes": passes,
            "funded_fails": funded_fails, "payouts": payouts}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=float, default=3000.0)
    ap.add_argument("--edge", type=float, default=1.0, help="scale the average daily edge (0.5 = half the edge)")
    ap.add_argument("--sims", type=int, default=10000)
    args = ap.parse_args()
    r = dict(TRADEIFY_SELECT_FLEX_50K, target=args.target)
    df = load_bars("15min")
    days = sorted(set(df["trade_day"]))
    _, live = run(df, risk_fixed=200, ecf_len=20, a_scale_r=1.0)
    d = daily_pnl(live, days).to_numpy()
    d = d - (1 - args.edge) * d.mean()       # remove part of the edge, keep the volatility
    rng = np.random.default_rng(42)
    res = [simulate_year(bootstrap_path(d, 250, rng), r) for _ in range(args.sims)]
    net = np.array([x["net"] for x in res])
    pct = lambda a, q: float(np.percentile(a, q))
    print(f"target ${args.target:.0f}, edge x{args.edge}: mean day at $200 risk ${d.mean():.1f}")
    print(f"  net/year (payouts after split - eval fees): mean ${net.mean():,.0f} | median ${pct(net, 50):,.0f} | "
          f"p10 ${pct(net, 10):,.0f} | p90 ${pct(net, 90):,.0f} | P(net>0) {100 * (net > 0).mean():.0f}%")
    for key in ("paid", "fees", "evals", "passes", "funded_fails", "payouts"):
        a = np.array([x[key] for x in res])
        print(f"  {key:12} mean {a.mean():8.1f}   median {pct(a, 50):8.1f}")
    first = []
    for _ in range(2000):
        p = bootstrap_path(d, 250, rng)
        # day of first payout
        rr = simulate_year(p, r)
        first.append(rr["payouts"] > 0)
    print(f"  P(at least one payout in the year) {100 * np.mean(first):.0f}%")


if __name__ == "__main__":
    main()
