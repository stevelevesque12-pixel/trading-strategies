"""Expected dollars per year from running the combined ORB setup (MNQ orb_trend + MES orb_vwap) on a
Tradeify Growth 50K, as a function of risk. Rules as commonly reported (verify on tradeify.co):

  Eval:   +$3,000 target, $2,000 EOD-trailing drawdown (assumed to lock at +$100), $1,250 daily loss limit
          (soft: stops the day), no consistency rule, no minimum days. Fee: $FEE per ~21 sessions while open;
          a bust means buying a new eval.
  Funded: same $2,000 EOD-trailing drawdown (locks at +$100). Payout when balance >= $53,000, >= 5 winning
          days of >= $150 since the last payout, and largest day <= 35% of the cycle's profit. Payout =
          min(cap, balance - $52,000); caps $1,500 / $2,000 / $2,500 / $3,000+; trader keeps 90%.

Daily P&L comes from the Python engine at risk x k (risk, daily loss/profit guards all scaled by k), on both
markets, then resampled in 5-session blocks to build 1-year paths. Strategy keeps running after a bust
(new eval). Reported: mean and median net $ per year (payouts kept minus fees), bust counts.

    python -m research.experiments.tradeify_ev [--fee 139] [--period all|2023]
"""

import argparse
from dataclasses import replace

import numpy as np

from research.candidate import ORB, ORB_VWAP
from trend.data import load
from trend.strategy import LucidRules, backtest, daily_series

R = LucidRules(max_micros=40)
CAPS = [1500, 2000, 2500, 3000]


def daily(k, start, end):
    maps = []
    for ds, pv, spec in (("nq_15m", 2.0, ORB), ("es_15m", 5.0, ORB_VWAP)):
        m = load(ds).slice(start=start, end=end)
        s = replace(spec, risk_usd=spec.risk_usd * k, daily_loss_limit=spec.daily_loss_limit * k,
                    daily_profit_cap=spec.daily_profit_cap * k, cushion_sizing=False)
        d, p, lo, n = daily_series(m, backtest(m, s, R, point_value=pv))
        maps.append(dict(zip(m.day_dates[d], zip(p, lo, n))))
    days = sorted(set(maps[0]) & set(maps[1]))
    A = np.array([[mp[x] for x in days] for mp in maps])
    return A[:, :, 0].sum(0), A[:, :, 1].sum(0)


ONE_TIME = False
CUSHION = False
RESET = None


def year_path(p, lo, fee, rng, sessions=252):
    nb = len(p) // 5
    idx = (rng.integers(0, nb, sessions // 5 + 1)[:, None] * 5 + np.arange(5)).ravel()[:sessions]
    P, L = p[idx], lo[idx]
    net, fees, evals, payouts, funded_busts = 0.0, 0.0, 0, 0, 0
    stage, bal, peak, mll, days_open = "eval", 0.0, 0.0, -2000.0, 0
    cyc_prof, cyc_best, cyc_wins, n_pay = 0.0, 0.0, 0, 0
    fees += fee; evals += 1
    for t in range(sessions):
        # soft daily loss limit: if the day's open drawdown reaches -$1,250 the platform liquidates there,
        # so that day books -$1,250 even if the trade would have recovered later
        if L[t] <= -1250.0:
            day_p = day_lo = -1250.0
        else:
            day_p, day_lo = P[t], L[t]
        if CUSHION:  # scale size by remaining drawdown cushion (quarters, floor 25%), like the Pine script
            f = max(0.25, min(1.0, np.floor((bal - mll) / 2000.0 * 4) / 4))
            day_p, day_lo = f * day_p, f * day_lo
        if stage == "eval":
            days_open += 1
            if days_open % 21 == 0 and not ONE_TIME:
                fees += fee
        if bal + day_lo <= mll:  # breach
            if stage == "funded":
                funded_busts += 1
            stage, bal, peak, mll, days_open = "eval", 0.0, 0.0, -2000.0, 0
            cyc_prof = cyc_best = 0.0; cyc_wins = 0; n_pay = 0
            fees += (RESET if RESET is not None else fee); evals += 1
            continue
        bal += day_p
        if bal > peak:
            peak = bal
            mll = min(peak - 2000.0, 100.0)
        if stage == "eval":
            if bal >= 3000.0:
                stage, bal, peak, mll = "funded", 0.0, 0.0, -2000.0
                cyc_prof = cyc_best = 0.0; cyc_wins = 0
        else:
            cyc_prof += day_p
            cyc_best = max(cyc_best, day_p)
            cyc_wins += day_p >= 150
            if bal >= 3000.0 and cyc_wins >= 5 and cyc_prof > 0 and cyc_best <= 0.35 * cyc_prof:
                amt = min(CAPS[min(n_pay, 3)], bal - 2000.0)
                if amt >= 500:
                    net += 0.9 * amt; bal -= amt; n_pay += 1; payouts += 1
                    cyc_prof = cyc_best = 0.0; cyc_wins = 0
    return net - fees, payouts, evals, funded_busts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fee", type=float, default=139.0)
    ap.add_argument("--period", default="all", choices=["all", "2023"])
    ap.add_argument("--paths", type=int, default=3000)
    ap.add_argument("--one-time", action="store_true", help="eval fee charged once per eval, not monthly")
    ap.add_argument("--risks", default="0.5,1,1.5,2,3,4,6")
    ap.add_argument("--cushion", action="store_true")
    ap.add_argument("--reset", type=float, default=None, help="fee for each new eval after a bust")
    a = ap.parse_args()
    global ONE_TIME, CUSHION, RESET
    ONE_TIME, CUSHION, RESET = a.one_time, a.cushion, a.reset
    start = "2023-01-01" if a.period == "2023" else None
    rng = np.random.default_rng(1)
    print(f"Tradeify Growth 50K, eval fee ${a.fee:.0f}/{'eval' if a.one_time else 'month'}, history {a.period}, {a.paths} one-year paths")
    print("risk x | MNQ $/MES $ per trade | mean net $/yr | median | P(net<0) | payouts/yr | evals bought/yr | funded busts/yr")
    for k in [float(x) for x in a.risks.split(",")]:
        p, lo = daily(k, start, None)
        res = np.array([year_path(p, lo, a.fee, rng) for _ in range(a.paths)])
        print(f"{k:>5}  | {500 * k:>5.0f} / {400 * k:<5.0f}         | {res[:, 0].mean():>9.0f}     | {np.median(res[:, 0]):>7.0f} | "
              f"{(res[:, 0] < 0).mean():>6.0%}   | {res[:, 1].mean():>6.2f}     | {res[:, 2].mean():>6.2f}          | {res[:, 3].mean():.2f}")


if __name__ == "__main__":
    main()
