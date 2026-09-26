"""
Monte Carlo for the TR1 breakout: resample historical *days* (not trades,
so a day's 1-3 correlated entries stay together) and replay them as new
random sequences.

Two questions:
  1. Personal account: fixed $ risk per trade -- distribution of 1-year P&L
     and max drawdown.
  2. Prop evaluation: probability of hitting the target before the trailing
     max loss, per $ risk level.

Sampling: iid days, or --block N consecutive-day blocks (keeps volatility
regimes and losing streaks clustered the way they really happen -- the
more honest of the two).

    python -m tr_breakout.monte_carlo --data <parquet> --from 2022-01-01 --block 20
"""

import argparse

import numpy as np
import pandas as pd

from backtest.data import load_1m_csv
from tr_breakout.prop_sim import PropRules
from tr_breakout.strategy import Config, run_backtest, trades_frame


def day_table(df, cfg, start=None):
    """One row per RTH day: total P&L and worst intraday closed-trade P&L (0 on no-trade days)."""
    tf = trades_frame(run_backtest(df, cfg))
    t = df.index.time
    days = pd.Index(sorted(set(df.index[(t >= cfg.rth_open) & (t < cfg.rth_close)].date)))
    if start:
        days = days[days >= pd.Timestamp(start).date()]
    if tf.empty:
        return np.zeros(len(days)), np.zeros(len(days))
    g = tf.groupby("day")["pnl"]
    total = g.sum().reindex(days, fill_value=0.0)
    worst = g.apply(lambda s: min(0.0, s.cumsum().min())).reindex(days, fill_value=0.0)
    return total.to_numpy(), worst.to_numpy()


def sample_idx(rng, n_pool, n_paths, n_days, block):
    if block <= 1:
        return rng.integers(0, n_pool, size=(n_paths, n_days))
    n_blocks = -(-n_days // block)
    starts = rng.integers(0, n_pool - block + 1, size=(n_paths, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)).reshape(n_paths, -1)
    return idx[:, :n_days]


def account_mc(total, rng, n_paths, n_days, block):
    idx = sample_idx(rng, len(total), n_paths, n_days, block)
    eq = np.cumsum(total[idx], axis=1)
    dd = (eq - np.maximum.accumulate(np.maximum(eq, 0), axis=1)).min(axis=1)
    return eq[:, -1], dd


def prop_mc(total, worst, rules: PropRules, rng, n_paths, max_days, block):
    idx = sample_idx(rng, len(total), n_paths, max_days, block)
    tot, wst = total[idx], worst[idx]
    bal = np.zeros(n_paths)
    peak = np.zeros(n_paths)
    status = np.zeros(n_paths, dtype=int)  # 0 open, 1 pass, -1 fail
    days_used = np.full(n_paths, max_days)
    for d in range(max_days):
        live = status == 0
        if not live.any():
            break
        floor_ = peak - rules.max_loss
        if rules.lock_at_start:
            floor_ = np.minimum(floor_, 0.0)
        day_tot = tot[:, d]
        if rules.daily_loss is not None:
            day_tot = np.maximum(day_tot, -rules.daily_loss)  # approx: halted at the limit
        fail = live & (bal + np.minimum(wst[:, d], day_tot) <= floor_)
        bal = np.where(live & ~fail, bal + day_tot, bal)
        peak = np.maximum(peak, bal)
        ok = live & ~fail & (bal >= rules.target)
        status[fail], status[ok] = -1, 1
        days_used[fail | ok] = d + 1
    return status, days_used


def pct(a, qs=(5, 25, 50, 75, 95)):
    return "  ".join(f"p{q}={np.percentile(a, q):>9,.0f}" for q in qs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--paths", type=int, default=20_000)
    ap.add_argument("--block", type=int, default=1, help="block length in days (1 = iid)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--haircut", type=float, default=0.0,
                    help="stress test: remove this fraction of the average daily P&L from every day (0.5 = half the edge)")
    ap.add_argument("--account-risk", type=float, default=1000)
    ap.add_argument("--account-size", type=float, default=100_000)
    ap.add_argument("--contracts", type=int, default=None, help="fixed contracts per trade instead of --account-risk")
    ap.add_argument("--skip-prop", action="store_true")
    ap.add_argument("--horizon", type=int, default=252, help="trading days for the account sim")
    ap.add_argument("--prop-risk", type=float, nargs="+", default=[200, 250, 300, 400])
    ap.add_argument("--target", type=float, default=3000)
    ap.add_argument("--max-loss", type=float, default=2000)
    ap.add_argument("--daily-loss", type=float, default=None)
    ap.add_argument("--max-contracts", type=int, default=50)
    ap.add_argument("--max-days", type=int, default=120, help="give up on an eval after this many trading days")
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    df = load_1m_csv(args.data)
    mode = "iid days" if args.block <= 1 else f"{args.block}-day blocks"
    if args.haircut:
        mode += f", edge cut {100 * args.haircut:.0f}%"

    cfg = Config(start_equity=args.account_size, risk_pct=args.account_risk / args.account_size,
                 fixed_contracts=args.contracts)
    total, _ = day_table(df, cfg, args.start)
    total = total - args.haircut * total.mean()
    final, dd = account_mc(total, rng, args.paths, args.horizon, args.block)
    sizing = f"{args.contracts} contract(s) fixed" if args.contracts else f"${args.account_risk:,.0f} fixed risk"
    print(f"== ${args.account_size:,.0f} account, {sizing}, {args.horizon} trading days, "
          f"{args.paths:,} paths ({mode}, pool {len(total)} days) ==")
    print(f"  P&L      {pct(final)}")
    print(f"  max DD   {pct(dd)}")
    print(f"  P(losing year) {100 * (final < 0).mean():.1f}%   "
          + "   ".join(f"P(DD worse than -${x:,.0f}) {100 * (dd <= -x).mean():.1f}%"
                       for x in (0.2 * args.account_size, 0.3 * args.account_size, 0.5 * args.account_size)))
    if args.skip_prop:
        return

    rules = PropRules(target=args.target, max_loss=args.max_loss,
                      daily_loss=args.daily_loss, max_contracts=args.max_contracts)
    print(f"\n== Prop eval: +${rules.target:,.0f} target, ${rules.max_loss:,.0f} EOD trailing max loss, "
          f"cap {rules.max_contracts} MNQ, give up after {args.max_days} days ==")
    rows = []
    for risk in args.prop_risk:
        cfg = Config(start_equity=rules.account, risk_pct=risk / rules.account,
                     max_contracts=rules.max_contracts)
        total, worst = day_table(df, cfg, args.start)
        shift = args.haircut * total.mean()
        total, worst = total - shift, worst - shift
        status, used = prop_mc(total, worst, rules, rng, args.paths, args.max_days, args.block)
        p = (status == 1).mean()
        rows.append({
            "risk_$": int(risk),
            "pass_%": round(100 * p, 1),
            "fail_%": round(100 * (status == -1).mean(), 1),
            "timeout_%": round(100 * (status == 0).mean(), 1),
            "median_days_pass": int(np.median(used[status == 1])) if p else None,
            "p90_days_pass": int(np.percentile(used[status == 1], 90)) if p else None,
            "evals_to_1_pass": round(1 / p, 1) if p else None,
            "P(pass within 3 tries)_%": round(100 * (1 - (1 - p) ** 3), 1),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
