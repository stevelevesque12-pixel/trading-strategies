"""Tradeify account simulator for the ma_limit strategy: which contract counts maximize $ paid out per year
when a failed account is simply replaced by a new evaluation?

Plan rules are the ones used in stevelevesque12-pixel/AI-personal-hedge-fund research/account_structure
(third-party summaries, Oct 2026 -- verify on Tradeify's site before relying on them):
  Select Flex 50K : $165 one-time, $3,000 target, $2,000 EOD trailing DD (locks at the start balance),
                    40% eval consistency, 3 days min, no DLL. Funded: same DD, payout after 5 winning days
                    (>= $150), 50% of profit up to $2,500, $250 min, 90% split.
  Growth 150K     : $369 one-time, $9,000 target, $5,000 EOD trailing DD, soft $3,750 DLL, no eval consistency,
                    3 days min. Funded: payout after 5 winning days (>= $250) when the best day of the cycle is
                    <= 35% of the balance, up to $2,500 (payouts 1-3) / $5,000 after, keeping $6,500 in the account,
                    90% split, $250 min.
  Phidias Premium (search summaries Jul 2026; phidiaspropfirm.com was unreachable -- verify): overnight and weekend
                    holds allowed, EOD trailing DD, no eval consistency, 1 day min. Funded: payout every 5 trading
                    days when the best day since the last payout is <= 30% of that profit, $500 min, split 75% ->
                    80% -> 85% -> 90% -> 100% by payout 5. Payout size isn't published in what I found: assumed all
                    profit above the start balance (optimistic) -- see PHIDIAS_HALF.
The drawdown is checked against each day's worst point: closed P&L so far plus the open trade's worst excursion.
Trades must be flat by the daily close (Tradeify auto-closes positions), so run the backtest with flatten_at.

  python -m ma_limit.propsim [--sims 5000]
"""
import argparse

import numpy as np
import pandas as pd

from .strategy import Params, backtest

DATA = "sample_data/real_multi_instrument/tv_mcl1_30m_2025-01-02_2026-10.parquet"
PV = 100.0  # MCL $ per point

PLANS = {
    "Select Flex 50K": dict(fee=165, target=3000, dd=2000, eval_cons=0.40, dll=None, win=150, mode="flex",
                            caps=(2500, 2500), buffer=0),
    "Growth 150K": dict(fee=369, target=9000, dd=5000, eval_cons=None, dll=3750, win=250, mode="growth",
                        caps=(2500, 5000), buffer=6500, f_cons=0.35),
}
PHIDIAS = {
    "Phidias Premium 50K": dict(fee=150, target=4000, dd=2500, eval_cons=None, dll=None, mode="phidias",
                                min_days=1, min_pay=500, f_cons=0.30),
    "Phidias Premium 150K": dict(fee=400, target=9000, dd=4500, eval_cons=None, dll=None, mode="phidias",
                                 min_days=1, min_pay=500, f_cons=0.30),
}
PHIDIAS_SPLIT = (0.75, 0.80, 0.85, 0.90, 1.00)
PHIDIAS_HALF = False  # True: payouts limited to 50% of profit (pessimistic variant)
SPLIT, MIN_PAY, PAY_DAYS, MIN_EVAL_DAYS = 0.9, 250.0, 5, 3


def daily_paths_df(t, days):
    """Same as daily_paths for a frame with day (exit session day), exit, pnl, mae_usd (per contract)."""
    tot = pd.Series(0.0, index=days); low = pd.Series(0.0, index=days)
    for d, g in t.sort_values("exit").groupby("day"):
        if d not in tot.index:
            continue
        run, worst = 0.0, 0.0
        for pnl, mae in zip(g.pnl, g.mae_usd):
            worst = min(worst, run - mae); run += pnl; worst = min(worst, run)
        tot[d], low[d] = run, worst
    return tot.to_numpy(), low.to_numpy()


def daily_paths(trades, days):
    """Per-contract (day_total, day_worst) arrays over `days`: closed P&L at the close and the lowest point
    reached during the day (closed P&L before each trade minus that trade's worst excursion)."""
    tot = pd.Series(0.0, index=days)
    low = pd.Series(0.0, index=days)
    by_day = {}
    for t in trades:
        by_day.setdefault(t.entry_time.normalize().tz_localize(None), []).append(t)
    for d, ts in by_day.items():
        if d not in tot.index:
            continue
        run, worst = 0.0, 0.0
        for t in sorted(ts, key=lambda x: x.exit_time):
            worst = min(worst, run - t.mae * PV)
            run += t.pnl_dollars
            worst = min(worst, run)
        tot[d], low[d] = run, worst
    return tot.to_numpy(), low.to_numpy()


def simulate(plan, tot, low, n_eval, n_fund, idx):
    """idx: (sims, days) day indices into tot/low. Returns per-sim arrays."""
    p = plan
    sims, days = idx.shape
    mode = np.zeros(sims, int)  # 0 buy new eval, 1 eval, 2 funded
    bal = np.zeros(sims); peak = np.zeros(sims); best = np.zeros(sims); nd = np.zeros(sims, int)
    wins = np.zeros(sims, int); cyc_best = np.zeros(sims); npay = np.zeros(sims, int)
    fees = np.zeros(sims); paid = np.zeros(sims); evals = np.zeros(sims, int); passed = np.zeros(sims, int)
    lost = np.zeros(sims, int); pays = np.zeros(sims, int)
    for k in range(days):
        new = mode == 0
        fees[new] += p["fee"]; evals += new; mode[new] = 1
        for a in (bal, peak, best, cyc_best):
            a[new] = 0
        nd[new] = 0; wins[new] = 0; npay[new] = 0
        e = mode == 1
        q = np.where(e, n_eval, n_fund)
        t, m = tot[idx[:, k]] * q, low[idx[:, k]] * q
        if p["dll"] is not None:  # soft daily loss limit: the day stops there, the account survives
            hit = m <= -p["dll"]  # liquidated at the limit: the day ends there even if the trade later recovered
            t = np.where(hit, -p["dll"], t); m = np.maximum(m, -p["dll"])
        line = np.minimum(peak - p["dd"], 0.0)
        dead = bal + m <= line
        lost += dead & ~e
        mode[dead] = 0
        al = ~dead
        bal[al] += t[al]; peak[al] = np.maximum(peak[al], bal[al])
        ea = e & al
        nd[ea] += 1; best[ea] = np.maximum(best[ea], t[ea])
        ok = ea & (bal >= p["target"]) & (nd >= p.get("min_days", MIN_EVAL_DAYS))
        if p["eval_cons"] is not None:
            ok &= best <= p["eval_cons"] * bal
        passed += ok
        mode[ok] = 2; bal[ok] = 0; peak[ok] = 0; wins[ok] = 0; cyc_best[ok] = 0
        fa = (mode == 2) & al & ~ok
        cyc_best[fa] = np.maximum(cyc_best[fa], t[fa])
        if p["mode"] == "phidias":
            wins[fa] += 1  # trading days since the last payout
            profit = np.maximum(bal, 0)
            amt = np.minimum(0.5 * profit if PHIDIAS_HALF else profit, p.get("cap", np.inf))  # cap: optional per-payout limit
            go = fa & (wins >= PAY_DAYS) & (cyc_best <= p["f_cons"] * np.maximum(profit, 1e-9)) & (amt >= p["min_pay"])
            split = np.array(PHIDIAS_SPLIT)[np.minimum(npay, len(PHIDIAS_SPLIT) - 1)]
            paid[go] += amt[go] * split[go]; pays += go; npay += go
            bal[go] -= amt[go]; peak[go] = bal[go]; wins[go] = 0; cyc_best[go] = 0
            continue
        wins[fa] += t[fa] >= p["win"]
        cap = np.where(npay < 3, p["caps"][0], p["caps"][1])
        if p["mode"] == "flex":
            amt = np.minimum(cap, 0.5 * np.maximum(bal, 0))
            go = fa & (wins >= PAY_DAYS) & (amt >= MIN_PAY)
        else:
            amt = np.minimum(cap, bal - p["buffer"])
            go = fa & (wins >= PAY_DAYS) & (cyc_best <= p["f_cons"] * np.maximum(bal, 1e-9)) & (amt >= MIN_PAY)
        paid[go] += amt[go] * SPLIT; pays += go; npay += go
        bal[go] -= amt[go]; peak[go] = bal[go]; wins[go] = 0; cyc_best[go] = 0
    return dict(net=paid - fees, paid=paid, fees=fees, evals=evals, passed=passed, lost=lost, payouts=pays)


def boot(n_pool, sims, days, rng, block=5):
    """Block bootstrap of day indices (keeps runs of good/bad days together)."""
    idx = np.empty((sims, days), int)
    cur = rng.integers(n_pool, size=sims)
    for k in range(days):
        jump = rng.random(sims) < 1 / block
        cur = np.where(jump, rng.integers(n_pool, size=sims), cur)
        idx[:, k] = cur
        cur = (cur + 1) % n_pool
    return idx


def summary(r):
    n = r["net"]
    return dict(mean_net=n.mean(), median_net=np.median(n), p10=np.percentile(n, 10), p90=np.percentile(n, 90),
                p_profit=(n > 0).mean(), paid=r["paid"].mean(), fees=r["fees"].mean(), evals=r["evals"].mean(),
                passed=r["passed"].mean(), funded_lost=r["lost"].mean(), payouts=r["payouts"].mean())


# Contract counts searched, up to each plan's micro limit (third-party summaries; verify).
GRID = {"Select Flex 50K": range(4, 41, 4), "Growth 150K": range(10, 121, 10),
        "Phidias Premium 50K": range(4, 61, 4), "Phidias Premium 150K": range(10, 171, 10)}
TV_EXPORT = "sample_data/real_multi_instrument/tv_trades_ma_limit_mcl_5ct_2021-07_2026-10.parquet"


def tv_trades(path=TV_EXPORT, contracts=5):
    """TradingView 'List of trades' export -> per-contract frame (day = exit session day, ET)."""
    d = pd.read_parquet(path) if str(path).endswith(".parquet") else pd.read_csv(path, encoding="utf-8-sig")
    en = d[d.Type.str.startswith("Entry")].set_index("Trade number")
    ex = d[d.Type.str.startswith("Exit")].set_index("Trade number")
    t = pd.DataFrame({"entry": pd.to_datetime(en["Date and time"]), "exit": pd.to_datetime(ex["Date and time"]),
                      "pnl": ex["Net PnL USD"] / contracts, "mae_usd": -ex["Adverse excursion USD"] / contracts})
    t["day"] = (t.exit + pd.Timedelta(hours=7)).dt.normalize()  # 18:00 reopen starts the next session day
    return t


def search(plan, tot, low, sims, grid, seed=7):
    idx = boot(len(tot), sims, 250, np.random.default_rng(seed))
    rows = [dict(eval_ct=ne, funded_ct=nf, **summary(simulate(plan, tot, low, ne, nf, idx))) for ne in grid for nf in grid]
    return pd.DataFrame(rows).sort_values("mean_net", ascending=False)


def report(name, plan, tot, low, days, sims, grid):
    res = search(plan, tot, low, sims, grid)
    print(f"\n===== {name}: best eval / funded contract counts, {sims:,} simulated years =====")
    print(res.head(8).to_string(index=False, float_format=lambda v: f"{v:,.2f}"))
    b = res.iloc[0]
    rep = summary(simulate(plan, tot, low, int(b.eval_ct), int(b.funded_ct), np.arange(len(days))[None, :]))
    yrs = len(days) / 252
    print(f"Replay of the real days in order ({days[0].date()} - {days[-1].date()}, {yrs:.1f} yrs) at "
          f"{int(b.eval_ct)}/{int(b.funded_ct)}: net ${rep['mean_net']:,.0f} (${rep['mean_net'] / yrs:,.0f}/yr), "
          f"paid ${rep['paid']:,.0f}, fees ${rep['fees']:,.0f}, evals {rep['evals']:.0f}, payouts {rep['payouts']:.0f}")
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=3000)
    a = ap.parse_args()

    # Tradeify: flat by the close -> backtest with flatten_at, 2x ATR stop, 1R target, window 2 (best when flat by 16:30).
    bars = pd.read_parquet(DATA)[["open", "high", "low", "close"]]
    trades = backtest(bars, "30min", Params(expiry_bars=2, live_mode="realtime", atr_mult=2.0, rr=1.0,
                                            hours=("10:00", "15:00"), flatten_at="16:30"))
    days = pd.DatetimeIndex(sorted({d for d in bars.index.normalize().tz_localize(None) if d.weekday() < 5}))
    tot, low = daily_paths(trades, days)
    print(f"Tradeify version (flat 16:30, 2x ATR / 1R): {len(trades)} trades, ${tot.sum():,.0f}/contract, worst day ${low.min():,.0f}")
    for name, plan in PLANS.items():
        report(name, plan, tot, low, days, a.sims, GRID[name])

    # Phidias Premium: overnight allowed -> the TradingView export (2x ATR / 2R, window 2, 10-14h), 2021-2026.
    t = tv_trades()
    pdays = pd.bdate_range(t.day.min(), t.day.max())
    ptot, plow = daily_paths_df(t, pdays)
    print(f"\nPhidias version (TradingView export, overnight holds): {len(t)} trades, ${ptot.sum():,.0f}/contract")
    for name, plan in PHIDIAS.items():
        report(name, plan, ptot, plow, pdays, a.sims, GRID[name])
    oos = pdays < "2025-01-01"
    print("\nPhidias, 2021-2024 days only (before any tuning):")
    for name, plan in PHIDIAS.items():
        report(name, plan, ptot[oos], plow[oos], pdays[oos], a.sims, GRID[name])


if __name__ == "__main__":
    main()
