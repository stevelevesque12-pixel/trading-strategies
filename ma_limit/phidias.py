"""Phidias Premium simulator for the ma_limit strategy, with open trades marked to market on every bar.

Rules (Phidias site, Premium 50K, Oct 2026 screenshots):
  Evaluation : $4,000 target, $2,500 EOD trailing drawdown, 1 day minimum, max 100 micros, overnight allowed.
  Funded     : $2,500 EOD trailing drawdown, $52,600 minimum threshold (payouts only from balance above it),
               payouts every 5 valid days (a valid day makes >= $150), 30% consistency rule,
               progressive split 75% -> 100%. Reset: $108.45.
Assumptions where the page is silent (each is a constant below):
  - the drawdown stops trailing at start + $100 (so it locks at $50,100, which is $52,600 - $2,500);
  - 30% rule: the best day since the last payout is at most 30% of the profit made since then;
  - split 75/80/85/90/100% for payouts 1-5+ (Phidias rules page), applied to the withdrawal; max $2,000 per
    period on 50K ($2,500 100K, $2,750 150K); the balance after a withdrawal may not fall below start + $100;
    $500 minimum payout (assumed);
  - a lost funded account goes back to a new evaluation at the reset price.
Session days run 18:00 -> 17:00 ET. The drawdown is checked against each day's worst equity (closed P&L plus
the open trade at its worst bar extreme), and the trailing line moves with the day's closing equity.
"""
import numpy as np
import pandas as pd

RULES = dict(target=4000.0, dd=2500.0, lock=100.0, threshold=2600.0, valid_day=150.0, pay_days=5,
             cons=0.30, min_pay=500.0, max_pay=2000.0, keep=100.0, split=(0.75, 0.80, 0.85, 0.90, 1.00), reset=108.45, max_qty=100)


def session_day(ts):
    return (ts + pd.Timedelta(hours=6)).normalize().tz_localize(None)


def bar_paths(trades, bars, pv):
    """Per contract, per session day: (day change in closing equity, day's worst equity vs the prior close)."""
    n = len(bars)
    idx = bars.index
    pos = pd.Series(np.arange(n), index=idx)
    closed = np.zeros(n)  # realized P&L booked at each bar
    open_c = np.zeros(n)  # open-trade mark at bar close
    open_w = np.zeros(n)  # open-trade mark at the bar's adverse extreme
    lo, hi, cl = bars.low.to_numpy(), bars.high.to_numpy(), bars.close.to_numpy()
    for t in trades:
        a = pos[t.entry_time.floor("15min") if t.entry_time not in pos.index else t.entry_time]
        b = pos[t.exit_time]
        d = t.direction
        if b > a:
            sl = slice(a, b)  # bars where the trade is still open at the close
            open_c[sl] += (cl[sl] - t.entry) * d * pv
            adverse = np.where(d == 1, lo[sl], hi[sl])
            open_w[sl] += np.minimum((adverse - t.entry) * d * pv, (cl[sl] - t.entry) * d * pv)
        # exit bar: worst point is the trade's MAE at most, then realized
        open_w[b] += min(-t.mae * pv, 0.0) if b == a else min(((lo[b] if d == 1 else hi[b]) - t.entry) * d * pv, 0.0)
        closed[b] += t.pnl_dollars
    eq_close = np.cumsum(closed) + open_c
    eq_worst = np.cumsum(closed) - closed + np.minimum(open_w, 0.0) + np.where(open_c != 0, 0.0, 0.0)
    eq_worst = np.minimum(eq_worst, eq_close)
    day = np.asarray(session_day(idx))
    df = pd.DataFrame({"day": day, "close": eq_close, "worst": eq_worst})
    g = df.groupby("day")
    last = g.close.last()
    prev = last.shift(fill_value=0.0)
    days = last.index[pd.DatetimeIndex(last.index).weekday < 5]
    return (last - prev)[days].to_numpy(), (g.worst.min() - prev)[days].to_numpy(), pd.DatetimeIndex(days)


def trade_paths(t, days):
    """Approximation from a trade list (TradingView export): each trade's result on its exit day, and its worst
    excursion counted on every session day it was open. Per contract; t has entry, exit, pnl, mae_usd."""
    tot = pd.Series(0.0, index=days); low = pd.Series(0.0, index=days)
    run_by_day = {}
    for e, x, pnl, mae in zip(t.entry, t.exit, t.pnl, t.mae_usd):
        d0, d1 = session_day(pd.Timestamp(e).tz_localize("America/New_York")), session_day(pd.Timestamp(x).tz_localize("America/New_York"))
        for d in pd.bdate_range(d0, d1):
            if d in low.index:
                low[d] = min(low[d], tot[d] - mae)
        if d1 in tot.index:
            tot[d1] += pnl; low[d1] = min(low[d1], tot[d1])
    return tot.to_numpy(), low.to_numpy()


def simulate(day_ret, day_low, n_eval, n_fund, idx, price=None, r=RULES):
    """idx: (sims, days) day indices. Returns per-sim arrays. price: first evaluation cost (None = reset price)."""
    sims, days = idx.shape
    mode = np.zeros(sims, int)  # 0 buy, 1 eval, 2 funded
    bal = np.zeros(sims); peak = np.zeros(sims); valid = np.zeros(sims, int); cyc_best = np.zeros(sims)
    cyc_start = np.zeros(sims); npay = np.zeros(sims, int)
    fees = np.zeros(sims); paid = np.zeros(sims); evals = np.zeros(sims, int); passed = np.zeros(sims, int)
    lost = np.zeros(sims, int); pays = np.zeros(sims, int); first_pay = np.full(sims, np.nan); bought = np.zeros(sims, bool)
    for k in range(days):
        new = mode == 0
        fees[new] += np.where(bought[new] | (price is None), r["reset"], price if price is not None else 0)
        bought[new] = True; evals += new; mode[new] = 1
        bal[new] = 0; peak[new] = 0
        e = mode == 1
        q = np.where(e, n_eval, n_fund)
        t, m = day_ret[idx[:, k]] * q, day_low[idx[:, k]] * q
        line = np.minimum(peak - r["dd"], r["lock"])
        dead = bal + m <= line
        lost += dead & ~e
        mode[dead] = 0
        al = ~dead
        bal[al] += t[al]
        peak[al] = np.maximum(peak[al], bal[al])
        ok = e & al & (bal >= r["target"])
        passed += ok
        mode[ok] = 2; bal[ok] = 0; peak[ok] = 0; valid[ok] = 0; cyc_best[ok] = 0; cyc_start[ok] = 0; npay[ok] = 0
        fa = (mode == 2) & al & ~ok
        valid[fa] += t[fa] >= r["valid_day"]
        cyc_best[fa] = np.maximum(cyc_best[fa], t[fa])
        # Request needs the balance at the threshold; withdraw up to the period cap, keeping start + $100 in the account.
        amt = np.where(bal >= r["threshold"], bal - r["keep"], 0.0)
        if r.get("max_pay"):
            amt = np.minimum(amt, r["max_pay"])
        cyc_profit = bal - cyc_start
        go = fa & (valid >= r["pay_days"]) & (amt >= r["min_pay"]) & (cyc_best <= r["cons"] * np.maximum(cyc_profit, 1e-9))
        split = np.array(r["split"])[np.minimum(npay, len(r["split"]) - 1)]
        paid[go] += amt[go] * split[go]; pays += go; npay += go
        first_pay = np.where(go & np.isnan(first_pay), k + 1, first_pay)
        bal[go] -= amt[go]; cyc_start[go] = bal[go]; valid[go] = 0; cyc_best[go] = 0
    return dict(net=paid - fees, paid=paid, fees=fees, evals=evals, passed=passed, lost=lost, payouts=pays, first_pay=first_pay)


def summary(res, years):
    n = res["net"]
    fp = res["first_pay"]
    return dict(net_yr=n.mean() / years, median_yr=np.median(n) / years, p10=np.percentile(n, 10) / years,
                p90=np.percentile(n, 90) / years, p_profit=(n > 0).mean(), paid_yr=res["paid"].mean() / years,
                fees_yr=res["fees"].mean() / years, evals_yr=res["evals"].mean() / years,
                passed_yr=res["passed"].mean() / years, funded_lost_yr=res["lost"].mean() / years,
                payouts_yr=res["payouts"].mean() / years,
                first_pay_days=float(np.nanmedian(fp)) if np.isfinite(fp).any() else np.nan)
