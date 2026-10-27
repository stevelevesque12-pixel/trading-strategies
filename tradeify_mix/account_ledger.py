"""One Tradeify Growth 50K account, the user's own sizes: a real-order replay with every trade, evaluation and payout,
plus simulated-year statistics, risk scenarios and strategy combinations at the same sizes.

  python -m tradeify_mix.account_ledger [growth50|lucid150|selectflex150]   # -> results/account*/ *.csv

Sizes: eval Saty 3 MNQ -> funded Saty 2 MNQ; Opening Drive 1 MNQ (fixed contracts: each Opening Drive trade's P&L per
contract from the combo list); Halyard 2 MNQ; Gold Breakout 1 MGC. Rules: compare.PLANS["Growth 50K"]. The replay
trades the real days in order; a new evaluation is bought the day after any fail or blown account.
"""
import itertools

import numpy as np
import pandas as pd

from tradeify_mix import compare as C
from tradeify_mix.risk_year import transform

PRESETS = {
    "growth50": ("Growth 50K", dict(S=3, OD=1, H=2, G=1), dict(S=2, OD=1, H=2, G=1), "account"),
    "lucid150": ("LucidFlex 150K", dict(S=9, OD=2, H=4, G=4), dict(S=7, OD=2, H=4, G=3), "account_lucid150"),
    "selectflex150": ("Select Flex 150K", dict(S=7, OD=2, H=6, G=3), dict(S=7, OD=2, H=6, G=3), "account_selectflex150"),
}
OUT = C.RESULTS / "account"
R = C.PLANS["Growth 50K"]
EVAL = dict(S=3, OD=1, H=2, G=1)
FUNDED = dict(S=2, OD=1, H=2, G=1)


def configure(preset):
    global OUT, R, EVAL, FUNDED
    plan, EVAL, FUNDED, sub = PRESETS[preset]
    R = C.PLANS[plan]; OUT = C.RESULTS / sub
NAMES = {"S": "Saty", "OD": "Opening Drive", "H": "Halyard", "G": "Gold Breakout"}
REPLAY = ("2025-10-17", "2026-10-16")
FULL = ("2023-11-01", "2026-10-16")
NO2026 = ("2023-11-01", "2025-12-31")
SIMS = 4000


def with_fixed_od(L):
    od = L["OD"].copy(); q = od.qty.astype(float)
    od["pnl"] = od.pnl / q; od["mae"] = od.mae / q; od["qty"] = 1
    L = dict(L); L["OD1"] = od
    return L


def trades_for(L, c):
    """Every trade the account would take at sizes c, with $ P&L, worst point and contracts at that size."""
    fr = [(L["CS"].assign(strat="Saty"), c["S"] / 6), (L["OD1"].assign(strat="Opening Drive"), c["OD"]),
          (L["H"].assign(strat="Halyard"), c["H"] / 5)]
    parts = []
    fr = [f for f in fr if f[1]]
    if fr:
        parts.append(C.nq_book(fr))
    if c["G"]:
        parts.append(L["G"].assign(strat="Gold Breakout", q=float(c["G"])))
    t = pd.concat(parts, ignore_index=True)
    t["contracts"] = (t.qty * t.q).round().astype(int)
    t["pnl"] = t.pnl * t.q; t["mae"] = t.mae * t.q
    return t


def day_paths(t, days):
    return C.prefix(C.events(t, 1.0), days)


def replay(L, start, end):
    days = pd.bdate_range(start, end)
    te, tf = trades_for(L, EVAL), trades_for(L, FUNDED)
    Pe, Pf = day_paths(te, days), day_paths(tf, days)
    mode, acct = 0, 0
    bal = peak = best = since = 0.0; wins = npay = nd = 0; locked = False
    trade_rows, day_rows, event_rows = [], [], []
    for i, d in enumerate(days):
        if mode == 0:
            acct += 1; mode = 1; bal = peak = best = since = 0.0; wins = npay = nd = 0; locked = False
            event_rows.append(dict(date=d.date(), account=acct, event="Evaluation bought", amount=-R["fee"], balance=0.0))
        e = mode == 1
        P = Pe if e else Pf
        raw = float(P[i, -1])
        if R["dll"]:
            tot, mn = C.day_dll(P[i:i + 1], R["dll"]); tot, mn = float(tot[0]), float(mn[0])
            dll_hit = bool(P[i].min() <= -R["dll"])
        else:
            tot, mn, dll_hit = raw, float(P[i].min()), False
        tr = (te if e else tf); tr = tr[tr.day == d]
        phase = "Evaluation" if e else "Funded"
        for _, r in tr.sort_values("entry").iterrows():
            trade_rows.append(dict(date=d.date(), account=acct, phase=phase, strategy=r.strat,
                                   side="Long" if r.dir > 0 else "Short", contracts=int(r.contracts),
                                   entry=r.entry.strftime("%Y-%m-%d %H:%M"), exit=r.exit.strftime("%Y-%m-%d %H:%M"),
                                   pnl=round(r.pnl, 2), worst_point=round(r.mae, 2)))
        floor = min(peak - R["dd"], 100.0) if (not e and locked) else peak - R["dd"]
        event = ""
        if bal + mn <= floor:
            event = "Evaluation failed" if e else "Funded account blown"
            event_rows.append(dict(date=d.date(), account=acct, event=event, amount=0.0, balance=round(bal + tot, 2)))
            day_rows.append(dict(date=d.date(), account=acct, phase=phase, trades=len(tr), day_pnl=round(tot, 2),
                                 raw_pnl=round(raw, 2), dll_hit=dll_hit, balance=round(bal + tot, 2), floor=round(floor, 2), event=event))
            mode = 0
            continue
        bal += tot; peak = max(peak, bal)
        if e:
            nd += tot != 0; best = max(best, tot)
            if bal >= R["target"] and nd >= R["ev_days"] and best <= R["ev_cons"] * bal:
                event = "Evaluation passed"
                event_rows.append(dict(date=d.date(), account=acct, event=event, amount=0.0, balance=round(bal, 2)))
                day_rows.append(dict(date=d.date(), account=acct, phase=phase, trades=len(tr), day_pnl=round(tot, 2),
                                     raw_pnl=round(raw, 2), dll_hit=dll_hit, balance=round(bal, 2), floor=round(floor, 2), event=event))
                mode = 2; bal = peak = best = 0.0
                continue
        else:
            locked = locked or peak >= R["lock"]
            wins += tot > R["day_min"]; since += tot; best = max(best, tot)
            cap = R["caps"][min(npay, len(R["caps"]) - 1)]
            amt = min(cap, bal - R["keep"], R["frac"] * bal)
            if bal >= R["pay_bal"] and wins >= C.NWIN and best <= R["cons"] * max(since, 1e-9) and amt >= R["min_pay"]:
                event = "Payout"
                event_rows.append(dict(date=d.date(), account=acct, event="Payout", amount=round(amt * C.SPLIT, 2),
                                       balance=round(bal - amt, 2), requested=round(amt, 2)))
                bal -= amt; npay += 1; wins = 0; best = 0.0; since = 0.0
                if R["lock_on_pay"]: locked = True
                if R.get("max_payouts") and npay >= R["max_payouts"]:
                    event = "Payout; account moves to live review"
                    event_rows.append(dict(date=d.date(), account=acct, event="Account closed after last payout", amount=0.0,
                                           balance=round(bal, 2)))
                    day_rows.append(dict(date=d.date(), account=acct, phase=phase, trades=len(tr), day_pnl=round(tot, 2),
                                         raw_pnl=round(raw, 2), dll_hit=dll_hit, balance=round(bal, 2), floor=round(floor, 2), event=event))
                    mode = 0
                    continue
        day_rows.append(dict(date=d.date(), account=acct, phase=phase, trades=len(tr), day_pnl=round(tot, 2), raw_pnl=round(raw, 2),
                             dll_hit=dll_hit, balance=round(bal, 2), floor=round(floor, 2), event=event))
    return pd.DataFrame(trade_rows), pd.DataFrame(day_rows), pd.DataFrame(event_rows)


def boot_stats(L, e, f, window, sims=SIMS, seed=5):
    days = pd.bdate_range(*window)
    ix = C.boot(len(days), sims, 250, np.random.default_rng(seed))
    Pe, Pf = day_paths(trades_for(L, e), days), day_paths(trades_for(L, f), days)
    return C.run_plan(Pe, Pf, ix, R, return_cash=True)


def summarize(net, cash, st):
    cum = cash.cumsum(1); first = np.array([np.argmax(r > 0) + 1 if (r > 0).any() else 999 for r in cash])
    outlay = -np.minimum(cum.min(1), 0)
    return dict(net_avg=round(net.mean()), net_median=round(np.median(net)), net_p10=round(np.percentile(net, 10)),
                net_p90=round(np.percentile(net, 90)), loss_year_share=round(float((net < 0).mean()), 3),
                evals=round(st["evals"], 1), eval_fees=round(st["fees"]), payouts=round(st["payouts"], 1),
                paid_within_year=round(float((first < 999).mean()), 3),
                first_payout_month=round(float(np.median(np.minimum(first, 250)) * 12 / 250), 1),
                outlay_median=round(np.median(outlay)), outlay_p90=round(np.percentile(outlay, 90)))


def main(preset="growth50"):
    configure(preset)
    OUT.mkdir(parents=True, exist_ok=True)
    L0 = C.load_all(); L = with_fixed_od(L0)

    # 1) real-order replay of the last 365 days
    trades, daysdf, events = replay(L, *REPLAY)
    trades.to_csv(OUT / "replay_trades.csv", index=False); daysdf.to_csv(OUT / "replay_days.csv", index=False)
    events.to_csv(OUT / "replay_events.csv", index=False)
    daysdf["month"] = pd.to_datetime(daysdf.date).dt.strftime("%Y-%m")
    ev = events.assign(month=pd.to_datetime(events.date).dt.strftime("%Y-%m"))
    months = daysdf.groupby("month").agg(trading_pnl=("day_pnl", "sum"), trades=("trades", "sum"),
                                         dll_days=("dll_hit", "sum")).reset_index()
    months["evals_bought"] = months.month.map(ev[ev.event == "Evaluation bought"].groupby("month").size()).fillna(0).astype(int)
    months["eval_fees"] = -months.evals_bought * R["fee"]
    months["payouts"] = months.month.map(ev[ev.event == "Payout"].groupby("month").amount.sum()).fillna(0).round(2)
    months["net_cash"] = months.payouts + months.eval_fees
    months["cumulative_net_cash"] = months.net_cash.cumsum().round(2)
    months.to_csv(OUT / "replay_months.csv", index=False)
    strat = trades.groupby("strategy").agg(trades=("pnl", "size"), win_rate=("pnl", lambda s: round(float((s > 0).mean()), 3)),
                                           avg_win=("pnl", lambda s: round(s[s > 0].mean(), 2)),
                                           avg_loss=("pnl", lambda s: round(s[s <= 0].mean(), 2)),
                                           total_pnl=("pnl", lambda s: round(s.sum(), 2)),
                                           worst_trade=("pnl", "min")).reset_index()
    strat.to_csv(OUT / "replay_strategies.csv", index=False)
    paid = events[events.event == "Payout"].amount.sum(); fees = (events.event == "Evaluation bought").sum() * R["fee"]
    summary = dict(period="Oct 17 2025 - Oct 16 2026", trading_days=len(daysdf), trades=len(trades),
                   evals_bought=int((events.event == "Evaluation bought").sum()), evals_passed=int((events.event == "Evaluation passed").sum()),
                   evals_failed=int((events.event == "Evaluation failed").sum()), funded_blown=int((events.event == "Funded account blown").sum()),
                   payouts=int((events.event == "Payout").sum()), payouts_received=round(paid, 2), eval_fees=fees,
                   net_profit=round(paid - fees, 2), trading_pnl=round(daysdf.day_pnl.sum(), 2),
                   dll_days=int(daysdf.dll_hit.sum()), ending_balance=round(daysdf.balance.iloc[-1], 2),
                   ending_phase=daysdf.phase.iloc[-1])
    pd.DataFrame([summary]).to_csv(OUT / "replay_summary.csv", index=False)
    print(summary, flush=True)

    # 2) simulated years at these sizes
    rows, dist = [], []
    for lab, w in (("3 years (Nov 2023 - Oct 2026)", FULL), ("Without 2026 (Nov 2023 - Dec 2025)", NO2026), ("Last 12 months", REPLAY)):
        st, cash = boot_stats(L, EVAL, FUNDED, w); net = cash.sum(1)
        rows.append(dict(window=lab, **summarize(net, cash, st)))
        for q in range(5, 100, 5):
            dist.append(dict(window=lab, percentile=q, net=round(float(np.percentile(net, q)))))
        if w == FULL:
            cum = cash.cumsum(1); month = np.minimum(np.arange(250) * 12 // 250, 11)
            mrows = []
            for m in range(12):
                c = cum[:, np.nonzero(month == m)[0][-1]]
                mrows.append(dict(month=m + 1, p10=round(np.percentile(c, 10)), median=round(np.median(c)), p90=round(np.percentile(c, 90))))
            pd.DataFrame(mrows).to_csv(OUT / "sim_cumulative.csv", index=False)
    pd.DataFrame(rows).to_csv(OUT / "sim_summary.csv", index=False); pd.DataFrame(dist).to_csv(OUT / "sim_distribution.csv", index=False)
    print(pd.DataFrame(rows).to_string(index=False), flush=True)

    # 3) risk scenarios
    sc_rows = []
    for name, sc, special in (("Base case", {}, None), ("Winners 10% smaller", dict(win=0.9), None), ("Winners 25% smaller", dict(win=0.75), None),
                              ("Losses 25% bigger", dict(loss=1.25), None), ("One more tick of slippage", dict(slip=1), None),
                              ("Gold Breakout stops working", {}, "G"), ("Halyard stops working", {}, "H"), ("Opening Drive stops working", {}, "OD"),
                              ("Saty's winners halved", dict(win_leg={"CS": 0.5}), None),
                              ("Without 2026", {}, NO2026), ("Weak market: 2025 only", {}, ("2025-01-01", "2025-12-31"))):
        Ls = with_fixed_od(transform(L0, sc)) if sc else L
        if "win_leg" in sc:
            Ls["OD1"] = with_fixed_od(L0)["OD1"]
        e, f, w = dict(EVAL), dict(FUNDED), FULL
        if special in ("G", "H", "OD"):
            e[special] = 0; f[special] = 0
        elif special:
            w = special
        st, cash = boot_stats(Ls, e, f, w); net = cash.sum(1)
        sc_rows.append(dict(scenario=name, **summarize(net, cash, st)))
    pd.DataFrame(sc_rows).to_csv(OUT / "scenarios.csv", index=False)
    print(pd.DataFrame(sc_rows)[["scenario", "net_avg", "net_median", "net_p10", "loss_year_share"]].to_string(index=False), flush=True)

    # 4) strategy combinations at these sizes
    cb = []
    for mask in itertools.product((0, 1), repeat=4):
        if not any(mask): continue
        e = {k: v * m for (k, v), m in zip(EVAL.items(), mask)}; f = {k: v * m for (k, v), m in zip(FUNDED.items(), mask)}
        st, cash = boot_stats(L, e, f, FULL, sims=2500); net = cash.sum(1)
        st2, cash2 = boot_stats(L, e, f, NO2026, sims=2500); net2 = cash2.sum(1)
        cb.append(dict(strategies=" + ".join(NAMES[k] for k, m in zip(EVAL, mask) if m), count=sum(mask),
                       net_avg=round(net.mean()), net_median=round(np.median(net)), net_p10=round(np.percentile(net, 10)),
                       loss_year_share=round(float((net < 0).mean()), 3), net_avg_without_2026=round(net2.mean()),
                       loss_year_share_without_2026=round(float((net2 < 0).mean()), 3)))
    pd.DataFrame(cb).sort_values("net_avg", ascending=False).to_csv(OUT / "combinations.csv", index=False)

    # 5) sizes and stop risk (last 365 days)
    t = trades_for(L, FUNDED); t = t[t.entry >= REPLAY[0]]
    sz = []
    for k, n in NAMES.items():
        for phase, c in (("Evaluation", EVAL), ("Funded", FUNDED)):
            sz.append(dict(strategy=n, phase=phase, contracts=c[k], instrument="MGC" if k == "G" else "MNQ"))
    pd.DataFrame(sz).to_csv(OUT / "sizes.csv", index=False)
    risk = t.rename(columns={"strat": "strategy"}).groupby("strategy").agg(trades=("pnl", "size"), worst_loss=("pnl", "min"),
                                     median_loss=("pnl", lambda s: round(s[s < 0].median(), 2)),
                                     loss_p90=("pnl", lambda s: round(s[s < 0].quantile(0.1), 2))).reset_index()
    risk.to_csv(OUT / "stop_risk.csv", index=False)
    print("done")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "growth50")
