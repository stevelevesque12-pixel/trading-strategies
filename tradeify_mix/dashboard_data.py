"""Every table behind the Tradeify plan dashboard, written to results/dashboard/.

  python -m tradeify_mix.dashboard_data

Final sizes per plan are FINAL below (the --plans grid winners, with Opening Drive at $1,250 instead of $1,500 in the Select Flex 150K
funded account (Saty + Opening Drive + Gold can be open together), so open contracts stay within 30 micros). Other tables are copied from earlier runs in results/
(compare_best, saty_one_partner, pm_window).
"""
import numpy as np
import pandas as pd

from tradeify_mix import compare as C

OUT = C.RESULTS / "dashboard"
PLAN_ORDER = ["Growth 50K", "Select Flex 50K", "Growth 150K", "Select Flex 150K"]
FINAL = {
    "Growth 50K": (dict(CS=5, OD=750, H=4, G=3), dict(CS=2, OD=500, H=4, G=1)),
    "Select Flex 50K": (dict(CS=2, OD=750, H=4, G=1), dict(CS=5, OD=750, H=4, G=2)),
    "Growth 150K": (dict(CS=9, OD=1500, H=8, G=6), dict(CS=5, OD=1000, H=4, G=2)),
    "Select Flex 150K": (dict(CS=5, OD=1500, H=8, G=4), dict(CS=9, OD=1250, H=8, G=4)),
}
CONTRACT_LIMIT = {"Growth 50K": 40, "Select Flex 50K": 40, "Growth 150K": None, "Select Flex 150K": 30}
WLAB = {"Nov 2023 - Oct 2026": "3 years", "last 12 months": "Last 12 months"}
SIMS = 2000


def sizes(c):
    return f"Saty {c.get('CS', 0)} MNQ · OD ${c.get('OD', 0):,} · Halyard {c.get('H', 0)} MNQ · Gold {c.get('G', 0)} MGC"


def peak_micros(L, cfg):
    fr = [(L[k], cfg[k] / C.LIST_SIZE[k]) for k in ("CS", "OD", "H") if cfg.get(k)]
    nq = C.nq_book(fr); q = nq.q * nq.qty
    g = L["G"]; gq = cfg.get("G", 0)
    ev = pd.concat([pd.DataFrame(dict(t=nq.entry, d=q)), pd.DataFrame(dict(t=nq.exit, d=-q)),
                    pd.DataFrame(dict(t=g.entry, d=gq)), pd.DataFrame(dict(t=g.exit, d=-gq))]).sort_values(["t", "d"])
    return int(round(ev.d.cumsum().max()))


def plans_and_outcomes(sim):
    rows, dist = [], []
    for p in PLAN_ORDER:
        ce, cf = FINAL[p]
        pe, pf = peak_micros(sim.L, ce), peak_micros(sim.L, cf)
        for w, lab in WLAB.items():
            st, net = C.run_plan(sim.P(ce, w), sim.P(cf, w), sim.ix[w], C.PLANS[p], return_net=True)
            rows.append(dict(plan=p, window=lab, net_per_year=round(st["net"]), bad_run_p10=round(st["p10"]),
                             median=round(float(np.median(net))), good_run_p90=round(st["p90"]),
                             loss_year_share=round(float((net < 0).mean()), 3), evals_per_year=round(st["evals"], 1),
                             eval_fees=round(st["fees"]), payouts_per_year=round(st["payouts"], 1),
                             first_payout_days=int(st["first_pay"]), net_per_fee_dollar=round(st["net"] / st["fees"], 2),
                             eval_sizes=sizes(ce), funded_sizes=sizes(cf), peak_micros_eval=pe, peak_micros_funded=pf,
                             contract_limit=CONTRACT_LIMIT[p] or ""))
            for q in range(5, 100, 5):
                dist.append(dict(plan=p, window=lab, percentile=q, net=round(float(np.percentile(net, q)))))
    return pd.DataFrame(rows), pd.DataFrame(dist)


def yearly(sim):
    """Three years from today, bootstrapped from Nov 2023 - Oct 2026: net per year 1/2/3 and the monthly running total."""
    w = "Nov 2023 - Oct 2026"
    ix = C.boot(len(sim.spans[w]), SIMS, 750, np.random.default_rng(21))
    yr, mo = [], []
    for p in PLAN_ORDER:
        ce, cf = FINAL[p]
        _, cash = C.run_plan(sim.P(ce, w), sim.P(cf, w), ix, C.PLANS[p], days=750, return_cash=True)
        for y in range(3):
            v = cash[:, y * 250:(y + 1) * 250].sum(1)
            yr.append(dict(plan=p, year=f"Year {y + 1}", mean=round(v.mean()), p10=round(np.percentile(v, 10)),
                           median=round(np.median(v)), p90=round(np.percentile(v, 90)),
                           loss_share=round(float((v < 0).mean()), 3)))
        cum = cash.cumsum(1)
        for m in range(1, 37):
            v = cum[:, min(750, round(m * 750 / 36)) - 1]
            mo.append(dict(plan=p, month=m, p10=round(np.percentile(v, 10)), median=round(np.median(v)),
                           p90=round(np.percentile(v, 90)), mean=round(v.mean())))
    return pd.DataFrame(yr), pd.DataFrame(mo)


def calendar(sim):
    """Real trading order, one account per plan started on the first trading day of each calendar year."""
    w = "Nov 2023 - Oct 2026"; span = sim.spans[w]
    rows = []
    for p in PLAN_ORDER:
        ce, cf = FINAL[p]
        Pe, Pf = sim.P(ce, w), sim.P(cf, w)
        for y, (a, b) in {"2024": ("2024-01-01", "2024-12-31"), "2025": ("2025-01-01", "2025-12-31"),
                          "2026 (to Oct 16)": ("2026-01-01", "2026-10-16")}.items():
            sel = np.nonzero((span >= a) & (span <= b))[0]
            st, cash = C.run_plan(Pe, Pf, sel[None, :], C.PLANS[p], days=len(sel), return_cash=True)
            rows.append(dict(plan=p, year=y, net=round(cash.sum()), payouts=int(st["payouts"]), evals=int(st["evals"]),
                             eval_fees=round(st["fees"]), first_payout_days=int(st["first_pay"]) if st["first_pay"] < 999 else ""))
    return pd.DataFrame(rows)


def saty_dll(L):
    """Saty P&L alone vs with the other three trading before it on the same day, under each plan's daily loss limit.
    Select Flex has no daily loss limit, so for it the table shows the morning dip before Saty against its drawdown."""
    CS = L["CS"].copy(); CS["t_in"] = CS.entry + pd.Timedelta(minutes=1); CS = CS[CS.entry >= "2023-11-01"]
    oth = {k: L[k].assign(t_in=L[k].entry + pd.Timedelta(minutes=1 if k == "OD" else 0)) for k in ("OD", "H", "G")}

    def morning(s, c):
        ev = []
        for k in ("OD", "H", "G"):
            if not c.get(k): continue
            q = c[k] / C.LIST_SIZE[k]; t = oth[k]; t = t[(t.day == s.day) & (t.t_in < s.t_in)]
            for _, r in t.iterrows():
                if r.exit <= s.t_in: ev += [(r.exit - pd.Timedelta(seconds=1), r.mae * q), (r.exit, (r.pnl - r.mae) * q)]
                else: ev += [(s.t_in - pd.Timedelta(seconds=1), r.mae * q)]
        ev.sort(key=lambda x: x[0]); cum = np.cumsum([v for _, v in ev]) if ev else np.array([0.0])
        return float(sum(v for _, v in ev)), min(0.0, float(cum.min()))

    rows = []
    for p in PLAN_ORDER:
        R = C.PLANS[p]
        for phase, c in zip(("Eval", "Funded"), FINAL[p]):
            alone = withc = 0.0; hurt = missed = 0; dips = []
            for _, s in CS.iterrows():
                sp = s.pnl * c["CS"] / 6; smae = s.mae * c["CS"] / 6
                closed, low = morning(s, c); dips.append(low)
                if R["dll"]:
                    dll = R["dll"]
                    a = -dll if smae <= -dll else sp
                    wv = 0.0 if low <= -dll else (-dll - closed if closed + smae <= -dll else sp)
                    missed += low <= -dll
                else:
                    a = wv = sp
                alone += a; withc += wv; hurt += wv < a - 1
            dips = pd.Series(dips)
            rows.append(dict(plan=p, phase=phase, saty_size=f"{c['CS']} MNQ", daily_loss_limit=R["dll"] or "None",
                             saty_alone=round(alone), saty_with_others=round(withc), cost=round(alone - withc),
                             cost_share=round((alone - withc) / alone, 3) if alone else 0.0, days_hurt=hurt,
                             days_missed=missed, saty_days=len(CS), morning_dip_median=round(dips.median()),
                             morning_dip_worst=round(dips.min()), drawdown=R["dd"]))
    return pd.DataFrame(rows)


def entry_times(L):
    rows = []
    for lab, t, shift in (("Saty + Opening Drive script, Nov 2023 - Oct 2026", L["CS"], 1),
                          ("Saty standalone list, May 2019 - Oct 2026", L["S"], 10)):
        tm = (t.entry + pd.Timedelta(minutes=shift)).dt.strftime("%H:%M")
        g = t.assign(slot=tm).groupby("slot").pnl
        per = 6 if shift == 1 else 3
        for slot, n in g.size().items():
            rows.append(dict(source=lab, entry_time=slot, trades=int(n), share=round(n / len(t), 3),
                             avg_pnl_per_mnq=round(g.mean()[slot] / per)))
    return pd.DataFrame(rows)


def strategy_days(L):
    start = pd.Timestamp("2023-11-01"); yrs = len(pd.bdate_range(start, "2026-10-16")) / 252
    size = {"CS": 2, "OD": 500, "H": 4, "G": 1}
    name = {"CS": "Saty", "OD": "Opening Drive", "H": "Halyard", "G": "Gold BO"}
    days = {k: L[k][L[k].entry >= start].groupby("day").pnl.sum() * size[k] / C.LIST_SIZE[k] for k in size}
    rows = []
    for ks in (["CS"], ["OD"], ["H"], ["G"], ["CS", "OD"], ["CS", "G"], ["CS", "H", "G"], ["CS", "OD", "H", "G"]):
        d = pd.concat([days[k] for k in ks], axis=1, sort=True).fillna(0).sum(axis=1)
        rows.append(dict(set=" + ".join(name[k] for k in ks), legs=len(ks), trade_days_per_year=round(len(d) / yrs),
                         days_over_150_per_year=round((d > 150).sum() / yrs), win_day_share=round(float((d > 0).mean()), 3),
                         profit_per_year=round(d.sum() / yrs)))
    return pd.DataFrame(rows)


def copies():
    a = pd.read_csv(C.RESULTS / "compare_best.csv")
    names = {"A  Saty+Halyard+Gold": "Saty + Halyard + Gold BO", "B  Saty+Opening Drive": "Saty + Opening Drive",
             "MIX all four": "All four"}
    sets = []
    for _, r in a.iterrows():
        for w, lab in WLAB.items():
            sets.append(dict(set=names[r.family], window=lab, net_per_year=round(r[f"{w}|net"]),
                             bad_run_p10=round(r[f"{w}|p10"]), evals_per_year=round(r[f"{w}|evals"], 1),
                             first_payout_days=int(r[f"{w}|first_pay"]), eval_sizes=r["eval"], funded_sizes=r["funded"]))
    sets = pd.DataFrame(sets)
    split = sets[sets.set != "All four"].groupby("window", as_index=False).net_per_year.mean().round()
    for _, r in split.iterrows():
        sets.loc[len(sets)] = dict(set="Split accounts: half A and half B", window=r.window, net_per_year=int(r.net_per_year),
                                   bad_run_p10=np.nan, evals_per_year=np.nan, first_payout_days=np.nan, eval_sizes="", funded_sizes="")
    part = pd.read_csv(C.RESULTS / "saty_one_partner.csv")
    pm = pd.read_csv(C.RESULTS / "pm_window.csv")
    return sets, part, pm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sim = C.Sim(SIMS, 7)
    plans, dist = plans_and_outcomes(sim); plans.to_csv(OUT / "plans.csv", index=False); dist.to_csv(OUT / "plan_outcomes.csv", index=False)
    print(plans[["plan", "window", "net_per_year", "bad_run_p10", "peak_micros_funded"]].to_string(index=False), flush=True)
    yr, mo = yearly(sim); yr.to_csv(OUT / "yearly.csv", index=False); mo.to_csv(OUT / "monthly_cumulative.csv", index=False)
    print(yr.to_string(index=False), flush=True)
    cal = calendar(sim); cal.to_csv(OUT / "calendar_years.csv", index=False); print(cal.to_string(index=False), flush=True)
    sd = saty_dll(sim.L); sd.to_csv(OUT / "saty_dll.csv", index=False); print(sd.to_string(index=False), flush=True)
    entry_times(sim.L).to_csv(OUT / "saty_entry_times.csv", index=False)
    strategy_days(sim.L).to_csv(OUT / "strategy_days.csv", index=False)
    sets, part, pm = copies()
    sets.to_csv(OUT / "strategy_sets.csv", index=False); part.to_csv(OUT / "saty_partner.csv", index=False)
    pm.to_csv(OUT / "pm_window.csv", index=False)


if __name__ == "__main__":
    main()
