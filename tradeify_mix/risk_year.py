"""Risk scenarios and a typical year for each Tradeify plan, four-strategy mix at the final sizes (dashboard_data.FINAL).

  python -m tradeify_mix.risk_year      # -> results/risk/*.csv

One simulated year = 250 trading days drawn in 5-day blocks, buying a new evaluation the day after any fail or blown
account. Scenarios change the trade lists before simulating:
  winners x f / losers x g   every strategy's winning (losing) trades scaled, the dip inside losers scaled with them
  extra slippage             one more tick per side on every contract
  market regime              days drawn only from a calendar window
  strategy stops working     that strategy removed
  no edge                    each strategy's winners scaled so its trades sum to zero
"""
import numpy as np
import pandas as pd

from tradeify_mix import compare as C
from tradeify_mix.dashboard_data import FINAL, PLAN_ORDER

OUT = C.RESULTS / "risk"
SIMS = 3000
BASE = ("2023-11-01", "2026-10-16")
TICK = dict(CS=0.5, OD=0.5, H=0.5, G=1.0, S=0.5)      # $ per tick per contract

SCENARIOS = [
    ("Base case (backtest as is)", "Base", {}),
    ("Winners 10% smaller", "Live drag", dict(win=0.9)),
    ("Winners 25% smaller", "Live drag", dict(win=0.75)),
    ("Losses 25% bigger", "Live drag", dict(loss=1.25)),
    ("Winners 25% smaller and losses 25% bigger", "Live drag", dict(win=0.75, loss=1.25)),
    ("One more tick of slippage per side", "Live drag", dict(slip=1)),
    ("Weak market: 2025 days only", "Market", dict(window=("2025-01-01", "2025-12-31"))),
    ("Without 2026: Nov 2023 - Dec 2025", "Market", dict(window=("2023-11-01", "2025-12-31"))),
    ("Strong market: last 12 months", "Market", dict(window=("2025-10-17", "2026-10-16"))),
    ("Saty's winners halved", "Strategy", dict(win_leg={"CS": 0.5})),
    ("Gold BO stops working", "Strategy", dict(drop="G")),
    ("Opening Drive stops working", "Strategy", dict(drop="OD")),
    ("Halyard stops working", "Strategy", dict(drop="H")),
    ("No edge in any strategy", "Strategy", dict(no_edge=True)),
]


def transform(L0, sc):
    L = {}
    for k, t in L0.items():
        t = t.copy()
        win = sc.get("win_leg", {}).get(k, sc.get("win", 1.0))
        loss = sc.get("loss", 1.0)
        if sc.get("no_edge"):
            pos, neg = t.pnl.clip(lower=0).sum(), -t.pnl.clip(upper=0).sum()
            win = neg / pos if pos > 0 else 1.0
        w = t.pnl > 0
        t.loc[w, "pnl"] *= win
        t.loc[~w, "pnl"] *= loss
        t.loc[~w, "mae"] *= loss
        if sc.get("slip"):
            extra = 2 * sc["slip"] * TICK.get(k, 0.5) * t.qty
            t["pnl"] -= extra; t["mae"] -= extra
        t["mae"] = np.minimum(t.mae, np.minimum(t.pnl, 0))
        L[k] = t
    return L


def cfg_for(cfg, sc):
    d = sc.get("drop")
    return {k: v for k, v in cfg.items() if k != d}


def simulate(L, cfg_e, cfg_f, plan, window, ix):
    days = pd.bdate_range(*window)
    Pe = C.prefix(C.events_for(L, cfg_e), days); Pf = C.prefix(C.events_for(L, cfg_f), days)
    return C.run_plan(Pe, Pf, ix, C.PLANS[plan], return_cash=True)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    L0 = C.load_all()
    rng_cache = {}

    def ix_for(window):
        if window not in rng_cache:
            rng_cache[window] = C.boot(len(pd.bdate_range(*window)), SIMS, 250, np.random.default_rng(17))
        return rng_cache[window]

    scen_rows, typ_rows, month_rows, first_rows, outlay_rows, events_rows = [], [], [], [], [], []
    for name, group, sc in SCENARIOS:
        L = transform(L0, sc)
        window = sc.get("window", BASE)
        for plan in PLAN_ORDER:
            ce, cf = (cfg_for(c, sc) for c in FINAL[plan])
            st, cash = simulate(L, ce, cf, plan, window, ix_for(window))
            net = cash.sum(1); cum = cash.cumsum(1); outlay = -np.minimum(cum.min(1), 0)
            scen_rows.append(dict(scenario=name, group=group, plan=plan, net_per_year=round(net.mean()),
                                  median=round(np.median(net)), bad_year_p10=round(np.percentile(net, 10)),
                                  good_year_p90=round(np.percentile(net, 90)), loss_year_share=round(float((net < 0).mean()), 3),
                                  evals_per_year=round(st["evals"], 1), payouts_per_year=round(st["payouts"], 1),
                                  max_outlay_p90=round(np.percentile(outlay, 90))))
            if name != SCENARIOS[0][0]:
                continue
            # typical year: the path closest to the median net with a median number of evaluations
            evals = (cash < 0).sum(1)
            score = np.abs(net - np.median(net)) / max(np.std(net), 1) + np.abs(evals - np.median(evals)) / max(np.std(evals), 1)
            p = int(np.argmin(score))
            month = np.minimum(np.arange(250) * 12 // 250, 11)
            for m in range(12):
                sel = month == m
                mc = cash[:, sel].sum(1); cm = cum[:, np.nonzero(sel)[0][-1]]
                month_rows.append(dict(plan=plan, month=m + 1, typical_cash=round(cash[p, sel].sum()),
                                       typical_cumulative=round(cum[p, np.nonzero(sel)[0][-1]]),
                                       cum_p10=round(np.percentile(cm, 10)), cum_median=round(np.median(cm)),
                                       cum_p90=round(np.percentile(cm, 90)), month_cash_mean=round(mc.mean())))
            for d in np.nonzero(cash[p] != 0)[0]:
                v = cash[p, d]
                events_rows.append(dict(plan=plan, trading_day=int(d + 1), month=int(month[d] + 1),
                                        event="Payout" if v > 0 else "Evaluation bought", amount=round(v)))
            firstpay = np.array([np.argmax(r > 0) + 1 if (r > 0).any() else 999 for r in cash])
            for m in range(1, 13):
                first_rows.append(dict(plan=plan, month=m, share_paid=round(float((firstpay <= m * 250 / 12).mean()), 3)))
            for q in (10, 25, 50, 75, 90, 95):
                outlay_rows.append(dict(plan=plan, percentile=q, max_outlay=round(np.percentile(outlay, q))))
            typ_rows.append(dict(plan=plan, typical_net=round(net[p]), median_net=round(np.median(net)), mean_net=round(net.mean()),
                                 loss_year_share=round(float((net < 0).mean()), 3), typical_evals=int(evals[p]),
                                 typical_payouts=int((cash[p] > 0).sum()), first_payout_month_median=round(float(np.median(np.minimum(firstpay, 250)) * 12 / 250), 1),
                                 paid_within_year=round(float((firstpay < 999).mean()), 3), max_outlay_median=round(np.median(outlay)),
                                 max_outlay_p90=round(np.percentile(outlay, 90)), eval_sizes=FINAL[plan][0], funded_sizes=FINAL[plan][1]))
        print(name, flush=True)
    pd.DataFrame(scen_rows).to_csv(OUT / "scenarios.csv", index=False)
    t = pd.DataFrame(typ_rows)
    t["eval_sizes"] = t.eval_sizes.map(lambda c: f"Saty {c.get('CS', 0)} MNQ · OD ${c.get('OD', 0):,} · Halyard {c.get('H', 0)} MNQ · Gold {c.get('G', 0)} MGC")
    t["funded_sizes"] = t.funded_sizes.map(lambda c: f"Saty {c.get('CS', 0)} MNQ · OD ${c.get('OD', 0):,} · Halyard {c.get('H', 0)} MNQ · Gold {c.get('G', 0)} MGC")
    t.to_csv(OUT / "typical_year.csv", index=False)
    pd.DataFrame(month_rows).to_csv(OUT / "typical_months.csv", index=False)
    pd.DataFrame(events_rows).to_csv(OUT / "typical_events.csv", index=False)
    pd.DataFrame(first_rows).to_csv(OUT / "first_payout.csv", index=False)
    pd.DataFrame(outlay_rows).to_csv(OUT / "outlay.csv", index=False)
    print(t.drop(columns=["eval_sizes", "funded_sizes"]).to_string(index=False))
    s = pd.DataFrame(scen_rows)
    print(s.pivot(index="scenario", columns="plan", values="net_per_year")[PLAN_ORDER].loc[[x[0] for x in SCENARIOS]].to_string())


if __name__ == "__main__":
    main()
