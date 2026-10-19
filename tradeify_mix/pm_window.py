"""Net payouts per year with and without the Saty afternoon (PM) window, four-strategy mix at each plan's best sizes.

The PM window (10m crosses closing 13:00-14:30 ET, no VWAP filter, flat 15:50) is not in the TradingView lists, so its
effect comes from the 1-minute NQ backtest in AI-personal-hedge-fund (scripts/saty_pm_window_payouts.py, saty_days with
and without PM_WIN): results/pm_delta_1mnq.csv holds, per day, the Saty P&L and worst-open-P&L change for 1 MNQ. It is
added as an afternoon trade scaled by each plan's Saty size. Window: Nov 2023 - Dec 11 2025 (where the 1-minute data
ends).
  python -m tradeify_mix.pm_window
"""
import numpy as np
import pandas as pd

from tradeify_mix import compare as C


def pm_trades():
    d = pd.read_csv(C.RESULTS / "pm_delta_1mnq.csv", index_col=0, parse_dates=True)
    day = d.index.normalize()
    t = pd.DataFrame(dict(entry=day + pd.Timedelta(hours=13), exit=day + pd.Timedelta(hours=15, minutes=50),
                          pnl=d.pnl.values, mae=np.minimum(d.low.values, 0), dir=1, signal="PM", qty=1, day=day))
    t["key"] = ((t.exit - pd.Timedelta(hours=18)).dt.hour * 60 + t.exit.dt.minute).astype(float)
    return t


def parse(s):
    c = {}
    for tok in s.split():
        k = tok.rstrip("0123456789"); c[k] = int(tok[len(k):])
    return c


def main(sims=2000):
    C.WINDOWS = {"Nov 2023 - Dec 2025": ("2023-11-01", "2025-12-11")}
    sim, w, pmt = C.Sim(sims, 7), "Nov 2023 - Dec 2025", pm_trades()
    yrs = len(sim.spans[w]) / 252

    def P(c, pm):
        ev = C.events_for(sim.L, c)
        if pm:
            ev = pd.concat([ev, C.events(pmt, c["CS"])])
        return C.prefix(ev, sim.spans[w])

    rows = []
    for _, r in pd.read_csv(C.RESULTS / "plans_best.csv").iterrows():
        ce, cf = parse(r["eval"]), parse(r["funded"])
        for pm in (False, True):
            s = C.run_plan(P(ce, pm), P(cf, pm), sim.ix[w], C.PLANS[r.plan])
            rows.append(dict(plan=r.plan, pm="with PM" if pm else "without", raw_profit_yr=round(P(cf, pm)[:, -1].sum() / yrs),
                             net_payouts_yr=round(s["net"]), bad_year=round(s["p10"]), evals=round(s["evals"], 1),
                             first_pay=int(s["first_pay"])))
    df = pd.DataFrame(rows); df.to_csv(C.RESULTS / "pm_window.csv", index=False)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
