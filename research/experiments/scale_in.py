"""Scale-in on ORB trend days: a second entry that fires only once the day is clearly trending.

Add-on leg (same market as the base ORB leg): after the base ORB direction is set, enter again when price
has run >= k x ATR beyond the opening-range edge, still in the base direction, confirmed by the same daily-EMA
filter, signal bars up to `last` ET. Own bracket: stop sl_k x ATR, target tp_k x ATR, flat 15:55.
Chosen on ES+NQ 2016-22 only; 2023-26 reported once. Lucid odds for base ORB alone vs base + add-on.

    python -m research.experiments.scale_in
"""

import itertools

import numpy as np

from research.candidate import ORB, ORB_VWAP
from research.experiments.fast_pass import sim
from trend import components as comp, engine, indicators as ind
from trend.data import WINDOWS, load
from trend.strategy import FEE_RT, SLIP_TICKS, LucidRules, backtest, daily_series, metrics
from datetime import time

R = LucidRules()


def or_levels(m, minutes=60):
    om = (m.close_minute - m.tf_min) % 1440
    hi = np.full(len(m.c), np.nan)
    lo = np.full(len(m.c), np.nan)
    cur, h, l = -1, -np.inf, np.inf
    for i in range(len(m.c)):
        if m.day_id[i] != cur:
            cur, h, l = m.day_id[i], -np.inf, np.inf
        if 570 <= om[i] < 570 + minutes:
            h, l = max(h, m.h[i]), min(l, m.l[i])
        elif 570 + minutes <= om[i] < 1020 and h > -np.inf:
            hi[i], lo[i] = h, l
    return hi, lo


def addon_trades(m, pv, k, sl_k, tp_k, last, dema, risk=300.0, atr_n=14):
    a = ind.atr(m.h, m.l, m.c, atr_n)
    hi, lo = or_levels(m)
    d = comp.trend_orb(m, 60)
    de = comp._daily_ema(m, dema)
    lg = (d == 1) & (de == 1) & (m.c >= hi + k * a)
    sh = (d == -1) & (de == -1) & (m.c <= lo - k * a)
    lg = lg & ~np.r_[False, lg[:-1]]
    sh = sh & ~np.r_[False, sh[:-1]]
    WINDOWS["addon"] = (time(10, 45), last)
    return engine.run(m.o, m.h, m.l, m.c, a, lg, sh, d.astype(float), np.full(len(m.c), 2.0), m.entry_mask("addon"),
                      m.flatten_mask(), m.day_id, float(sl_k), float(tp_k), float(risk), 1.0, pv, 0.25, FEE_RT,
                      SLIP_TICKS, R.max_micros, 1e9, 1e9, True, 0.0)


def dsh(m, tr):
    _, p, _, _ = daily_series(m, tr)
    return p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0


def main():
    es, nq = load("es_15m"), load("nq_15m")
    S = {"es_is": (es.slice(end="2023-01-01"), 5.0), "nq_is": (nq.slice(end="2023-01-01"), 2.0),
         "es_oos": (es.slice(start="2023-01-01"), 5.0), "nq_oos": (nq.slice(start="2023-01-01"), 2.0)}
    res = []
    for k, sl, tp, last, dema in itertools.product([0.5, 1.0, 1.5, 2.0], [1.0, 1.5, 2.0, 3.0], [2.0, 3.0, 5.0, 8.0],
                                                   [time(12, 0), time(13, 30), time(15, 0)], [20, 50]):
        sc = []
        for key in ("es_is", "nq_is"):
            m, pv = S[key]
            tr = addon_trades(m, pv, k, sl, tp, last, dema)
            sc.append(dsh(m, tr) if len(tr) >= 150 else -9)
        res.append((min(sc), (k, sl, tp, last, dema)))
    res.sort(reverse=True)
    print("add-on leg, top by worse-of-ES/NQ 2016-22 daily Sharpe:")
    for s, p in res[:6]:
        row = []
        for key in S:
            m, pv = S[key]
            r = metrics(m, addon_trades(m, pv, *p), R, curve_points=0)
            row.append(f"{key} n{r['trades']} PF {r['profit_factor']:.2f}")
        print(f"  {s:.2f} k={p[0]} sl={p[1]} tp={p[2]} until {p[3]} dEMA{p[4]} | " + " | ".join(row))
    p = res[0][1]
    print("\nLucid odds (MNQ base orb_trend + MES base orb_vwap, add-on on both at $300):")
    for per in ("is", "oos"):
        maps = []
        for key, spec in ((f"nq_{per}", ORB), (f"es_{per}", ORB_VWAP)):
            m, pv = S[key]
            dd, pp, ll, nn = daily_series(m, backtest(m, spec, R, point_value=pv))
            maps.append(dict(zip(m.day_dates[dd], zip(pp, ll, nn))))
        for key in (f"nq_{per}", f"es_{per}"):
            m, pv = S[key]
            dd, pp, ll, nn = daily_series(m, addon_trades(m, pv, *p))
            maps.append(dict(zip(m.day_dates[dd], zip(pp, ll, nn))))
        days = sorted(set.intersection(*[set(x) for x in maps]))
        A = np.array([[x[d] for d in days] for x in maps])
        for lab, sl_ in (("base ORB", slice(0, 2)), ("ORB + add-on", slice(0, 4))):
            P, L, N = A[sl_, :, 0].sum(0), A[sl_, :, 1].sum(0), A[sl_, :, 2].sum(0)
            a42, a1y = sim(P, L, N, 42), sim(P, L, N, 250)
            print(f"  {per:<3} {lab:<13} Sharpe {P.mean() / P.std() * np.sqrt(252):.2f} | pass<=42 {a42[0]:.0%} bust {a42[1]:.0%} | "
                  f"1y pass {a1y[0]:.0%} bust {a1y[1]:.0%} median {a1y[2]}")


if __name__ == "__main__":
    main()
