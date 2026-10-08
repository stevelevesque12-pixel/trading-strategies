"""Sideways-day mean reversion, to pair with the ORB trend strategies.

Trades only when the chop filter says SIDEWAYS (the days the ORB strategies sit out): fade a stretch
away from session VWAP back toward it.

  long : regime sideways and close < VWAP - k x ATR and RSI(n) < 50 - band   (short = mirror)
  exit : take-profit tp_k x ATR (about the distance back to VWAP), stop sl_k x ATR, flat 15:55 ET
  first bar of each stretch only ('fresh')

Grid-searched on ES + NQ 2016-22 (score = the worse market's daily Sharpe, >= 200 trades each), then
reported once on 2023-26, real MES 2025-26, and combined with the ORB strategies for Lucid odds.

    python -m research.experiments.mean_revert
"""

import itertools

import numpy as np

from research.candidate import ORB, ORB_VWAP
from research.experiments.fast_pass import sim
from trend import engine, indicators as ind
from trend.data import load
from trend.strategy import FEE_RT, SLIP_TICKS, LucidRules, backtest, daily_series, metrics

R = LucidRules()


def mr_trades(m, pv, k, rsi_n, band, chop_hi, sl_k, tp_k, window="rth", risk=300.0, atr_n=14):
    a = ind.atr(m.h, m.l, m.c, atr_n)
    vw = ind.session_vwap(m.h, m.l, m.c, m.v, m.day_id)
    rs = ind.rsi(m.c, rsi_n)
    chop = np.nan_to_num(ind.choppiness(m.h, m.l, m.c, 30), nan=0)
    side = chop > chop_hi
    lg = side & (m.c < vw - k * a) & (rs < 50 - band)
    sh = side & (m.c > vw + k * a) & (rs > 50 + band)
    lg = lg & ~np.r_[False, lg[:-1]]
    sh = sh & ~np.r_[False, sh[:-1]]
    regime = np.full(len(m.c), 2.0)  # always "full size" -- the filter is already applied above
    return engine.run(m.o, m.h, m.l, m.c, a, lg, sh, np.zeros(len(m.c)), regime, m.entry_mask(window),
                      m.flatten_mask(), m.day_id, float(sl_k), float(tp_k), float(risk), 1.0, pv, 0.25,
                      FEE_RT, SLIP_TICKS, R.max_micros, 1e9, 1e9, False, 0.0)


def sharpe(m, tr):
    _, p, _, _ = daily_series(m, tr)
    return p.mean() / p.std() * np.sqrt(252) if p.std() > 0 else 0.0


def main():
    es, nq = load("es_15m"), load("nq_15m")
    sets = {"es_is": (es.slice(end="2023-01-01"), 5.0), "nq_is": (nq.slice(end="2023-01-01"), 2.0),
            "es_oos": (es.slice(start="2023-01-01"), 5.0), "nq_oos": (nq.slice(start="2023-01-01"), 2.0),
            "mes": (load("mes_15m"), 5.0)}
    grid = itertools.product([1.0, 1.5, 2.0, 2.5], [9, 14], [10, 20], [50, 55, 61.8], [1.0, 1.5, 2.0], [0.75, 1.0, 1.5, 2.0],
                             ["rth", "ny_am", "pm"])
    best = []
    for k, rn, band, ch, sl, tp, win in grid:
        sc = []
        for key in ("es_is", "nq_is"):
            m, pv = sets[key]
            tr = mr_trades(m, pv, k, rn, band, ch, sl, tp, win)
            sc.append(sharpe(m, tr) if len(tr) >= 200 else -9)
        best.append((min(sc), (k, rn, band, ch, sl, tp, win)))
    best.sort(reverse=True)
    print("top in-sample (worse of ES/NQ 2016-22 daily Sharpe):")
    for s, p in best[:8]:
        row = []
        for key in ("es_is", "nq_is", "es_oos", "nq_oos", "mes"):
            m, pv = sets[key]
            r = metrics(m, mr_trades(m, pv, *p), R, curve_points=0)
            row.append(f"{key} n{r['trades']} PF {r['profit_factor']:.2f}")
        print(f"  IS-Sharpe {s:.2f} k={p[0]} rsi={p[1]}/{p[2]} chop>{p[3]} sl={p[4]} tp={p[5]} {p[6]} | " + " | ".join(row))
    p = best[0][1]
    # correlation with ORB legs and combined Lucid odds
    print("\ncombined with ORB (MNQ settings on NQ, MES settings on ES), MR at $300/leg on both:")
    for per in ("is", "oos"):
        legs = []
        for key, spec in ((f"nq_{per}", ORB), (f"es_{per}", ORB_VWAP)):
            m, pv = sets[key]
            legs.append(daily_series(m, backtest(m, spec, R, point_value=pv)))
        for key in (f"es_{per}", f"nq_{per}"):
            m, pv = sets[key]
            legs.append(daily_series(m, mr_trades(m, pv, *p)))
        # align on dates
        maps = [dict(zip(sets[f"{'nq' if i in (0, 3) else 'es'}_{per}"][0].day_dates[d], zip(pp, ll, nn)))
                for i, (d, pp, ll, nn) in enumerate(legs)]
        days = sorted(set.intersection(*[set(x) for x in maps]))
        A = np.array([[x[d] for d in days] for x in maps])
        orb_p = A[:2, :, 0].sum(0)
        mr_p = A[2:, :, 0].sum(0)
        print(f"  {per}: corr(ORB daily, MR daily) = {np.corrcoef(orb_p, mr_p)[0, 1]:.2f}; "
              f"Sharpe ORB {orb_p.mean() / orb_p.std() * np.sqrt(252):.2f}, MR {mr_p.mean() / mr_p.std() * np.sqrt(252):.2f}, "
              f"both {(orb_p + mr_p).mean() / (orb_p + mr_p).std() * np.sqrt(252):.2f}")
        for lab, P, L, N in (("ORB only", A[:2, :, 0].sum(0), A[:2, :, 1].sum(0), A[:2, :, 2].sum(0)),
                             ("ORB + MR", A[:, :, 0].sum(0), A[:, :, 1].sum(0), A[:, :, 2].sum(0))):
            a42 = sim(P, L, N, 42)
            a1y = sim(P, L, N, 250)
            print(f"    {lab:<9} pass<=42 sessions {a42[0]:.0%} bust {a42[1]:.0%} | within 1y pass {a1y[0]:.0%} "
                  f"bust {a1y[1]:.0%} median {a1y[2]} | trades/wk {N.sum() / (len(N) / 5):.1f}")


if __name__ == "__main__":
    main()
