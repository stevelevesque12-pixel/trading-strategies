"""Overlay experiments on the live candidate: daily volatility regime and daily trend alignment.

    python -m research.experiments.overlays
"""

from dataclasses import replace

import numpy as np

from research.candidate import LIVE
from research.validate import mc_lucid
from trend import components as comp
from trend import engine, indicators as ind
from trend.data import load
from trend.strategy import MES_POINT_VALUE, MES_TICK, FEE_RT, SLIP_TICKS, LucidRules, _atr, daily_series, metrics, signals


def daily_vol_rank(m, n=10, look=250):
    """Yesterday's n-day average daily range, as a percentile of the prior `look` days (0..1), per bar."""
    _, sess_idx, _ = comp._session_last_close(m)
    first = np.r_[True, m.day_id[1:] != m.day_id[:-1]]
    starts = np.flatnonzero(first)
    ends = np.r_[starts[1:], len(m.c)]
    rng = np.array([m.h[a:b].max() - m.l[a:b].min() for a, b in zip(starts, ends)])
    avg = ind.sma(rng, n)
    rank = ind.percent_rank(np.nan_to_num(avg), look)
    out = np.full(len(m.c), np.nan)
    ok = sess_idx >= 1
    out[ok] = rank[sess_idx[ok] - 1]
    return out


def run(m, s, long_mask=None, short_mask=None, rules=LucidRules()):
    t, r, lg, sh = signals(m, s)
    if long_mask is not None:
        lg = lg & long_mask
    if short_mask is not None:
        sh = sh & short_mask
    tr = engine.run(m.o, m.h, m.l, m.c, _atr(m, s.atr_n), lg, sh, t.astype(float), r.astype(float),
                    m.entry_mask(s.window), m.flatten_mask(), m.day_id, s.sl_k, s.tp_k, s.risk_usd, s.small_mult,
                    MES_POINT_VALUE, MES_TICK, FEE_RT, SLIP_TICKS, rules.max_micros, s.daily_loss_limit,
                    s.daily_profit_cap, s.exit_on_flip, s.trail_k)
    return tr


def report(label, m, tr, rules=LucidRules()):
    r = metrics(m, tr, rules, curve_points=0, cushion_sizing=True)
    _, p, l, n = daily_series(m, tr)
    mc = mc_lucid(p, l, n, rules, True, n=500) or {}
    print(f"  {label:<34} n{r['trades']:<4} wr {r['win_rate']:.2f} pf {r['profit_factor']:.2f} net {r['net']:>7.0f} "
          f"dd {r['max_dd']:>5.0f} | MC pass {mc.get('pass', 0):.0%} bust {mc.get('bust', 0):.0%}")
    return r


def main():
    full = load("es_15m")
    periods = {"IS": full.slice(end="2023-01-01"), "OOS": full.slice(start="2023-01-01"), "MES": load("mes_15m")}
    for name, m in periods.items():
        print(name)
        vr = daily_vol_rank(m)
        report("baseline", m, run(m, LIVE))
        for th in (0.2, 0.33, 0.5):
            mask = np.nan_to_num(vr, nan=1.0) >= th
            report(f"daily vol rank >= {th}", m, run(m, LIVE, mask, mask))
        for n in (10, 20, 50):
            d = comp._daily_ema(m, n)
            report(f"daily EMA{n} aligned", m, run(m, LIVE, d == 1, d == -1))
        for n in (20, 50):
            d = comp.trend_daily_slope(m, n)
            report(f"daily EMA{n} slope aligned", m, run(m, LIVE, d == 1, d == -1))


if __name__ == "__main__":
    main()
