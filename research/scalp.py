"""5-minute scalping track for the research loop.

Data is short (all we have): MES 5m 2026-05-10..2026-10-07, MNQ/MYM/M2K 5m 2026-05-10..2026-08-25.

  * Fit (in-sample): MES and MNQ, 2026-05-10..2026-08-15 -- score = the worse market's daily Sharpe.
  * Out-of-sample:   MES 2026-08-15..2026-10-07 (never seen during fitting).
  * Out-of-instrument: MYM and M2K over their whole files (never fitted).
  * Sizing: Monte Carlo Lucid 50K pass-minus-bust within 42 sessions, on in-sample daily P&L blocks.

With ~5 months of data this can only screen ideas; anything that passes needs forward testing.
"""

import random
import time
from dataclasses import asdict, replace

import numpy as np

from research.experiments.fast_pass import sim
from trend import components as comp
from trend.data import load
from trend.strategy import LucidRules, Spec, backtest, daily_series, evaluate, metrics

SPLIT = "2026-08-15"
ATR_N = [10, 14, 20]
SL_K = [0.75, 1.0, 1.5, 2.0]
TP_K = [1.0, 1.5, 2.0, 3.0, 4.0]
TRAIL_K = [0.0, 0.0, 1.0, 1.5]
TRIGGERS = ["any", "any", "pullback", "fresh"]
PB_N = [9, 20, 34]
WINDOWS = ["rth", "ny_am", "all_rth", "pm"]
RISK = [150, 250, 400, 600]
DLL = [300, 450, 600]
CAP = [900, 1400, 99999]
MIN_TRADES = 100  # in-sample, per market (~1+/day)


def _pv(name):
    return {"mes": (5.0, 0.25), "mnq": (2.0, 0.25), "mym": (0.5, 1.0), "m2k": (5.0, 0.1)}[name.split("_")[0]]


def _eval(m, s, rules, **kw):
    pv, tick = _pv(m.name)
    return metrics(m, backtest(m, s, rules, point_value=pv, tick=tick), rules, cushion_sizing=s.cushion_sizing, **kw)


def _rand(family, rng):
    t, c1, c2, r = family
    sp = lambda tbl, n: {k: rng.choice(v) for k, v in tbl[n][1].items()}
    return Spec(trend=t, trend_p=sp(comp.TREND, t), conf1=c1, conf1_p=sp(comp.CONFIRM, c1),
                conf2=c2, conf2_p=sp(comp.CONFIRM, c2), regime=r, regime_p=sp(comp.REGIME, r),
                atr_n=rng.choice(ATR_N), sl_k=rng.choice(SL_K), tp_k=rng.choice(TP_K), trail_k=rng.choice(TRAIL_K),
                trigger=rng.choice(TRIGGERS), pb_n=rng.choice(PB_N), exit_on_flip=rng.choice([True, False]),
                window=rng.choice(WINDOWS), risk_usd=250)


def _mut(s, family, rng):
    o = _rand(family, rng)
    attrs = ["trend_p", "conf1_p", "conf2_p", "regime_p", "atr_n", "sl_k", "tp_k", "trail_k", "trigger", "window",
             "exit_on_flip", "pb_n"]
    k = rng.choice(attrs)
    return replace(s, **{k: getattr(o, k)})


def _ok(s):
    if s.trend == "ema_cross" and s.trend_p["fast"] >= s.trend_p["slow"]:
        return False
    for n, p in (("conf1", s.conf1_p), ("conf2", s.conf2_p)):
        if getattr(s, n) == "macd" and p["fast"] >= p["slow"]:
            return False
    return True


def _score(r):
    return r["sharpe"] if r["trades"] >= MIN_TRADES else -5 + r["trades"] / MIN_TRADES


def _mc42(m, s, rules, seed=0, length=4000):
    """Monte Carlo: resample 5-session blocks of daily P&L into one long path, start an eval every 5
    sessions, count passes / busts within 42 sessions."""
    pv, tick = _pv(m.name)
    _, p, lo, n = daily_series(m, backtest(m, s, rules, point_value=pv, tick=tick))
    nb = len(p) // 5
    if nb < 4 or n.sum() == 0:
        return {"pass": 0.0, "bust": 0.0, "timeout": 1.0}
    rng = np.random.default_rng(seed)
    idx = (rng.integers(0, nb, length // 5)[:, None] * 5 + np.arange(5)).ravel()
    pr, br, _ = sim(p[idx], lo[idx], n[idx], 42, cushion=s.cushion_sizing)
    return {"pass": pr, "bust": br, "timeout": 1 - pr - br}


def optimise(family, seed=None, n_random=300, n_mutate=300):
    rng = random.Random(seed)
    rules = LucidRules()
    mes, mnq = load("mes_5m_m"), load("mnq_5m")
    mes_is, mes_oos = mes.slice(end=SPLIT), mes.slice(start=SPLIT)
    nq_is = mnq.slice(end=SPLIT)
    t0, tried = time.time(), 0
    best, best_sc, seen = None, -1e9, set()

    def consider(s):
        nonlocal best, best_sc, tried
        if not _ok(s) or s.key() in seen:
            return
        seen.add(s.key())
        tried += 1
        sc = min(_score(_eval(mes_is, s, rules, curve_points=0)), _score(_eval(nq_is, s, rules, curve_points=0)))
        if sc > best_sc:
            best, best_sc = s, sc

    for _ in range(n_random):
        consider(_rand(family, rng))
    for _ in range(n_mutate):
        consider(_mut(best, family, rng))
    if best is None or best_sc < 0:
        return None  # nothing tradeable in-sample on both markets; don't log

    sized, sized_sc = best, -1e9
    for risk in RISK:
        for dll in DLL:
            for cap in CAP:
                for cush in (False, True):
                    s = replace(best, risk_usd=risk, daily_loss_limit=dll, daily_profit_cap=cap, cushion_sizing=cush)
                    tried += 1
                    if _eval(mes_is, s, rules, curve_points=0)["trades"] < MIN_TRADES:
                        continue  # too small to place most trades
                    mc = _mc42(mes_is, s, rules)
                    sc = mc["pass"] - mc["bust"]
                    if sc > sized_sc:
                        sized, sized_sc = s, sc

    r_is, r_oos = _eval(mes_is, sized, rules, curve_points=0), _eval(mes_oos, sized, rules, curve_points=0)
    rec = {
        "id": f"5m_scalp:{sized.name()}:{abs(hash(sized.key())) % 10**8}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "v": 6, "track": "5m_scalp", "dataset": "mes_5m_m",
        "split": SPLIT, "tf_min": 5, "name": sized.name(), "spec": asdict(sized), "configs_tried": tried,
        "seconds": round(time.time() - t0, 1),
        "is": r_is, "oos": r_oos, "full": _eval(mes, sized, rules, curve_points=100),
        "nq_is": _eval(nq_is, sized, rules, curve_points=0),
        "nq_oos": _eval(mnq.slice(start=SPLIT), sized, rules, curve_points=0),
        "oo_instrument": {k: _eval(load(k), sized, rules, curve_points=0) for k in ("mym_5m", "m2k_5m")},
        "fast": {k: {"pass_rate": v["pass"], "bust_rate": v["bust"], "median_days_to_pass": None}
                 for k, v in (("es_is", _mc42(mes_is, sized, rules)), ("es_oos", _mc42(mes_oos, sized, rules)),
                              ("nq_is", _mc42(nq_is, sized, rules)), ("nq_oos", {"pass": None, "bust": None}))},
    }
    rec["mes_check"] = rec["oos"]
    oi = rec["oo_instrument"]
    rec["robust"] = bool(
        r_is["trades"] >= MIN_TRADES and r_oos["trades"] >= 30
        and r_is["profit_factor"] > 1.1 and rec["nq_is"]["profit_factor"] > 1.1 and r_oos["profit_factor"] > 1.1
        and max(oi["mym_5m"]["profit_factor"], oi["m2k_5m"]["profit_factor"]) > 1.0
    )
    return rec
