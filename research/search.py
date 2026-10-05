"""Strategy search: pick a component family, optimise it in-sample, score it out-of-sample, log it.

Protocol (per family = trend + 2 confirmations + regime filter):
  1. Stage 1 (signal quality): random search + local mutation over component params,
     ATR length, ATR band multiples, trigger mode, exit-on-flip and session window,
     at a fixed $200 risk. Objective (v3): the worse of the two in-sample halves' daily Sharpe,
     penalised below 75 trades per half.
  2. Stage 2 (prop sizing): with the signal frozen, sweep risk $/trade, small-size
     multiplier, daily loss limit and daily profit cap to maximise in-sample Lucid 50K
     pass rate minus bust rate, with and without drawdown-cushion sizing (v4).
  3. The frozen spec is then run once on the out-of-sample period (never seen during
     optimisation) and on the real MES 15m contract (Sep 2025 - Aug 2026), and logged.
"""

import json
import random
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np

from trend import components as comp
from trend.data import load
from trend.strategy import LucidRules, Spec, backtest, evaluate, metrics

RESULTS = Path(__file__).resolve().parent / "results.jsonl"

TRACKS = {
    # name: (dataset, in-sample end, description)
    "15m_full": ("es_15m", "2023-01-01", "ES 15m 2016-05..2026-08 priced as MES; IS < 2023, OOS >= 2023"),
    "5m_recent": ("mes_5m", "2026-07-15", "MES 5m 2026-05..2026-08; IS < Jul 15, OOS after (short, low confidence)"),
}

ATR_N = [10, 14, 20, 30]
SL_K = [1.0, 1.5, 2.0, 2.5, 3.0]
TP_K = [1.5, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0]
RISK = [100, 150, 200, 300, 400, 500, 650]
TRAIL_K = [0.0, 0.0, 1.5, 2.0, 3.0]
VERSION = 6  # v6: + whole-tick stop/trail distances (matches TradingView trade-for-trade)
TRIGGERS = ["fresh", "any", "pullback", "pullback"]
PB_N = [9, 20, 34, 50]
SMALL = [0.33, 0.5, 0.66]
DLL = [300, 450, 600, 900]
CAP = [900, 1200, 1400, 99999]


def sample_params(space, rng):
    return {k: rng.choice(v) for k, v in space.items()}


def random_spec(family, rng):
    t, c1, c2, r = family
    return Spec(
        trend=t, trend_p=sample_params(comp.TREND[t][1], rng),
        conf1=c1, conf1_p=sample_params(comp.CONFIRM[c1][1], rng),
        conf2=c2, conf2_p=sample_params(comp.CONFIRM[c2][1], rng),
        regime=r, regime_p=sample_params(comp.REGIME[r][1], rng),
        atr_n=rng.choice(ATR_N), sl_k=rng.choice(SL_K), tp_k=rng.choice(TP_K),
        trigger=rng.choice(TRIGGERS), pb_n=rng.choice(PB_N), exit_on_flip=rng.choice([True, False]),
        window=rng.choice(["rth", "ny_am", "ext"]), trail_k=rng.choice(TRAIL_K),
    )


def mutate(s: Spec, rng):
    s = replace(s, trend_p=dict(s.trend_p), conf1_p=dict(s.conf1_p), conf2_p=dict(s.conf2_p), regime_p=dict(s.regime_p))
    what = rng.randrange(6)
    if what < 4:
        attr, table, name = [("trend_p", comp.TREND, s.trend), ("conf1_p", comp.CONFIRM, s.conf1),
                             ("conf2_p", comp.CONFIRM, s.conf2), ("regime_p", comp.REGIME, s.regime)][what]
        space = table[name][1]
        if space:
            k = rng.choice(list(space))
            getattr(s, attr)[k] = rng.choice(space[k])
    elif what == 4:
        s = replace(s, sl_k=rng.choice(SL_K), tp_k=rng.choice(TP_K), atr_n=rng.choice(ATR_N), trail_k=rng.choice(TRAIL_K))
    else:
        s = replace(s, trigger=rng.choice(TRIGGERS), pb_n=rng.choice(PB_N), exit_on_flip=rng.choice([True, False]),
                    window=rng.choice(["rth", "ny_am", "ext"]))
    return s


def valid(s: Spec):
    if s.trend == "ema_cross" and s.trend_p["fast"] >= s.trend_p["slow"]:
        return False
    for name, p in (("conf1", s.conf1_p), ("conf2", s.conf2_p)):
        if getattr(s, name) == "macd" and p["fast"] >= p["slow"]:
            return False
    return s.tp_k >= s.sl_k * 0.75


def signal_score(r, min_trades):
    if r["trades"] < min_trades:
        return -5 + r["trades"] / min_trades
    return r["sharpe"]


def prop_score(r):
    # a bust costs a fresh eval fee and as much time as a pass is worth: weigh them equally
    lu = r["lucid"]
    return lu["pass_rate"] - lu["bust_rate"]


def optimise_family(family, track="15m_full", n_random=120, n_mutate=120, seed=None):
    rng = random.Random(seed)
    ds, split, _ = TRACKS[track]
    full = load(ds)
    is_m, oos_m = full.slice(end=split), full.slice(start=split)
    mid = is_m.index[len(is_m) // 2]
    is_a, is_b = is_m.slice(end=mid), is_m.slice(start=mid)
    min_trades = 150 if track == "15m_full" else 60
    rules = LucidRules()
    tried = 0
    t0 = time.time()

    best, best_sc, seen = None, -1e9, set()

    def consider(s):
        nonlocal best, best_sc, tried
        if not valid(s) or s.key() in seen:
            return
        seen.add(s.key())
        tried += 1
        # score = the worse of the two in-sample halves: a fit that only works in one regime loses
        sc = min(signal_score(evaluate(is_a, s, rules, curve_points=0), min_trades // 2),
                 signal_score(evaluate(is_b, s, rules, curve_points=0), min_trades // 2))
        if sc > best_sc:
            best, best_sc = s, sc

    for _ in range(n_random):
        consider(random_spec(family, rng))
    for _ in range(n_mutate):
        consider(mutate(best, rng))
    if best is None:
        return None

    # stage 2: prop sizing on the frozen signal
    sized, sized_sc = best, -1e9
    for risk in RISK:
        for small in SMALL:
            for dll in DLL:
                for cap in CAP:
                    s = replace(best, risk_usd=risk, small_mult=small, daily_loss_limit=dll, daily_profit_cap=cap)
                    tried += 1
                    trades = backtest(is_m, s, rules)
                    for cush in (False, True):  # cushion sizing is a sim-side overlay: reuse the trades
                        sc = prop_score(metrics(is_m, trades, rules, curve_points=0, cushion_sizing=cush))
                        if sc > sized_sc:
                            sized, sized_sc = replace(s, cushion_sizing=cush), sc

    r_is = evaluate(is_m, sized, rules, curve_points=0)
    r_oos = evaluate(oos_m, sized, rules, curve_points=0)
    r_all = evaluate(full, sized, rules, curve_points=100)
    rec = {
        "id": f"{track}:{sized.name()}:{abs(hash(sized.key())) % 10**8}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "v": VERSION, "track": track, "dataset": ds, "split": split, "tf_min": full.tf_min,
        "name": sized.name(), "spec": asdict(sized), "configs_tried": tried,
        "seconds": round(time.time() - t0, 1),
        "is": r_is, "oos": r_oos, "full": r_all,
    }
    if track == "15m_full":
        rec["mes_check"] = evaluate(load("mes_15m"), sized, rules, curve_points=0)
    rec["robust"] = bool(
        r_is["trades"] >= min_trades and r_oos["trades"] >= min_trades // 3
        and r_is["profit_factor"] > 1.05 and r_oos["profit_factor"] > 1.05
        and r_oos["sharpe"] > 0.3
    )
    return rec


def all_families():
    confs = sorted(comp.CONFIRM)
    fams = []
    for t in sorted(comp.TREND):
        for i, a in enumerate(confs):
            for b in confs[i + 1:]:
                for r in sorted(comp.REGIME):
                    fams.append((t, a, b, r))
    return fams


def done_families(track):
    out = set()
    if RESULTS.exists():
        for line in RESULTS.read_text().splitlines():
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if d["track"] == track and d.get("v", 1) == VERSION:
                s = d["spec"]
                out.add((s["trend"], s["conf1"], s["conf2"], s["regime"]))
    return out


def append(rec):
    with RESULTS.open("a") as f:
        f.write(json.dumps(rec, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)) + "\n")
