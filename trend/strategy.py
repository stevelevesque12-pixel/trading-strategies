"""Strategy spec -> signals -> backtest -> metrics, including a Lucid 50K evaluation simulator."""

import json
from dataclasses import asdict, dataclass, field
from typing import Dict, Optional

import numpy as np

from . import components as comp
from . import engine
from . import indicators as ind
from .data import Market

MES_POINT_VALUE = 5.0
MES_TICK = 0.25

# Cost model (per MES contract). Commission+exchange+NFA through a Tradovate/
# Rithmic prop account is roughly $1.0-1.4 per side; $2.50 round turn is
# deliberately on the high side. Plus 1 tick ($1.25) of slippage on every
# market-order fill (entries, stops, flattens); limit targets fill at price.
FEE_RT = 2.50
SLIP_TICKS = 1.0


@dataclass
class LucidRules:
    """Lucid Trading LucidFlex 50K evaluation (as published Oct 2026)."""
    profit_target: float = 3000.0
    max_loss: float = 2000.0         # EOD-trailing max loss limit
    lock_at_profit: float = 100.0    # MLL stops trailing once it reaches start + $100
    consistency: float = 0.50        # largest day <= 50% of total profit (eval only)
    min_days: int = 2
    max_micros: int = 40             # 4 minis / 40 micros
    max_days: int = 250              # LucidFlex has no time limit (one-time fee); ~1 year horizon, unresolved = timeout


@dataclass
class Spec:
    trend: str
    trend_p: Dict
    conf1: str
    conf1_p: Dict
    conf2: str
    conf2_p: Dict
    regime: str
    regime_p: Dict
    atr_n: int = 14
    sl_k: float = 2.0
    tp_k: float = 4.0
    trigger: str = "fresh"           # "fresh" = enter on the bar everything first lines up; "any" = whenever aligned & flat
    exit_on_flip: bool = True
    window: str = "rth"
    risk_usd: float = 200.0          # $ risked at full size (strong trend); sideways-ish trend uses small_mult
    small_mult: float = 0.5
    daily_loss_limit: float = 600.0
    daily_profit_cap: float = 1400.0
    trail_k: float = 0.0             # 0 = fixed bracket; >0 = chandelier trail at trail_k*ATR

    def key(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    def name(self) -> str:
        return f"{self.trend}+{self.conf1}+{self.conf2}|{self.regime}"


_cache: Dict = {}


def _component(m: Market, table, name, params):
    k = (m.name, len(m), int(m.index[0].value), table is comp.TREND, table is comp.CONFIRM, name, tuple(sorted(params.items())))
    if k not in _cache:
        if len(_cache) > 600:
            _cache.clear()
        _cache[k] = table[name][0](m, **params)
    return _cache[k]


def _atr(m: Market, n):
    k = (m.name, len(m), int(m.index[0].value), "atr", n)
    if k not in _cache:
        _cache[k] = ind.atr(m.h, m.l, m.c, n)
    return _cache[k]


def signals(m: Market, s: Spec):
    t = _component(m, comp.TREND, s.trend, s.trend_p)
    a = _component(m, comp.CONFIRM, s.conf1, s.conf1_p)
    b = _component(m, comp.CONFIRM, s.conf2, s.conf2_p)
    r = _component(m, comp.REGIME, s.regime, s.regime_p)
    on = r > 0
    long_ok = (t == 1) & (a == 1) & (b == 1) & on
    short_ok = (t == -1) & (a == -1) & (b == -1) & on
    if s.trigger == "fresh":
        long_ok = long_ok & ~np.r_[False, long_ok[:-1]]
        short_ok = short_ok & ~np.r_[False, short_ok[:-1]]
    return t, r, long_ok, short_ok


def backtest(m: Market, s: Spec, rules: Optional[LucidRules] = None, fee_rt=FEE_RT, slip_ticks=SLIP_TICKS):
    rules = rules or LucidRules()
    t, r, lg, sh = signals(m, s)
    trades = engine.run(
        m.o, m.h, m.l, m.c, _atr(m, s.atr_n), lg, sh, t.astype(np.float64), r.astype(np.float64),
        m.entry_mask(s.window), m.flatten_mask(), m.day_id,
        float(s.sl_k), float(s.tp_k), float(s.risk_usd), float(s.small_mult),
        MES_POINT_VALUE, MES_TICK, float(fee_rt), float(slip_ticks), int(rules.max_micros),
        float(s.daily_loss_limit), float(s.daily_profit_cap), bool(s.exit_on_flip), float(s.trail_k),
    )
    return trades


# ------------------------------------------------------------------ metrics
def daily_series(m: Market, trades: np.ndarray):
    """Per trading day (every session in the data, traded or not): pnl, intraday worst equity vs day start."""
    days = np.unique(m.day_id)
    pos = {d: i for i, d in enumerate(days)}
    pnl = np.zeros(len(days))
    low = np.zeros(len(days))
    ntr = np.zeros(len(days), int)
    for tr in trades:
        d = pos[m.day_id[int(tr[engine.T_EXIT_I])]]
        low[d] = min(low[d], pnl[d] + tr[engine.T_MAE])
        pnl[d] += tr[engine.T_PNL]
        ntr[d] += 1
    return days, pnl, low, ntr


def lucid_sim(pnl, low, ntr, rules: LucidRules, step: int = 5):
    """Start a fresh eval every `step` sessions; report pass / bust / timeout rates."""
    res = {"pass": 0, "bust": 0, "timeout": 0}
    days_to_pass = []
    starts = range(0, max(0, len(pnl) - 40), step)
    for s0 in starts:
        cum = peak = 0.0
        mll = -rules.max_loss
        best_day = 0.0
        traded = 0
        outcome = "timeout"
        for k in range(s0, min(len(pnl), s0 + rules.max_days)):
            if cum + low[k] <= mll:
                outcome = "bust"
                break
            cum += pnl[k]
            traded += ntr[k] > 0
            best_day = max(best_day, pnl[k])
            if cum <= mll:
                outcome = "bust"
                break
            if cum > peak:
                peak = cum
                mll = min(peak - rules.max_loss, rules.lock_at_profit)
            if cum >= rules.profit_target and best_day <= rules.consistency * cum and traded >= rules.min_days:
                outcome = "pass"
                days_to_pass.append(k - s0 + 1)
                break
        res[outcome] += 1
    n = max(1, sum(res.values()))
    return {
        "evals": n,
        "pass_rate": res["pass"] / n,
        "bust_rate": res["bust"] / n,
        "timeout_rate": res["timeout"] / n,
        "median_days_to_pass": float(np.median(days_to_pass)) if days_to_pass else None,
    }


def metrics(m: Market, trades: np.ndarray, rules: Optional[LucidRules] = None, curve_points: int = 300):
    rules = rules or LucidRules()
    n = len(trades)
    out = {"trades": n}
    days, dpnl, dlow, dntr = daily_series(m, trades)
    years = max(1e-9, (m.index[-1] - m.index[0]).days / 365.25)
    out["years"] = round(years, 2)
    if n == 0:
        out.update(win_rate=0, profit_factor=0, net=0, max_dd=0, avg_trade=0, sharpe=0, trades_per_week=0,
                   equity=[], lucid=lucid_sim(dpnl, dlow, dntr, rules))
        return out
    p = trades[:, engine.T_PNL]
    wins = p[p > 0].sum()
    loss = -p[p <= 0].sum()
    eq = np.cumsum(p)
    dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    sd = dpnl.std()
    out.update(
        win_rate=float((p > 0).mean()),
        profit_factor=float(wins / loss) if loss > 0 else 99.0,
        net=float(p.sum()),
        max_dd=float(dd),
        avg_trade=float(p.mean()),
        avg_qty=float(trades[:, engine.T_QTY].mean()),
        avg_r=float((p / np.maximum(trades[:, engine.T_RISK], 1)).mean()),
        sharpe=float(dpnl.mean() / sd * np.sqrt(252)) if sd > 0 else 0.0,
        trades_per_week=float(n / (years * 52)),
        pct_full_size=float((trades[:, engine.T_REGIME] >= 2).mean()),
        exit_mix={engine.REASONS[k]: int((trades[:, engine.T_REASON] == k).sum()) for k in engine.REASONS},
        pnl_full_size=float(p[trades[:, engine.T_REGIME] >= 2].sum()),
        pnl_small_size=float(p[trades[:, engine.T_REGIME] < 2].sum()),
        lucid=lucid_sim(dpnl, dlow, dntr, rules),
    )
    # equity curve, down-sampled, as [unix_ms, equity]
    t_ms = m.index[trades[:, engine.T_EXIT_I].astype(int)].as_unit("ms").asi8
    idx = np.unique(np.linspace(0, n - 1, min(n, curve_points)).astype(int))
    out["equity"] = [[int(t_ms[i]), round(float(eq[i]), 2)] for i in idx]
    return out


def evaluate(m: Market, s: Spec, rules: Optional[LucidRules] = None, **kw):
    return metrics(m, backtest(m, s, rules), rules, **kw)
