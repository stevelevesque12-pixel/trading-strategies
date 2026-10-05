"""
Cross-market robustness check: run the FROZEN MCL Trend Dip settings (no
re-fitting at all) on other futures. If the dip-in-trend edge is a real
market behavior rather than an MCL curve-fit, it should be at least mildly
positive elsewhere. Sized like the real strategy: equal risk per trade (huge
notional so contract rounding vanishes), daily loss stop 2.25R, daily profit
lock 4R, and max stop = 2.1% of price (MCL's 1.5 points at ~$70).
Metrics are therefore in R terms; "net" is in units of $1,000 risk.

  python -m trend_lab.cross_market
"""

import json
from datetime import datetime, timezone

from .data import load_bars
from .metrics import compute, equity_points
from .optimize import load_registry, save_registry
from .portfolio import merge
from .sim import SimConfig, simulate
from .strategies import ALL_FAMILIES

# frozen walk-forward picks (identical to tradingview/mcl_trend_dip.pine)
ENGINE_A = ("trend_dip_atr", {"session": "us", "trail_k": 2.5, "target_r": 2.0, "be_r": 1.0, "htf_rule": "60min",
                              "htf_len": 50, "slope_bars": 12, "base_ema": 100, "fast": 13, "dip_k": 1.0,
                              "dip_bars": 6, "swing_lb": 8})
ENGINE_B = ("trend_dip_rsi", {"session": "ny", "trail_k": 3.0, "target_r": 3.0, "be_r": None, "htf_rule": "240min",
                              "htf_len": 50, "base_ema": 100, "rsi_n": 3, "rsi_lo": 30, "swing_lb": 5,
                              "exit_rsi": 70})

# symbol -> (point value $, tick, commission RT per contract)
SPECS = {
    "mcl": (100.0, 0.01, 1.24), "mgc": (10.0, 0.1, 1.24), "sil": (1000.0, 0.005, 1.24),
    "mes": (5.0, 0.25, 1.24), "mnq": (2.0, 0.25, 1.24), "mym": (0.5, 1.0, 1.24), "m2k": (5.0, 0.1, 1.24),
    "gc": (100.0, 0.1, 4.5), "si": (5000.0, 0.005, 4.5), "es": (50.0, 0.25, 4.5), "nq": (20.0, 0.25, 4.5),
}


def run_symbol(sym):
    df = load_bars("15min", symbol=sym)
    pv, tick, comm = SPECS[sym]
    days = sorted(set(df["trade_day"]))
    risk = 1000.0
    max_stop = 0.021 * float(df["close"].median())
    lists = []
    for name, p in (ENGINE_A, ENGINE_B):
        cfg = SimConfig(session=p["session"], risk_usd=risk, max_contracts=10**6, point_value=pv, tick=tick,
                        commission_rt=comm, max_stop_dist=max_stop, daily_loss_stop=2.25 * risk,
                        daily_profit_lock=4 * risk)
        # $1,000 risk with an uncapped contract count keeps rounding error small (approximately equal-R trades)
        lists.append(simulate(df, ALL_FAMILIES[name].generate(df, p), cfg))
    trades = merge(lists, daily_loss_stop=2.25 * risk, exclusive=True)
    m = compute(trades, days)
    return m, trades, days


def main():
    out = {}
    for sym in SPECS:
        m, trades, days = run_symbol(sym)
        out[sym] = {"metrics": {k: m.get(k) for k in ("trades", "win_rate", "profit_factor", "net", "max_dd",
                                                        "avg_r", "trades_per_week")},
                    "start": str(days[0]), "end": str(days[-1]), "equity": equity_points(trades)}
        print(f"{sym:>4} {days[0]}..{days[-1]}  {json.dumps(out[sym]['metrics'])}", flush=True)
    reg = load_registry()
    reg["cross_market"] = {"created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                           "engines": [ENGINE_A, ENGINE_B], "results": out}
    save_registry(reg)


if __name__ == "__main__":
    main()
