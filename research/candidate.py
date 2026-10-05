"""Log the current live candidate (the exact variant in tradingview/trend_atr_mes.pine) to the dashboard.

    python -m research.candidate
"""

import time
from dataclasses import asdict

from research import validate
from research.search import append
from trend.data import load
from trend.strategy import LucidRules, Spec, evaluate

LIVE = Spec(
    trend="ema_slope", trend_p={"n": 300},
    conf1="dmi", conf1_p={"n": 20},
    conf2="roc", conf2_p={"n": 40},
    regime="er", regime_p={"n": 30, "lo": 0.25, "hi": 0.4},
    atr_n=20, sl_k=3.0, tp_k=8.0, trail_k=3.0, trigger="fresh", exit_on_flip=True, window="ny_am",
    risk_usd=800, small_mult=0.33, daily_loss_limit=600, daily_profit_cap=900, cushion_sizing=True,
)


def main():
    full = load("es_15m")
    rules = LucidRules()
    split = "2023-01-01"
    rec = {
        "id": f"candidate:{LIVE.name()}:{time.strftime('%Y%m%d%H%M')}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "v": 0, "candidate": True,
        "track": "15m_full", "dataset": "es_15m", "split": split, "tf_min": 15,
        "name": "★ LIVE " + LIVE.name(), "spec": asdict(LIVE), "configs_tried": 0, "seconds": 0,
        "is": evaluate(full.slice(end=split), LIVE, rules, curve_points=0),
        "oos": evaluate(full.slice(start=split), LIVE, rules, curve_points=0),
        "full": evaluate(full, LIVE, rules, curve_points=100),
        "mes_check": evaluate(load("mes_15m"), LIVE, rules, curve_points=0),
    }
    rec["robust"] = rec["is"]["profit_factor"] > 1.05 and rec["oos"]["profit_factor"] > 1.05
    append(rec)
    v = validate.save(rec)
    print(rec["id"], "OOS PF", round(rec["oos"]["profit_factor"], 2), "MC", v["mc_lucid_oos"])


if __name__ == "__main__":
    main()
