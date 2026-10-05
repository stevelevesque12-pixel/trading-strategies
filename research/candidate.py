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


# Opening-range trend: consensus of 12 joint ES+NQ runs (tradingview/orb_trend.pine)
ORB = Spec(
    trend="orb", trend_p={"minutes": 60},
    conf1="daily_ema", conf1_p={"n": 50},
    conf2="fast_ema", conf2_p={"n": 20},
    regime="er", regime_p={"n": 30, "lo": 0.3, "hi": 0.4},
    atr_n=30, sl_k=2.5, tp_k=8.0, trail_k=0.0, trigger="fresh", exit_on_flip=True, window="ny_am",
    risk_usd=500, small_mult=0.66, daily_loss_limit=300, daily_profit_cap=900, cushion_sizing=True,
)

CANDIDATES = {"live": (LIVE, "es_15m", "★ LIVE "), "orb": (ORB, "es_15m", "★ ORB "), "orb_mnq": (ORB, "nq_15m", "★ ORB (MNQ) ")}


def main(which=("orb", "orb_mnq")):
    for w in which:
        log(*CANDIDATES[w])


def log(spec, dataset, prefix):
    LIVE_ = spec
    full = load(dataset)
    rules = LucidRules()
    split = "2023-01-01"
    rec = {
        "id": f"candidate:{prefix.strip()}:{LIVE_.name()}:{time.strftime('%Y%m%d%H%M')}",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"), "v": 0, "candidate": True,
        "track": "15m_full" if dataset == "es_15m" else "15m_joint", "dataset": dataset, "split": split, "tf_min": 15,
        "name": prefix + LIVE_.name(), "spec": asdict(LIVE_), "configs_tried": 0, "seconds": 0,
        "is": evaluate(full.slice(end=split), LIVE_, rules, curve_points=0),
        "oos": evaluate(full.slice(start=split), LIVE_, rules, curve_points=0),
        "full": evaluate(full, LIVE_, rules, curve_points=100),
        "mes_check": evaluate(load("mes_15m"), LIVE_, rules, curve_points=0),
    }
    if dataset == "es_15m":
        nq = load("nq_15m")
        rec["nq_is"] = evaluate(nq.slice(end=split), LIVE_, rules, curve_points=0)
        rec["nq_oos"] = evaluate(nq.slice(start=split), LIVE_, rules, curve_points=0)

    rec["robust"] = rec["is"]["profit_factor"] > 1.05 and rec["oos"]["profit_factor"] > 1.05
    append(rec)
    v = validate.save(rec)
    print(rec["id"], "OOS PF", round(rec["oos"]["profit_factor"], 2), "MC", v["mc_lucid_oos"])


if __name__ == "__main__":
    import sys
    main(tuple(sys.argv[1:]) or ("orb", "orb_mnq"))
