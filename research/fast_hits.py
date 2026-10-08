"""List strategies that meet the fast-pass bar AND survive the 10-year 15m sanity check.

    python -m research.fast_hits

Bar: Lucid pass within 42 sessions >= 30% with bust <= 10% on out-of-sample data
(15m_fast: ES and NQ 2023-26; 5m_scalp: MES Aug 15-Oct 7 2026 and robust), then for 5m hits the same rules
on 15m ES/NQ 2016-2026 (bar lengths / 3) must have PF > 1.05 in all four (market x period) slices.
"""

import json
from dataclasses import replace

from research.search import RESULTS
from trend.data import load
from trend.strategy import LucidRules, Spec, backtest, metrics

EXCLUDE = {"donchian+prev_close+vwap|atr_rank", "donchian+heikin+overnight|or_width",
           "linreg_slope+overnight+rsi|or_width",
           "hma_slope+heikin+overnight|or_width"}
LEN_KEYS = {"n", "fast", "slow", "mid", "look", "sig"}


def scaled(spec, f=3):
    def sc(p):
        return {k: (max(2, int(v) // f) if k in LEN_KEYS and isinstance(v, (int, float)) and v >= 6 else v)
                for k, v in p.items()}
    return replace(spec, trend_p=sc(spec.trend_p), conf1_p=sc(spec.conf1_p), conf2_p=sc(spec.conf2_p),
                   regime_p=sc(spec.regime_p), atr_n=max(5, spec.atr_n // f), pb_n=max(3, spec.pb_n // f),
                   daily_loss_limit=1e9, daily_profit_cap=1e9, cushion_sizing=False, risk_usd=200)


def long_history_ok(spec, rules):
    out = []
    for ds, pv in (("es_15m", 5.0), ("nq_15m", 2.0)):
        m = load(ds)
        for a, b in ((None, "2023-01-01"), ("2023-01-01", None)):
            mm = m.slice(start=a, end=b)
            out.append(metrics(mm, backtest(mm, spec, rules, point_value=pv), rules, curve_points=0)["profit_factor"])
    return all(x > 1.05 for x in out), out


def main():
    rules = LucidRules()
    seen, hits = set(), []
    for line in RESULTS.read_text().splitlines():
        r = json.loads(line)
        f = r.get("fast")
        if not f or r["name"] in EXCLUDE:
            continue
        ok = False
        if r["track"] == "15m_fast":
            a, b = f["es_oos"], f["nq_oos"]
            ok = a["pass_rate"] >= .3 and a["bust_rate"] <= .1 and b["pass_rate"] >= .3 and b["bust_rate"] <= .1
        elif r["track"] == "5m_scalp":
            a = f["es_oos"]
            ok = r["robust"] and a["pass_rate"] >= .3 and a["bust_rate"] <= .1
        if not ok or r["id"] in seen:
            continue
        seen.add(r["id"])
        s = Spec(**r["spec"])
        lh, pfs = long_history_ok(scaled(s) if r["track"] == "5m_scalp" else s, rules)
        print(f"{r['track']:<9} {r['name']:<44} long-history PFs {[round(x, 2) for x in pfs]} -> {'PASS' if lh else 'fail'}")
        if lh:
            hits.append(r)
    print(f"\n{len(hits)} strategies pass everything")


if __name__ == "__main__":
    main()
