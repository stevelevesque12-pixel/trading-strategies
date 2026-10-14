"""Backtest the Carver fast trend-following strategy on every futures contract
in sample_data/ and write carver_trend/RESULTS.md (+ equity chart).

    python -m carver_trend.run_all
"""

from dataclasses import replace
from pathlib import Path

import pandas as pd

from .backtest import SPECS, run, summarize
from .strategy import Params

OUT_DIR = Path(__file__).resolve().parent
LONG_HISTORY = ["ES", "NQ", "GC", "SI"]  # 15m, 2016-05 -> 2026-08
MICROS = ["MES", "MNQ", "MYM", "M2K", "MGC", "SIL", "MCL"]  # 15m, 2025-09 -> 2026-08
QC_START = "2021-08-25"  # QC: set_start_date(end_date - 5 years)
MICRO_WARMUP_DAYS = 120  # micros only have ~11 months, so the 3y QC warm-up is impossible

COLS = ["start", "end", "years", "total_return_pct", "cagr_pct", "ann_vol_pct", "sharpe",
        "max_dd_pct", "calmar", "costs_usd", "contracts_per_year", "pct_days_long"]


def table(rows: dict) -> str:
    df = pd.DataFrame(rows).T[COLS]
    return df.to_markdown()


def main():
    p = Params()
    sections = []
    curves = {}

    full = {}
    for s in LONG_HISTORY:
        full[s] = summarize(run([s], p))
    res = run(LONG_HISTORY, p)
    full["PORTFOLIO (ES+NQ+GC+SI)"] = summarize(res)
    curves["Portfolio ES+NQ+GC+SI (full)"] = res.equity
    contrib = res.pnl_by_instrument.sum().round(-3)
    sections.append("## 1. Full history (2016 data, trading after the QC 3-year warm-up)\n\n"
                    "Each instrument alone (IDM 1, weight 1) and as the 4-instrument portfolio "
                    "(IDM 1.56, weight 0.25).\n\n" + table(full)
                    + "\n\nPortfolio gross P&L by instrument (USD): "
                    + ", ".join(f"{k} {v:+,.0f}" for k, v in contrib.items()))

    qc = {}
    for s in LONG_HISTORY:
        qc[s] = summarize(run([s], p, start=QC_START))
    res_qc = run(LONG_HISTORY, p, start=QC_START)
    qc["PORTFOLIO (ES+NQ+GC+SI)"] = summarize(res_qc)
    curves["Portfolio ES+NQ+GC+SI (QC 5y window)"] = res_qc.equity
    sections.append(f"## 2. QC's window: last 5 years ({QC_START} -> 2026-08-25)\n\n" + table(qc))

    micro = {}
    for s in MICROS:
        micro[s] = summarize(run([s], p, warmup_days=MICRO_WARMUP_DAYS))
    res_m = run(MICROS, p, warmup_days=MICRO_WARMUP_DAYS)
    micro["PORTFOLIO (7 micros)"] = summarize(res_m)
    curves["Portfolio 7 micros (short window)"] = res_m.equity
    sections.append("## 3. Micro futures: adds Dow (MYM), Russell (M2K) and crude oil (MCL)\n\n"
                    f"Only about 11 months of data, so the warm-up is {MICRO_WARMUP_DAYS} calendar days instead of "
                    "3 years, and the vol blend uses the history available so far. "
                    "Treat these as **indicative only**: a few months of trend-following P&L is noise.\n\n"
                    + table(micro))

    variants = {}
    for exp in [4, 5, 6, 7]:
        pv = replace(p, slow_ema_span=2 ** exp)
        variants[f"EWMAC {2 ** exp // 4}/{2 ** exp}"] = summarize(run(LONG_HISTORY, pv))
    sections.append("## 4. Sensitivity: `slow_ema_span_exponent` 4-7 (ES+NQ+GC+SI portfolio, full history)\n\n"
                    + table(variants))

    chart = OUT_DIR / "equity_curves.png"
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(10, 5))
        for name, eq in curves.items():
            ax.plot(eq.index, eq / 1e6, label=name)
        ax.set_ylabel("Equity ($M, start $1M)")
        ax.set_title(f"Carver EWMAC {p.fast_ema_span}/{p.slow_ema_span} trend strength, buffered")
        ax.grid(alpha=0.3)
        ax.legend()
        fig.tight_layout()
        fig.savefig(chart, dpi=110)
    except ImportError:
        chart = None

    specs = pd.DataFrame([{"symbol": s.symbol, "class": s.asset_class, "multiplier": s.multiplier,
                           "tick": s.tick_size, "cost/side $": round(s.cost_per_side, 2),
                           "rolls/yr": s.rolls_per_year} for s in SPECS.values()]).set_index("symbol")

    header = f"""# Carver fast trend following (long/short, trend strength): local backtest

This is a port of the QuantConnect algorithm `FuturesFastTrendFollowingLongAndShortWithTrendStrenthAlgorithm`
with `slow_ema_span_exponent = 7` (QC default is 6) and otherwise default parameters: EWMAC {p.fast_ema_span}/{p.slow_ema_span}, forecast scalar {p.forecast_scalar},
cap +/-{p.abs_forecast_cap:g}, sigma span {p.sigma_span}, 70/30 blend with the {p.blend_years}-year average vol,
tau {p.target_risk}, buffer {p.buffer_scaler}, IDM from Carver Table 16, $1M starting capital (compounding).
It runs on daily session bars built from the repo's 15-minute data. A signal at day t's close fills at
day t+1's open. Costs are commission plus 1 tick of slippage per side, plus an estimated roll cost.

Regenerate with `python -m carver_trend.run_all`.

**Coverage.** "Every major future" isn't possible here because the environment has no
QuantConnect/LEAN and no outside data access. The tests use every futures series in `sample_data/`:
ES, NQ, GC and SI with 10 years of data, and MES, MNQ, MYM, M2K, MGC, SIL and MCL with about 11 months.
Rates (ZN/ZB), FX (6E/6J), grains, softs and full-size energies aren't in the repo. Running the
original algorithm on QuantConnect is still the way to get the full universe.

**Data caveat.** The continuous series look unadjusted (not Panama back-adjusted the way QC is), so
roll gaps show up as small fake returns.

![equity]({chart.name if chart else ''})

### Contract specs used

{specs.to_markdown()}
"""
    (OUT_DIR / "RESULTS.md").write_text(header + "\n\n" + "\n\n".join(sections) + "\n")
    print((OUT_DIR / "RESULTS.md").read_text())


if __name__ == "__main__":
    main()
