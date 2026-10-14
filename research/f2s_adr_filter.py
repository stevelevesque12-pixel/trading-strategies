"""
Out-of-sample test of the "skip Failed-2s signals once the day has used >= 80%
of its average daily range" filter found in research/f2s_vwap_extension.py
(NQ 1m, Dec 2022 - Dec 2025, 1m-15m pair).

adr_used = today's RTH range so far (at the entry bar's close) / mean RTH range
of the prior 14 sessions. Daily ranges come from the longest file available
for each symbol (15m files for the micros), so the 3-week 1m files don't lose
most of their days to warm-up. Each signal is re-simulated on the base bars
with the strategy's own stop and 1R target (stop first inside a bar, flat at
15:55 ET), minus 2 ticks of cost per round trip.

Usage:  python -m research.f2s_adr_filter
"""

import numpy as np
import pandas as pd

from backtest.data import load_1m_csv
from backtest.engine import BacktestEngine
from failed2s.instruments import Instrument
from failed2s.risk import RiskManager
from failed2s.strategy import PAIRS, Failed2sStrategy

from .f2s_vwap_extension import simulate

D = "sample_data/real_multi_instrument/"
TICK = {"NQ": 0.25, "ES": 0.25, "GC": 0.1, "SI": 0.005, "MNQ": 0.25, "MES": 0.25, "MYM": 1.0,
        "M2K": 0.1, "MGC": 0.1, "MCL": 0.01, "SIL": 0.005}
MICROS = ["MNQ", "MES", "MYM", "M2K", "MGC", "MCL", "SIL"]
CUT = 0.8

# (label, symbol, pair, base file, daily-range file, in_sample_window or None)
RUNS = (
    [("NQ 1m 2022-25 (in-sample)", "NQ", "1m-15m", "real_nq_1m_2022-12-26_2025-12-11.parquet",
      "real_nq_1m_2022-12-26_2025-12-11.parquet")]
    + [(f"{s} 1m Aug-2026", s, "1m-15m", f"real_{s.lower()}_1m_2026-08-02_2026-08-25.parquet",
        f"real_{s.lower()}_15m_2025-09-30_2026-08-25.parquet") for s in MICROS]
    + [(f"{s} 5m May-Aug-2026", s, "5m-1h", f"real_{s.lower()}_5m_2026-05-10_2026-08-25.parquet",
        f"real_{s.lower()}_15m_2025-09-30_2026-08-25.parquet") for s in MICROS]
    + [(f"{s} 15m 2016-26", s, "15m-4h", f"real_{s.lower()}_15m_{start}_2026-08-25.parquet",
        f"real_{s.lower()}_15m_{start}_2026-08-25.parquet")
       for s, start in [("NQ", "2016-05-29"), ("ES", "2016-05-29"), ("GC", "2016-05-26"), ("SI", "2016-05-26")]]
)


def rth(df: pd.DataFrame) -> pd.DataFrame:
    return df.between_time("09:30", "15:59")


def run(label, sym, pair_name, base_file, daily_file) -> pd.DataFrame:
    tick = TICK[sym]
    base = rth(load_1m_csv(D + base_file))
    res = base.index.to_series().diff().mode().iloc[0]
    day = base.index.date
    rng_so_far = base.high.groupby(day).cummax() - base.low.groupby(day).cummin()

    dly = rth(load_1m_csv(D + daily_file))
    dr = dly.groupby(dly.index.date).agg(h=("high", "max"), l=("low", "min"))
    adr = (dr.h - dr.l).rolling(14).mean().shift()

    eng = BacktestEngine(
        pair=PAIRS[pair_name], instrument=Instrument(sym, tick, 1.0),
        strategy=Failed2sStrategy(tick_size=tick),
        risk=RiskManager(daily_loss_limit=float("inf"), max_daily_trades=10_000),
    )
    trades = eng.run(D + base_file)
    entry_len = pd.Timedelta(PAIRS[pair_name].entry_tf)
    rows = []
    for t in trades:
        ts = t.entry_time + entry_len - res
        if ts not in base.index or not np.isfinite(adr.get(ts.date(), np.nan)):
            continue
        risk = abs(t.entry_price - t.stop_price)
        pnl = simulate(base, ts, t.direction, t.entry_price, t.stop_price, t.target_price)
        rows.append(dict(run=label, sym=sym, pair=pair_name, entry_time=ts, hour=ts.hour,
                         adr_used=rng_so_far.loc[ts] / adr[ts.date()],
                         r=(pnl - 2 * tick) / risk))
    return pd.DataFrame(rows)


def line(name, d):
    lo, hi = d[d.adr_used < CUT].r, d[d.adr_used >= CUT].r
    se = lambda s: s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
    diff = lo.mean() - hi.mean()
    dse = np.sqrt(se(lo) ** 2 + se(hi) ** 2)
    print(f"{name:30s} all n={len(d):5d} {d.r.mean():+.3f} | <{CUT}: n={len(lo):5d} {lo.mean():+.3f} | "
          f">={CUT}: n={len(hi):5d} {hi.mean():+.3f} | diff {diff:+.3f} ±{dse:.3f}  (t={diff / dse:+.1f})")


def main() -> None:
    allt = pd.concat([run(*r) for r in RUNS], ignore_index=True)
    allt.to_csv("research/f2s_adr_filter_trades.csv", index=False)

    print("=== per run ===")
    for name, d in allt.groupby("run", sort=False):
        line(name, d)

    ins = allt[allt.run.str.contains("in-sample")]
    nq15 = allt[allt.run == "NQ 15m 2016-26"]
    overlap = (nq15.entry_time >= pd.Timestamp("2022-12-26", tz="America/New_York")) & (
        nq15.entry_time < pd.Timestamp("2025-12-12", tz="America/New_York"))
    oos = pd.concat([allt[~allt.run.str.contains("in-sample") & (allt.run != "NQ 15m 2016-26")], nq15[~overlap]])

    print("\n=== pooled out-of-sample, by pair ===")
    for p, d in oos.groupby("pair", sort=False):
        line(f"OOS {p}", d)
    line("OOS all pairs", oos)

    print("\n=== in-sample cutoff sweep (NQ 1m-15m): avg R above/below cutoff ===")
    for c in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2]:
        lo, hi = ins[ins.adr_used < c].r, ins[ins.adr_used >= c].r
        print(f"  cut {c:.1f}: below n={len(lo):4d} {lo.mean():+.3f} | above n={len(hi):4d} {hi.mean():+.3f}")

    print("\n=== time-of-day control: filter effect within each entry hour ===")
    for name, d in [("in-sample", ins), ("OOS 1m+5m", oos[oos.pair != "15m-4h"])]:
        print(f"  {name}")
        for h, g in d.groupby("hour"):
            if len(g) >= 40:
                lo, hi = g[g.adr_used < CUT].r, g[g.adr_used >= CUT].r
                print(f"    {h:02d}h: hour avg {g.r.mean():+.3f} | <{CUT} n={len(lo):4d} {lo.mean():+.3f} | "
                      f">={CUT} n={len(hi):4d} {hi.mean():+.3f}")


if __name__ == "__main__":
    main()
