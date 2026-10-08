"""Failed 2 + 8/21 ribbon on 3 years of 1m NQ (Dec 2022 - Dec 2025). 1 NQ, $4 RT, 1 tick slippage/side."""
import sys
import pandas as pd
import bt_failed2 as b
from bt_failed2 import NY, resample, run, stats

b.PV, b.COMM = 20.0, 2.0  # NQ: $20/pt, $4 round trip
m1 = b.load("real_nq_1m_2022-12-26_2025-12-11.parquet")
pd.set_option("display.width", 220)
PERIODS = [("2023", "2023-01-01", "2024-01-01"), ("2024", "2024-01-01", "2025-01-01"), ("2025", "2025-01-01", "2026-01-01")]


def by_year(t):
    r = {}
    for lab, a, z in PERIODS:
        s = stats(t[(t.time >= pd.Timestamp(a, tz=NY)) & (t.time < pd.Timestamp(z, tz=NY))])
        r[f"pf{lab}"] = s.get("pf")
    return r


def cell(t):
    s = stats(t)
    return f'{s.get("pf")} ({s["trades"]}, ${s.get("net", 0):,})'


if __name__ == "__main__":
    print("== Timeframes, 0.5R and magnitude ==")
    rows = []
    for n in (1, 2, 3, 4, 5, 6, 8, 10, 15):
        d = resample(m1, n)
        for tm in (0.5, "mag"):
            t = run(d, tmode=tm)
            rows.append(dict(tf=n, tgt=tm, **stats(t), **by_year(t)))
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n== Bar-alignment offsets, 0.5R: PF (trades, net) ==")
    rows = []
    for n in (2, 3, 4, 5):
        rows.append({"tf": n, **{f"off{o}": cell(run(resample(m1, n, o), tmode=0.5)) for o in range(n)}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n== Fast EMA, 0.5R ==")
    rows = []
    for n in (2, 4, 5):
        rows.append({"tf": n, **{f"ema{f}": cell(run(resample(m1, n), tmode=0.5, fast=f)) for f in (6, 7, 8, 9, 10)}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n== Target sweep ==")
    rows = []
    for n in (2, 4, 5):
        rows.append({"tf": n, **{f"{tm}R": cell(run(resample(m1, n), tmode=tm)) for tm in (0.3, 0.5, 0.75, 1.0, 1.5, 2.0)}})
    print(pd.DataFrame(rows).to_string(index=False))

    print("\n== 4m 0.5R by half-year and direction ==")
    t = run(resample(m1, 4), tmode=0.5)
    t["h"] = t.time.dt.year.astype(str) + "H" + ((t.time.dt.month > 6) + 1).astype(str)
    print(pd.DataFrame([dict(h=h, **stats(g)) for h, g in t.groupby("h")]).to_string(index=False))
    print(pd.DataFrame([dict(dir=k, **stats(g)) for k, g in t.groupby("dir")]).to_string(index=False))
