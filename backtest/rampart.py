"""Backtest of the PQX_Rampart_5min_MNQ NinjaTrader 8 strategy.

A bar-by-bar port of the NT8 OnBarUpdate logic with its default inputs:

- Range = high/low of every 5m bar overlapping 09:15-09:50 ET.
- Entry window 09:50-12:00 ET (bar OPEN time). The first bar whose close is
  >= range high + 6 ticks (<= range low - 6 ticks) is the signal; it must be a
  rising edge of that condition, and the "previous bar" flag carries across days.
- Market entry at the NEXT bar's open (NT OnBarClose semantics). Stop and
  target are priced off the SIGNAL bar: stop = signal low - 1 tick (long),
  target = signal close + 0.3 * (signal close - stop).
- One long and one short per day, flat only, >= 1 bar since the last entry.
- Anything open on the bar opening at 16:00 ET is closed at the next bar's open.
- Size = floor($500 / (risk points * point value)), min 1, max 1000.

Exits are resolved on 1-minute bars when they exist (stop assumed first if a
single 1m bar touches both), else on the 5m bar itself with the `tie` rule.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parent.parent / "sample_data" / "real_multi_instrument"
TICK = 0.25


@dataclass
class Params:
    range_start: int = 9 * 60 + 15
    range_end: int = 9 * 60 + 50
    break_ticks: int = 6
    stop_buf_ticks: int = 1
    rr: float = 0.3
    last_entry: int = 12 * 60
    force_close: int = 16 * 60
    one_per_side: bool = True
    min_bars_between: int = 1
    risk_usd: float = 500.0
    max_contracts: int = 1000
    point_value: float = 2.0  # MNQ


def load(path):
    df = pd.read_parquet(path).tz_convert("America/New_York")
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close"]].sort_index()


def to_5m(df1):
    return df1.resample("5min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()


def rnd(px):
    return round(px / TICK) * TICK


_1M_CACHE = {}


def _index_1m(bars5, bars1):
    """For each 5m bar, the [start, end) slice of the 1m bars inside it. Cached: sweeps reuse it."""
    key = (id(bars5), id(bars1))
    if key not in _1M_CACHE:
        idx = bars5.index
        b1 = bars1.loc[(bars1.index >= idx[0]) & (bars1.index < idx[-1] + pd.Timedelta("5min"))]
        key1 = b1.index.floor("5min").to_numpy()
        _1M_CACHE[key] = (np.searchsorted(key1, idx.to_numpy(), side="left"),
                          np.searchsorted(key1, idx.to_numpy(), side="right"),
                          b1["high"].to_numpy(), b1["low"].to_numpy())
    return _1M_CACHE[key]


def run(bars5, bars1=None, p=Params(), tie="stop"):
    """Walk the 5m bars. bars1 (optional) resolves intrabar stop/target order."""
    idx = bars5.index
    o, h, l, c = (bars5[k].to_numpy() for k in ("open", "high", "low", "close"))
    tod = idx.hour * 60 + idx.minute
    day = idx.year * 10000 + idx.month * 100 + idx.day
    dow = idx.dayofweek

    if bars1 is not None:
        pos1, end1, h1, l1 = _index_1m(bars5, bars1)
        has1 = end1 > pos1
    trades = []
    cur_day = None
    rng_hi = rng_lo = 0.0
    seen = False
    longs = shorts = 0
    last_trade_bar = None
    prev_ql = prev_qs = False
    pos = None          # open position dict
    pending = None      # order placed on bar close, fills next bar open
    pending_flat = False

    def close_pos(i, px, reason):
        nonlocal pos
        sgn = 1 if pos["dir"] == "long" else -1
        pts = (px - pos["fill"]) * sgn
        trades.append(dict(entry_time=pos["time"], exit_time=idx[i], dir=pos["dir"], signal_close=pos["sig"],
                           fill=pos["fill"], stop=pos["stop"], target=pos["target"], exit=px, reason=reason,
                           qty=pos["qty"], risk_pts=pos["risk"], pnl_pts=pts, pnl_usd=pts * pos["qty"] * p.point_value))
        pos = None

    def check_exit(hi, lo):
        """Return exit price/reason if stop/target touched within [lo, hi]."""
        if pos["dir"] == "long":
            hs, ht = lo <= pos["stop"], hi >= pos["target"]
        else:
            hs, ht = hi >= pos["stop"], lo <= pos["target"]
        return hs, ht

    for i in range(len(idx)):
        # 1) orders from the previous bar fill at this bar's open
        if pending_flat and pos is not None:
            close_pos(i, o[i], "flat")
        pending_flat = False
        if pending is not None:
            pos = dict(pending, fill=o[i], time=idx[i])
            pending = None
            # opened through the stop/target: stop-market / limit fills at the open
            if pos is not None:
                if (pos["dir"] == "long" and o[i] <= pos["stop"]) or (pos["dir"] == "short" and o[i] >= pos["stop"]):
                    close_pos(i, o[i], "stop_gap")
                elif (pos["dir"] == "long" and o[i] >= pos["target"]) or (pos["dir"] == "short" and o[i] <= pos["target"]):
                    close_pos(i, o[i], "target_gap")
        # 2) intrabar stop/target
        if pos is not None:
            if bars1 is not None and has1[i]:
                for j in range(pos1[i], end1[i]):
                    hs, ht = check_exit(h1[j], l1[j])
                    if hs:
                        close_pos(i, pos["stop"], "stop"); break
                    if ht:
                        close_pos(i, pos["target"], "target"); break
            else:
                hs, ht = check_exit(h[i], l[i])
                if hs and (not ht or tie == "stop"):
                    close_pos(i, pos["stop"], "stop")
                elif ht:
                    close_pos(i, pos["target"], "target")

        # 3) OnBarUpdate at this bar's close
        if day[i] != cur_day:
            if cur_day is not None and pos is not None:
                pending_flat = True
            cur_day = day[i]
            rng_hi = rng_lo = 0.0; seen = False; longs = shorts = 0
        t = tod[i]
        if t + 5 > p.range_start and t < p.range_end:
            rng_hi = max(rng_hi, h[i]) if seen else h[i]
            rng_lo = min(rng_lo, l[i]) if seen else l[i]
            seen = True
        ready = seen and rng_hi > rng_lo
        m = p.break_ticks * TICK
        ql = ready and c[i] >= rng_hi + m
        qs = ready and c[i] <= rng_lo - m
        fl, fs = ql and not prev_ql, qs and not prev_qs
        prev_ql, prev_qs = ql, qs

        if t >= p.force_close:
            if pos is not None:
                pending_flat = True
            continue
        last_entry = min(p.last_entry, p.force_close)
        bars_since = math.inf if last_trade_bar is None else i - last_trade_bar
        can = (dow[i] < 5 and p.range_end <= t < last_entry and ready and pos is None and pending is None
               and longs + shorts < (2 if p.one_per_side else 1) and bars_since >= p.min_bars_between)
        for is_long, fire in ((True, fl), (False, fs)):
            if not (can and fire and (not p.one_per_side or (longs if is_long else shorts) == 0)):
                continue
            entry = c[i]
            buf = max(p.stop_buf_ticks * TICK, TICK)
            stop = rnd(l[i] - buf if is_long else h[i] + buf)
            dist = abs(entry - stop)
            if dist < TICK * 0.5:
                break
            tp = dist * p.rr
            target = rnd(entry + tp if is_long else entry - tp)
            qty = max(1, min(max(int(math.floor(p.risk_usd / (dist * p.point_value))), 1), p.max_contracts))
            pending = dict(dir="long" if is_long else "short", sig=entry, stop=stop, target=target, qty=qty, risk=dist)
            if is_long: longs += 1
            else: shorts += 1
            last_trade_bar = i
            break
    return pd.DataFrame(trades)


def summary(t, rt_cost_per_contract=0.0):
    if t.empty:
        return {"trades": 0}
    pnl = t.pnl_usd - rt_cost_per_contract * t.qty
    eq = pnl.cumsum()
    daily = pnl.groupby(t.entry_time.dt.date).sum()
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    return {
        "trades": len(t),
        "win%": round((pnl > 0).mean() * 100, 1),
        "net$": round(pnl.sum()),
        "PF": round(gp / gl, 2) if gl else float("inf"),
        "avg$": round(pnl.mean(), 1),
        "avgW$": round(pnl[pnl > 0].mean(), 1),
        "avgL$": round(pnl[pnl <= 0].mean(), 1),
        "maxDD$": round((eq.cummax().clip(lower=0) - eq).max()),
        "avg_qty": round(t.qty.mean(), 1),
        "avg_risk_pts": round(t.risk_pts.mean(), 1),
        "sharpe_daily": round(daily.mean() / daily.std() * np.sqrt(252), 2) if daily.std() else None,
    }


def sweep_rr(grid, cost):
    """Net-of-cost results per R:R on every dataset, plus each year of NQ."""
    nq1 = load(DATA / "real_nq_1m_2022-12-26_2025-12-11.parquet")
    nq5 = to_5m(nq1)
    mnq5 = load(DATA / "real_mnq_5m_2026-05-10_2026-08-25.parquet")
    mnq1 = load(DATA / "real_mnq_1m_2026-08-02_2026-08-25.parquet")
    rows = []
    for rr in grid:
        p = Params(rr=rr)
        t = run(nq5, nq1, p)
        s = summary(t, cost)
        row = {"rr": rr, "trades": s["trades"], "win%": s["win%"], "PF": s["PF"], "net$": s["net$"],
               "maxDD$": s["maxDD$"], "sharpe": s["sharpe_daily"]}
        for y, g in t.groupby(t.entry_time.dt.year):
            if y >= 2023:
                row[f"{y}$"] = summary(g, cost)["net$"]
        row["flat%"] = round((t.reason == "flat").mean() * 100, 1)
        row["NQ stop1st$"] = summary(run(nq5, None, p, tie="stop"), cost)["net$"]
        m = summary(run(mnq5, mnq1, p), cost)
        row["MNQ26$"], row["MNQ26 PF"] = m["net$"], m["PF"]
        row["MNQ26 stop1st$"] = summary(run(mnq5, None, p, tie="stop"), cost)["net$"]
        rows.append(row)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(f"Net of ${cost}/RT/contract. NQ = Dec 2022-Dec 2025 (1m exits); MNQ26 = May-Aug 2026.")
    print(pd.DataFrame(rows).to_string(index=False))


WINDOWS = [(8 * 60 + 30, 9 * 60 + 30), (9 * 60, 9 * 60 + 30), (9 * 60, 9 * 60 + 50),
           (9 * 60 + 15, 9 * 60 + 45), (9 * 60 + 15, 9 * 60 + 50), (9 * 60 + 15, 10 * 60),
           (9 * 60 + 30, 9 * 60 + 35), (9 * 60 + 30, 9 * 60 + 45), (9 * 60 + 30, 9 * 60 + 50),
           (9 * 60 + 30, 10 * 60), (9 * 60 + 30, 10 * 60 + 15), (9 * 60 + 30, 10 * 60 + 30)]
MARGINS = [0, 2, 4, 6, 8, 12, 16, 24, 32, 48]


def hhmm(m):
    return f"{m // 60:02d}:{m % 60:02d}"


def grid(cost, out=None, rrs=(0.3, 0.55)):
    """Breakout margin x range window x R:R. NQ 2023-24 is the in-sample; 2025 and MNQ 2026 are held out."""
    nq1 = load(DATA / "real_nq_1m_2022-12-26_2025-12-11.parquet")
    nq5 = to_5m(nq1)
    mnq5 = load(DATA / "real_mnq_5m_2026-05-10_2026-08-25.parquet")
    mnq1 = load(DATA / "real_mnq_1m_2026-08-02_2026-08-25.parquet")
    rows = []
    for rr in rrs:
        for ws, we in WINDOWS:
            for mt in MARGINS:
                p = Params(range_start=ws, range_end=we, break_ticks=mt, rr=rr)
                t = run(nq5, nq1, p)
                net = t.pnl_usd - cost * t.qty
                yr = t.entry_time.dt.year
                ins = summary(t[yr.isin([2023, 2024])], cost)
                m = summary(run(mnq5, mnq1, p), cost)
                rows.append({"rr": rr, "window": f"{hhmm(ws)}-{hhmm(we)}", "margin": mt,
                             "trades": len(t), **{k: summary(t, cost)[k] for k in ("win%", "PF", "net$", "maxDD$", "sharpe_daily")},
                             "IS23-24$": ins["net$"], "IS PF": ins["PF"], "OOS25$": round(net[yr == 2025].sum()),
                             "stop1st$": summary(run(nq5, None, p, tie="stop"), cost)["net$"],
                             "MNQ26$": m.get("net$", 0), "MNQ26 n": m["trades"]})
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250, "display.max_columns", 30)
    for rr in rrs:
        d = df[df.rr == rr]
        print(f"\n=== R:R {rr}: net $ (NQ Dec22-Dec25, ${cost}/RT/ct), rows = window, cols = margin ticks ===")
        print(d.pivot(index="window", columns="margin", values="net$").to_string())
        print(f"--- R:R {rr}: profit factor ---")
        print(d.pivot(index="window", columns="margin", values="PF").to_string())
    print("\n=== Top 15 by in-sample (2023-24) net, with held-out results ===")
    print(df.sort_values("IS23-24$", ascending=False).head(15).to_string(index=False))
    if out:
        Path(out).mkdir(parents=True, exist_ok=True)
        df.to_csv(Path(out) / "grid.csv", index=False)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="directory for trade CSVs")
    ap.add_argument("--cost", type=float, default=1.24, help="round-turn $ per contract for the net line")
    ap.add_argument("--sweep-rr", nargs="*", type=float, default=None,
                    help="sweep the R:R target (no values = a default grid) instead of the default report")
    ap.add_argument("--grid", action="store_true",
                    help="sweep breakout margin x range window (x R:R 0.3/0.55); writes --out/grid.csv")
    args = ap.parse_args()
    if args.grid:
        return grid(args.cost, args.out)
    if args.sweep_rr is not None:
        return sweep_rr(args.sweep_rr or [0.2, 0.25, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0], args.cost)

    runs = {}
    nq1 = load(DATA / "real_nq_1m_2022-12-26_2025-12-11.parquet")
    runs["NQ 2022-12..2025-12 (5m from 1m, 1m exits)"] = run(to_5m(nq1), nq1)
    runs["NQ 2022-12..2025-12 (5m only, stop-first)"] = run(to_5m(nq1), None, tie="stop")
    runs["NQ 2022-12..2025-12 (5m only, target-first)"] = run(to_5m(nq1), None, tie="target")
    mnq5 = load(DATA / "real_mnq_5m_2026-05-10_2026-08-25.parquet")
    mnq1 = load(DATA / "real_mnq_1m_2026-08-02_2026-08-25.parquet")
    runs["MNQ 2026-05..2026-08 (native 5m, 1m exits in Aug)"] = run(mnq5, mnq1)
    runs["MNQ 2026-05..2026-08 (native 5m, stop-first)"] = run(mnq5, None, tie="stop")

    rows = []
    for name, t in runs.items():
        rows.append({"run": name, "costs": "gross", **summary(t)})
        rows.append({"run": name, "costs": f"${args.cost}/RT/ct", **summary(t, args.cost)})
    pd.set_option("display.width", 250, "display.max_columns", 30)
    print(pd.DataFrame(rows).to_string(index=False))

    main_t = runs["NQ 2022-12..2025-12 (5m from 1m, 1m exits)"]
    print("\nExit reasons (NQ, 1m exits):\n", main_t.reason.value_counts().to_string())
    print("\nBy direction:\n", main_t.groupby("dir").apply(lambda g: pd.Series(summary(g, args.cost))).to_string())
    yr = main_t.groupby(main_t.entry_time.dt.year).apply(lambda g: pd.Series(summary(g, args.cost)))
    print("\nBy year (net):\n", yr.to_string())
    if args.out:
        out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
        for name, t in runs.items():
            slug = "".join(ch if ch.isalnum() else "_" for ch in name.lower()).strip("_")
            t.to_csv(out / f"{slug}.csv", index=False)


if __name__ == "__main__":
    main()
