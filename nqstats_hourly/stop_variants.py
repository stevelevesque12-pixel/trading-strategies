"""Usage: python nqstats_hourly/stop_variants.py <nq_1m.parquet>

Compares three ways of trading the NQStats hourly breach (08:00-15:59 ET, hour opens
inside the previous hour's range, first breach only, one trade per hour, target = hour
open, anything still open is closed at the hour's last bar):

  A  fade the breach: resting limit one tick beyond the previous hour high/low, stop a
     fixed % of the hour open beyond the entry (or none)
  B  the Pine strategy's iFVG entry, stop = sweep extreme pushed out by a % buffer
  C  the Pine strategy's iFVG entry, no price stop (hour end only)

Costs as in backtest_replica.py: 1 MNQ, $1.82 round trip, 1 tick slippage on market/stop
fills. On A's entry bar the order of the breach and a touch of the hour open is unknown,
so only the stop is checked on that bar (conservative).
"""
import sys
import numpy as np
import pandas as pd

TICK, PT_VALUE, RT_COMM, SLIP = 0.25, 2.0, 1.82, 0.25
MAX_FVG_AGE, MAX_TRACKED = 60, 200

df = pd.read_parquet(sys.argv[1]).tz_convert("America/New_York")
O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
HR, MN, TS = df.index.hour.to_numpy(), df.index.minute.to_numpy(), df.index

# ---- One pass to collect, per session hour: breach and the first iFVG signal ----
events = []
cur_hi = cur_lo = prev_hi = prev_lo = hour_open = np.nan
breached, bdir, sweep, ev = False, 0, np.nan, None
fvgs, lbt, lbb, lbr, lbrb = [], np.nan, -10**9, np.nan, -10**9
for i in range(len(df)):
    o, h, l, c, hr = O[i], H[i], L[i], C[i], HR[i]
    if i == 0 or hr != HR[i - 1]:
        if ev is not None:
            ev["last"] = i - 1
            events.append(ev)
        prev_hi, prev_lo, cur_hi, cur_lo, hour_open = cur_hi, cur_lo, h, l, o
        breached, bdir, sweep = False, 0, np.nan
        ev = dict(hour=hr, open=o, prev_hi=prev_hi, prev_lo=prev_lo) if 8 <= hr <= 15 else None
    else:
        cur_hi, cur_lo = max(cur_hi, h), min(cur_lo, l)
    if bdir == 1: sweep = min(sweep, l)
    elif bdir == -1: sweep = max(sweep, h)
    if ev is not None and not breached and prev_lo < hour_open < prev_hi:
        hb, lb = h >= prev_hi + TICK, l <= prev_lo - TICK
        if hb or lb:
            breached = True
            bdir = 0 if (hb and lb) else (-1 if hb else 1)
            sweep = h if hb else l
            ev.update(b_idx=i, dir=bdir, seg=MN[i] // 20 + 1)
    bull = bear = False
    for k in range(len(fvgs) - 1, -1, -1):
        sb, top, bot, bearish = fvgs[k]
        if i - sb > MAX_FVG_AGE: del fvgs[k]
        elif bearish and c > top: bull, lbt, lbb = True, top, i; del fvgs[k]
        elif not bearish and c < bot: bear, lbr, lbrb = True, bot, i; del fvgs[k]
    bull = bull or (i - lbb <= MAX_FVG_AGE and c > lbt and C[i - 1] <= lbt)
    bear = bear or (i - lbrb <= MAX_FVG_AGE and c < lbr and C[i - 1] >= lbr)
    if i >= 2:
        if L[i - 2] - h >= TICK: fvgs.append([i - 1, L[i - 2], h, True])
        if l - H[i - 2] >= TICK: fvgs.append([i - 1, l, H[i - 2], False])
        del fvgs[:max(0, len(fvgs) - MAX_TRACKED)]
    if ev is not None and bdir != 0 and "s_idx" not in ev and MN[i] != 59:
        if (bdir == 1 and bull and sweep < c < hour_open) or (bdir == -1 and bear and hour_open < c < sweep):
            ev.update(s_idx=i, s_close=c, s_sweep=sweep)
ev_df = pd.DataFrame([e for e in events if e.get("dir", 0) != 0])


def simulate(d, entry, stop, target, first_exit_bar, last, entry_idx, stop_only_on_entry_bar):
    """Walk bars until stop, target or hour end. Returns (exit_price, reason)."""
    if stop_only_on_entry_bar and stop is not None:
        if (d == 1 and L[entry_idx] <= stop) or (d == -1 and H[entry_idx] >= stop):
            return stop - SLIP * d, "stop"
    for j in range(first_exit_bar, last + 1):
        o, h, l = O[j], H[j], L[j]
        hs = stop is not None and (l <= stop if d == 1 else h >= stop)
        ht = h >= target if d == 1 else l <= target
        if stop is not None and (o <= stop if d == 1 else o >= stop): return o - SLIP * d, "stop"
        if (o >= target if d == 1 else o <= target): return o, "target"
        toward_target_first = (h - o) < (o - l) if d == 1 else (o - l) < (h - o)
        if ht and (toward_target_first or not hs): return target, "target"
        if hs: return stop - SLIP * d, "stop"
    return C[last] - SLIP * d, "hour end"


def run_A(stop_pct):
    out = []
    for e in ev_df.itertuples():
        d, i = e.dir, int(e.b_idx)
        lvl = e.prev_lo - TICK if d == 1 else e.prev_hi + TICK
        entry = min(O[i], lvl) if d == 1 else max(O[i], lvl)  # gap through = better fill
        stop = None if stop_pct is None else entry - d * e.open * stop_pct / 100
        px, why = simulate(d, entry, stop, e.open, i + 1, int(e.last), i, True)
        out.append((e.hour, e.seg, d, entry, px, why))
    return out


def run_B(buf_pct, no_stop=False):
    out = []
    for e in ev_df[ev_df.s_idx.notna()].itertuples():
        d, i = e.dir, int(e.s_idx)
        entry = e.s_close + SLIP * d
        stop = None if no_stop else e.s_sweep - d * e.open * buf_pct / 100
        px, why = simulate(d, entry, stop, e.open, i + 1, int(e.last), i, False)
        out.append((e.hour, e.seg, d, entry, px, why))
    return out


def frame(rows):
    t = pd.DataFrame(rows, columns=["hour", "seg", "dir", "entry", "exit", "reason"])
    t["pts"] = (t.exit - t.entry) * t.dir
    t["usd"] = t.pts * PT_VALUE - RT_COMM
    return t


def summ(t):
    w, lo = t.usd[t.usd > 0].sum(), -t.usd[t.usd < 0].sum()
    eq = t.usd.cumsum()
    return {"trades": len(t), "win%": (t.usd > 0).mean() * 100, "target%": (t.reason == "target").mean() * 100,
            "exp $": t.usd.mean(), "net $": t.usd.sum(), "PF": w / lo if lo else np.inf,
            "worst $": t.usd.min(), "maxDD $": (eq.cummax() - eq).max()}


variants = {f"A stop {p}%": frame(run_A(p)) for p in (0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50)}
variants["A no stop"] = frame(run_A(None))
variants.update({f"B buffer {p}%": frame(run_B(p)) for p in (0.0, 0.05, 0.10, 0.15, 0.20, 0.30)})
variants["C no stop"] = frame(run_B(0, no_stop=True))

pd.set_option("display.width", 220)
pd.set_option("display.max_columns", 20)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
print(f"{TS[0].date()} -> {TS[-1].date()} | breaches traded by A: {len(ev_df)}, iFVG signals: {ev_df.s_idx.notna().sum()}")
print(f"avg hour open {ev_df.open.mean():,.0f} -> 0.10% ~ {ev_df.open.mean() * 0.001:.1f} NQ pts\n")
print(pd.DataFrame({k: summ(v) for k, v in variants.items()}).T)

best = max(variants, key=lambda k: variants[k].usd.sum())
for name in dict.fromkeys([best, "A no stop", "C no stop"]):
    t = variants[name]
    print(f"\n==== {name} ====")
    print("NET $ hour x segment\n", t.pivot_table("usd", "hour", "seg", "sum", margins=True))
    print("EXPECTANCY $ hour x segment\n", t.pivot_table("usd", "hour", "seg", "mean"))
    print("TRADES hour x segment\n", t.pivot_table("usd", "hour", "seg", "count"))

# Reward = distance from entry to the hour open. A fade from a level that sits right next
# to the open has almost nothing to gain, so break the results down by that distance.
print("\n==== by reward (entry -> hour open, NQ pts) ====")
for name in ("A stop 0.25%", "A no stop", "B buffer 0.2%"):
    t = variants[name]
    tgt = ev_df.open.to_numpy() if name.startswith("A") else ev_df[ev_df.s_idx.notna()].open.to_numpy()
    t["reward"] = (tgt - t.entry.to_numpy()) * t.dir.to_numpy()
    b = pd.cut(t.reward, [0, 5, 10, 15, 20, 30, 50, 1e9])
    print(f"\n{name}\n", t.groupby(b, observed=True).usd.agg(trades="count", win=lambda x: (x > 0).mean() * 100,
                                                               exp="mean", net="sum"))
