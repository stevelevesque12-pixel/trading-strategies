"""Usage: python nqstats_hourly/backtest_replica.py <nq_1m.parquet> <trades_out.csv>

Bar-by-bar Python replica of tradingview/nqstats_hourly_retracement.pine (default inputs).

Fills follow TradingView's broker emulator: process_orders_on_close entries at the
signal bar's close, exit orders active from the next bar, intrabar path open->nearer
extreme->other extreme, gaps through a level fill at the open. 1 tick slippage on
market and stop fills (not on the limit target). PnL reported in NQ points and as
1 MNQ contract ($2/pt, $1.82 round-trip commission).
"""
import sys
import numpy as np
import pandas as pd

TICK, PT_VALUE, RT_COMM, SLIP = 0.25, 2.0, 1.82, 0.25
FIRST_HOUR, LAST_HOUR, MAX_FVG_AGE, MAX_TRACKED, MIN_GAP = 8, 15, 60, 200, 0.25

df = pd.read_parquet(sys.argv[1]).tz_convert("America/New_York")
O, H, L, C = (df[c].to_numpy() for c in ("open", "high", "low", "close"))
HR, MN = df.index.hour.to_numpy(), df.index.minute.to_numpy()
TS = df.index

cur_hi = cur_lo = prev_hi = prev_lo = hour_open = np.nan
breached, breach_dir, sweep, breach_seg = False, 0, np.nan, 0
fvgs = []  # [start_bar, top, bot, bearish]
last_bull_top = last_bear_bot = np.nan
last_bull_bar = last_bear_bar = -10**9
trades_this_hour = 0
pos = None  # dict(dir, entry, stop, target, ...)
trades = []


def close_pos(i, px, reason):
    global pos
    pts = (px - pos["entry"]) * pos["dir"]
    trades.append({**pos, "exit_time": TS[i], "exit": px, "reason": reason, "pts": pts,
                   "usd": pts * PT_VALUE - RT_COMM})
    pos = None


for i in range(len(df)):
    o, h, l, c, hr = O[i], H[i], L[i], C[i], HR[i]

    # 1) Exit orders working from the bar after entry.
    if pos is not None:
        d, s, t = pos["dir"], pos["stop"], pos["target"]
        if d == 1:
            if o <= s: close_pos(i, o - SLIP, "stop")
            elif o >= t: close_pos(i, o, "target")
            else:
                high_first = (h - o) < (o - l)
                hit_t, hit_s = h >= t, l <= s
                if hit_t and (high_first or not hit_s): close_pos(i, t, "target")
                elif hit_s: close_pos(i, s - SLIP, "stop")
        else:
            if o >= s: close_pos(i, o + SLIP, "stop")
            elif o <= t: close_pos(i, o, "target")
            else:
                low_first = (o - l) < (h - o)
                hit_t, hit_s = l <= t, h >= s
                if hit_t and (low_first or not hit_s): close_pos(i, t, "target")
                elif hit_s: close_pos(i, s + SLIP, "stop")

    # 2) Script logic at the bar's close (mirrors the Pine file top to bottom).
    new_hour = i == 0 or hr != HR[i - 1]
    in_session = FIRST_HOUR <= hr <= LAST_HOUR
    if new_hour:
        prev_hi, prev_lo = cur_hi, cur_lo
        cur_hi, cur_lo = h, l
        hour_open = o
        breached, breach_dir, sweep, breach_seg = False, 0, np.nan, 0
        trades_this_hour = 0
    else:
        cur_hi, cur_lo = max(cur_hi, h), min(cur_lo, l)

    if breach_dir == 1: sweep = min(sweep, l)
    elif breach_dir == -1: sweep = max(sweep, h)

    open_inside = prev_lo < hour_open < prev_hi
    eligible = in_session and not breached and open_inside
    hi_b = eligible and h >= prev_hi + TICK
    lo_b = eligible and l <= prev_lo - TICK
    if hi_b or lo_b:
        breached, breach_seg = True, MN[i] // 20 + 1
        breach_dir = 0 if (hi_b and lo_b) else (-1 if hi_b else 1)
        sweep = h if hi_b else l

    bull_ifvg = bear_ifvg = False
    for k in range(len(fvgs) - 1, -1, -1):
        sb, top, bot, bearish = fvgs[k]
        if i - sb > MAX_FVG_AGE: del fvgs[k]
        elif bearish and c > top:
            bull_ifvg, last_bull_top, last_bull_bar = True, top, i; del fvgs[k]
        elif not bearish and c < bot:
            bear_ifvg, last_bear_bot, last_bear_bar = True, bot, i; del fvgs[k]
    bull_reclaim = (i - last_bull_bar <= MAX_FVG_AGE and c > last_bull_top and C[i - 1] <= last_bull_top)
    bear_reclaim = (i - last_bear_bar <= MAX_FVG_AGE and c < last_bear_bot and C[i - 1] >= last_bear_bot)

    if i >= 2:
        if L[i - 2] - h >= MIN_GAP: fvgs.append([i - 1, L[i - 2], h, True])
        if l - H[i - 2] >= MIN_GAP: fvgs.append([i - 1, l, H[i - 2], False])
        del fvgs[:max(0, len(fvgs) - MAX_TRACKED)]

    last_bar = MN[i] == 59
    can_enter = in_session and breach_seg > 0 and trades_this_hour < 1 and pos is None and not last_bar
    long_sig = can_enter and breach_dir == 1 and (bull_ifvg or bull_reclaim) and sweep < c < hour_open
    short_sig = can_enter and breach_dir == -1 and (bear_ifvg or bear_reclaim) and hour_open < c < sweep
    if long_sig or short_sig:
        trades_this_hour += 1
        d = 1 if long_sig else -1
        pos = dict(dir=d, entry_time=TS[i], hour=hr, seg=breach_seg,
                   entry=c + SLIP * d, stop=sweep, target=hour_open,
                   kind="new iFVG" if (bull_ifvg if d == 1 else bear_ifvg) else "reclaim")
    elif pos is not None and last_bar:
        close_pos(i, c - SLIP * pos["dir"], "hour end")

t = pd.DataFrame(trades)
t.to_csv(sys.argv[2], index=False)


def summ(x):
    wins, losses = x.usd[x.usd > 0].sum(), -x.usd[x.usd < 0].sum()
    return pd.Series({"trades": len(x), "win%": (x.usd > 0).mean() * 100,
                      "target%": (x.reason == "target").mean() * 100,
                      "avg pts": x.pts.mean(), "avg risk pts": ((x.entry - x.stop) * x.dir).mean(),
                      "exp $": x.usd.mean(), "net $": x.usd.sum(),
                      "PF": wins / losses if losses else np.inf})


pd.set_option("display.width", 200)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
eq = t.usd.cumsum()
print(f"{TS[0].date()} -> {TS[-1].date()} | 1 MNQ, $1.82 RT, 1-tick slippage")
print(summ(t).to_string(), f"\nmax drawdown $: {(eq.cummax() - eq).max():,.2f}")
print("\nexit reasons:", t.reason.value_counts().to_dict(), "| entry kinds:", t.kind.value_counts().to_dict())
print("\nBY HOUR\n", t.groupby("hour").apply(summ, include_groups=False))
print("\nBY SEGMENT\n", t.groupby("seg").apply(summ, include_groups=False))
print("\nNET $ hour x segment\n", t.pivot_table("usd", "hour", "seg", "sum", margins=True))
print("\nTRADES hour x segment\n", t.pivot_table("usd", "hour", "seg", "count", margins=True))
print("\nEXPECTANCY $ hour x segment\n", t.pivot_table("usd", "hour", "seg", "mean"))
print("\nBY YEAR\n", t.groupby(t.entry_time.dt.year).apply(summ, include_groups=False))
