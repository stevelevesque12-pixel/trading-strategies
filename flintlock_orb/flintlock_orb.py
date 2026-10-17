"""Flintlock ORB - a 09:30 New York opening-range breakout taken only on a solid-bodied candle.

Entry rules (New York time; a signal on a candle means: enter at the NEXT candle's open):
1. The opening range is the high and low of the candles that open in the first
   OR_MINUTES minutes from 09:30.
2. After the range is complete, a bullish candle that closes above the range high is a
   long signal; a bearish candle that closes below the range low is a short signal.
   No signals from the candle whose close reaches 16:00 until midnight.
3. The candle must be solid: its wick on the breakout side is at most MAX_WICK_BODY x its
   body, and both wicks together are at most MAX_TOTAL_WICK x its body.
4. One position at a time, at most MAX_TRADES entries per calendar day, and DIRECTION
   picks long, short or both.

Exits:
* Stop: STOP_ATR_MULT x ATR(ATR_PERIOD) away from the signal candle's close
  (STOP_MODE "ATR"), or the far side of the opening range (STOP_MODE "Range").
  Target: TARGET_R times that risk. Both are fixed at the signal and rounded to the tick.
* Any open position is closed at the open of the candle whose close reaches 16:00, and at
  the last candle of each 18:00-17:00 futures session at the latest.
* When a candle touches both the stop and the target, the stop is taken as hit first.
The entry list already respects these exits (a new entry only fires when flat).

Input: a CSV with header time,open,high,low,close,volume, where time is the candle OPEN
time in UTC (or with an explicit UTC offset, e.g. 2025-03-04 14:30:00+00:00), covering the
full 24h session, oldest first. Tick size is a setting (0.25 for ES/NQ).

Run: python flintlock_orb.py bars.csv -> writes flintlock_orb_signals.csv, prints a count.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---- settings -------------------------------------------------------------
OR_MINUTES = 5          # opening range length, minutes from 09:30
MAX_WICK_BODY = 0.09    # breakout-side wick / body, at most
MAX_TOTAL_WICK = 1.5    # both wicks / body, at most
STOP_MODE = "ATR"       # "ATR" or "Range"
ATR_PERIOD = 14
STOP_ATR_MULT = 1.4
TARGET_R = 2.0
MAX_TRADES = 1          # entries per calendar day
DIRECTION = "both"      # "long", "short" or "both"
TICK_SIZE = 0.25

RANGE_START = 570       # 09:30 in minutes of the day
FLAT_AT = 960           # 16:00


# ---- helpers --------------------------------------------------------------
def _clock(index):
    """(New York calendar day, minute of day, futures session day) per candle."""
    if index.tz is None:
        index = index.tz_localize("UTC")
    et = index.tz_convert("America/New_York")
    day = et.normalize().asi8
    minute = np.asarray(et.hour * 60 + et.minute)
    session = (et + pd.Timedelta(hours=6)).normalize().asi8  # the 18:00 roll
    return day, minute, session


def _bar_minutes(index):
    """Candle length in minutes (median spacing)."""
    if len(index) < 2:
        return 1
    d = np.diff(index.as_unit("ns").asi8)
    d = d[d > 0]
    if not d.size:
        return 1
    return max(1, int(round(float(np.median(d)) / 60_000_000_000)))


def _snap(x, tick):
    """Round prices to the nearest tick."""
    x = np.asarray(x, dtype=np.float64)
    inv = round(1.0 / tick)
    s = np.round(x / tick)
    return s / inv if inv > 0 and abs(inv * tick - 1.0) < 1e-9 else s * tick


def _atr(h, l, c, length):
    """Wilder ATR: true range smoothed with a running average seeded by a simple mean."""
    n = len(h)
    out = np.full(n, np.nan)
    if n == 0 or length <= 0 or n < length:
        return out
    tr = np.empty(n)
    tr[0] = h[0] - l[0]
    for i in range(1, n):
        pc = c[i - 1]
        tr[i] = max(h[i] - l[i], abs(h[i] - pc), abs(l[i] - pc))
    alpha = 1.0 / length
    s = 0.0
    for i in range(length):
        s += tr[i]
    out[length - 1] = s / length
    for i in range(length, n):
        out[i] = alpha * tr[i] + (1.0 - alpha) * out[i - 1]
    return out


def _range(day, minute, x, how, start, end):
    """High (or low) of the candles opening in [start, end) of each day; NaN until end."""
    win = (minute >= start) & (minute < end)
    v = pd.Series(np.where(win, x, np.nan)).groupby(day).transform(how).to_numpy()
    return np.where(minute >= end, v, np.nan)


def _positions(o, h, l, day, session, cl, cs, sl_l, tp_l, sl_s, tp_s, flat, max_day, tick):
    """Walk the candles holding one position at a time and keep only the entries taken."""
    n = len(o)
    le = np.zeros(n, bool)
    se = np.zeros(n, bool)
    sp = np.full(n, np.nan)
    tp = np.full(n, np.nan)
    pos, pend, fills = 0, 0, 0
    stop = tgt = np.nan
    for t in range(n):
        if t == 0 or day[t] != day[t - 1]:
            fills = 0
        if t > 0:
            if pos != 0 and flat[t - 1]:
                pos = 0  # closed at this candle's open
            if pend != 0:
                if pos == 0 and session[t] == session[t - 1]:
                    pos, entry, stop, tgt = pend, o[t], sp[t - 1], tp[t - 1]
                    if stop == stop:  # opened past the stop: one tick beyond
                        if pos == 1 and entry - stop < tick:
                            stop = entry - tick
                        elif pos == -1 and stop - entry < tick:
                            stop = entry + tick
                    if tgt == tgt:
                        if pos == 1 and tgt - entry < tick:
                            tgt = entry + tick
                        elif pos == -1 and entry - tgt < tick:
                            tgt = entry - tick
                    fills += 1
                pend = 0
            if pos == 1 and ((stop == stop and l[t] <= stop) or (tgt == tgt and h[t] >= tgt)):
                pos = 0
            elif pos == -1 and ((stop == stop and h[t] >= stop) or (tgt == tgt and l[t] <= tgt)):
                pos = 0
        if t < n - 1 and session[t + 1] != session[t]:
            pos = 0  # flat at the session's last candle
            continue
        if not (pos == 0 or flat[t]):
            continue
        if max_day > 0 and fills >= max_day:
            continue
        if cl[t] and not flat[t]:
            le[t], sp[t], tp[t], pend = True, sl_l[t], tp_l[t], 1
        elif cs[t] and not flat[t]:
            se[t], sp[t], tp[t], pend = True, sl_s[t], tp_s[t], -1
    return le, se, sp, tp


# ---- the strategy ---------------------------------------------------------
def signals(df, tick_size=TICK_SIZE):
    """Entry flags plus the stop and target price set on each entry candle."""
    o = df["open"].to_numpy(np.float64)
    h = df["high"].to_numpy(np.float64)
    l = df["low"].to_numpy(np.float64)
    c = df["close"].to_numpy(np.float64)
    day, minute, session = _clock(df.index)
    bm = _bar_minutes(df.index)
    end = RANGE_START + int(OR_MINUTES)
    rh = _range(day, minute, h, "max", RANGE_START, end)
    rl = _range(day, minute, l, "min", RANGE_START, end)
    body = np.abs(c - o)
    up = h - np.maximum(o, c)
    dn = np.minimum(o, c) - l
    k = float(MAX_WICK_BODY)
    solid = (body > 0) & (up + dn <= float(MAX_TOTAL_WICK) * body)
    ok = (minute >= end) & (minute + bm < FLAT_AT) & solid
    with np.errstate(invalid="ignore"):
        cl = ok & (c > o) & (c > rh) & (up <= k * body)
        cs = ok & (c < o) & (c < rl) & (dn <= k * body)
    if STOP_MODE == "ATR":
        a = _atr(h, l, c, int(ATR_PERIOD)) * float(STOP_ATR_MULT)
        sl_l, sl_s = c - a, c + a
    else:
        sl_l, sl_s = rl, rh
    rr = float(TARGET_R)
    tp_l, tp_s = c + rr * (c - sl_l), c - rr * (sl_s - c)
    flat = minute + bm >= FLAT_AT
    sl_l, tp_l, sl_s, tp_s = (_snap(x, tick_size) for x in (sl_l, tp_l, sl_s, tp_s))
    cl = cl & (DIRECTION in ("long", "both")) & np.isfinite(sl_l)
    cs = cs & (DIRECTION in ("short", "both")) & np.isfinite(sl_s)
    le, se, sp, tp = _positions(o, h, l, day, np.asarray(session, np.int64), cl, cs,
                                sl_l, tp_l, sl_s, tp_s, flat, int(MAX_TRADES), tick_size)
    return {"long_entry": le, "short_entry": se, "long_exit": flat, "short_exit": flat.copy(),
            "stop_price": _snap(sp, tick_size), "target_price": _snap(tp, tick_size),
            "flat_at_session_end": True}


def load_candles(path):
    df = pd.read_csv(path, float_precision="round_trip")
    df.index = pd.to_datetime(df.pop("time"), utc=True)
    return df[["open", "high", "low", "close", "volume"]]


def main():
    if len(sys.argv) < 2:
        print("usage: python flintlock_orb.py bars.csv")
        sys.exit(1)
    df = load_candles(sys.argv[1])
    s = signals(df)
    rows = np.flatnonzero(s["long_entry"] | s["short_entry"])
    out = pd.DataFrame({"time": df.index[rows],
                        "side": np.where(s["long_entry"][rows], "long", "short"),
                        "stop_price": s["stop_price"][rows],
                        "target_price": s["target_price"][rows]})
    path = Path("flintlock_orb_signals.csv")
    out.to_csv(path, index=False)
    print(f"{len(out)} entries -> {path}")


if __name__ == "__main__":
    main()
