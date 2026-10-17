"""Bellows - an opening-range breakout taken only on days whose opening range is
"normal size" for the market's recent volatility.

The idea: a 15-minute opening range can be unusually narrow, ordinary or unusually wide
compared with how much the market has been moving lately. Bellows measures that, sorts
each day into one of three groups (terciles), and trades the breakout of the range only
on days in the chosen group. These settings trade the middle group.

All times are New York time.

Regime label, decided at OR_END (09:45):
  1. Opening range (OR) = the highest high and lowest low of the candles from 09:30 to
     OR_END. Width = OR high - OR low.
  2. Session ATR = Wilder ATR(ATR_LEN) of daily regular-session candles (the high, low and
     last close of each day's 09:30-16:00 candles), taken from YESTERDAY back, so today
     is never part of it.
  3. Ratio w = OR width / session ATR.
  4. Today's w is compared with the 1/3 and 2/3 quantiles of the previous
     LOOKBACK_SESSIONS days' w values (today excluded). Grading starts once 63 earlier
     values exist. low = w <= 1/3 quantile, high = w > 2/3 quantile, mid = in between.
     TRADE_REGIME picks the group that may trade ("all" ignores the label).

Entry rules, checked at each candle's close, from OR_END until FORCE_FLAT:
  1. Long when the candle's high reaches OR high + ENTRY_BUFFER_TICKS ticks.
     Short when the candle's low reaches OR low - ENTRY_BUFFER_TICKS ticks.
  2. If one candle reaches both levels, an up candle (close >= open) counts as long,
     a down candle as short.
  3. One trade per day: the first trigger wins.
  4. Entry = market order at the next candle's open.

Exits (distances count from the entry fill):
  - Stop distance = from the entry level to the opposite side of the OR plus the same buffer,
    kept between 0.10 x and STOP_CAP_ATR x the session ATR, rounded to the tick.
  - Target distance = TP_MULT x the OR width, rounded to the tick.
  - Flat: from FORCE_FLAT on, any open trade exits at the next candle's open.

Two parts you can use separately:
  - `signals()` is the strategy: per candle, long/short entry, stop and target distances
    and the exit flag. Plug it into your own platform.
  - `backtest()` is one fill model for it, at 1 contract with costs:
      * Market entries and exits (and stop fills) pay SLIPPAGE_TICKS against you; targets
        fill at their price. COMMISSION is charged per side per contract.
      * A stop or target the candle opens through fills at the open.
      * When one candle reaches both stop and target, the 1-minute candles inside it decide
        the order (if you pass them); inside one 1-minute candle (or without 1-minute data)
        the stop counts first, unless the candle opened beyond one of them.
    Another platform's fill rules can give other results.

Input: a CSV with header `time,open,high,low,close` (more columns are ignored), where
`time` is the candle OPEN time in UTC or with an explicit UTC offset
(e.g. `2025-03-04 14:30:00+00:00`), covering the full 24h Globex session, oldest first.
These settings are for 5-minute NQ candles (tick 0.25, $20 per point). The optional
second CSV is 1-minute candles of the same contract in the same format.

Needs Python 3 with numpy and pandas.
Run: python bellows.py bars_5m.csv [bars_1m.csv]
(writes bellows_trades.csv next to the first input and prints a summary)
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# The Bellows settings.
TRADE_REGIME = "mid"        # "all", "low", "mid" or "high"
LOOKBACK_SESSIONS = 504     # days of w the terciles are taken over
ATR_LEN = 28                # session ATR length (days)
ENTRY_BUFFER_TICKS = 15
OR_END = "09:45"            # opening range 09:30 -> this time
FORCE_FLAT = "15:50"
DIRECTION = "both"          # "both", "long" or "short"
STOP_CAP_ATR = 0.6          # stop at most this x session ATR
TP_MULT = 3.0               # target = this x OR width
TICK_SIZE = 0.25

# Backtest settings.
POINT_VALUE = 20.0          # NQ: $20 per point
COMMISSION = 2.18           # $ per side per contract
SLIPPAGE_TICKS = 2          # on market entries, market exits and stop fills
CONTRACTS = 1
START_EQUITY = 100_000.0

RTH_OPEN, RTH_CLOSE = "09:30", "16:00"


def _mins(hhmm):
    h, m = hhmm.split(":")
    return int(h) * 60 + int(m)


def load_candles(path):
    df = pd.read_csv(path, float_precision="round_trip")  # exact, not the fast reader
    df.columns = [str(c).strip().lower() for c in df.columns]
    df.index = pd.to_datetime(df["time"], utc=True)
    return df[["open", "high", "low", "close"]].astype(np.float64)


def _et(index):
    return index.tz_convert("America/New_York") if index.tz is not None \
        else index.tz_localize("UTC").tz_convert("America/New_York")


def _wilder_atr(high, low, close, length):
    h, l, c = (s.to_numpy(dtype=float) for s in (high, low, close))
    pc = np.empty_like(c)
    if pc.size:
        pc[0] = np.nan
        pc[1:] = c[:-1]
    tr = np.fmax(h - l, np.fmax(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(pd.Series(tr).ewm(alpha=1 / length, adjust=False).mean().to_numpy(), index=high.index)


def signals(df, tick_size=TICK_SIZE):
    """Entry flags, stop / target distances and the flat flag, decided at each candle's close."""
    n = len(df)
    tick = float(tick_size) or 0.25
    et = _et(df.index)
    day = et.normalize().asi8  # calendar day in New York
    mod = (et.hour * 60 + et.minute).to_numpy()
    high, low, close, open_ = (df[k].to_numpy(np.float64) for k in ("high", "low", "close", "open"))

    open_min, rth_end = _mins(RTH_OPEN), _mins(RTH_CLOSE)
    or_end, flat_min = _mins(OR_END), _mins(FORCE_FLAT)

    # Per-day series: yesterday's session ATR, today's OR, and the trailing terciles of w.
    f = pd.DataFrame({"high": high, "low": low, "close": close, "day": day})
    r = f.loc[(mod >= open_min) & (mod < rth_end)].groupby("day")
    a_prior = _wilder_atr(r["high"].max(), r["low"].min(), r["close"].last(), int(ATR_LEN)).shift(1)
    o = f.loc[(mod >= open_min) & (mod < or_end)].groupby("day")
    or_hi_s, or_lo_s = o["high"].max(), o["low"].min()
    w_s = (or_hi_s - or_lo_s).divide(a_prior.where(a_prior > 0))
    roll = w_s.shift(1).rolling(int(LOOKBACK_SESSIONS), min_periods=min(int(LOOKBACK_SESSIONS), 63))
    q33_s, q66_s = roll.quantile(1 / 3), roll.quantile(2 / 3)

    dser = pd.Series(day)
    or_hi = dser.map(or_hi_s).to_numpy(dtype=float)
    or_lo = dser.map(or_lo_s).to_numpy(dtype=float)
    a = dser.map(a_prior).to_numpy(dtype=float)
    w = dser.map(w_s).to_numpy(dtype=float)
    q33 = dser.map(q33_s).to_numpy(dtype=float)
    q66 = dser.map(q66_s).to_numpy(dtype=float)
    width = or_hi - or_lo

    with np.errstate(invalid="ignore"):
        ok = np.isfinite(width) & (width > 0) & np.isfinite(a) & (a > 0)
        graded = ok & np.isfinite(w) & np.isfinite(q33) & np.isfinite(q66)
        if TRADE_REGIME == "low":
            ok = graded & (w <= q33)
        elif TRADE_REGIME == "high":
            ok = graded & (w > q66)
        elif TRADE_REGIME == "mid":
            ok = graded & (w > q33) & (w <= q66)

    buf = max(1.0, float(ENTRY_BUFFER_TICKS)) * tick
    long_level = or_hi + buf
    short_level = or_lo - buf

    armed = ok & (mod >= or_end) & (mod < flat_min)
    with np.errstate(invalid="ignore"):
        hit_long = armed & (high >= long_level)
        hit_short = armed & (low <= short_level)
    both = hit_long & hit_short  # both levels on one candle: its direction decides
    up_candle = close >= open_
    hit_long = hit_long & ~(both & ~up_candle)
    hit_short = hit_short & ~(both & up_candle)
    if DIRECTION == "long":
        hit_short = np.zeros(n, dtype=bool)
    elif DIRECTION == "short":
        hit_long = np.zeros(n, dtype=bool)

    anyb = hit_long | hit_short
    first = anyb & (pd.Series(anyb).groupby(day).cumsum().to_numpy() == 1)
    long_entry = hit_long & first
    short_entry = hit_short & first
    any_e = long_entry | short_entry

    # Stop: the opposite side of the range, kept between 0.10x and STOP_CAP_ATR x the session ATR.
    level = np.where(long_entry, long_level, np.where(short_entry, short_level, np.nan))
    opp = np.where(long_entry, or_lo - buf, np.where(short_entry, or_hi + buf, np.nan))
    with np.errstate(invalid="ignore"):
        stop = np.clip(np.abs(level - opp), 0.10 * a, float(STOP_CAP_ATR) * a)
        stop = np.maximum(tick, np.round(stop / tick) * tick)
        tgt = np.maximum(tick, np.round(float(TP_MULT) * width / tick) * tick)

    flat = mod >= flat_min
    return {
        "long_entry": long_entry,
        "short_entry": short_entry,
        "long_exit": flat,
        "short_exit": flat,
        "stop_dist": np.where(any_e, stop, np.nan),
        "target_dist": np.where(any_e, tgt, np.nan),
    }


def _on_tick(x, tick):
    """x rounded to the tick (removes float noise only)."""
    k = np.round(x / tick)
    if abs(x - k * tick) > tick * 1e-6:
        return x
    inv = np.round(1.0 / tick)
    return k / inv if inv > 0 and abs(inv * tick - 1.0) < 1e-9 else k * tick


def _sub_windows(index, sub):
    """For each candle, the [start, end) rows of the 1-minute candles inside it."""
    if sub is None or not len(sub):
        return None
    ts, sts = index.asi8, sub.index.asi8
    nxt = np.empty_like(ts)
    nxt[:-1] = ts[1:]
    nxt[-1] = ts[-1] + (int(np.median(np.diff(ts))) if len(ts) > 1 else 60_000_000_000)
    return (np.searchsorted(sts, ts, side="left"), np.searchsorted(sts, nxt, side="left"),
            *(sub[k].to_numpy(np.float64) for k in ("open", "high", "low", "close")))


def backtest(df, sig, sub=None, tick_size=TICK_SIZE, point_value=POINT_VALUE):
    """One fill model for the signals (see the top of the file), CONTRACTS contracts.
    `sub` = optional 1-minute candles. Returns (trades, equity at each candle's close)."""
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    et = _et(df.index)
    sday = (et + pd.Timedelta(hours=6)).normalize().asi8  # trading day, rolls at 18:00
    le, se, lx, sx = sig["long_entry"], sig["short_entry"], sig["long_exit"], sig["short_exit"]
    sdist, tdist = sig["stop_dist"], sig["target_dist"]
    win = _sub_windows(df.index, sub)
    slip = SLIPPAGE_TICKS * tick_size
    qty = float(CONTRACTS)
    n = len(c)
    trades, equity = [], np.empty(n)
    realized = 0.0
    pos, entry_i, entry_px, stop, tgt = 0, 0, 0.0, 0.0, 0.0
    entries_today, day = 0, None

    def close_trade(t, exit_px, reason):
        nonlocal realized, pos
        pnl = ((exit_px - entry_px) if pos == 1 else (entry_px - exit_px)) * qty * point_value \
            - COMMISSION * 2.0 * qty
        realized += pnl
        trades.append((entry_i, t, "long" if pos == 1 else "short", entry_px, exit_px, stop, tgt, pnl, reason))
        pos = 0

    for t in range(n):
        if sday[t] != day:
            day, entries_today = sday[t], 0
        if t >= 1:
            s = t - 1
            # Flat signal from the previous candle: exit at this open.
            if (pos == 1 and lx[s]) or (pos == -1 and sx[s]):
                close_trade(t, o[t] - slip if pos == 1 else o[t] + slip, "flat")
            # Entry from the previous candle's signal, same trading day, one per day.
            if pos == 0 and (le[s] or se[s]) and sday[s] == day and entries_today < 1:
                d = 1 if le[s] else -1
                entry_px = o[t] + slip if d == 1 else o[t] - slip
                stop = _on_tick(entry_px - d * sdist[s], tick_size)
                tgt = _on_tick(entry_px + d * tdist[s], tick_size)
                pos, entry_i = d, t
                entries_today += 1
        if pos != 0:
            hit_stop = l[t] <= stop if pos == 1 else h[t] >= stop
            hit_tgt = h[t] >= tgt if pos == 1 else l[t] <= tgt
            done = False
            if hit_stop and hit_tgt and win is not None and win[1][t] > win[0][t]:
                so, sh, sl, sc = win[2:]
                for k in range(win[0][t], win[1][t]):
                    hs = sl[k] <= stop if pos == 1 else sh[k] >= stop
                    ht = sh[k] >= tgt if pos == 1 else sl[k] <= tgt
                    if hs and ht:
                        if (so[k] <= stop) if pos == 1 else (so[k] >= stop):
                            ht = False
                        elif (so[k] >= tgt) if pos == 1 else (so[k] <= tgt):
                            hs = False
                        elif (sc[k] >= so[k]) == (pos == 1):
                            ht = False  # up 1-min candle: low (long stop) came first
                        else:
                            hs = False
                    if hs:
                        lvl = min(stop, so[k]) if pos == 1 else max(stop, so[k])
                        close_trade(t, lvl - slip if pos == 1 else lvl + slip, "stop")
                        done = True
                        break
                    if ht:
                        close_trade(t, max(tgt, so[k]) if pos == 1 else min(tgt, so[k]), "target")
                        done = True
                        break
            if not done and (hit_stop or hit_tgt):
                if hit_stop and hit_tgt:
                    if (o[t] >= tgt) if pos == 1 else (o[t] <= tgt):
                        hit_stop = (o[t] <= stop) if pos == 1 else (o[t] >= stop)
                    hit_tgt = not hit_stop  # otherwise the stop counts first
                if hit_stop:
                    lvl = min(stop, o[t]) if pos == 1 else max(stop, o[t])
                    close_trade(t, lvl - slip if pos == 1 else lvl + slip, "stop")
                else:
                    close_trade(t, max(tgt, o[t]) if pos == 1 else min(tgt, o[t]), "target")
        unreal = ((c[t] - entry_px) if pos == 1 else (entry_px - c[t])) * qty * point_value if pos else 0.0
        equity[t] = START_EQUITY + realized + unreal
    if pos != 0:
        close_trade(n - 1, c[n - 1] - slip if pos == 1 else c[n - 1] + slip, "end of data")
        equity[n - 1] = START_EQUITY + realized
    return trades, equity


def summary(trades, equity):
    pnl = np.array([t[7] for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl < 0]
    max_dd = abs(float((equity - np.maximum.accumulate(equity)).min())) if len(equity) else 0.0
    return {"trades": len(pnl), "net": float(pnl.sum()), "max_dd": max_dd,
            "win_rate": len(wins) / max(len(pnl), 1),
            "profit_factor": float(wins.sum()) / float(-losses.sum()) if len(losses) else float("inf")}


def main(path, sub_path=None, out=None):
    df = load_candles(path)
    sub = load_candles(sub_path) if sub_path else None
    trades, equity = backtest(df, signals(df), sub)
    out = out or Path(path).with_name("bellows_trades.csv")
    t = df.index
    pd.DataFrame([(t[e], t[x], side, ep, xp, sp, tp, pnl, r) for e, x, side, ep, xp, sp, tp, pnl, r in trades],
                 columns=["entry_time", "exit_time", "side", "entry_price", "exit_price", "stop_price",
                          "target_price", "pnl", "exit_reason"]).to_csv(out, index=False)
    s = summary(trades, equity)
    print(f"Bellows | {len(df):,} candles | {s['trades']:,} trades | win {100 * s['win_rate']:.1f}% | "
          f"net ${s['net']:,.0f} | profit factor {s['profit_factor']:.2f} | "
          f"max drawdown ${s['max_dd']:,.0f} | trades in {out}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        sys.exit("usage: python bellows.py bars_5m.csv [bars_1m.csv]")
    main(*sys.argv[1:3])
