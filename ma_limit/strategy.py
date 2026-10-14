"""9/20 SMA "close through both, limit back at the 20" strategy.

Rules (from the TradingView screenshot):
  Short: 9 SMA above 20 SMA, a bar closes below both -> sell limit at the 20 SMA.
  Long:  9 SMA below 20 SMA, a bar closes above both -> buy limit at the 20 SMA.
  Stop = 2 x ATR(14) from the limit price, target = 1:1.

Signals come from signal-timeframe bars (e.g. 30m); fills and exits are walked on
finer bars (e.g. 15m or 5m) so stop/target ordering inside a 30m bar is resolved
where the data allows. When a single fine bar could have hit both levels, the
stop is assumed to come first, and on the bar that fills the order only the stop
is checked (we don't know if the target-side extreme came before the fill).
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class Params:
    fast: int = 9
    slow: int = 20
    atr_len: int = 14
    atr_mult: float = 2.0
    rr: float = 1.0
    expiry_bars: int = 5  # "x": the limit must fill within this many candles of the latest close beyond both MAs
    track_ma: bool = False  # move the working limit to each new bar's 20 SMA
    # Enter at the 20 SMA of the candle that touches it, with that candle's ATR for the stop --
    # how trades are marked by hand on the chart. Uses the touching candle's close (the MA
    # value live trading would chase intrabar ends there), so it's a slight look-ahead.
    live_ma: bool = True
    # With live_ma, "realtime" places the limit where price meets the forming candle's live 20 SMA
    # (= mean of the previous 19 closes, known at the candle's open) and sizes the stop from the
    # last closed candle's ATR -- no look-ahead. "chart" uses the touching candle's closed values.
    live_mode: str = "chart"
    cancel_on_flip: bool = False  # pull an unfilled limit once the 9/20 order flips (default: keep it until x expires)
    tick_size: float = 0.01
    point_value: float = 100.0
    commission_rt: float = 2.12  # $ per contract round trip (Tradeify MCL; see failed2s/instruments.py)
    stop_slip_ticks: int = 1
    session_gap_hours: Optional[float] = None  # flatten / cancel before a gap this long (None: hold over weekends)
    hours: Optional[tuple] = None  # (start, end) "HH:MM" ET: only setup candles opening in this window count
    # "HH:MM" ET: close any open trade at the open of the first bar at/after this time and cancel working orders
    # (prop firms like Tradeify close everything before the 17:00 daily break). None: hold overnight.
    flatten_at: Optional[str] = None
    # Intraday refinements (all off by default):
    be_at: Optional[float] = None  # move the stop to the entry once price has gone this many R in favour
    vol_min: Optional[float] = None  # only take setups when ATR / its vol_len-bar average is at least this
    vol_len: int = 200
    sides: str = "both"  # "both", "long" or "short"
    skip_days: tuple = ()  # weekdays to skip setups on, 0 = Monday
    # Side swapping on an opposite setup while in a trade:
    #   None     - ignore it (stay in the trade until stop/target)
    #   "limit"  - work the opposite limit at the 20 SMA; its fill closes the trade and flips
    #   "market" - close at market on the signal, then work the opposite limit as usual
    reverse: Optional[str] = None  # default: no side swapping; ride to SL/TP, then look for the next setup


@dataclass
class Trade:
    direction: int
    signal_time: pd.Timestamp
    entry_time: pd.Timestamp
    entry: float
    stop: float
    target: float
    exit_time: Optional[pd.Timestamp] = None
    exit: Optional[float] = None
    reason: str = ""
    pnl_dollars: float = 0.0
    r_multiple: float = 0.0
    mae: float = 0.0  # worst adverse move while open, in points (bar extremes; capped at the stop)
    risk0: float = 0.0  # initial stop distance in points (the stop can move to breakeven)


def resample(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Bar-open-labelled OHLCV resample (matches the source data's convention)."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    if "volume" in df:
        agg["volume"] = "sum"
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(subset=["open"])


def indicators(bars: pd.DataFrame, p: Params) -> pd.DataFrame:
    out = bars.copy()
    out["fast"] = out["close"].rolling(p.fast).mean()
    out["slow"] = out["close"].rolling(p.slow).mean()
    prev_close = out["close"].shift()
    tr = pd.concat(
        [out["high"] - out["low"], (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()], axis=1
    ).max(axis=1)
    out["atr"] = _rma(tr.to_numpy(), p.atr_len)  # Wilder / TradingView ta.atr
    out["touch"] = out["close"].shift().rolling(p.slow - 1).mean()  # price that meets the live 20 SMA
    out["atr_prev"] = out["atr"].shift()
    out["atr_ratio"] = out["atr"] / out["atr"].rolling(p.vol_len, min_periods=20).mean()
    return out


def _rma(x: np.ndarray, n: int) -> np.ndarray:
    """TradingView ta.rma: seeded with the SMA of the first n values, then Wilder smoothing."""
    out = np.full(len(x), np.nan)
    if len(x) < n:
        return out
    out[n - 1] = x[:n].mean()
    for i in range(n, len(x)):
        out[i] = out[i - 1] + (x[i] - out[i - 1]) / n
    return out


def signals(bars: pd.DataFrame, p: Params) -> pd.Series:
    """+1 long setup, -1 short setup, 0 none, evaluated on each bar's close."""
    lo = np.minimum(bars["fast"], bars["slow"])
    hi = np.maximum(bars["fast"], bars["slow"])
    short = (bars["fast"] > bars["slow"]) & (bars["close"] < lo)
    long_ = (bars["fast"] < bars["slow"]) & (bars["close"] > hi)
    sig = pd.Series(0, index=bars.index)
    sig[short] = -1
    sig[long_] = 1
    if p.hours:
        t = bars.index  # bar-open timestamps
        mins = t.hour * 60 + t.minute
        s, e = (int(x[:2]) * 60 + int(x[3:]) for x in p.hours)
        inside = (mins >= s) & (mins < e) if s < e else (mins >= s) | (mins < e)  # e.g. 18:00-02:00 wraps midnight
        sig[~inside] = 0
    if p.sides != "both":
        sig[sig == (-1 if p.sides == "long" else 1)] = 0
    if p.skip_days:
        sig[bars.index.weekday.isin(p.skip_days)] = 0
    if p.vol_min is not None and "atr_ratio" in bars:
        sig[~(bars["atr_ratio"] >= p.vol_min)] = 0
    return sig


def backtest(fine: pd.DataFrame, rule: str, p: Params) -> List[Trade]:
    sig_bars = indicators(resample(fine, rule), p)
    sig = signals(sig_bars, p)
    period = pd.Timedelta(rule)
    gap = pd.Timedelta(hours=p.session_gap_hours) if p.session_gap_hours is not None else None
    flat_min = int(p.flatten_at[:2]) * 60 + int(p.flatten_at[3:]) if p.flatten_at else None

    # Map each signal bar's close time -> its values, to be acted on from the next fine bar.
    sb = sig_bars.assign(sig=sig)
    sb_close = {ts + period: row for ts, row in zip(sb.index, sb.itertuples())}
    sb_open = dict(zip(sb.index, sb.itertuples()))

    trades: List[Trade] = []
    pending = None  # dict(direction, limit, stop, target, bars_left, signal_time)
    pos: Optional[Trade] = None
    idx = fine.index
    o, h, l, c = (fine[k].to_numpy() for k in ("open", "high", "low", "close"))
    slip = p.stop_slip_ticks * p.tick_size

    def close_trade(t: Trade, when, price, reason):
        t.exit_time, t.exit, t.reason = when, price, reason
        pts = (price - t.entry) * t.direction
        t.pnl_dollars = pts * p.point_value - p.commission_rt
        t.r_multiple = pts / (t.risk0 or abs(t.entry - t.stop))
        trades.append(t)

    for i in range(len(idx)):
        ts = idx[i]
        # Signal-bar close events happen at the open of this fine bar.
        row = sb_close.get(ts)
        if row is not None and not np.isnan(row.atr) and not np.isnan(row.slow):
            if pending is not None:
                still_valid = (pending["direction"] == -1 and row.fast > row.slow) or (
                    pending["direction"] == 1 and row.fast < row.slow
                )
                pending["bars_left"] -= 1
                if (p.cancel_on_flip and not still_valid) or pending["bars_left"] <= 0:
                    pending = None
                elif p.track_ma:
                    d = pending["direction"]
                    risk = p.atr_mult * row.atr
                    pending.update(limit=row.slow, stop=row.slow - d * risk, target=row.slow + d * p.rr * risk)
            opposite = pos is not None and p.reverse and row.sig == -pos.direction
            if opposite and p.reverse == "market":
                close_trade(pos, ts, o[i], "reverse")
                pos = None
            if row.sig != 0 and (pos is None or opposite):
                d = int(row.sig)
                risk = p.atr_mult * row.atr
                pending = dict(
                    direction=d,
                    limit=row.slow,
                    stop=row.slow - d * risk,
                    target=row.slow + d * p.rr * risk,
                    bars_left=p.expiry_bars,
                    signal_time=row.Index,
                )

        def check_exit(filled_now: bool):
            nonlocal pos
            d = pos.direction
            adverse = (h[i] - pos.entry) if d == -1 else (pos.entry - l[i])
            pos.mae = max(pos.mae, min(adverse, (pos.risk0 or abs(pos.entry - pos.stop)) + slip))
            stop_hit = h[i] >= pos.stop if d == -1 else l[i] <= pos.stop
            tgt_hit = (l[i] <= pos.target if d == -1 else h[i] >= pos.target) and not filled_now
            if stop_hit:
                px = max(o[i], pos.stop) if d == -1 else min(o[i], pos.stop)  # gap through stop -> worse fill
                close_trade(pos, ts, px - d * slip, "stop")
                pos = None
            elif tgt_hit:
                px = min(o[i], pos.target) if d == -1 else max(o[i], pos.target)
                close_trade(pos, ts, px, "target")
                pos = None
            elif p.be_at is not None:
                # Breakeven from the next bar on, once this bar reached +be_at R (stop/target checked first).
                fav = (pos.entry - l[i]) if d == -1 else (h[i] - pos.entry)
                if fav >= p.be_at * pos.risk0 and (pos.stop - pos.entry) * d < 0:
                    pos.stop = pos.entry

        # Prop-firm end of day: out at this bar's open, no working orders until the evening reopen.
        if flat_min is not None and flat_min <= ts.hour * 60 + ts.minute < 18 * 60:
            if pos is not None:
                pos.mae = max(pos.mae, (o[i] - pos.entry) * -pos.direction)
                close_trade(pos, ts, o[i], "eod_flat")
                pos = None
            pending = None

        # An open trade's stop/target is checked before a same-bar reversal fill (conservative).
        if pos is not None:
            check_exit(False)

        if pending is not None and (pos is None or pending["direction"] == -pos.direction):
            d, lim, stop, target = pending["direction"], pending["limit"], pending["stop"], pending["target"]
            cur = sb_open.get(ts.floor(period)) if p.live_ma else None
            if cur is not None and not np.isnan(cur.slow):
                ma, atr = (cur.touch, cur.atr_prev) if p.live_mode == "realtime" else (cur.slow, cur.atr)
                risk = p.atr_mult * atr
                lim, stop, target = ma, ma - d * risk, ma + d * p.rr * risk
            # Like the chart's position tool: entry and stop distance on the tick grid, target mirrors the stop.
            tick = lambda x: round(round(x / p.tick_size) * p.tick_size, 10)
            lim, risk = tick(lim), tick(abs(lim - stop))
            stop, target = round(lim - d * risk, 10), round(lim + d * p.rr * risk, 10)
            hit = h[i] >= lim if d == -1 else l[i] <= lim
            if hit:
                if p.live_ma:
                    fill = lim  # marked at the MA, even when the candle opened through it
                else:
                    fill = max(o[i], lim) if d == -1 else min(o[i], lim)  # gap through -> better price
                if pos is not None:
                    close_trade(pos, ts, fill, "reverse")
                pos = Trade(d, pending["signal_time"], ts, fill, stop, target, risk0=abs(fill - stop))
                pending = None
                check_exit(True)

        # Early (holiday) close: the session stops before the flatten time, so get out on this bar's close.
        if flat_min is not None and i + 1 < len(idx):
            nxt = idx[i + 1]
            ends_early = nxt - ts > period and (nxt.date() != ts.date() or nxt.hour * 60 + nxt.minute >= 18 * 60)
            if ends_early and ts.hour * 60 + ts.minute < flat_min and ts.hour < 17:
                if pos is not None:
                    pos.mae = max(pos.mae, (c[i] - pos.entry) * -pos.direction)
                    close_trade(pos, ts, c[i], "eod_flat")
                    pos = None
                pending = None

        # Flatten / cancel ahead of a long data gap (weekend / holiday).
        if gap is not None and i + 1 < len(idx) and idx[i + 1] - ts > gap:
            if pos is not None:
                close_trade(pos, ts, c[i], "gap_flat")
                pos = None
            pending = None

    return trades


def summarize(trades: List[Trade]) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = np.array([t.pnl_dollars for t in trades])
    eq = pnl.cumsum()
    dd = (np.maximum.accumulate(np.r_[0, eq])[1:] - eq).max()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    return {
        "trades": len(trades),
        "win%": round(100 * len(wins) / len(pnl), 1),
        "pnl$": round(pnl.sum(), 0),
        "pf": round(wins.sum() / -losses.sum(), 2) if losses.sum() else float("inf"),
        "avgR": round(np.mean([t.r_multiple for t in trades]), 3),
        "exp$": round(pnl.mean(), 2),
        "maxDD$": round(dd, 0),
    }
