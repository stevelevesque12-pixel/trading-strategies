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
    expiry_bars: int = 3  # "x": the limit must fill within this many candles of the latest close beyond both MAs
    track_ma: bool = False  # move the working limit to each new bar's 20 SMA
    cancel_on_flip: bool = False  # pull an unfilled limit once the 9/20 order flips (default: keep it until x expires)
    tick_size: float = 0.01
    point_value: float = 100.0
    commission_rt: float = 1.50  # $ per contract round trip
    stop_slip_ticks: int = 1
    session_gap_hours: Optional[float] = None  # flatten / cancel before a gap this long (None: hold over weekends)
    hours: Optional[tuple] = None  # (start, end) "HH:MM" ET window for new signals
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
    out["atr"] = tr.ewm(alpha=1 / p.atr_len, adjust=False).mean()  # Wilder / TradingView ta.atr
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
        sig[~((mins >= s) & (mins < e))] = 0
    return sig


def backtest(fine: pd.DataFrame, rule: str, p: Params) -> List[Trade]:
    sig_bars = indicators(resample(fine, rule), p)
    sig = signals(sig_bars, p)
    period = pd.Timedelta(rule)
    gap = pd.Timedelta(hours=p.session_gap_hours) if p.session_gap_hours is not None else None

    # Map each signal bar's close time -> its values, to be acted on from the next fine bar.
    sb = sig_bars.assign(sig=sig)
    sb_close = {ts + period: row for ts, row in zip(sb.index, sb.itertuples())}

    trades: List[Trade] = []
    pending = None  # dict(direction, limit, stop, target, expires, signal_time)
    pos: Optional[Trade] = None
    idx = fine.index
    o, h, l, c = (fine[k].to_numpy() for k in ("open", "high", "low", "close"))
    slip = p.stop_slip_ticks * p.tick_size

    def close_trade(t: Trade, when, price, reason):
        t.exit_time, t.exit, t.reason = when, price, reason
        pts = (price - t.entry) * t.direction
        t.pnl_dollars = pts * p.point_value - p.commission_rt
        t.r_multiple = pts / abs(t.entry - t.stop)
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
                if (p.cancel_on_flip and not still_valid) or ts >= pending["expires"]:
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
                    expires=ts + p.expiry_bars * period,
                    signal_time=row.Index,
                )

        def check_exit(filled_now: bool):
            nonlocal pos
            d = pos.direction
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

        # An open trade's stop/target is checked before a same-bar reversal fill (conservative).
        if pos is not None:
            check_exit(False)

        if pending is not None and (pos is None or pending["direction"] == -pos.direction):
            d, lim = pending["direction"], pending["limit"]
            hit = h[i] >= lim if d == -1 else l[i] <= lim
            if hit:
                fill = max(o[i], lim) if d == -1 else min(o[i], lim)  # gap through -> better price
                if pos is not None:
                    close_trade(pos, ts, fill, "reverse")
                pos = Trade(d, pending["signal_time"], ts, fill, pending["stop"], pending["target"])
                pending = None
                check_exit(True)

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
