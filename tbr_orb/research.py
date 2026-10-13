"""
Python replica of tradingview/tbr_orb_strategy.pine, used to research its
defaults on real data. Mirrors the "Time Range ORB Detector [34ematrader]"
indicator's signal styles (Breakout / Fade / Breakout-then-reversal / NS ORB)
and adds the strategy's stop/target/time-exit handling.

Fill model (deliberately a bit conservative):
  - market entries at the signal bar's close + 1 tick slippage;
  - stops fill at the stop price - 1 tick; targets fill exactly at the limit;
  - if a bar touches both stop and target, the stop wins;
  - NS limit entries need price to trade 1 tick THROUGH the limit, and on the
    fill bar itself only the stop can be hit (the target needs a close beyond);
  - commission is charged per round turn.

Usage:
    python -m tbr_orb.research                       # default config, NQ 1m
    python -m tbr_orb.research --window ORB15 --target-r 2 --side long
"""

import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WINDOWS = {
    "TBR812": (8 * 60 + 12, 9 * 60 + 12),
    "ORB5": (570, 575),
    "ORB15": (570, 585),
    "ORB30": (570, 600),
    "IB": (570, 630),
    "LDN": (72, 132),
}

DEFAULT_DATA = "sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet"


@dataclass
class Config:
    window: Tuple[int, int] = WINDOWS["ORB30"]
    style: str = "breakout"  # breakout | fade | breakout_rev | ns
    confirm: str = "sweep"  # sweep | reclaim | opposing  (non-NS styles)
    stop_mode: str = "range"  # range | atr
    stop_frac: float = 1.0  # range mode: long stop = high - frac * range
    stop_atr: float = 0.5
    target_r: Optional[float] = 3.0  # None = hold to the time exit
    target_atr: Optional[float] = None  # if set, overrides target_r
    cutoff: int = 14 * 60 + 30  # no new entries at/after
    flatten: int = 15 * 60 + 55
    ns_entry: float = 0.125
    ns_sl: float = 0.95
    ns_tp_r: float = 0.4
    ns_full_body: bool = True
    max_range_pct: float = 0.0  # 0 = off
    tick: float = 0.25
    point_value: float = 20.0
    commission_rt: float = 5.0  # $ per contract round turn


@dataclass
class Trade:
    date: object
    direction: int
    entry: float
    stop: float
    exit: float
    reason: str

    def points(self, cfg: Config) -> float:
        return (self.exit - self.entry) * self.direction - cfg.commission_rt / cfg.point_value

    def risk(self) -> float:
        return abs(self.entry - self.stop)


def load_days(path: str, timeframe: Optional[str] = None, atr_len: int = 14) -> Dict[object, tuple]:
    """Load bars into {date: (open, high, low, close, atr, minute_of_day)}, NY time."""
    df = pd.read_parquet(path) if path.endswith(".parquet") else pd.read_csv(path, index_col=0, parse_dates=True)
    df = df.tz_convert("America/New_York") if df.index.tz is not None else df.tz_localize("America/New_York")
    if timeframe:
        df = df.resample(timeframe, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
        ).dropna()
    prev_close = df["close"].shift()
    tr = np.maximum(df["high"] - df["low"],
                    np.maximum((df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()))
    df["atr"] = tr.ewm(alpha=1 / atr_len, adjust=False).mean()  # Wilder/RMA, like ta.atr
    df["mod"] = df.index.hour * 60 + df.index.minute
    df = df[df.index.dayofweek < 5]
    return {
        d: (g["open"].values, g["high"].values, g["low"].values, g["close"].values, g["atr"].values, g["mod"].values)
        for d, g in df.groupby(df.index.date)
    }


def _manage(arr, j0, d, entry, stop, target, cfg, date, fill_bar=False, until=None) -> Trade:
    o, h, l, c, _, m = arr
    for j in range(j0, len(c)):
        if m[j] >= cfg.flatten:
            return Trade(date, d, entry, stop, c[j], "time")
        if until is not None and j >= until:
            return Trade(date, d, entry, stop, c[j], "reverse")
        hit_stop = l[j] <= stop if d > 0 else h[j] >= stop
        if target is None:
            hit_tgt = False
        elif fill_bar and j == j0:
            hit_tgt = c[j] >= target if d > 0 else c[j] <= target
        else:
            hit_tgt = h[j] >= target if d > 0 else l[j] <= target
        if hit_stop:
            return Trade(date, d, entry, stop, stop - d * cfg.tick, "stop")
        if hit_tgt:
            return Trade(date, d, entry, stop, target, "target")
    return Trade(date, d, entry, stop, c[-1], "end")


def day_trades(arr, cfg: Config, date=None) -> List[Trade]:
    o, h, l, c, atr, m = arr
    s, e = cfg.window
    build = (m >= s) & (m < e)
    if not build.any():
        return []
    hi, lo = h[build].max(), l[build].min()
    rng = hi - lo
    if rng <= 0:
        return []
    if cfg.max_range_pct and rng / c[build][-1] * 100 > cfg.max_range_pct:
        return []
    sig_bars = np.where((m >= e) & (m < cfg.cutoff))[0]

    if cfg.style == "ns":
        for i in sig_bars:
            up = c[i] > hi and (not cfg.ns_full_body or o[i] > hi)
            dn = c[i] < lo and (not cfg.ns_full_body or o[i] < lo)
            if up or dn:
                d = 1 if up else -1
                break
        else:
            return []
        lim = hi - cfg.ns_entry * rng if d > 0 else lo + cfg.ns_entry * rng
        stop = hi - cfg.ns_sl * rng if d > 0 else lo + cfg.ns_sl * rng
        tgt = lim + d * cfg.ns_tp_r * abs(lim - stop)
        for j in range(i + 1, len(c)):
            if m[j] >= cfg.cutoff:
                return []
            if (d > 0 and l[j] <= lim - cfg.tick) or (d < 0 and h[j] >= lim + cfg.tick):
                px = min(o[j], lim) if d > 0 else max(o[j], lim)
                return [_manage(arr, j, d, px, stop, tgt, cfg, date, fill_bar=True)]
        return []

    first, sweep, second = 0, None, None
    swing_hi = swing_lo = None
    signals: List[Tuple[int, int]] = []
    fade = cfg.style == "fade"
    for i in sig_bars:
        up, dn = c[i] > hi, c[i] < lo
        sweep_now = False
        if first == 0:
            if up != dn:
                first, sweep, sweep_now = (1 if up else -1), i, True
                swing_hi, swing_lo = h[i], l[i]
            else:
                continue
        else:
            swing_hi, swing_lo = max(swing_hi, h[i]), min(swing_lo, l[i])
            if second is None and ((first == 1 and dn) or (first == -1 and up)):
                second = i
        if not signals:
            raw = -first if fade else first
            if cfg.confirm == "sweep":
                ok = sweep_now
            elif cfg.confirm == "reclaim":
                ok = i > sweep and lo <= c[i] <= hi
            else:
                ok = i > sweep and (c[i] > o[i] if raw > 0 else c[i] < o[i])
            if ok:
                signals.append((i, raw))
        elif cfg.style in ("breakout_rev", "fade") and second == i and len(signals) == 1:
            # indicator's REV signal goes in the direction of the second break
            signals.append((i, 1 if up else -1))

    trades = []
    for k, (i, d) in enumerate(signals):
        nxt = signals[k + 1][0] if k + 1 < len(signals) else None
        entry = c[i] + d * cfg.tick
        atr_stop = entry - d * cfg.stop_atr * atr[i]
        stop = (hi - cfg.stop_frac * rng if d > 0 else lo + cfg.stop_frac * rng) if cfg.stop_mode == "range" else atr_stop
        if (stop - entry) * d >= 0:
            stop = atr_stop
        risk = abs(entry - stop)
        if cfg.target_atr:
            tgt = entry + d * cfg.target_atr * atr[i]
        elif cfg.target_r:
            tgt = entry + d * cfg.target_r * risk
        else:
            tgt = None
        trades.append(_manage(arr, i + 1, d, entry, stop, tgt, cfg, date, until=nxt))
    return trades


def run(days: Dict[object, tuple], cfg: Config) -> pd.DataFrame:
    rows = []
    for d, arr in days.items():
        for t in day_trades(arr, cfg, d):
            rows.append(dict(date=t.date, dir=t.direction, entry=t.entry, stop=t.stop, exit=t.exit,
                             reason=t.reason, pts=t.points(cfg), risk=t.risk()))
    return pd.DataFrame(rows, columns=["date", "dir", "entry", "stop", "exit", "reason", "pts", "risk"])


def summarize(trades: pd.DataFrame, point_value: float) -> dict:
    if trades.empty:
        return dict(n=0)
    pnl = trades["pts"] * point_value
    eq = pnl.cumsum()
    gross_win, gross_loss = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    return dict(
        n=len(trades),
        win_rate=round((pnl > 0).mean(), 3),
        profit_factor=round(gross_win / gross_loss, 2) if gross_loss else float("inf"),
        avg_r=round((trades["pts"] / trades["risk"]).mean(), 3),
        net_usd=round(pnl.sum()),
        max_dd_usd=round((eq - eq.cummax()).min()),
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=DEFAULT_DATA)
    ap.add_argument("--timeframe", default=None, help="resample, e.g. 5min (default: use as-is)")
    ap.add_argument("--window", default="ORB30", choices=list(WINDOWS))
    ap.add_argument("--style", default="breakout", choices=["breakout", "fade", "breakout_rev", "ns"])
    ap.add_argument("--target-r", type=float, default=3.0, help="0 = hold to the time exit")
    ap.add_argument("--max-range-pct", type=float, default=0.0)
    ap.add_argument("--side", default="both", choices=["both", "long", "short"])
    ap.add_argument("--point-value", type=float, default=20.0)
    args = ap.parse_args()

    cfg = Config(window=WINDOWS[args.window], style=args.style, target_r=args.target_r or None,
                 max_range_pct=args.max_range_pct, point_value=args.point_value)
    t = run(load_days(args.data, args.timeframe), cfg)
    if args.side != "both":
        t = t[t["dir"] == (1 if args.side == "long" else -1)]
    t["year"] = pd.to_datetime(t["date"]).dt.year
    print("ALL ", summarize(t, cfg.point_value))
    for y, g in t.groupby("year"):
        print(y, summarize(g, cfg.point_value))


if __name__ == "__main__":
    main()
