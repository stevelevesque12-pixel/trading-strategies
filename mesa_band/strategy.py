"""
Python port of tradingview/mesa_phase_band_trend.pine for backtesting.

Mirrors the Pine script bar for bar:
  - the same MESA/Hilbert phase-adaptive band indicator and regime flips,
  - the same stop/target/sizing rules and prop-firm guards (daily loss
    limit, daily profit cap, trailing drawdown with lock, profit target,
    max trades per day, entry window, flatten time),
  - TradingView broker-emulator fill semantics: orders decided on a bar's
    close fill at the next bar's open; `immediately=true` closes fill at the
    current close; stop/limit exits are checked intrabar along the
    open -> nearer extreme -> farther extreme path; slippage is applied
    to market and stop fills, not to limit fills.

Two ways to run it (see run.py):
  - "continuous": daily guards only (no drawdown halt / profit target), to
    measure the raw edge over the whole data set.
  - "evaluation": one prop evaluation from a given start day on a fresh
    account. It ends PASSED when the profit target is reached, FAILED when
    the firm's trailing max-loss floor is touched intrabar, or TIMEOUT after
    a maximum number of trading days. run.py starts one every week to get
    many independent samples.
"""

from dataclasses import dataclass, field
from datetime import timedelta
from typing import List, Optional

import numpy as np
import pandas as pd

NY = "America/New_York"


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick: float
    point_value: float
    commission_per_side: float = 0.62  # typical micro commission + exchange fees at prop firms


INSTRUMENTS = {
    "MES": Instrument("MES", 0.25, 5.0),
    "MNQ": Instrument("MNQ", 0.25, 2.0),
    "MGC": Instrument("MGC", 0.10, 10.0),
    "MCL": Instrument("MCL", 0.01, 100.0),
}


@dataclass
class Params:
    # indicator
    fast_limit: float = 0.50
    slow_limit: float = 0.05
    trend_length: int = 20
    vol_length: int = 20
    active_mult: float = 1.5
    slow_mult: float = 3.0
    # trade management
    direction: str = "Both"  # "Both" | "Long only" | "Short only"
    stop_mode: str = "ATR"  # "ATR" | "Trend line" | "Opposite band"
    atr_length: int = 14
    atr_stop_mult: float = 2.0
    stop_buf_ticks: int = 2
    min_stop_ticks: int = 8
    target_r: float = 2.0
    trail_on_trend: bool = False
    # sizing
    risk_per_trade: float = 250.0
    max_contracts: int = 50
    round_turn_cost: Optional[float] = None  # None -> 2 * commission + 2 ticks of slippage
    slippage_ticks: int = 1
    # prop rules
    start_balance: float = 50_000.0
    daily_loss_limit: float = 900.0
    daily_profit_cap: float = 0.0
    max_drawdown: float = 2_000.0
    dd_mode: str = "End of day"  # "End of day" | "Intraday"
    dd_lock_offset: float = 0.0
    dd_buffer: float = 200.0
    profit_target: float = 3_000.0
    max_trades_day: int = 4
    # session (America/New_York)
    entry_start: int = 930  # HHMM, bar OPEN time must be in [start, end)
    entry_end: int = 1530
    flatten_hhmm: int = 1555


# ---------------------------------------------------------------------------
# Indicator
# ---------------------------------------------------------------------------

def compute_indicator(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)
    src = (high + low) / 2.0
    n = len(df)

    smooth = np.zeros(n)
    detr = np.zeros(n)
    q1 = np.zeros(n)
    i1 = np.zeros(n)
    i2 = np.zeros(n)
    q2 = np.zeros(n)
    re = np.zeros(n)
    im = np.zeros(n)
    period = np.zeros(n)
    phase = np.zeros(n)
    activity = np.zeros(n)

    def hb(a, i, pf):  # Hilbert FIR with nz() on missing history
        return (0.0962 * a[i]
                + 0.5769 * (a[i - 2] if i >= 2 else 0.0)
                - 0.5769 * (a[i - 4] if i >= 4 else 0.0)
                - 0.0962 * (a[i - 6] if i >= 6 else 0.0)) * pf

    rng = max(p.fast_limit - p.slow_limit, 0.0001)
    for i in range(n):
        s1 = src[i - 1] if i >= 1 else src[i]
        s2 = src[i - 2] if i >= 2 else src[i]
        s3 = src[i - 3] if i >= 3 else src[i]
        smooth[i] = (4.0 * src[i] + 3.0 * s1 + 2.0 * s2 + s3) / 10.0
        prev_period = period[i - 1] if i >= 1 else 10.0
        pf = 0.075 * prev_period + 0.54

        detr[i] = hb(smooth, i, pf)
        q1[i] = hb(detr, i, pf)
        i1[i] = detr[i - 3] if i >= 3 else 0.0
        jI = hb(i1, i, pf)
        jQ = hb(q1, i, pf)

        i2p = i2[i - 1] if i >= 1 else 0.0
        q2p = q2[i - 1] if i >= 1 else 0.0
        i2[i] = 0.2 * (i1[i] - jQ) + 0.8 * i2p
        q2[i] = 0.2 * (q1[i] + jI) + 0.8 * q2p

        re_raw = i2[i] * i2p + q2[i] * q2p
        im_raw = i2[i] * q2p - q2[i] * i2p
        re[i] = 0.2 * re_raw + 0.8 * (re[i - 1] if i >= 1 else 0.0)
        im[i] = 0.2 * im_raw + 0.8 * (im[i - 1] if i >= 1 else 0.0)

        angle = np.degrees(np.arctan(im[i] / re[i])) if re[i] != 0.0 else 0.0
        raw_period = abs(360.0 / angle) if angle != 0.0 else prev_period
        lp = min(raw_period, 1.5 * prev_period)
        lp = max(lp, 0.67 * prev_period)
        lp = max(6.0, min(50.0, lp))
        period[i] = 0.2 * lp + 0.8 * prev_period

        prev_phase = phase[i - 1] if i >= 1 else 0.0
        phase[i] = np.degrees(np.arctan(q1[i] / i1[i])) if i1[i] != 0.0 else prev_phase
        delta = max(prev_phase - phase[i], 1.0)
        alpha = max(p.fast_limit / delta, p.slow_limit)
        activity[i] = max(0.0, min(1.0, (alpha - p.slow_limit) / rng))

    s = pd.Series(src, index=df.index)
    trend = s.ewm(span=p.trend_length, adjust=False).mean()
    ret = (s - s.shift(1)) / s.shift(1)
    ret = ret.fillna(0.0)
    vol = ret.rolling(p.vol_length).std(ddof=0).shift(1)
    mult = p.slow_mult - (p.slow_mult - p.active_mult) * activity
    upper = trend + mult * vol * trend
    lower = trend - mult * vol * trend

    prev_close = pd.Series(close, index=df.index).shift(1)
    tr = pd.concat([df["high"] - df["low"],
                    (df["high"] - prev_close).abs(),
                    (df["low"] - prev_close).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1.0 / p.atr_length, adjust=False).mean()

    sq = np.zeros(n, dtype=int)
    up = upper.to_numpy()
    lo = lower.to_numpy()
    for i in range(n):
        prev = sq[i - 1] if i >= 1 else 0
        if close[i] > up[i]:
            sq[i] = 1
        elif close[i] < lo[i]:
            sq[i] = -1
        else:
            sq[i] = prev
    prev_sq = np.concatenate([[0], sq[:-1]])

    out = pd.DataFrame(index=df.index)
    out["trend"] = trend
    out["upper"] = upper
    out["lower"] = lower
    out["atr"] = atr
    out["sq"] = sq
    out["long_flip"] = (sq == 1) & (prev_sq != 1)
    out["short_flip"] = (sq == -1) & (prev_sq != -1)
    return out


# ---------------------------------------------------------------------------
# Simulator
# ---------------------------------------------------------------------------

@dataclass
class Trade:
    direction: int
    qty: int
    entry_time: pd.Timestamp
    entry_price: float
    stop: float
    target: Optional[float]
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    reason: str = ""
    pnl: float = 0.0  # net of commission
    risk_usd: float = 0.0  # initial risk incl. costs at the fill price

    @property
    def r_multiple(self) -> float:
        return self.pnl / self.risk_usd if self.risk_usd > 0 else 0.0


@dataclass
class Attempt:
    start: object
    end: object = None
    result: str = "OPEN"
    days: int = 0
    trades: int = 0
    pnl: float = 0.0
    best_day: float = 0.0


@dataclass
class Result:
    trades: List[Trade] = field(default_factory=list)
    daily_pnl: pd.Series = None
    attempts: List[Attempt] = field(default_factory=list)


def _hhmm_to_min(hhmm: int) -> int:
    return (hhmm // 100) * 60 + hhmm % 100


def prepare(df: pd.DataFrame, p: Params) -> dict:
    """Indicator + calendar arrays, computed once and shared by every run over `df`."""
    ind = compute_indicator(df, p)

    idx_ny = df.index.tz_convert(NY)
    bar_delta = pd.Series(df.index).diff().mode().iloc[0]
    tf_min = bar_delta.total_seconds() / 60.0
    close_ny = idx_ny + bar_delta
    open_min = (idx_ny.hour * 60 + idx_ny.minute).to_numpy()
    close_min = (close_ny.hour * 60 + close_ny.minute).to_numpy()
    # CME trading day: 18:00 ET -> 17:00 ET next day.
    tday = (idx_ny + timedelta(hours=6)).normalize().tz_localize(None).to_numpy()

    o = df["open"].to_numpy(float)
    h = df["high"].to_numpy(float)
    l = df["low"].to_numpy(float)
    c = df["close"].to_numpy(float)
    trend = ind["trend"].to_numpy()
    upper = ind["upper"].to_numpy()
    lower = ind["lower"].to_numpy()
    atr = ind["atr"].to_numpy()
    lflip = ind["long_flip"].to_numpy()
    sflip = ind["short_flip"].to_numpy()
    # Last bar before a trading halt (daily break, holiday early close): the
    # equivalent of Pine's session.islastbar, which TradingView derives from
    # the exchange calendar. Any gap of 2+ bars counts.
    gaps = np.r_[np.diff(df.index.asi8), np.iinfo(np.int64).max]
    session_last = gaps > 2 * bar_delta.value
    return dict(tf_min=tf_min, open_min=open_min, close_min=close_min, tday=tday, o=o, h=h, l=l, c=c,
                session_last=session_last,
                trend=trend, upper=upper, lower=lower, atr=atr, lflip=lflip, sflip=sflip)


def run(df: pd.DataFrame, inst: Instrument, p: Params, mode: str = "continuous",
        start: int = 0, max_days: Optional[int] = None, prep: Optional[dict] = None) -> Result:
    """
    mode="continuous": whole series, daily guards only.
    mode="evaluation": ONE prop evaluation starting at bar `start` (should be
    the first bar of a trading day). Ends PASSED at the profit target, FAILED
    when the trailing max-loss floor is touched intrabar, or TIMEOUT after
    `max_days` trading days.
    """
    assert mode in ("continuous", "evaluation")
    evaluation = mode == "evaluation"
    pr = prep if prep is not None else prepare(df, p)
    tf_min, open_min, close_min, tday = pr["tf_min"], pr["open_min"], pr["close_min"], pr["tday"]
    o, h, l, c = pr["o"], pr["h"], pr["l"], pr["c"]
    trend, upper, lower, atr = pr["trend"], pr["upper"], pr["lower"], pr["atr"]
    lflip, sflip = pr["lflip"], pr["sflip"]
    session_last = pr["session_last"]

    pv = inst.point_value
    tick = inst.tick
    slip = p.slippage_ticks * tick
    comm = inst.commission_per_side
    rt_cost = p.round_turn_cost if p.round_turn_cost is not None else 2 * comm + 2 * slip * pv
    entry_lo, entry_hi = _hhmm_to_min(p.entry_start), _hhmm_to_min(p.entry_end)
    flat_min = _hhmm_to_min(p.flatten_hhmm)
    warmup = max(p.trend_length, p.vol_length) + 50
    min_stop_pts = p.min_stop_ticks * tick
    buf_pts = p.stop_buf_ticks * tick
    max_dd = p.max_drawdown if evaluation else 0.0
    profit_target = p.profit_target if evaluation else 0.0

    def rnd(x):
        return round(x / tick) * tick

    res = Result()

    # account state
    cash = p.start_balance
    prev_eq = p.start_balance
    day_start_eq = p.start_balance
    peak_eq = p.start_balance
    trades_today = 0
    halt_day = False
    halt_acct = False
    attempt: Optional[Attempt] = Attempt(start=tday[start]) if evaluation else None
    days_seen = 1
    last_i = start

    # position state
    pos: Optional[Trade] = None
    pending: Optional[tuple] = None  # ("close",) or ("enter", dir, qty, stop, target)

    daily = {}

    def fill_exit(t: Trade, price: float, time, reason: str):
        nonlocal cash
        pnl = (price - t.entry_price) * t.direction * t.qty * pv - comm * t.qty
        cash += pnl
        t.exit_price, t.exit_time, t.reason = price, time, reason
        t.pnl += pnl
        res.trades.append(t)
        if attempt is not None:
            attempt.trades += 1

    def fill_entry(direction, qty, price, time, st, tg):
        nonlocal cash
        cash -= comm * qty
        risk = (abs(price - st) * pv) * qty + rt_cost * qty
        return Trade(direction, qty, time, price, st, tg, pnl=-comm * qty, risk_usd=risk)

    for i in range(start, len(df)):
        t_i = df.index[i]
        new_day = i > start and tday[i] != tday[i - 1]
        if new_day:
            days_seen += 1
            if evaluation and max_days is not None and days_seen > max_days:
                attempt.result = "TIMEOUT"
                break
        last_i = i

        # ---- fills at this bar's open ----
        if pending is not None:
            kind = pending[0]
            if pos is not None and (kind == "close" or pending[1] != pos.direction):
                fill_exit(pos, o[i] - slip * pos.direction, t_i, "Regime flip")
                pos = None
            if kind == "enter" and pos is None:
                _, d, q, st, tg = pending
                pos = fill_entry(d, q, o[i] + slip * d, t_i, st, tg)
            pending = None

        # ---- intrabar stop / target ----
        worst_eq = None
        if pos is not None:
            d = pos.direction
            st, tg = pos.stop, pos.target
            adverse = l[i] if d > 0 else h[i]
            exit_px = reason = None
            if (o[i] - st) * d <= 0:
                exit_px, reason = o[i] - slip * d, "Stop"
            elif tg is not None and (o[i] - tg) * d >= 0:
                exit_px, reason = o[i], "Target"
            else:
                high_first = (h[i] - o[i]) < (o[i] - l[i])
                legs = ("h", "l") if high_first else ("l", "h")
                for leg in legs:
                    px = h[i] if leg == "h" else l[i]
                    fav = (leg == "h") == (d > 0)
                    if fav and tg is not None and (px - tg) * d >= 0:
                        exit_px, reason = tg, "Target"
                        break
                    if not fav and (px - st) * d <= 0:
                        exit_px, reason = st - slip * d, "Stop"
                        break
            # firm-side intrabar low-water mark (before the exit, if any)
            worst_px = adverse if reason != "Stop" else exit_px
            worst_eq = cash + (worst_px - pos.entry_price) * d * pos.qty * pv
            if exit_px is not None:
                fill_exit(pos, exit_px, t_i, reason)
                pos = None

        # ---- bar close: Pine logic ----
        open_pnl = (c[i] - pos.entry_price) * pos.direction * pos.qty * pv if pos else 0.0
        equity = cash + open_pnl

        if new_day:
            day_start_eq = prev_eq
            trades_today = 0
            halt_day = False
            if p.dd_mode == "End of day":
                peak_eq = max(peak_eq, prev_eq)
        if p.dd_mode == "Intraday":
            peak_eq = max(peak_eq, equity)

        dd_floor = min(peak_eq - max_dd, p.start_balance + p.dd_lock_offset) if max_dd > 0 else None
        daily_pnl = equity - day_start_eq

        flatten_rules = False
        reason = ""
        if not halt_acct:
            breached = dd_floor is not None and (
                equity <= dd_floor or (worst_eq is not None and worst_eq <= dd_floor))
            if breached:
                halt_acct, flatten_rules, reason = True, True, "Max drawdown"
                if attempt is not None:
                    attempt.result = "FAILED"
            elif profit_target > 0 and equity - p.start_balance >= profit_target:
                halt_acct, flatten_rules, reason = True, True, "Profit target"
                if attempt is not None:
                    attempt.result = "PASSED"
        if not halt_acct and not halt_day:
            if p.daily_loss_limit > 0 and daily_pnl <= -p.daily_loss_limit:
                halt_day, flatten_rules, reason = True, True, "Daily loss limit"
            elif p.daily_profit_cap > 0 and daily_pnl >= p.daily_profit_cap:
                halt_day, flatten_rules, reason = True, True, "Daily profit cap"

        flatten_now = (close_min[i] + tf_min > flat_min and close_min[i] < 18 * 60) or session_last[i]
        flattened = False
        if pos is not None and (flatten_rules or flatten_now):
            fill_exit(pos, c[i] - slip * pos.direction, t_i, reason or "Session flatten")
            pos = None
            pending = None
            flattened = True
            equity = cash

        in_window = entry_lo <= open_min[i] < entry_hi
        can_trade = (i > warmup and in_window and not flatten_now and not flattened
                     and not halt_day and not halt_acct and trades_today < p.max_trades_day)

        def stop_for(is_long):
            if p.stop_mode == "ATR":
                raw = c[i] - p.atr_stop_mult * atr[i] if is_long else c[i] + p.atr_stop_mult * atr[i]
            elif p.stop_mode == "Trend line":
                raw = trend[i] - buf_pts if is_long else trend[i] + buf_pts
            else:
                raw = lower[i] - buf_pts if is_long else upper[i] + buf_pts
            s = min(raw, c[i] - min_stop_pts) if is_long else max(raw, c[i] + min_stop_pts)
            return rnd(s)

        def qty_for(st):
            per = abs(c[i] - st) * pv + rt_cost
            budget = p.risk_per_trade
            if p.daily_loss_limit > 0:
                budget = min(budget, p.daily_loss_limit + daily_pnl)
            if dd_floor is not None:
                budget = min(budget, equity - dd_floor - p.dd_buffer)
            if per <= 0 or budget <= 0 or np.isnan(per):
                return 0
            return int(min(np.floor(budget / per), p.max_contracts))

        enter_long = enter_short = False
        if lflip[i] or sflip[i]:
            if lflip[i] and p.direction != "Short only" and can_trade:
                ls = stop_for(True)
                lq = qty_for(ls)
                enter_long = lq >= 1
            if sflip[i] and p.direction != "Long only" and can_trade:
                ss = stop_for(False)
                sq_ = qty_for(ss)
                enter_short = sq_ >= 1

        if not flattened and pos is not None:
            if (lflip[i] and pos.direction < 0) or (sflip[i] and pos.direction > 0):
                pending = ("close",)

        if enter_long:
            tg = rnd(c[i] + (c[i] - ls) * p.target_r) if p.target_r > 0 else None
            pending = ("enter", 1, lq, ls, tg)
            trades_today += 1
        elif enter_short:
            tg = rnd(c[i] - (ss - c[i]) * p.target_r) if p.target_r > 0 else None
            pending = ("enter", -1, sq_, ss, tg)
            trades_today += 1

        if p.trail_on_trend and pos is not None and not (enter_long or enter_short):
            if pos.direction > 0:
                pos.stop = max(pos.stop, rnd(trend[i] - buf_pts))
            else:
                pos.stop = min(pos.stop, rnd(trend[i] + buf_pts))

        # ---- bookkeeping ----
        daily[tday[i]] = equity - day_start_eq
        if attempt is not None:
            attempt.end = tday[i]
            attempt.pnl = equity - p.start_balance
        prev_eq = equity
        if evaluation and halt_acct:
            break

    if pos is not None:
        fill_exit(pos, c[last_i], df.index[last_i], "End of data")
    if attempt is not None:
        attempt.pnl = cash - p.start_balance
        res.attempts.append(attempt)

    dp = pd.Series(daily)
    dp.index = pd.to_datetime(dp.index)
    res.daily_pnl = dp
    if evaluation:
        # per-attempt trading days + best day (consistency rules)
        for a in res.attempts:
            a.days = int(len(dp))
            a.best_day = float(dp.max()) if len(dp) else 0.0
    return res


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def summarize(trades: List[Trade], daily: pd.Series) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = np.array([t.pnl for t in trades])
    wins = pnl[pnl > 0]
    losses = pnl[pnl <= 0]
    eq = np.cumsum(pnl)
    max_dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
    active = daily[daily != 0]
    sharpe = float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() > 0 else 0.0
    return {
        "trades": len(trades),
        "win_rate": 100.0 * len(wins) / len(trades),
        "net": float(pnl.sum()),
        "profit_factor": float(wins.sum() / -losses.sum()) if losses.sum() < 0 else float("inf"),
        "avg_trade": float(pnl.mean()),
        "avg_r": float(np.mean([t.r_multiple for t in trades])),
        "avg_qty": float(np.mean([t.qty for t in trades])),
        "max_dd": max_dd,
        "worst_day": float(daily.min()),
        "best_day": float(daily.max()),
        "trading_days": int(len(active)),
        "sharpe": sharpe,
        "long_net": float(sum(t.pnl for t in trades if t.direction > 0)),
        "short_net": float(sum(t.pnl for t in trades if t.direction < 0)),
    }
