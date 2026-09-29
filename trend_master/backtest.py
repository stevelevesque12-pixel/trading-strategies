"""
Python port of tradingview/trend_master_prop.pine, for backtesting on the
parquet files in sample_data/real_multi_instrument/.

Mirrors the Pine script bar by bar:
  * orders process on bar close (entries and market exits fill at the close,
    plus 1 tick of slippage, like process_orders_on_close + slippage=1),
  * the stop/target bracket is checked from the next bar on; if a bar reaches
    both, the stop is assumed to fill first (conservative); a bar that opens
    through the stop fills at its open,
  * $0.75 per contract per side commission.

Two modes:
  raw   - the strategy with its personal risk limits (risk/trade, $600 daily
          stop, 4 trades/day, 2 losses in a row) but no account-level
          drawdown/target, so you see the edge itself.
  eval  - a fresh Tradeify Select 50K evaluation started every Monday;
          reports how many passed / failed / were still running at the end
          of the data.

Usage:
    python -m trend_master.backtest
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "sample_data", "real_multi_instrument")

# point value (USD/point), tick size
SPECS = {
    "mes": (5.0, 0.25),
    "mnq": (2.0, 0.25),
    "mgc": (10.0, 0.10),
    "mcl": (100.0, 0.01),
}

COMMISSION_SIDE = 0.75


@dataclass
class Params:
    fast_len: int = 21
    slow_len: int = 55
    adaptive: bool = True
    confirm_bars: int = 3
    chop_win: int = 50
    chop_max: int = 2
    pull_depth: int = 40
    pull_gap: int = 12
    min_age: int = 8
    trade_starts: bool = True
    trade_pulls: bool = True
    min_quality: float = 0.0
    entry_start: int = 9 * 60 + 30
    entry_end: int = 15 * 60 + 30
    flatten_at: int = 15 * 60 + 55
    stop_mode: str = "Ribbon"
    stop_buf: float = 0.5
    atr_mult: float = 2.0
    target_r: float = 2.0
    be_r: float = 1.0
    be_ticks: int = 1
    trail_slow: bool = False
    exit_flip: bool = True
    # account
    initial: float = 50000.0
    trail_dd: float = 2000.0
    lock_offset: float = 100.0
    firm_daily: float = 0.0
    profit_target: float = 3000.0
    consist_pct: float = 40.0
    min_days: int = 3
    max_minis: int = 4
    guard_buf: float = 100.0
    my_daily_stop: float = 600.0
    day_profit_cap: float = 0.0
    max_trades: int = 4
    max_losses: int = 2
    risk_usd: float = 300.0
    user_cts: int = 40
    rt_comm: float = 1.5


SELECT_50K_EVAL = dict()  # the Params defaults are the Select 50K eval preset
RAW = dict(trail_dd=1e12, lock_offset=1e12, firm_daily=0.0, profit_target=0.0, consist_pct=0.0, min_days=0)


# ───────────────────────────── indicators / signals ─────────────────────────────

def _rma(x: np.ndarray, n: int) -> np.ndarray:
    out = np.full_like(x, np.nan)
    if len(x) < n:
        return out
    out[n - 1] = np.nanmean(x[:n])
    a = 1.0 / n
    for i in range(n, len(x)):
        out[i] = a * x[i] + (1 - a) * out[i - 1]
    return out


def _ema(x: np.ndarray, n: int) -> np.ndarray:
    """Pine ta.ema: SMA seed on the first n valid values, then EMA."""
    out = np.full_like(x, np.nan)
    a = 2.0 / (n + 1)
    valid = np.where(~np.isnan(x))[0]
    if len(valid) < n:
        return out
    start = valid[0]
    seed_end = start + n - 1
    out[seed_end] = np.mean(x[start:seed_end + 1])
    for i in range(seed_end + 1, len(x)):
        xi = x[i] if not np.isnan(x[i]) else out[i - 1]
        out[i] = a * xi + (1 - a) * out[i - 1]
    return out


def compute_signals(df: pd.DataFrame, p: Params) -> pd.DataFrame:
    o, h, l, c, v = (df[k].to_numpy(float) for k in ("open", "high", "low", "close", "volume"))
    n = len(c)
    s = pd.Series(c)
    chg = (s - s.shift(p.slow_len)).abs()
    noise = s.diff().abs().rolling(p.slow_len).sum()
    eff = np.where(noise > 0, chg / noise, 0.0)
    eff = np.where(np.isnan(chg), np.nan, eff)
    eff_s = _ema(eff, 5)
    mult = (1.6 - np.nan_to_num(eff_s)) if p.adaptive else np.ones(n)
    f_len = np.maximum(2.0, p.fast_len * mult)
    s_len = np.maximum(3.0, p.slow_len * mult)

    fast = np.empty(n)
    slow = np.empty(n)
    fast[0] = slow[0] = c[0]
    for i in range(1, n):
        af = 2.0 / (max(f_len[i], 1.0) + 1.0)
        as_ = 2.0 / (max(s_len[i], 1.0) + 1.0)
        fast[i] = af * c[i] + (1 - af) * fast[i - 1]
        slow[i] = as_ * c[i] + (1 - as_) * slow[i - 1]

    d = np.where(fast > slow, 1, -1)
    flipped = np.r_[False, d[1:] != d[:-1]]
    flips = pd.Series(flipped.astype(float)).rolling(p.chop_win).sum().to_numpy()
    choppy = flips > p.chop_max  # nan -> False, like Pine's na comparison

    prev_c = np.r_[np.nan, c[:-1]]
    tr = np.nanmax(np.vstack([h - l, np.abs(h - prev_c), np.abs(l - prev_c)]), axis=0)
    atr = _rma(tr, 14)

    sep = np.where(atr > 0, np.minimum(np.abs(fast - slow) / atr, 2.0) / 2.0, 0.0)
    vol_ma = pd.Series(v).rolling(20).mean().to_numpy()
    signed = np.where(c > o, v, -v)
    dir_vol = pd.Series(signed).rolling(20).sum().to_numpy() / (vol_ma * 20)
    has_vol = ~np.isnan(vol_ma) & (vol_ma > 0)
    vol_fit = np.where(has_vol, np.clip(np.where(d == 1, dir_vol, -dir_vol), 0.0, 1.0), 0.0)
    e0 = np.nan_to_num(eff_s)
    score = np.where(has_vol, 0.4 * sep + 0.35 * e0 * 1.5 + 0.25 * vol_fit, 0.55 * sep + 0.45 * e0 * 1.5)
    score = np.clip(np.nan_to_num(score), 0.0, 1.0)
    score_s = _ema(score, 3)

    rib_top = np.maximum(fast, slow)
    rib_btm = np.minimum(fast, slow)

    start = np.zeros(n, int)
    pull = np.zeros(n, int)
    trend_start = 0
    last_label = 0
    in_pull = False
    last_pull = -100000
    for i in range(n):
        if flipped[i]:
            trend_start = i
        age = i - trend_start
        if np.isnan(atr[i]):
            continue
        if flipped[i]:
            in_pull = False
        di = d[i]
        if age >= p.confirm_bars and di != last_label and (c[i] > rib_top[i] if di == 1 else c[i] < rib_btm[i]):
            last_label = di
            if not choppy[i]:
                start[i] = di
        if not flipped[i] and age >= p.min_age and last_label == di:
            depth = (rib_top[i] - rib_btm[i]) * p.pull_depth / 100.0
            into = l[i] <= rib_top[i] - depth if di == 1 else h[i] >= rib_btm[i] + depth
            back = c[i] > rib_top[i] if di == 1 else c[i] < rib_btm[i]
            with_trend = c[i] > o[i] if di == 1 else c[i] < o[i]
            if into and not back:
                in_pull = True
            elif in_pull and back:
                in_pull = False
                if not choppy[i] and with_trend and i - last_pull >= p.pull_gap and score_s[i] >= 0.32:
                    last_pull = i
                    pull[i] = di

    out = df.copy()
    out["fast"], out["slow"], out["dir"], out["flipped"] = fast, slow, d, flipped
    out["atr"], out["eff_s"], out["start"], out["pull"] = atr, eff_s, start, pull
    return out


# ───────────────────────────── account simulation ─────────────────────────────

@dataclass
class Result:
    trades: list = field(default_factory=list)   # dicts
    status: str = "running"                       # running / passed / failed
    end_ts: pd.Timestamp | None = None
    days: int = 0
    breaches: int = 0


def simulate(sig: pd.DataFrame, p: Params, pv: float, tick: float, start_idx: int = 0, is_micro: bool = True) -> Result:
    ts = sig.index
    ny = ts.tz_convert("America/New_York")
    mins_open = (ny.hour * 60 + ny.minute).to_numpy()
    bar_mins = (ts[1] - ts[0]).total_seconds() / 60.0 if len(ts) > 1 else 1.0
    close_ny = ny + pd.Timedelta(minutes=bar_mins)
    mins_close = (close_ny.hour * 60 + close_ny.minute).to_numpy()
    tday = (ny + pd.Timedelta(hours=6)).normalize()  # CME day starts 18:00 NY
    tday = tday.to_numpy()

    o, h, l, c = (sig[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    slow, atr, d, flipped = sig["slow"].to_numpy(), sig["atr"].to_numpy(), sig["dir"].to_numpy(), sig["flipped"].to_numpy()
    eff_s, start, pull = np.nan_to_num(sig["eff_s"].to_numpy()), sig["start"].to_numpy(), sig["pull"].to_numpy()

    daily_loss = min(p.firm_daily, p.my_daily_stop) if p.firm_daily > 0 and p.my_daily_stop > 0 else max(p.firm_daily, p.my_daily_stop)
    max_cts = min(p.user_cts, p.max_minis * 10 if is_micro else p.max_minis)
    consist_cap = p.profit_target * p.consist_pct / 100.0 if p.consist_pct > 0 and p.profit_target > 0 else 0.0
    day_cap = min(consist_cap, p.day_profit_cap) if consist_cap > 0 and p.day_profit_cap > 0 else max(consist_cap, p.day_profit_cap)
    cost_ct = p.rt_comm + 2 * tick * pv

    res = Result()
    closed_eq = p.initial
    day_start = p.initial
    peak = p.initial
    trades_today = 0
    loss_streak = 0
    trading_days = 0
    best_day = 0.0
    halted = False

    pos = 0          # +1 / -1
    qty = 0
    avg = 0.0
    t_stop = t_tgt = t_risk = np.nan
    be_done = False
    pend_stop = np.nan
    entry_bar = -1
    entry_ts = None
    entry_comm = 0.0
    closed_on_close = -1  # bar where a market exit filled at the close (no re-entry that bar, like Pine)

    def close_pos(i, px, reason):
        nonlocal closed_eq, pos, qty, loss_streak, closed_on_close
        if reason not in ("stop", "target"):
            closed_on_close = i
        gross = (px - avg) * qty * pv * pos
        comm = COMMISSION_SIDE * qty
        closed_eq += gross - comm
        profit = gross - comm - entry_comm
        loss_streak = loss_streak + 1 if profit < 0 else 0
        res.trades.append(dict(entry=entry_ts, exit=ts[i], side=pos, qty=qty, entry_px=avg, exit_px=px,
                               risk_usd=t_risk * qty * pv, pnl=profit, reason=reason))
        pos, qty = 0, 0

    for i in range(start_idx, len(c)):
        # A. intrabar bracket fills (orders placed on an earlier bar)
        if pos != 0 and entry_bar < i and not np.isnan(pend_stop):
            if pos == 1:
                if o[i] <= pend_stop:
                    close_pos(i, o[i] - tick, "stop")
                elif l[i] <= pend_stop:
                    close_pos(i, pend_stop - tick, "stop")
                elif not np.isnan(t_tgt) and o[i] >= t_tgt:
                    close_pos(i, o[i], "target")
                elif not np.isnan(t_tgt) and h[i] >= t_tgt:
                    close_pos(i, t_tgt, "target")
            else:
                if o[i] >= pend_stop:
                    close_pos(i, o[i] + tick, "stop")
                elif h[i] >= pend_stop:
                    close_pos(i, pend_stop + tick, "stop")
                elif not np.isnan(t_tgt) and o[i] <= t_tgt:
                    close_pos(i, o[i], "target")
                elif not np.isnan(t_tgt) and l[i] <= t_tgt:
                    close_pos(i, t_tgt, "target")

        # B. new trading day
        if i > start_idx and tday[i] != tday[i - 1]:
            peak = max(peak, closed_eq)
            if trades_today > 0:
                trading_days += 1
            best_day = max(best_day, closed_eq - day_start)
            day_start = closed_eq
            trades_today = 0
            loss_streak = 0

        if halted:
            continue

        # C-E. equity, floor, rooms
        open_pnl = (c[i] - avg) * qty * pv * pos if pos else 0.0
        worst_open = ((l[i] - avg) if pos == 1 else (avg - h[i])) * qty * pv if pos else 0.0
        cur_eq = closed_eq + open_pnl
        lock = p.initial + p.lock_offset
        floor = min(peak - p.trail_dd, lock)
        dd_room = cur_eq - floor
        day_pnl = cur_eq - day_start
        day_room = daily_loss + day_pnl if daily_loss > 0 else 1e12
        limit_eq = max(floor + p.guard_buf, day_start - daily_loss + p.guard_buf if daily_loss > 0 else -1e12)

        # F. real rule breaches (gap through the guard stop)
        breached = False
        if pos and closed_eq + worst_open <= floor:
            breached = True
            res.breaches += 1
            close_pos(i, c[i] - tick * pos, "dd breach")
            halted = True
            res.status, res.end_ts = "failed", ts[i]
            continue
        if pos and p.firm_daily > 0 and closed_eq + worst_open - day_start <= -p.firm_daily:
            breached = True
            res.breaches += 1
            close_pos(i, c[i] - tick * pos, "daily breach")

        # G. evaluation passed?
        total = closed_eq - p.initial
        best_now = max(best_day, closed_eq - day_start)
        days_now = trading_days + (1 if trades_today > 0 else 0)
        consist_ok = p.consist_pct <= 0 or best_now <= total * p.consist_pct / 100.0
        if p.profit_target > 0 and pos == 0 and total >= p.profit_target and days_now >= p.min_days and consist_ok:
            halted = True
            res.status, res.end_ts, res.days = "passed", ts[i], days_now
            continue
        # Within 2x the safety buffer of the floor the guard stop leaves no room to trade: the
        # account is effectively dead (in reality it touched floor + buffer), so count it as failed.
        if pos == 0 and dd_room < 2 * p.guard_buf:
            halted = True
            res.status, res.end_ts, res.days = "failed", ts[i], days_now
            continue

        flatten_now = mins_close[i] + bar_mins > p.flatten_at

        # H. manage open position
        pend_stop = np.nan
        if pos and not breached:
            is_long = pos == 1
            if p.be_r > 0 and not be_done and (h[i] >= avg + p.be_r * t_risk if is_long else l[i] <= avg - p.be_r * t_risk):
                t_stop = max(t_stop, avg + p.be_ticks * tick) if is_long else min(t_stop, avg - p.be_ticks * tick)
                be_done = True
            if p.trail_slow:
                tr = slow[i] - p.stop_buf * atr[i] if is_long else slow[i] + p.stop_buf * atr[i]
                t_stop = max(t_stop, tr) if is_long else min(t_stop, tr)
            if is_long:
                guard = np.ceil((avg + (limit_eq - closed_eq) / (qty * pv)) / tick) * tick
                stop_now = max(t_stop, guard)
            else:
                guard = np.floor((avg - (limit_eq - closed_eq) / (qty * pv)) / tick) * tick
                stop_now = min(t_stop, guard)
            if flatten_now:
                close_pos(i, c[i] - tick * pos, "eod")
            elif p.exit_flip and flipped[i] and d[i] != pos:
                close_pos(i, c[i] - tick * pos, "flip")
            elif (c[i] <= stop_now) if is_long else (c[i] >= stop_now):
                close_pos(i, c[i] - tick * pos, "guard")
            else:
                pend_stop = stop_now

        # I. entries (only when flat at the start of this bar's script run)
        if pos == 0 and entry_bar != i and not np.isnan(atr[i]):
            in_window = p.entry_start <= mins_open[i] < p.entry_end
            day_cap_hit = day_cap > 0 and day_pnl >= day_cap
            streak_hit = p.max_losses > 0 and loss_streak >= p.max_losses
            if in_window and not flatten_now and not day_cap_hit and not streak_hit and trades_today < p.max_trades \
                    and closed_on_close != i:
                long_sig = (p.trade_starts and start[i] == 1 and eff_s[i] >= p.min_quality) or (p.trade_pulls and pull[i] == 1)
                short_sig = (p.trade_starts and start[i] == -1 and eff_s[i] >= p.min_quality) or (p.trade_pulls and pull[i] == -1)
                side = 1 if long_sig else -1 if short_sig else 0
                if side:
                    budget = min(p.risk_usd, dd_room - p.guard_buf, day_room - p.guard_buf if daily_loss > 0 else 1e12)
                    if side == 1:
                        raw = min(slow[i], l[i]) - p.stop_buf * atr[i] if p.stop_mode == "Ribbon" else c[i] - p.atr_mult * atr[i]
                        stop = np.floor(raw / tick) * tick
                        risk = c[i] - stop
                    else:
                        raw = max(slow[i], h[i]) + p.stop_buf * atr[i] if p.stop_mode == "Ribbon" else c[i] + p.atr_mult * atr[i]
                        stop = np.ceil(raw / tick) * tick
                        risk = stop - c[i]
                    q = int(min(np.floor(budget / (risk * pv + cost_ct)), max_cts)) if risk > 0 and budget > 0 else 0
                    if q >= 1:
                        pos, qty = side, q
                        avg = c[i] + tick * side
                        entry_comm = COMMISSION_SIDE * q
                        closed_eq -= entry_comm
                        t_stop, t_risk, be_done = stop, risk, False
                        t_tgt = c[i] + side * p.target_r * risk if p.target_r > 0 else np.nan
                        pend_stop = stop
                        entry_bar, entry_ts = i, ts[i]
                        trades_today += 1

    if res.status == "running":
        res.days = trading_days + (1 if trades_today > 0 else 0)
    return res


# ───────────────────────────── reporting ─────────────────────────────

def load(sym: str, tf: str) -> pd.DataFrame | None:
    files = glob.glob(os.path.join(DATA_DIR, f"real_{sym}_{tf}_*.parquet"))
    if not files:
        return None
    df = pd.read_parquet(files[0]).sort_index()
    return df[~df.index.duplicated()]


def raw_stats(res: Result) -> dict:
    t = pd.DataFrame(res.trades)
    if t.empty:
        return dict(trades=0)
    pnl = t["pnl"]
    eq = pnl.cumsum()
    dd = (eq - eq.cummax()).min()
    wins, losses = pnl[pnl > 0].sum(), -pnl[pnl < 0].sum()
    r = pnl / t["risk_usd"]
    days = t["exit"].dt.tz_convert("America/New_York").dt.date
    daily = pnl.groupby(days).sum()
    return dict(
        trades=len(t),
        win_pct=100 * (pnl > 0).mean(),
        pf=wins / losses if losses > 0 else np.inf,
        net=pnl.sum(),
        avg_r=r.mean(),
        max_dd=dd,
        worst_day=daily.min(),
        green_days=f"{(daily > 0).sum()}/{len(daily)}",
    )


def eval_runs(sig: pd.DataFrame, p: Params, pv: float, tick: float, warmup: int) -> dict:
    ny = sig.index.tz_convert("America/New_York")
    tday = (ny + pd.Timedelta(hours=6)).normalize()
    starts = []
    prev = None
    for i in range(warmup, len(sig)):
        if prev is not None and tday[i] != prev and tday[i].weekday() == 0:
            starts.append(i)
        prev = tday[i]
    out = dict(runs=len(starts), passed=0, failed=0, running=0, days_to_pass=[])
    for s in starts:
        r = simulate(sig, p, pv, tick, start_idx=s)
        out[r.status] += 1
        if r.status == "passed":
            out["days_to_pass"].append(r.days)
    return out


def main() -> None:
    base = Params()
    raw_p = Params(**RAW)
    rows, evals = [], []
    for tf in ("1m", "5m", "15m"):
        for sym, (pv, tick) in SPECS.items():
            df = load(sym, tf)
            if df is None:
                continue
            sig = compute_signals(df, base)
            warm = 300
            r = simulate(sig, raw_p, pv, tick, start_idx=warm)
            st = raw_stats(r)
            span = f"{df.index[warm].date()} to {df.index[-1].date()}"
            rows.append(dict(tf=tf, sym=sym.upper(), span=span, **st))
            ev = eval_runs(sig, base, pv, tick, warm)
            dtp = ev.pop("days_to_pass")
            evals.append(dict(tf=tf, sym=sym.upper(), **ev, median_days=float(np.median(dtp)) if dtp else np.nan))
    pd.set_option("display.width", 200)
    pd.set_option("display.float_format", lambda x: f"{x:,.2f}")
    print("RAW EDGE ($300 risk/trade, personal limits only)")
    print(pd.DataFrame(rows).to_string(index=False))
    print()
    print("TRADEIFY SELECT 50K EVAL, fresh attempt started every Monday")
    print(pd.DataFrame(evals).to_string(index=False))


if __name__ == "__main__":
    main()
