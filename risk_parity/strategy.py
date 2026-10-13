"""
Low-vol cross-asset futures portfolio, rebuilt for a prop-firm swing account
(e.g. Phidias Swing/Premium: overnight + weekend holds allowed, end-of-day
trailing drawdown, no daily loss limit).

Origin: a QuantConnect "inverse volatility futures" algorithm -- every Monday
10:00 hold the 5 lowest-volatility continuous futures out of 12, long-only,
weighted by 1/vol, 50% gross. That doesn't fit a $50K prop account: one
micro is already ~$30K+ of notional, so percentage-of-equity weights round
to 0 or 1 contract, and the trailing drawdown is too tight for its
unhedged, stop-less exposure. This version keeps the idea (hold the calm
markets, size inversely to their volatility, rebalance weekly) and changes
the mechanics:

  - Universe: micros only (MES, MNQ, MCL, MGC by default).
  - Equity cluster: MES and MNQ are ~0.9 correlated, so they share one
    slot -- only the lower-volatility one of the two is held.
  - Trend filter: long only while the daily close is above its
    `trend_len`-day SMA (the original went long unconditionally, which is
    what hurt in 2022).
  - Sizing = equal dollar risk per market: contracts =
    floor(risk_per_market / (stop_atr * ATR * point_value)). Since ATR is a
    volatility measure this *is* inverse-volatility weighting, but expressed
    in whole contracts and dollars of risk instead of % of notional.
  - Protective stop per position at entry - stop_atr * ATR, checked on
    every intraday bar (gaps fill at the bar's open).
  - Portfolio weekly stop: if the week's P&L (marked on bar closes) falls to
    -weekly_stop_usd, everything is flattened until the next rebalance.
  - Rebalance on the first bar at/after 10:00 ET on the first trading day
    of each week, at that bar's open, using only completed daily sessions.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from failed2s.instruments import INSTRUMENTS, Instrument

EQUITY_CLUSTER = ("MES", "MNQ")


@dataclass
class RiskParityConfig:
    symbols: Tuple[str, ...] = ("MES", "MNQ", "MCL", "MGC")
    vol_len: int = 30              # daily-return stdev lookback (ranking)
    atr_len: int = 20              # daily ATR lookback (sizing + stops)
    trend_len: int = 100           # SMA trend filter; 0 disables it
    max_positions: int = 3
    one_equity_slot: bool = True
    # One micro with a 2x daily-ATR stop risks ~$800 (MES, MCL) to ~$2,100
    # (MNQ, MGC) at 2026 volatility, so a small per-market budget buys
    # nothing. Size per market up to this, then stop adding positions
    # (calmest first) once total open risk would exceed max_total_risk_usd.
    risk_per_market_usd: float = 1000.0
    max_total_risk_usd: float = 2000.0
    stop_atr: float = 2.0
    max_contracts: int = 10
    weekly_stop_usd: Optional[float] = 1000.0
    rebalance_hour: int = 10
    commission_per_side: float = 1.25   # per contract, incl. exchange fees
    slippage_ticks: float = 1.0         # per side


@dataclass
class BacktestResult:
    eod_pnl: pd.Series                 # cumulative $ P&L at each session close
    trades: pd.DataFrame
    rebalances: pd.DataFrame
    config: RiskParityConfig = field(repr=False)


def session_date(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """CME trade date: the session that opens 18:00 ET belongs to the next day."""
    return (index + pd.Timedelta(hours=7)).normalize().tz_localize(None)


def daily_bars(bars: pd.DataFrame) -> pd.DataFrame:
    """Aggregate intraday bars (bar-open stamps, ET) into CME trade-date bars."""
    g = bars.groupby(session_date(bars.index))
    d = pd.DataFrame({"open": g["open"].first(), "high": g["high"].max(),
                      "low": g["low"].min(), "close": g["close"].last()})
    return d[d.index.dayofweek < 5]


def daily_indicators(d: pd.DataFrame, cfg: RiskParityConfig) -> pd.DataFrame:
    prev_close = d["close"].shift(1)
    tr = pd.concat([d["high"] - d["low"], (d["high"] - prev_close).abs(),
                    (d["low"] - prev_close).abs()], axis=1).max(axis=1)
    out = pd.DataFrame(index=d.index)
    out["close"] = d["close"]
    # Point change / prior close: back-adjusted continuous prices keep point
    # differences exact, so this is only a mild distortion of % vol.
    out["vol"] = (d["close"].diff() / prev_close).rolling(cfg.vol_len).std()
    out["atr"] = tr.rolling(cfg.atr_len).mean()
    out["sma"] = d["close"].rolling(cfg.trend_len).mean() if cfg.trend_len else np.nan
    return out


def select_targets(ind: Dict[str, pd.Series], cfg: RiskParityConfig,
                   instruments: Dict[str, Instrument]) -> Dict[str, int]:
    """Target contracts per symbol from one row of daily indicators per symbol."""
    eligible = {}
    for sym, row in ind.items():
        if not np.isfinite(row["vol"]) or not np.isfinite(row["atr"]) or row["atr"] <= 0:
            continue
        if cfg.trend_len and not (np.isfinite(row["sma"]) and row["close"] > row["sma"]):
            continue
        eligible[sym] = row
    if cfg.one_equity_slot:
        eq = [s for s in eligible if s in EQUITY_CLUSTER]
        if len(eq) > 1:
            keep = min(eq, key=lambda s: eligible[s]["vol"])
            eligible = {s: r for s, r in eligible.items() if s not in EQUITY_CLUSTER or s == keep}
    chosen = sorted(eligible, key=lambda s: eligible[s]["vol"])[: cfg.max_positions]
    targets, total_risk = {}, 0.0
    for sym in chosen:
        risk_per_contract = cfg.stop_atr * eligible[sym]["atr"] * instruments[sym].point_value
        n = min(int(cfg.risk_per_market_usd // risk_per_contract), cfg.max_contracts,
                int((cfg.max_total_risk_usd - total_risk) // risk_per_contract))
        if n > 0:
            targets[sym] = n
            total_risk += n * risk_per_contract
    return targets


def run_backtest(data: Dict[str, pd.DataFrame], cfg: Optional[RiskParityConfig] = None,
                 instruments: Optional[Dict[str, Instrument]] = None) -> BacktestResult:
    """
    `data`: symbol -> intraday OHLC bars (tz-aware ET, bar-open stamps).
    Symbols are looked up in `instruments` (default failed2s INSTRUMENTS) for
    point value / tick size, so pass e.g. ES prices under the key "MES" to
    run a longer proxy history at micro sizing.
    """
    cfg = cfg or RiskParityConfig()
    instruments = instruments or INSTRUMENTS
    syms = [s for s in cfg.symbols if s in data]
    pv = np.array([instruments[s].point_value for s in syms])
    slip = np.array([instruments[s].tick_size * cfg.slippage_ticks for s in syms])

    idx = data[syms[0]].index
    for s in syms[1:]:
        idx = idx.union(data[s].index)
    # Forward-fill so a symbol with a missing bar just shows no movement.
    o = np.column_stack([data[s]["open"].reindex(idx).ffill().to_numpy() for s in syms])
    h = np.column_stack([data[s]["high"].reindex(idx).ffill().to_numpy() for s in syms])
    lo = np.column_stack([data[s]["low"].reindex(idx).ffill().to_numpy() for s in syms])
    c = np.column_stack([data[s]["close"].reindex(idx).ffill().to_numpy() for s in syms])
    missing = {s: ~idx.isin(data[s].index) for s in syms}
    for j, s in enumerate(syms):  # bars where the symbol didn't trade can't trigger fills/stops
        h[missing[s], j] = np.maximum(o[missing[s], j], c[missing[s], j])
        lo[missing[s], j] = np.minimum(o[missing[s], j], c[missing[s], j])

    inds = {s: daily_indicators(daily_bars(data[s]), cfg) for s in syms}
    sdates = session_date(idx)
    week_key = sdates.to_period("W-SUN")

    pos = np.zeros(len(syms))
    stop = np.full(len(syms), np.nan)
    entry = np.full(len(syms), np.nan)
    realized = 0.0
    week_start_equity = 0.0
    week_halted = False
    last_rebal_week = None
    eod_dates, eod_vals = [], []
    trades, rebals = [], []

    def equity_at(price):
        return realized + float(np.sum(np.where(pos != 0, pos * (price - np.nan_to_num(entry)) * pv, 0.0)))

    def close_leg(j, qty, price, t, reason):
        nonlocal realized
        fill = price - slip[j] * np.sign(qty)
        realized += qty * (fill - entry[j]) * pv[j] - abs(qty) * cfg.commission_per_side
        trades.append({"time": t, "symbol": syms[j], "qty": -qty, "price": fill,
                       "entry": entry[j], "pnl": qty * (fill - entry[j]) * pv[j], "reason": reason})

    for i, t in enumerate(idx):
        # --- weekly rebalance at the bar's open ---
        wk = week_key[i]
        if wk != last_rebal_week and t.hour >= cfg.rebalance_hour and t.hour < 17:
            last_rebal_week = wk
            prior = {}
            for s in syms:
                past = inds[s][inds[s].index < sdates[i]]
                if len(past):
                    prior[s] = past.iloc[-1]
            targets = select_targets(prior, cfg, instruments) if len(prior) == len(syms) else {}
            for j, s in enumerate(syms):
                want = targets.get(s, 0)
                px = o[i, j]
                if pos[j] and want != pos[j]:
                    # Simplest bookkeeping: close the old leg, reopen the new size.
                    # Only the contract delta is charged costs below.
                    keep = min(pos[j], want)
                    if pos[j] > keep:
                        close_leg(j, pos[j] - keep, px, t, "rebalance")
                    pos[j] = keep
                if want > pos[j]:
                    add = want - pos[j]
                    fill = px + slip[j]
                    realized -= add * cfg.commission_per_side
                    entry[j] = fill if pos[j] == 0 else (entry[j] * pos[j] + fill * add) / want
                    pos[j] = want
                    trades.append({"time": t, "symbol": s, "qty": add, "price": fill,
                                   "entry": fill, "pnl": 0.0, "reason": "entry"})
                if pos[j]:
                    stop[j] = px - cfg.stop_atr * prior[s]["atr"]
                else:
                    stop[j], entry[j] = np.nan, np.nan
            week_start_equity = equity_at(o[i])
            week_halted = False
            rebals.append({"time": t, **{s: targets.get(s, 0) for s in syms}})

        # --- protective stops, intraday ---
        for j in range(len(syms)):
            if pos[j] and lo[i, j] <= stop[j]:
                close_leg(j, pos[j], min(o[i, j], stop[j]), t, "stop")
                pos[j], stop[j], entry[j] = 0, np.nan, np.nan

        # --- portfolio weekly stop, on the bar close ---
        if cfg.weekly_stop_usd and not week_halted and pos.any():
            if equity_at(c[i]) - week_start_equity <= -cfg.weekly_stop_usd:
                for j in range(len(syms)):
                    if pos[j]:
                        close_leg(j, pos[j], c[i, j], t, "weekly_stop")
                        pos[j], stop[j], entry[j] = 0, np.nan, np.nan
                week_halted = True

        # --- end of session: mark to market for the EOD trailing drawdown ---
        if i + 1 == len(idx) or sdates[i + 1] != sdates[i]:
            eod_dates.append(sdates[i])
            eod_vals.append(equity_at(c[i]))

    return BacktestResult(
        eod_pnl=pd.Series(eod_vals, index=pd.DatetimeIndex(eod_dates), name="pnl"),
        trades=pd.DataFrame(trades), rebalances=pd.DataFrame(rebals), config=cfg)


# ---------------------------------------------------------------- prop eval

@dataclass
class PropRules:
    profit_target: float = 4000.0      # $50K swing: 8%
    trailing_dd: float = 2500.0        # $50K swing: 5%, end-of-day trailing


def simulate_eval(eod_pnl: pd.Series, start: int, rules: PropRules) -> Tuple[str, int]:
    """Run one evaluation starting after session `start`. Returns (outcome, sessions)."""
    base = eod_pnl.iloc[start]
    peak = 0.0
    for n, v in enumerate(eod_pnl.iloc[start + 1:].to_numpy() - base, start=1):
        peak = max(peak, v)
        if v >= rules.profit_target:
            return "pass", n
        if peak - v >= rules.trailing_dd:
            return "fail", n
    return "open", len(eod_pnl) - start - 1


def eval_stats(eod_pnl: pd.Series, rules: PropRules, step: int = 5,
               first: int = 0) -> Dict[str, float]:
    """Start an evaluation every `step` sessions from `first` and tally outcomes."""
    outs = [simulate_eval(eod_pnl, s, rules) for s in range(first, len(eod_pnl) - 1, step)]
    n = len(outs)
    passed = [d for o, d in outs if o == "pass"]
    return {
        "evals": n,
        "pass": sum(o == "pass" for o, _ in outs) / n if n else np.nan,
        "fail": sum(o == "fail" for o, _ in outs) / n if n else np.nan,
        "open": sum(o == "open" for o, _ in outs) / n if n else np.nan,
        "median_sessions_to_pass": float(np.median(passed)) if passed else np.nan,
    }


def summary(res: BacktestResult, first_trade: Optional[pd.Timestamp] = None) -> Dict[str, float]:
    p = res.eod_pnl if first_trade is None else res.eod_pnl[res.eod_pnl.index >= first_trade]
    daily = p.diff().dropna()
    years = len(daily) / 252
    weekly = p.resample("W-FRI").last().diff().dropna()
    dd = (p.cummax() - p)
    tr = res.trades
    exits = tr[tr["reason"] != "entry"] if len(tr) else tr
    return {
        "sessions": len(daily),
        "net_pnl": float(p.iloc[-1] - p.iloc[0]) if len(p) else 0.0,
        "pnl_per_year": float(daily.sum() / years) if years else np.nan,
        "sharpe": float(daily.mean() / daily.std() * np.sqrt(252)) if daily.std() else np.nan,
        "max_eod_dd": float(dd.max()) if len(dd) else 0.0,
        "worst_day": float(daily.min()) if len(daily) else 0.0,
        "worst_week": float(weekly.min()) if len(weekly) else 0.0,
        "pct_weeks_invested": float((res.rebalances.drop(columns="time").sum(axis=1) > 0).mean())
        if len(res.rebalances) else 0.0,
        "stops_hit": int((exits["reason"] == "stop").sum()) if len(exits) else 0,
        "weekly_stops_hit": int((exits["reason"] == "weekly_stop").sum()) if len(exits) else 0,
    }
