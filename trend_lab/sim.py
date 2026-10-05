"""
Bar-by-bar simulator for MCL with realistic prop-firm frictions.

Contract between a strategy and the simulator: the strategy returns arrays
aligned to the bars, evaluated on each bar's CLOSE:

  long / short      bool   entry signal at the close of bar i
  stop_dist         float  initial stop distance in points (from the fill)
  exit_long/short   bool   optional discretionary exit signal at close of i
  trail_dist        float  optional chandelier trail distance (points)
  target_r          float  optional fixed target in R (None = no target)
  be_r              float  optional: move stop to breakeven after +be_r R
  max_bars          int    optional time stop: exit at next open after holding this many bars

Fills: entries/exit-signals fill at the NEXT bar's open, plus slippage.
Stops/targets are checked on each bar's high/low; if both are inside one
bar the stop is assumed first (conservative); a gap through the stop fills
at the open. Commission + slippage are charged on every trade.

Position size is risk-based: floor(risk_usd / (stop_dist * point_value +
per-contract cost)), capped at max_contracts.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

POINT_VALUE = 100.0  # MCL: 100 barrels, $1 per 0.01 tick
TICK = 0.01

# Entry windows in ET minutes-of-day (signal-bar CLOSE time), and flatten time.
SESSIONS = {
    # NYMEX pit hours, the most liquid / trendiest stretch for crude.
    "ny": dict(start=9 * 60, end=13 * 60 + 30, flatten=14 * 60 + 30),
    # Includes the pre-open ramp and the post-settle hour.
    "us": dict(start=8 * 60, end=14 * 60 + 30, flatten=16 * 60 + 30),
    # Globex nearly 23h; flat before Lucid's 16:45 ET cutoff.
    "all": dict(start=0, end=24 * 60, flatten=16 * 60 + 40),
}


@dataclass
class SimConfig:
    risk_usd: float = 200.0           # 10% of the $2,000 Lucid 50K drawdown
    max_contracts: int = 20           # well under the 50K micro cap
    commission_rt: float = 1.24       # per contract, round trip (typical Rithmic/Tradovate MCL)
    slippage_ticks: float = 1.0       # per side
    daily_loss_stop: float = 450.0    # stop entering after this realized loss (Lucid DLL is $1,200)
    daily_profit_lock: float = 1200.0 # stop entering after this gain (helps the 50% consistency rule)
    max_trades_day: int = 6
    session: str = "us"
    eia_filter: bool = True           # no entries Wed 10:15-10:45 ET (EIA crude inventories)
    max_stop_dist: float = 1.50       # skip signals with absurd stops (points)
    point_value: float = POINT_VALUE  # override for cross-market robustness tests
    tick: float = TICK


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    trade_day: object
    direction: int
    entry: float
    exit: float
    stop0: float
    contracts: int
    pnl: float
    r: float
    reason: str


def _arr(sig, key, n, default):
    v = sig.get(key)
    if v is None:
        return np.full(n, default)
    return np.asarray(v, dtype=type(default) if not isinstance(default, bool) else bool)


def simulate(df: pd.DataFrame, sig: Dict, cfg: SimConfig) -> List[Trade]:
    n = len(df)
    o = df["open"].to_numpy()
    h = df["high"].to_numpy()
    lo = df["low"].to_numpy()
    c = df["close"].to_numpy()
    idx = df.index
    tf_min = int(pd.Series(idx).diff().median().total_seconds() / 60)
    mod = (idx.hour * 60 + idx.minute).to_numpy()
    close_mod = (mod + tf_min) % (24 * 60)
    wday = idx.weekday.to_numpy()
    tday = pd.factorize(df["trade_day"])[0]
    tday_vals = df["trade_day"].to_numpy()

    long_sig = _arr(sig, "long", n, False)
    short_sig = _arr(sig, "short", n, False)
    stop_dist = _arr(sig, "stop_dist", n, np.nan)
    exit_long = _arr(sig, "exit_long", n, False)
    exit_short = _arr(sig, "exit_short", n, False)
    trail = _arr(sig, "trail_dist", n, np.nan)
    target_r = sig.get("target_r")
    be_r = sig.get("be_r")
    max_bars = sig.get("max_bars")

    s = SESSIONS[cfg.session]
    in_window = (close_mod >= s["start"]) & (close_mod <= s["end"]) & ~((mod >= 16 * 60) & (mod < 18 * 60))
    if cfg.session == "all":
        in_window &= ~((close_mod > s["flatten"] - 10) & (close_mod < 18 * 60 + 5))
    if cfg.eia_filter:
        in_window &= ~((wday == 2) & (close_mod >= 10 * 60 + 15) & (close_mod <= 10 * 60 + 45))
    past_flatten = (mod >= s["flatten"]) & (mod < 18 * 60)

    POINT_VALUE, TICK = cfg.point_value, cfg.tick
    slip = cfg.slippage_ticks * TICK
    cost_pc = cfg.commission_rt + 2 * slip * POINT_VALUE  # dollars per contract per round trip

    trades: List[Trade] = []
    pos = 0
    entry = stop = tgt = risk_pts = 0.0
    contracts = 0
    entry_i = -1
    best = 0.0
    pending = 0          # +1/-1 entry to fill at this bar's open
    pending_dist = 0.0
    pending_exit = False
    day = -1
    day_pnl = 0.0
    day_trades = 0

    def close(i, px, reason):
        nonlocal pos, day_pnl
        pnl = (px - entry) * pos * POINT_VALUE * contracts - cfg.commission_rt * contracts
        r = pnl / (risk_pts * POINT_VALUE * contracts + cost_pc * contracts)
        trades.append(Trade(idx[entry_i], idx[i], tday_vals[entry_i], pos, entry, px, entry - pos * risk_pts,
                            contracts, pnl, r, reason))
        day_pnl += pnl
        pos = 0

    for i in range(n):
        if tday[i] != day:
            day = tday[i]
            day_pnl = 0.0
            day_trades = 0

        # --- fills at this bar's open ---
        if pos != 0 and (pending_exit or tday[i] != tday[entry_i] or past_flatten[i]):
            close(i, o[i] - pos * slip, "signal" if pending_exit else "flatten")
        pending_exit = False

        if pending != 0 and pos == 0 and tday[i] == tday[i - 1] and not past_flatten[i]:
            pos = pending
            entry = o[i] + pos * slip
            risk_pts = pending_dist
            stop = entry - pos * risk_pts
            tgt = entry + pos * risk_pts * target_r if target_r else np.nan
            contracts = int(min(cfg.max_contracts, cfg.risk_usd // (risk_pts * POINT_VALUE + cost_pc)))
            entry_i = i
            best = entry
            day_trades += 1
        pending = 0

        # --- intrabar stop / target ---
        if pos != 0:
            if pos == 1:
                if o[i] <= stop:
                    close(i, o[i] - slip, "stop")
                elif lo[i] <= stop:
                    close(i, stop - slip, "stop")
                elif target_r and h[i] >= tgt:
                    close(i, tgt, "target")
            else:
                if o[i] >= stop:
                    close(i, o[i] + slip, "stop")
                elif h[i] >= stop:
                    close(i, stop + slip, "stop")
                elif target_r and lo[i] <= tgt:
                    close(i, tgt, "target")

        # --- end-of-bar management (uses close of bar i) ---
        if pos != 0:
            best = max(best, h[i]) if pos == 1 else min(best, lo[i])
            if be_r and (best - entry) * pos >= be_r * risk_pts:
                be = entry + pos * 2 * TICK
                stop = max(stop, be) if pos == 1 else min(stop, be)
            if not np.isnan(trail[i]):
                t = best - pos * trail[i]
                stop = max(stop, t) if pos == 1 else min(stop, t)
            if (pos == 1 and exit_long[i]) or (pos == -1 and exit_short[i]) or \
                    (max_bars and i - entry_i + 1 >= max_bars):
                pending_exit = True

        if pos == 0 and i + 1 < n and in_window[i] and day_trades < cfg.max_trades_day \
                and -day_pnl < cfg.daily_loss_stop and day_pnl < cfg.daily_profit_lock:
            d = stop_dist[i]
            if (long_sig[i] or short_sig[i]) and np.isfinite(d) and d > 0 and d <= cfg.max_stop_dist \
                    and cfg.risk_usd // (d * POINT_VALUE + cost_pc) >= 1:
                pending = 1 if long_sig[i] else -1
                pending_dist = max(d, 3 * TICK)

    if pos != 0:
        close(n - 1, c[n - 1], "end")
    return trades
