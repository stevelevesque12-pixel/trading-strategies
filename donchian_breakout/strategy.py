"""
Intraday lookback high/low (Donchian channel) breakout strategy.

Built on the same idea as tradingview/lookback_high_low.pine: the highest
high and lowest low of the last N candles. Long case (short is the mirror):

  1. Entry signal: a bar CLOSES above the highest high of the previous
     `entry_len` bars of the same RTH session. Fill at the next bar's open.
  2. Optional filter: only take longs above the session VWAP (anchored at
     the RTH open), shorts below it.
  3. Protective stop: signal close - `stop_atr` * ATR(`atr_len`), fixed for
     the life of the trade, checked intrabar against each bar's low.
  4. Trailing exit: a bar CLOSES below the lowest low of the previous
     `exit_len` bars -> exit at the next bar's open. This is the "let
     winners run" part: the exit channel ratchets up behind the trend.
  5. Intraday only: entries only inside [entry_start, entry_end), any open
     position is closed at the close of the first bar at/after `flatten_at`.
     At most `max_trades_per_day` entries per session.

All channels only look at bars from the current RTH session, so the overnight
gap never makes a level. ATR's true range likewise ignores the prior close on
a session's first bar. Signals are evaluated on closed bars only (no
lookahead); the backtest fills at the next bar's open, the same as
TradingView's strategy engine does for an order placed on bar close.
"""

from dataclasses import dataclass
from datetime import time
from typing import List, Optional

import numpy as np
import pandas as pd

from backtest.engine import Trade
from failed2s.instruments import Instrument


@dataclass(frozen=True)
class Params:
    timeframe: str = "5min"
    entry_len: int = 20
    exit_len: int = 10
    atr_len: int = 14
    stop_atr: float = 2.0
    vwap_filter: bool = True
    allow_long: bool = True
    allow_short: bool = True
    entry_start: time = time(9, 45)
    entry_end: time = time(15, 30)
    flatten_at: time = time(15, 50)
    session_start: time = time(9, 30)
    session_end: time = time(16, 0)
    max_trades_per_day: int = 3


@dataclass(frozen=True)
class Costs:
    slippage_ticks: float = 1.0           # per side, on market fills and stops
    commission_per_contract: float = 4.5  # round trip, dollars


def to_rth_bars(df_1m: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Resample (tz-aware, America/New_York) bars to `params.timeframe`, RTH only."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    bars = df_1m.resample(params.timeframe, label="left", closed="left").agg(agg).dropna(how="any")
    t = bars.index.time
    return bars[(t >= params.session_start) & (t < params.session_end)]


def add_indicators(bars: pd.DataFrame, params: Params) -> pd.DataFrame:
    """Channels, ATR and session VWAP, each value known at that bar's close."""
    out = bars.copy()
    day = pd.Series(out.index.date, index=out.index)
    g_high = out["high"].groupby(day)
    g_low = out["low"].groupby(day)

    # Previous-N-bar channel, current session only (NaN until N bars exist).
    out["entry_high"] = g_high.transform(lambda s: s.rolling(params.entry_len).max().shift(1))
    out["entry_low"] = g_low.transform(lambda s: s.rolling(params.entry_len).min().shift(1))
    out["exit_high"] = g_high.transform(lambda s: s.rolling(params.exit_len).max().shift(1))
    out["exit_low"] = g_low.transform(lambda s: s.rolling(params.exit_len).min().shift(1))

    prev_close = out["close"].shift(1)
    first_bar = day.ne(day.shift(1))
    tr = pd.concat([out["high"] - out["low"], (out["high"] - prev_close).abs(), (out["low"] - prev_close).abs()], axis=1).max(axis=1)
    tr[first_bar] = (out["high"] - out["low"])[first_bar]
    out["atr"] = tr.rolling(params.atr_len).mean()

    tp = (out["high"] + out["low"] + out["close"]) / 3.0
    vol = out["volume"].where(out["volume"] > 0, 1.0)  # volume-less data -> plain average
    out["vwap"] = (tp * vol).groupby(day).cumsum() / vol.groupby(day).cumsum()
    return out


def generate_signals(ind: pd.DataFrame, params: Params) -> pd.Series:
    """+1 / -1 / 0 per bar: a breakout close, before position/daily-count gating."""
    t = ind.index.time
    in_window = (t >= params.entry_start) & (t < params.entry_end)
    long_sig = (ind["close"] > ind["entry_high"]) & in_window
    short_sig = (ind["close"] < ind["entry_low"]) & in_window
    if params.vwap_filter:
        long_sig &= ind["close"] > ind["vwap"]
        short_sig &= ind["close"] < ind["vwap"]
    long_sig &= params.allow_long
    short_sig &= params.allow_short
    return pd.Series(np.where(long_sig, 1, np.where(short_sig, -1, 0)), index=ind.index)


def simulate(bars: pd.DataFrame, params: Params, instrument: Instrument,
             costs: Costs = Costs(), contracts: int = 1) -> List[Trade]:
    """Bar-by-bar fill simulation on RTH bars at `params.timeframe`."""
    ind = add_indicators(bars, params)
    sig = generate_signals(ind, params).to_numpy()
    idx = ind.index
    o, h, l, c = (ind[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = ind["atr"].to_numpy()
    exit_hi, exit_lo = ind["exit_high"].to_numpy(), ind["exit_low"].to_numpy()
    dates = np.array(idx.date)
    times = idx.time
    slip = costs.slippage_ticks * instrument.tick_size

    trades: List[Trade] = []
    pos: Optional[dict] = None
    pending_entry: Optional[dict] = None
    pending_exit = False
    trades_today, today = 0, None

    def close(price: float, i: int, reason: str) -> None:
        nonlocal pos
        d = pos["dir"]
        fill = price - slip * d
        pts = (fill - pos["entry"]) * d
        risk = abs(pos["entry"] - pos["stop"])
        trades.append(Trade(
            entry_time=pos["time"], exit_time=idx[i], direction="long" if d == 1 else "short",
            entry_price=pos["entry"], stop_price=pos["stop"], target_price=float("nan"),
            exit_price=fill, exit_reason=reason, contracts=contracts, pnl_points=pts,
            pnl_dollars=pts * instrument.point_value * contracts - costs.commission_per_contract * contracts,
            r_multiple=pts / risk if risk else 0.0,
        ))
        pos = None

    for i in range(len(idx)):
        if dates[i] != today:
            today, trades_today = dates[i], 0
            if pos is not None:  # data gap skipped the flatten bar
                close(o[i], i, "session_flatten")
            pending_entry, pending_exit = None, False

        # Orders placed on the previous bar's close fill at this bar's open.
        if pending_exit and pos is not None:
            close(o[i], i, "trail_exit")
        pending_exit = False
        if pending_entry is not None and pos is None:
            d = pending_entry["dir"]
            pos = {"dir": d, "entry": o[i] + slip * d, "stop": pending_entry["stop"], "time": idx[i]}
            trades_today += 1
        pending_entry = None

        if pos is not None:
            d = pos["dir"]
            if d == 1 and l[i] <= pos["stop"]:
                close(min(o[i], pos["stop"]), i, "stop")
            elif d == -1 and h[i] >= pos["stop"]:
                close(max(o[i], pos["stop"]), i, "stop")

        if pos is not None and times[i] >= params.flatten_at:
            close(c[i], i, "session_flatten")
            continue

        if pos is not None:
            if (pos["dir"] == 1 and c[i] < exit_lo[i]) or (pos["dir"] == -1 and c[i] > exit_hi[i]):
                pending_exit = True
        elif (sig[i] != 0 and trades_today < params.max_trades_per_day
              and not np.isnan(atr[i]) and i + 1 < len(idx) and dates[i + 1] == dates[i]):
            pending_entry = {"dir": int(sig[i]), "stop": c[i] - sig[i] * params.stop_atr * atr[i]}

    if pos is not None:
        close(c[-1], len(idx) - 1, "end_of_data")
    return trades
