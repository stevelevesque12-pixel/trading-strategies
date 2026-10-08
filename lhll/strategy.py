"""
Lower Highs and Lower Lows (LH/LL) time-exit strategy.

Rules:
  1. Signal: a bar whose high is below the previous bar's high AND whose low
     is below the previous bar's low.
  2. Entry: at that bar's close (long by default -- the classic version of
     this setup is a short-term mean-reversion buy; `direction="short"`
     trades the mirror for comparison).
  3. Exit: at the close `hold_bars` bars later (1-10 is the range of
     interest). No stop, no target -- purely a time exit.

One position at a time: signals that fire while a trade is open are
ignored. A signal on the exit bar itself is allowed (exit and re-enter on
the same close).

Intraday use: pass `session` (one label per bar, e.g. the trading date) and
the previous-bar comparison never reaches across sessions, and a trade
whose N-bar exit would land in the next session is closed at its own
session's last bar instead (no overnight holds).
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

import numpy as np
import pandas as pd


@dataclass
class Trade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: Literal["long", "short"]
    entry_price: float
    exit_price: float
    bars_held: int
    pnl_points: float  # after cost_points


def lhll_signal(df: pd.DataFrame, session: Optional[pd.Series] = None) -> pd.Series:
    """True on bars with a lower high and lower low than the previous bar."""
    sig = (df["high"] < df["high"].shift(1)) & (df["low"] < df["low"].shift(1))
    if session is not None:
        sess = pd.Series(np.asarray(session), index=df.index)
        sig &= sess.eq(sess.shift(1))
    return sig.fillna(False)


def backtest_lhll(
    df: pd.DataFrame,
    hold_bars: int,
    direction: Literal["long", "short"] = "long",
    cost_points: float = 0.0,
    session: Optional[pd.Series] = None,
) -> List[Trade]:
    if hold_bars < 1:
        raise ValueError("hold_bars must be >= 1")

    sig = lhll_signal(df, session).to_numpy()
    close = df["close"].to_numpy()
    idx = df.index
    n = len(df)
    sign = 1.0 if direction == "long" else -1.0

    if session is not None:
        sess = np.asarray(session)
        # index of the last bar of each bar's session
        breaks = np.flatnonzero(sess[1:] != sess[:-1])
        ends = np.append(breaks, n - 1)
        session_end = np.repeat(ends, np.diff(np.concatenate(([-1], ends))))
    else:
        session_end = np.full(n, n - 1)

    trades: List[Trade] = []
    free_at = 0  # first bar index a new entry may occur on
    for i in np.flatnonzero(sig):
        if i < free_at:
            continue
        target = i + hold_bars
        if session is None and target > n - 1:
            break  # not enough data left to complete the hold
        j = min(target, session_end[i])
        if j == i:
            continue  # signal on the session's last bar: nothing left to hold
        pnl = sign * (close[j] - close[i]) - cost_points
        trades.append(Trade(idx[i], idx[j], direction, close[i], close[j], j - i, pnl))
        free_at = j
    return trades


def summarize(trades: List[Trade], point_value: float = 1.0, baseline_pts: float = 0.0) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = np.array([t.pnl_points for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    equity = np.cumsum(pnl)
    max_dd = float(np.max(np.maximum.accumulate(np.concatenate(([0.0], equity)))[1:] - equity))
    gross_loss = -losses.sum()
    return {
        "trades": len(pnl),
        "win_pct": round(100 * len(wins) / len(pnl), 1),
        "avg_pts": round(pnl.mean(), 2),
        "total_pts": round(pnl.sum(), 1),
        "pf": round(wins.sum() / gross_loss, 2) if gross_loss else float("inf"),
        "max_dd_pts": round(max_dd, 1),
        "total_$": round(pnl.sum() * point_value, 0),
        # t-stat of mean trade P&L; > ~2 is the bare minimum to take an edge seriously
        "t_stat": _t(pnl),
        # same test against the market's own drift over the hold: is the *signal* adding anything?
        "edge_vs_base_pts": round(pnl.mean() - baseline_pts, 2),
        "edge_t": _t(pnl - baseline_pts),
    }


def _t(x: np.ndarray) -> float:
    if len(x) < 2 or x.std(ddof=1) == 0:
        return float("nan")
    return round(float(x.mean() / (x.std(ddof=1) / np.sqrt(len(x)))), 2)


def baseline_avg_points(df: pd.DataFrame, hold_bars: int, direction: str = "long",
                        session: Optional[pd.Series] = None) -> float:
    """Average N-bar forward move from *every* bar's close: the drift a signal has to beat."""
    fwd = df["close"].shift(-hold_bars) - df["close"]
    if session is not None:
        sess = pd.Series(np.asarray(session), index=df.index)
        fwd = fwd[sess.eq(sess.shift(-hold_bars))]
    avg = float(fwd.dropna().mean())
    return avg if direction == "long" else -avg
