"""
Futures order-flow imbalance strategy, ported from a QuantConnect algorithm.

Original rules (per instrument, 1-minute RTH bars, 1 contract):
  * SMA(100) of close, and a "price entropy" indicator: Shannon entropy
    (base 2) of the distribution of the last 10 closes (how many distinct
    prices, how evenly repeated), plus a 20-period average of that entropy.
  * Book imbalance: bid_size > 3 * ask_size -> bid imbalance;
    ask_size > 3 * bid_size -> ask imbalance (top-of-book sizes).
  * Flat and close > SMA and entropy < entropy-MA ("trending, low noise"):
        follow the imbalance  (bid imb -> long, ask imb -> short)
  * Flat and close < SMA and entropy > entropy-MA:
        fade the imbalance    (bid imb -> short, ask imb -> long)
  * Exit at +/-1% of the signal close (take profit / stop), checked on each
    1-minute close. No time stop, no end-of-day flatten.

Two things don't carry over 1:1:

1. **No quote data here.** The repo's datasets are OHLCV trade bars, so
   bid_size/ask_size don't exist. `imbalance_mode="clv"` approximates them by
   splitting bar volume by where the close sits in the range
   (buy = vol*(C-L)/(H-L), sell = vol*(H-C)/(H-L)). With that split
   `buy > 3*sell` is exactly `CLV > 0.75`, so the proxy is really a
   close-location filter, not true resting-book imbalance. Treat results as
   a test of the *rule structure*, not of the original quote-size signal.

2. **Entropy-MA bug in the original.** `_get_entropy_ma` sums 20 values but
   divides by `self._period` (10), so `ma_value` is 2x the real average.
   With entropy of 10 one-minute closes mostly in the 2.5-3.3 bit range,
   `entropy < 2*avg` is nearly always true and `entropy > 2*avg` is nearly
   never true. In practice the original only takes the "close > SMA,
   follow the imbalance" branch. `fix_entropy_ma=False` reproduces that;
   `True` divides by 20 as presumably intended.
"""

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import pandas as pd


@dataclass
class OrderFlowParams:
    sma_period: int = 100
    entropy_period: int = 10
    entropy_ma_period: int = 20
    imbalance_ratio: float = 3.0
    take_profit_pct: float = 0.01
    stop_loss_pct: float = 0.01
    fix_entropy_ma: bool = False
    imbalance_mode: str = "clv"  # "clv" (OHLCV proxy) or "quote" (needs bid_size/ask_size columns)
    rth_start: str = "09:30"
    rth_end: str = "15:59"  # last 1m bar-open of the regular session


@dataclass
class Trade:
    direction: int  # +1 long, -1 short
    entry_time: pd.Timestamp
    entry_price: float
    exit_time: Optional[pd.Timestamp] = None
    exit_price: Optional[float] = None
    reason: str = ""
    branch: str = ""  # "follow" (close>SMA, low entropy) or "fade" (close<SMA, high entropy)

    @property
    def points(self) -> float:
        return (self.exit_price - self.entry_price) * self.direction


def rolling_entropy(close: np.ndarray, period: int) -> np.ndarray:
    """Shannon entropy (bits) of the value distribution in each trailing window."""
    out = np.full(len(close), np.nan)
    for i in range(period - 1, len(close)):
        _, counts = np.unique(close[i - period + 1 : i + 1], return_counts=True)
        p = counts / period
        out[i] = -(p * np.log2(p)).sum()
    return out


def compute_indicators(df: pd.DataFrame, p: OrderFlowParams) -> pd.DataFrame:
    """Indicators on the RTH-only 1m series (the original ran with extended_market_hours=False)."""
    out = df.copy()
    close = out["close"].to_numpy(dtype=float)
    out["sma"] = out["close"].rolling(p.sma_period).mean()
    ent = rolling_entropy(close, p.entropy_period)
    out["entropy"] = ent
    divisor = p.entropy_ma_period if p.fix_entropy_ma else p.entropy_period
    out["entropy_ma"] = pd.Series(ent, index=out.index).rolling(p.entropy_ma_period).sum() / divisor

    if p.imbalance_mode == "quote":
        bid = out["bid_size"].astype(float)
        ask = out["ask_size"].astype(float)
    elif p.imbalance_mode == "clv":
        rng = (out["high"] - out["low"]).replace(0, np.nan)
        vol = out["volume"].astype(float)
        bid = (vol * (out["close"] - out["low"]) / rng).fillna(0.0)
        ask = (vol * (out["high"] - out["close"]) / rng).fillna(0.0)
    else:
        raise ValueError(f"unknown imbalance_mode {p.imbalance_mode!r}")
    delta = bid - ask
    out["bid_imb"] = (bid > ask * p.imbalance_ratio) & (delta > 0)
    out["ask_imb"] = (ask > bid * p.imbalance_ratio) & (delta < 0)
    return out


def run(df: pd.DataFrame, p: OrderFlowParams = OrderFlowParams()) -> List[Trade]:
    """
    Bar-by-bar simulation mirroring the QC on_data loop: signals and exits are
    evaluated on each 1m close and fill at that close (QC fills minute-data
    market orders at the current price). Positions carry across sessions.
    """
    rth = df.between_time(p.rth_start, p.rth_end)
    ind = compute_indicators(rth, p)

    times = ind.index
    c = ind["close"].to_numpy(dtype=float)
    sma = ind["sma"].to_numpy()
    en = ind["entropy"].to_numpy()
    en_ma = ind["entropy_ma"].to_numpy()
    bid_imb = ind["bid_imb"].to_numpy()
    ask_imb = ind["ask_imb"].to_numpy()

    trades: List[Trade] = []
    pos: Optional[Trade] = None
    for i in range(len(c)):
        if np.isnan(sma[i]) or np.isnan(en_ma[i]):
            continue
        if pos is None:
            direction = 0
            branch = ""
            if sma[i] < c[i] and en[i] < en_ma[i]:
                direction = 1 if bid_imb[i] else (-1 if ask_imb[i] else 0)
                branch = "follow"
            elif sma[i] > c[i] and en[i] > en_ma[i]:
                direction = -1 if bid_imb[i] else (1 if ask_imb[i] else 0)
                branch = "fade"
            if direction:
                pos = Trade(direction, times[i], c[i], branch=branch)
            continue

        ref = pos.entry_price
        if pos.direction == 1:
            hit_tp = c[i] >= ref * (1 + p.take_profit_pct)
            hit_sl = c[i] <= ref * (1 - p.stop_loss_pct)
        else:
            hit_tp = c[i] <= ref * (1 - p.take_profit_pct)
            hit_sl = c[i] >= ref * (1 + p.stop_loss_pct)
        if hit_tp or hit_sl:
            pos.exit_time, pos.exit_price = times[i], c[i]
            pos.reason = "tp" if hit_tp else "sl"
            trades.append(pos)
            pos = None
    return trades


def summarize(trades: List[Trade], point_value: float, cost_per_rt: float = 0.0) -> dict:
    """cost_per_rt: commission + slippage in dollars per round turn."""
    if not trades:
        return {"num_trades": 0}
    pnl = np.array([t.points * point_value - cost_per_rt for t in trades])
    equity = pnl.cumsum()
    dd = (np.maximum.accumulate(np.concatenate([[0.0], equity]))[1:] - equity).max()
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    hold = pd.Series([t.exit_time - t.entry_time for t in trades])
    return {
        "num_trades": len(trades),
        "longs": sum(t.direction == 1 for t in trades),
        "follow_branch": sum(t.branch == "follow" for t in trades),
        "win_rate_pct": round(100 * len(wins) / len(pnl), 1),
        "total_pnl": round(pnl.sum(), 0),
        "profit_factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() else float("inf"),
        "expectancy": round(pnl.mean(), 0),
        "max_drawdown": round(dd, 0),
        "median_hold": str(hold.median()),
    }
