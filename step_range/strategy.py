"""
Step Range Breakout & Trailing Stop -- Python port of the BigBeluga
TradingView indicator of the same name (CC BY-NC-SA 4.0,
https://creativecommons.org/licenses/by-nc-sa/4.0/). Personal research use.

Signal logic mirrors the Pine state machine bar-for-bar (all on bar close):

- SEARCHING: when the range midpoint ((highest(high, length) +
  lowest(low, length)) / 2) equals its value `consolidation_bars` ago,
  freeze the current highest/lowest as the zone -> ZONE_ACTIVE.
- ZONE_ACTIVE: close above the zone top -> long; close below the zone
  bottom -> short. Initial trail = breakout bar low - ATR*mult (long) /
  high + ATR*mult (short). -> TRAILING.
- TRAILING: the trail ratchets with each bar's low/high; a close through
  it ends the trade -> SEARCHING.

The Pine version only draws; the order-fill modeling lives in
`step_range.backtest`.
"""

from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class StepRangeParams:
    length: int = 20              # "Structure Length"
    consolidation_bars: int = 5   # "Consolidation Bars / Length"
    atr_length: int = 14          # "Trailing Stop ATR Length"
    atr_mult: float = 3.0         # "Trailing Stop Multiplier"


@dataclass
class Breakout:
    """One indicator trade: signal bar -> bar whose close crossed the trail."""
    direction: str          # "long" / "short"
    signal_idx: int         # breakout bar (close outside the zone)
    exit_signal_idx: int    # bar whose close crossed the trail (-1 = still open)
    zone_top: float
    zone_bottom: float
    initial_stop: float
    trail: np.ndarray       # trail value in force at the END of each bar, signal_idx..exit_signal_idx


def rma_atr(high: np.ndarray, low: np.ndarray, close: np.ndarray, length: int) -> np.ndarray:
    """Pine's ta.atr: Wilder's RMA of true range, seeded with an SMA."""
    prev_close = np.concatenate([[np.nan], close[:-1]])
    tr = np.nanmax(np.vstack([high - low, np.abs(high - prev_close), np.abs(low - prev_close)]), axis=0)
    tr[0] = high[0] - low[0]
    atr = np.full_like(tr, np.nan)
    if len(tr) < length:
        return atr
    atr[length - 1] = tr[:length].mean()
    alpha = 1.0 / length
    for i in range(length, len(tr)):
        atr[i] = alpha * tr[i] + (1 - alpha) * atr[i - 1]
    return atr


def find_breakouts(df: pd.DataFrame, p: StepRangeParams = StepRangeParams()) -> List[Breakout]:
    high = df["high"].to_numpy(float)
    low = df["low"].to_numpy(float)
    close = df["close"].to_numpy(float)

    hi = df["high"].rolling(p.length).max().to_numpy()
    lo = df["low"].rolling(p.length).min().to_numpy()
    mid = (hi + lo) / 2
    mid_prev = np.concatenate([np.full(p.consolidation_bars, np.nan), mid[:-p.consolidation_bars]])
    # Exact equality is intentional (matches `ta.change(rAvg, len1) == 0`);
    # futures prices sit on a tick grid, so the midpoint is exactly repeatable.
    stable = mid == mid_prev
    atr = rma_atr(high, low, close, p.atr_length)

    out: List[Breakout] = []
    state = "SEARCHING"
    top = bottom = stop = np.nan
    cur: dict = {}

    for i in range(len(close)):
        if state == "SEARCHING":
            if stable[i]:
                top, bottom = hi[i], lo[i]
                state = "ZONE_ACTIVE"
        elif state == "ZONE_ACTIVE":
            if np.isnan(atr[i]):
                continue
            if close[i] > top:
                stop = low[i] - atr[i] * p.atr_mult
                cur = dict(direction="long", signal_idx=i, zone_top=top, zone_bottom=bottom, initial_stop=stop, trail=[stop])
                state = "TRAILING"
            elif close[i] < bottom:
                stop = high[i] + atr[i] * p.atr_mult
                cur = dict(direction="short", signal_idx=i, zone_top=top, zone_bottom=bottom, initial_stop=stop, trail=[stop])
                state = "TRAILING"
        else:  # TRAILING
            if cur["direction"] == "long":
                stop = max(stop, low[i] - atr[i] * p.atr_mult)
                done = close[i] < stop
            else:
                stop = min(stop, high[i] + atr[i] * p.atr_mult)
                done = close[i] > stop
            cur["trail"].append(stop)
            if done:
                out.append(Breakout(exit_signal_idx=i, **{**cur, "trail": np.array(cur["trail"])}))
                state = "SEARCHING"

    if state == "TRAILING":
        out.append(Breakout(exit_signal_idx=-1, **{**cur, "trail": np.array(cur["trail"])}))
    return out
