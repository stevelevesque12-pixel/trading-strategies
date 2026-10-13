"""
MA pullback strategy (built for MCL, 30-minute by default).

Rules, short side (the optional long side is the exact mirror):

  1. Bias: fast MA (9) above slow MA (20) -> we only look for shorts.
  2. Setup: wait for a bar to CLOSE below both MAs.
  3. Entry: rest a sell limit at each MA. The first one hit is the trade;
     the other is cancelled (one-cancels-other). While the setup is armed
     and unfilled, both limits are re-priced to the MAs' values at every
     bar close.
  4. The armed setup is cancelled if the bias goes away (fast MA no longer
     above slow MA) or, if `expiry_bars` > 0, after that many bars unfilled.
  5. Exit: stop `stop_ticks` above the fill, target `target_r` * risk below.

The original spec only defined the bias, setup and entry. The stop, target,
expiry and MA type are added parameters, and their defaults are
placeholders rather than tested choices.

Fill model (matches TradingView's default broker emulator closely enough):
orders are placed at a bar's close and can fill on the next bar. A limit sell
fills when the bar's high reaches it, at the limit price, or at the open if
the bar gaps through it. If a bar reaches both the stop and the target,
we assume the stop was hit first. The fill bar itself is checked for a stop
only, because the bar's high came after the fill.
"""

from dataclasses import dataclass
from typing import List, Literal, Optional

import pandas as pd


@dataclass
class MAPullbackConfig:
    fast_len: int = 9
    slow_len: int = 20
    ma_type: Literal["SMA", "EMA"] = "SMA"
    stop_ticks: int = 30
    target_r: float = 1.5
    expiry_bars: int = 0          # 0 = setup never expires while bias holds
    enable_shorts: bool = True
    enable_longs: bool = False    # mirror side, off by default
    tick_size: float = 0.01       # MCL
    point_value: float = 100.0    # MCL: $100 per 1.00 move


@dataclass
class Trade:
    direction: str
    entry_time: object
    entry_price: float
    filled_at: str                # "fast" or "slow" MA
    stop_price: float
    target_price: float
    exit_time: object
    exit_price: float
    exit_reason: str
    pnl_points: float
    pnl_dollars: float
    r_multiple: float


def moving_average(close: pd.Series, length: int, ma_type: str) -> pd.Series:
    if ma_type == "SMA":
        return close.rolling(length).mean()
    if ma_type == "EMA":
        return close.ewm(span=length, adjust=False).mean()
    raise ValueError(f"unknown ma_type {ma_type!r}")


def run(bars: pd.DataFrame, cfg: Optional[MAPullbackConfig] = None) -> List[Trade]:
    """Simulate the strategy over OHLC bars already at the strategy timeframe."""
    cfg = cfg or MAPullbackConfig()
    fast = moving_average(bars["close"], cfg.fast_len, cfg.ma_type).to_numpy()
    slow = moving_average(bars["close"], cfg.slow_len, cfg.ma_type).to_numpy()
    o, h, l, c = (bars[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = bars.index
    stop_dist = cfg.stop_ticks * cfg.tick_size

    trades: List[Trade] = []
    armed: Optional[str] = None      # "short" / "long" while limits are resting
    armed_bars = 0
    orders: Optional[dict] = None    # {"fast": price, "slow": price} live for the current bar
    pos: Optional[dict] = None

    for i in range(len(bars)):
        # ---- 1. Resting entry orders placed at the previous close ----
        filled_this_bar = False
        if orders is not None and pos is None:
            if armed == "short":
                hit = {k: p for k, p in orders.items() if h[i] >= p}
                if hit:
                    which = min(hit, key=hit.get)          # lowest sell limit is reached first
                    fill = max(o[i], hit[which])
                    pos = _open("short", idx[i], fill, which, stop_dist, cfg.target_r)
            else:
                hit = {k: p for k, p in orders.items() if l[i] <= p}
                if hit:
                    which = max(hit, key=hit.get)          # highest buy limit is reached first
                    fill = min(o[i], hit[which])
                    pos = _open("long", idx[i], fill, which, stop_dist, cfg.target_r)
            if pos is not None:
                filled_this_bar = True
                armed, orders = None, None                 # OCO: the other order is cancelled
                # Fill bar: the move after the fill is unknown, so only the stop is checked.
                if (pos["dir"] == "short" and h[i] >= pos["stop"]) or (pos["dir"] == "long" and l[i] <= pos["stop"]):
                    trades.append(_close(pos, idx[i], pos["stop"], "stop", cfg))
                    pos = None

        # ---- 2. Manage an open position (bars after the fill bar) ----
        if pos is not None and not filled_this_bar:
            if pos["dir"] == "short":
                if h[i] >= pos["stop"]:
                    trades.append(_close(pos, idx[i], max(o[i], pos["stop"]), "stop", cfg))
                    pos = None
                elif l[i] <= pos["target"]:
                    trades.append(_close(pos, idx[i], min(o[i], pos["target"]), "target", cfg))
                    pos = None
            else:
                if l[i] <= pos["stop"]:
                    trades.append(_close(pos, idx[i], min(o[i], pos["stop"]), "stop", cfg))
                    pos = None
                elif h[i] >= pos["target"]:
                    trades.append(_close(pos, idx[i], max(o[i], pos["target"]), "target", cfg))
                    pos = None

        # ---- 3. Bar close: evaluate bias / setup and (re)price the limits ----
        orders = None
        if pd.isna(fast[i]) or pd.isna(slow[i]):
            continue
        short_bias = cfg.enable_shorts and fast[i] > slow[i]
        long_bias = cfg.enable_longs and fast[i] < slow[i]

        if (armed == "short" and not short_bias) or (armed == "long" and not long_bias):
            armed = None
        if armed is not None:
            armed_bars += 1
            if cfg.expiry_bars > 0 and armed_bars > cfg.expiry_bars:
                armed = None

        if armed is None and pos is None:
            if short_bias and c[i] < min(fast[i], slow[i]):
                armed, armed_bars = "short", 0
            elif long_bias and c[i] > max(fast[i], slow[i]):
                armed, armed_bars = "long", 0

        if armed is not None and pos is None:
            orders = {"fast": fast[i], "slow": slow[i]}

    return trades


def _open(direction, t, fill, which, stop_dist, target_r) -> dict:
    sign = -1 if direction == "short" else 1
    stop = fill - sign * stop_dist
    return {"dir": direction, "time": t, "entry": fill, "which": which,
            "stop": stop, "target": fill + sign * stop_dist * target_r, "risk": stop_dist}


def _close(pos: dict, t, price: float, reason: str, cfg: MAPullbackConfig) -> Trade:
    sign = -1 if pos["dir"] == "short" else 1
    pts = sign * (price - pos["entry"])
    return Trade(pos["dir"], pos["time"], pos["entry"], pos["which"], pos["stop"], pos["target"],
                 t, price, reason, pts, pts * cfg.point_value, pts / pos["risk"])
