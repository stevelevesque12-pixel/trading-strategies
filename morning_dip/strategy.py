"""
Morning Dip Limit -- an NQ long-only scalp (source: milkmantrades.com,
"Morning Dip Limit -- a promising lead", built Sep 27 2026).

Rules as published:
  1. Build 3-minute candles from 1-second NQ bars (UTC grid). ATR = simple
     mean of the last 14 true ranges. ER (Kaufman efficiency ratio) =
     |close change over 15 candles| / sum of |close-to-close changes|.
  2. From 09:00 to 11:00 CT, when a candle closes with ER <= 0.35 (choppy),
     arm a buy limit at close - 1.0 x ATR one second later.
  3. The limit fills only if price trades 1 tick through it. Cancel after
     9 minutes.
  4. Target: the signal candle's high-low midpoint. Stop: 1.5 x ATR below
     the limit. Time stop: 15 minutes. Flat at 11:00 CT. One order or
     position at a time.
  5. Fill model: stops and market exits slip 1 tick; the fill second can
     stop but not target; inside each second the adverse move comes first.

Each day is fed from 08:25 CT, so the indicators warm up inside the day
(state resets every session) and the first valid signal is usually after
09:15 CT.

This module is the pure signal side: candle bucketing, the ATR/ER
indicators, and turning a closed candle into a limit order. The fill
simulation (rules 3-5) lives in `morning_dip/backtest.py`.

The author's own caveats: a research lead, not armed. Positive on 2026 NQ,
flat on 2025 NQ, negative on ES, the short side loses, and there's no
bid/ask or queue modelling, so live fills on hard flushes may be worse.

Price rounding isn't specified in the source. Here the limit rounds to the
nearest tick, the stop rounds down (further away) and the target rounds up
(further away) -- the conservative choice for both exits. The author's
reference implementation may differ, so trades won't necessarily match
their published trade list to the tick.
"""

import math
from collections import deque
from dataclasses import dataclass
from datetime import time, timedelta
from typing import Deque, Optional

import pandas as pd

from failed2s.bars import Bar


@dataclass
class MorningDipConfig:
    candle_seconds: int = 180
    candle_offset_seconds: int = 0  # the candle start time: 0, 18 ... 162 s for 3-minute candles
    atr_length: int = 14
    er_length: int = 15
    er_max: float = 0.35  # arm only when ER <= this (choppy)
    dip_atr: float = 1.0  # limit = close - dip_atr * ATR
    stop_atr: float = 1.5  # stop = limit - stop_atr * ATR
    trade_through_ticks: int = 1  # the limit needs price to trade this many ticks through it
    arm_delay_seconds: int = 1
    cancel_after: timedelta = timedelta(minutes=9)
    time_stop: timedelta = timedelta(minutes=15)
    slippage_ticks: int = 1  # on stops and market exits
    commission_per_side: float = 2.25  # per contract
    feed_start: time = time(8, 25)
    signal_start: time = time(9, 0)
    flatten_at: time = time(11, 0)  # also the end of the signal window
    tz: str = "America/Chicago"


@dataclass
class LimitOrder:
    signal_time: object  # the signal candle's close time
    arm_time: object  # the order can fill from this time on
    expire_time: object
    limit_price: float
    stop_price: float
    target_price: float
    atr: float
    er: float


def floor_tick(price: float, tick: float) -> float:
    return math.floor(price / tick + 1e-9) * tick


def ceil_tick(price: float, tick: float) -> float:
    return math.ceil(price / tick - 1e-9) * tick


def round_tick(price: float, tick: float) -> float:
    return round(price / tick) * tick


class CandleIndicators:
    """Rolling ATR (simple mean of true ranges) and efficiency ratio over closed candles."""

    def __init__(self, atr_length: int = 14, er_length: int = 15):
        self.atr_length = atr_length
        self.er_length = er_length
        self.true_ranges: Deque[float] = deque(maxlen=atr_length)
        self.closes: Deque[float] = deque(maxlen=er_length + 1)
        self.prev_close: Optional[float] = None

    def update(self, candle: Bar) -> None:
        if self.prev_close is None:
            tr = candle.high - candle.low
        else:
            tr = max(candle.high, self.prev_close) - min(candle.low, self.prev_close)
        self.true_ranges.append(tr)
        self.closes.append(candle.close)
        self.prev_close = candle.close

    @property
    def atr(self) -> Optional[float]:
        if len(self.true_ranges) < self.atr_length:
            return None
        return sum(self.true_ranges) / self.atr_length

    @property
    def er(self) -> Optional[float]:
        if len(self.closes) < self.er_length + 1:
            return None
        closes = list(self.closes)
        path = sum(abs(b - a) for a, b in zip(closes, closes[1:]))
        if path == 0:
            return 0.0  # a dead-flat stretch is as choppy as it gets
        return abs(closes[-1] - closes[0]) / path


class MorningDipStrategy:
    """Turns each closed candle into an optional buy-limit order."""

    def __init__(self, tick_size: float = 0.25, config: Optional[MorningDipConfig] = None):
        self.tick_size = tick_size
        self.config = config or MorningDipConfig()
        self.indicators = CandleIndicators(self.config.atr_length, self.config.er_length)

    def reset(self) -> None:
        """Clear indicator state -- called at the start of every session day."""
        self.indicators = CandleIndicators(self.config.atr_length, self.config.er_length)

    def on_candle(self, candle: Bar, close_time: pd.Timestamp) -> Optional[LimitOrder]:
        """Feed a closed candle (timestamp = its start); returns an order to arm, if any."""
        cfg = self.config
        self.indicators.update(candle)

        t = close_time.time()
        if not (cfg.signal_start <= t < cfg.flatten_at):
            return None

        atr, er = self.indicators.atr, self.indicators.er
        if atr is None or er is None or atr <= 0 or er > cfg.er_max:
            return None

        limit = round_tick(candle.close - cfg.dip_atr * atr, self.tick_size)
        stop = floor_tick(limit - cfg.stop_atr * atr, self.tick_size)
        target = ceil_tick((candle.high + candle.low) / 2.0, self.tick_size)
        if target <= limit:
            return None  # can't happen with a positive dip, but never arm an inverted bracket

        arm_time = close_time + timedelta(seconds=cfg.arm_delay_seconds)
        return LimitOrder(
            signal_time=close_time,
            arm_time=arm_time,
            expire_time=arm_time + cfg.cancel_after,
            limit_price=limit,
            stop_price=stop,
            target_price=target,
            atr=atr,
            er=er,
        )
