"""
Morning Dip Limit -- an NQ long-only scalp (source: milkmantrades.com,
"Morning Dip Limit -- a promising lead", built Sep 27 2026).

Rules, matched to the author's reference implementation:
  1. Build 3-minute candles from 1-second NQ bars on a UTC grid (candle
     start = t // 180 * 180 + offset). ATR = simple mean of the last 14
     true ranges. ER (Kaufman efficiency ratio) = |close change over 15
     candles| / sum of |close-to-close changes|.
  2. From 09:00 to 11:00 CT (by candle close time), when a candle closes
     with ER <= 0.35 (choppy), rest a buy limit at close - 1.0 x ATR,
     rounded down to a tick, one second later.
  3. The limit fills only if price trades 1 tick through it. Cancel 9
     minutes after the signal candle closes (or at 11:00, if sooner).
  4. Target: the signal candle's high-low midpoint (rounded up to a tick,
     needs a 1-tick trade-through too). Stop: 1.5 x ATR below the limit
     (rounded up to a tick, at least 4 ticks). Time stop 15 minutes. Flat
     at 11:00 CT. One order or position at a time.
  5. Fill model: stops and market exits slip 1 tick; the fill second can
     stop but not target; inside each second the adverse move comes first.
  Skipped signals: a dip under 1 tick or a target under 2 ticks away.

Each day is fed from 08:25 CT, so the indicators warm up inside the day and
the first valid signal is usually after 09:15 CT.

Data health (also from the reference): a candle is unhealthy if its data has
a gap over 30 s, its last bar is more than 5 s before its close (1-second
data), it's the partial first/last candle of the day's feed, or -- on
1-minute data -- it's missing a bar. An unhealthy candle can't signal, and
it restarts the ATR/ER warm-up (as does a hole between two candles).

This module is the pure signal side: candles, indicators and the order a
candle produces. The fill simulation lives in `morning_dip/backtest.py`.
Times here are UTC epoch seconds.

The author's own caveats: a research lead, not armed. Positive on 2026 NQ,
flat on 2025 NQ, negative on ES, the short side loses, and there's no
bid/ask or queue modelling, so live fills on hard flushes may be worse.
"""

import math
from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import List, Optional
from zoneinfo import ZoneInfo

import numpy as np

EPS = 1e-8


@dataclass(frozen=True)
class MorningDipConfig:
    candle_seconds: int = 180
    candle_offset_seconds: int = 0  # the candle start time: 0, 18 ... 162 s for 3-minute candles
    atr_length: int = 14
    er_length: int = 15
    er_max: float = 0.35  # arm only when ER <= this (choppy)
    dip_atr: float = 1.0  # limit = close - dip_atr * ATR
    stop_atr: float = 1.5  # stop = limit - stop_atr * ATR
    min_stop_ticks: int = 4
    min_dip_ticks: int = 1
    min_reward_ticks: int = 2
    entry_through_ticks: int = 1  # the limit needs price to trade this far through it
    target_through_ticks: int = 1  # so does the target
    slippage_ticks: int = 1  # on stops and market exits
    arm_delay_seconds: int = 1  # 1-second data only; coarser data arms at the close
    cancel_after: timedelta = timedelta(minutes=9)  # from the signal candle's close
    time_stop: timedelta = timedelta(minutes=15)
    max_gap_seconds: int = 30
    max_stale_seconds: int = 5
    commission_per_side: float = 2.25  # per contract
    feed_start: time = time(8, 25)
    feed_end: time = time(15, 0)
    signal_start: time = time(9, 0)
    flatten_at: time = time(11, 0)  # also the end of the signal window
    tz: str = "America/Chicago"


@dataclass
class Candle:
    start: int
    end: int
    open: float
    high: float
    low: float
    close: float
    healthy: bool
    atr: Optional[float] = None
    er: Optional[float] = None


@dataclass
class LimitOrder:
    signal_end: int  # the signal candle's close
    arm: int  # can fill on bars starting at/after this
    expire: int
    limit_price: float
    stop_price: float
    target_price: float
    atr: float
    er: float


def floor_tick(price: float, tick: float) -> float:
    return math.floor(price / tick + EPS) * tick


def ceil_tick(price: float, tick: float) -> float:
    return math.ceil(price / tick - EPS) * tick


def build_candles(
    t: np.ndarray,
    o: np.ndarray,
    h: np.ndarray,
    l: np.ndarray,
    c: np.ndarray,
    config: MorningDipConfig,
    resolution: int = 1,
) -> List[Candle]:
    """One day's base bars (t = bar start, epoch seconds) -> candles with ATR/ER."""
    cfg = config
    if len(t) == 0:
        return []
    tf, off = cfg.candle_seconds, cfg.candle_offset_seconds
    buckets = (t - off) // tf * tf + off
    starts = np.r_[0, 1 + np.flatnonzero(np.diff(buckets))]
    ends = np.r_[starts[1:], len(t)]
    gaps = np.r_[resolution, np.diff(t)]  # each bar's gap from the bar before it

    candles: List[Candle] = []
    true_ranges: List[float] = []
    seg_start = 0
    prev_healthy = False
    for j, (s, e) in enumerate(zip(starts, ends)):
        start = int(buckets[s])
        end = start + tf
        hi, lo = float(h[s:e].max()), float(l[s:e].min())
        stale = end - int(t[e - 1]) - resolution
        healthy = (stale <= cfg.max_stale_seconds if resolution == 1 else stale == 0)
        healthy = healthy and gaps[s:e].max() <= max(cfg.max_gap_seconds, resolution)
        healthy = healthy and start >= t[0] and end <= t[-1] + resolution
        if resolution == 60:
            healthy = healthy and e - s == tf // 60

        if j == 0 or start != candles[-1].end or not healthy or not prev_healthy:
            seg_start = j
        if j == seg_start:
            tr = hi - lo
        else:
            pc = candles[-1].close
            tr = max(hi - lo, abs(hi - pc), abs(lo - pc))
        true_ranges.append(tr)

        cd = Candle(start, end, float(o[s]), hi, lo, float(c[e - 1]), healthy)
        if j - seg_start + 1 >= cfg.atr_length:
            cd.atr = sum(true_ranges[j - cfg.atr_length + 1 : j + 1]) / cfg.atr_length
        if j - seg_start >= cfg.er_length:
            closes = [x.close for x in candles[j - cfg.er_length :]] + [cd.close]
            path = sum(abs(b - a) for a, b in zip(closes, closes[1:]))
            if path > 0:  # a dead-flat stretch has no ER, so it can't signal
                cd.er = abs(closes[-1] - closes[0]) / path
        candles.append(cd)
        prev_healthy = healthy
    return candles


def order_for(
    candle: Candle,
    config: MorningDipConfig,
    tick: float,
    flat: int,
    resolution: int = 1,
) -> Optional[LimitOrder]:
    """The buy limit a closed candle arms, or None. `flat` = the 11:00 CT cutoff (epoch s)."""
    cfg = config
    if not candle.healthy or candle.atr is None or candle.atr <= 0 or candle.er is None:
        return None

    close_tod = datetime.fromtimestamp(candle.end, ZoneInfo(cfg.tz)).time()
    if not (cfg.signal_start <= close_tod < cfg.flatten_at):
        return None

    atr = candle.atr
    limit = floor_tick(candle.close - cfg.dip_atr * atr, tick)
    if candle.close - limit < cfg.min_dip_ticks * tick - EPS:
        return None
    reward = ceil_tick((candle.high + candle.low) / 2.0 - limit, tick)
    if reward < cfg.min_reward_ticks * tick - EPS:
        return None
    if candle.er > cfg.er_max:
        return None

    arm = candle.end + (cfg.arm_delay_seconds if resolution == 1 else 0)
    expire = min(candle.end + int(cfg.cancel_after.total_seconds()), flat)
    if arm >= expire:
        return None

    stop_distance = max(cfg.min_stop_ticks * tick, ceil_tick(cfg.stop_atr * atr, tick))
    return LimitOrder(
        signal_end=candle.end,
        arm=arm,
        expire=expire,
        limit_price=limit,
        stop_price=limit - stop_distance,
        target_price=limit + reward,
        atr=atr,
        er=candle.er,
    )
