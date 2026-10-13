from datetime import datetime, timedelta

import numpy as np
import pandas as pd

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS
from supertrend_atr.backtest import SuperTrendBacktester
from supertrend_atr.indicators import ATR, SuperTrend
from supertrend_atr.strategy import PendingStop, Position, StrategyParams, SuperTrendATRStrategy

T0 = datetime(2026, 1, 5, 9, 30)


def bar(i, o, h, l, c, v=1.0):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c, v)


def test_atr_seeds_with_sma_then_uses_wilder_smoothing():
    atr = ATR(period=3)
    assert atr.update(bar(0, 10, 12, 10, 11)) is None  # TR 2
    assert atr.update(bar(1, 11, 12, 11, 11)) is None  # TR max(1, 1, 0) = 1
    assert atr.update(bar(2, 11, 14, 11, 13)) == (2 + 1 + 3) / 3
    # TR = max(1, |14-13|, |13-13|) = 1 -> (2*2 + 1) / 3
    assert atr.update(bar(3, 13, 14, 13, 13)) == (2 * 2 + 1) / 3


def _reference_supertrend(df, period, factor):
    """Direct transcription of Pine's ta.supertrend (direction +1 = up here)."""
    h, l, c = df.high.values, df.low.values, df.close.values
    tr = np.maximum(h - l, np.maximum(abs(h - np.roll(c, 1)), abs(l - np.roll(c, 1))))
    tr[0] = h[0] - l[0]
    atr = np.full(len(df), np.nan)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, len(df)):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    dirs, up_prev, lo_prev, st_prev = [], None, None, None
    for i in range(len(df)):
        if np.isnan(atr[i]):
            dirs.append(None)
            continue
        src = (h[i] + l[i]) / 2
        up, lo = src + factor * atr[i], src - factor * atr[i]
        if lo_prev is not None and not (lo > lo_prev or c[i - 1] < lo_prev):
            lo = lo_prev
        if up_prev is not None and not (up < up_prev or c[i - 1] > up_prev):
            up = up_prev
        if i == 0 or np.isnan(atr[i - 1]):
            d = -1
        elif st_prev == up_prev:
            d = 1 if c[i] > up else -1
        else:
            d = -1 if c[i] < lo else 1
        st_prev = lo if d == 1 else up
        up_prev, lo_prev = up, lo
        dirs.append(d)
    return dirs


def test_supertrend_matches_pine_reference_on_random_walk():
    rng = np.random.default_rng(7)
    close = 100 + np.cumsum(rng.normal(0, 1, 400))
    df = pd.DataFrame({
        "open": np.r_[close[0], close[:-1]],
        "close": close,
    })
    df["high"] = df[["open", "close"]].max(axis=1) + rng.uniform(0, 1, len(df))
    df["low"] = df[["open", "close"]].min(axis=1) - rng.uniform(0, 1, len(df))
    st = SuperTrend(atr_period=10, factor=3.0)
    got = [st.update(bar(i, r.open, r.high, r.low, r.close)) for i, r in enumerate(df.itertuples())]
    assert got == _reference_supertrend(df, 10, 3.0)
    assert {1, -1} <= set(got)  # actually exercised both directions


def test_supertrend_flags_flip_only_on_the_change_bar():
    st = SuperTrend(atr_period=3, factor=1.0)
    prices = [100, 100, 100, 100, 99, 98, 97, 96, 110, 111, 112]
    flips = []
    for i, p in enumerate(prices):
        st.update(bar(i, p, p + 0.5, p - 0.5, p))
        flips.append(st.flipped)
    assert st.direction == 1
    assert flips.count(True) == 1 and flips[8]


def test_initial_stop_is_4_2_atr_from_fill():
    order = PendingStop("long", 101.0, atr_at_signal=2.0, placed_at=None, bars_left=3)
    assert order.initial_stop(101.25, 4.2) == 101.25 - 8.4
    short = PendingStop("short", 99.0, atr_at_signal=2.0, placed_at=None, bars_left=3)
    assert short.initial_stop(98.75, 4.2) == 98.75 + 8.4


def test_trailing_waits_for_activation_then_only_tightens():
    pos = Position("long", entry_price=100.0, entry_time=None, initial_stop=91.6, stop=91.6, best_price=100.0)
    # ATR 2 -> trail/activation distance 1.2. Best 101 < 100 + 1.2: not active yet.
    assert pos.trail(bar(0, 100, 101, 99.5, 100.5), atr=2.0, trail_mult=0.6, activation_mult=0.6) == 91.6
    assert pos.trail(bar(1, 100.5, 103, 100, 102.5), 2.0, 0.6, 0.6) == 103 - 1.2
    # Pullback bar with a lower high must not loosen the stop.
    assert pos.trail(bar(2, 102.5, 102.6, 102, 102.2), 2.0, 0.6, 0.6) == 103 - 1.2


def test_trailing_short_mirror_and_immediate_activation():
    pos = Position("short", 100.0, None, 108.4, 108.4, 100.0)
    assert pos.trail(bar(0, 100, 100.2, 99.5, 99.8), 2.0, 0.6, 0.0) == 99.5 + 1.2


def test_entry_signal_requires_htf_confirmation_and_volatility():
    p = StrategyParams(st_atr_period=3, st_factor=1.0, atr_period=3, vol_lookback=3, vol_min=0.0)
    s = SuperTrendATRStrategy(p)
    prices = [100, 100, 100, 100, 99, 98, 97, 96]
    for i, px in enumerate(prices):
        s.on_bar(bar(i, px, px + 0.5, px - 0.5, px))
    flip_bar = bar(8, 96, 110.5, 96, 110)
    s.on_bar(flip_bar)
    assert s.st.flipped and s.trend == "long"
    assert s.entry_signal(flip_bar) is None  # HTF SuperTrend not warmed up / not long

    s.p.htf_multiple = 0
    order = s.entry_signal(flip_bar)
    assert order.direction == "long" and order.price == 110.5 + 0.25

    s.p.vol_min = 100.0  # impossible volatility bar -> filtered out
    assert s.entry_signal(flip_bar) is None


def _bt():
    return SuperTrendBacktester(INSTRUMENTS["NQ"], slippage_ticks=1, commission_rt=0.0)


def test_stop_fills_gap_at_open_else_at_price_with_slippage():
    bt = _bt()
    order = PendingStop("long", 100.0, 1.0, None, 3)
    assert bt._entry_fill(order, bar(0, 99, 99.75, 98, 99)) is None
    assert bt._entry_fill(order, bar(0, 99, 101, 98, 100)) == 100.25
    assert bt._entry_fill(order, bar(0, 102, 103, 101, 102)) == 102.25  # gapped through

    pos = Position("long", 100, None, 95, 95, 100)
    assert bt._stop_fill(pos, bar(0, 96, 97, 94, 96)) == 94.75
    assert bt._stop_fill(pos, bar(0, 93, 94, 92, 93)) == 92.75


def test_backtester_end_to_end_on_synthetic_trend():
    # Chop, then a sharp sustained rally, then a selloff -- enough for a long
    # flip, a stop-order fill and a trailing-stop exit.
    idx = pd.date_range("2026-01-05 09:30", periods=300, freq="1min", tz="America/New_York")
    rng = np.random.default_rng(1)
    path = np.r_[100 + rng.normal(0, 0.3, 100).cumsum() * 0.1,
                 np.linspace(100, 140, 120), np.linspace(140, 120, 80)]
    df = pd.DataFrame({"open": path, "close": path + 0.05}, index=idx)
    df["high"] = df[["open", "close"]].max(axis=1) + 0.25
    df["low"] = df[["open", "close"]].min(axis=1) - 0.25
    df["volume"] = 1.0
    p = StrategyParams(st_atr_period=5, st_factor=2.0, atr_period=5, vol_lookback=5, vol_min=0.0, htf_multiple=0)
    trades = SuperTrendBacktester(INSTRUMENTS["NQ"], p, timeframe="1min").run(df)
    longs = [t for t in trades if t.direction == "long"]
    assert longs, "expected the rally to produce a long"
    best = max(longs, key=lambda t: t.pnl_points)
    assert best.exit_reason == "trail" and best.pnl_points > 10
    for t in trades:
        assert t.exit_time >= t.entry_time
