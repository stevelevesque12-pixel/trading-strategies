from dataclasses import replace
from datetime import datetime, time, timedelta

import numpy as np
import pandas as pd
import pytest

from failed2s.instruments import INSTRUMENTS
from morning_dip.backtest import MorningDipBacktest
from morning_dip.strategy import Candle, MorningDipConfig, build_candles, order_for

T0 = datetime(2026, 1, 5, 9, 0)
NQ = INSTRUMENTS["NQ"]

# Small lookbacks and 1-minute candles so a few minutes of data set up a signal.
# Candles closing 100, 101, 100 with a 2-point range each give ATR=2, ER=0, so
# the candle closing at 09:03:00 arms: limit 98, stop 95 (1.5 ATR = 3), target
# 100, live from 09:03:01 until 09:12:00. After that price sits flat at 99.5,
# which never signals (ER is undefined on a dead-flat stretch).
CFG = MorningDipConfig(
    candle_seconds=60,
    atr_length=2,
    er_length=2,
    feed_start=time(9, 0),
    signal_start=time(9, 0),
)
QUIET = 99.5


def series(overrides=None, end=1800, missing=()):
    """Dense 1-second bars from 09:00 CT; `overrides` maps second -> (o, h, l, c)."""
    rows = {}
    for s in range(end):
        k = s // 60
        p = (100, 101, 100)[k] if k < 3 else QUIET
        hi, lo = (p + 1, p - 1) if k < 3 and s % 60 == 30 else (p, p)
        rows[s] = (p, hi, lo, p)
    rows.update(overrides or {})
    secs = [s for s in sorted(rows) if s not in set(missing)]
    idx = pd.DatetimeIndex([T0 + timedelta(seconds=s) for s in secs]).tz_localize("America/Chicago")
    return pd.DataFrame([rows[s] for s in secs], index=idx, columns=["open", "high", "low", "close"])


def run(overrides=None, cfg=CFG, **kw):
    return MorningDipBacktest(NQ, cfg, bar_seconds=1).run(series(overrides, **kw))


def epoch(seconds_after_t0):
    return int(pd.Timestamp(T0 + timedelta(seconds=seconds_after_t0), tz="America/Chicago").timestamp())


# -- candles and indicators -------------------------------------------------

def candles_from(bars_by_minute, cfg=CFG):
    """One 1-second bar per candle is unhealthy, so fill every second of each minute."""
    t, o, h, l, c = [], [], [], [], []
    for k, (op, hi, lo, cl) in enumerate(bars_by_minute):
        for s in range(60):
            t.append(epoch(60 * k + s))
            o.append(op if s == 0 else cl)
            h.append(hi if s == 30 else max(op, cl))
            l.append(lo if s == 30 else min(op, cl))
            c.append(cl)
    t.append(epoch(60 * len(bars_by_minute)))  # one bar past the end so the last candle is complete
    for arr, v in ((o, c[-1]), (h, c[-1]), (l, c[-1]), (c, c[-1])):
        arr.append(v)
    return build_candles(np.array(t), *(np.array(x, float) for x in (o, h, l, c)), cfg)


def test_atr_is_simple_mean_of_true_ranges():
    cfg = replace(CFG, atr_length=3)
    cs = candles_from([(100, 102, 99, 101), (101, 101, 100, 100), (104, 106, 104, 105)], cfg)
    assert cs[1].atr is None
    # TRs: 3 (first candle: high - low), 1, then 106 - 100 = 6 across the gap up
    assert cs[2].atr == pytest.approx((3 + 1 + 6) / 3)


def test_efficiency_ratio():
    cfg = replace(CFG, er_length=3)
    cs = candles_from([(p, p, p, p) for p in (100, 102, 101, 104)], cfg)
    assert cs[2].er is None  # needs er_length + 1 closes
    assert cs[3].er == pytest.approx(4 / 6)  # net 100 -> 104 = 4; path 2 + 1 + 3


def test_flat_closes_have_no_er():
    cs = candles_from([(100, 101, 99, 100)] * 3)
    assert cs[2].er is None


def test_gap_inside_a_candle_makes_it_unhealthy_and_restarts_warmup():
    df = series(missing=range(125, 160))  # a 35 s hole in the 09:02 candle
    t = df.index.as_unit("s").asi8
    cs = build_candles(t, *(df[k].to_numpy() for k in ("open", "high", "low", "close")), CFG)
    assert [c.healthy for c in cs[:4]] == [True, True, False, True]
    assert cs[3].atr is None  # the segment restarted after the bad candle


def test_order_levels():
    cd = Candle(start=epoch(120), end=epoch(180), open=101, high=101, low=99, close=100, healthy=True, atr=2.1, er=0.2)
    order = order_for(cd, CFG, 0.25, flat=epoch(7200))
    assert order.limit_price == 97.75  # 100 - 2.1 = 97.9, rounded down
    assert order.stop_price == 97.75 - 3.25  # 1.5 x 2.1 = 3.15, rounded up
    assert order.target_price == 100.0  # 97.75 + ceil(100 - 97.75)
    assert (order.arm, order.expire) == (epoch(181), epoch(180 + 540))


def test_stop_is_at_least_4_ticks():
    cd = Candle(start=0, end=epoch(180), open=100, high=101, low=99, close=100, healthy=True, atr=0.5, er=0.2)
    order = order_for(cd, CFG, 0.25, flat=epoch(7200))
    assert order.limit_price - order.stop_price == 1.0


def test_skips_tiny_reward_trending_and_out_of_window():
    base = Candle(start=0, end=epoch(180), open=100, high=101, low=99, close=100, healthy=True, atr=2.0, er=0.2)
    assert order_for(replace(base, high=98.5, low=97.5), CFG, 0.25, epoch(7200)) is None  # target < 2 ticks away
    assert order_for(replace(base, er=0.36), CFG, 0.25, epoch(7200)) is None
    assert order_for(base, replace(CFG, signal_start=time(9, 30)), 0.25, epoch(7200)) is None
    assert order_for(replace(base, healthy=False), CFG, 0.25, epoch(7200)) is None


# -- fill simulation --------------------------------------------------------

def test_fill_and_target_both_need_one_tick_through():
    trades = run({181: (99, 99, 98.0, 98.5), 182: (98.5, 98.5, 97.75, 98),
                  200: (99, 100, 99, 99.5), 201: (99.5, 100.25, 99.5, 99.5)})
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_time.second == 2  # 98.00 touched at :01 but only 97.75 fills
    assert t.exit_time.second == 21  # 100.00 touched at :20 but only 100.25 exits
    assert (t.entry_price, t.exit_price, t.exit_reason) == (98.0, 100.0, "target")
    assert t.pnl_dollars == pytest.approx(2 * 20 - 2 * 2.25)


def test_signal_bar_itself_cannot_fill():
    # The 09:03:00 bar starts before the order arms at 09:03:01.
    assert run({180: (99, 99, 97, 98)}) == []


def test_fill_second_can_stop_but_not_target():
    trades = run({181: (99, 100.5, 97.75, 100), 182: (100, 100.25, 99.5, 100)})
    assert (trades[0].exit_time.second, trades[0].exit_reason) == (2, "target")

    trades = run({181: (99, 99, 94, 94.5)})
    assert trades[0].exit_reason == "stop"
    assert trades[0].exit_price == 95.0 - 0.25  # 1 tick slippage


def test_adverse_move_first_within_a_bar():
    trades = run({181: (98, 98, 97.5, 97.75), 190: (97, 100.5, 94.5, 99)})
    assert trades[0].exit_reason == "stop"


def test_stop_gap_fills_at_open():
    trades = run({181: (98, 98, 97.5, 97.75), 190: (93, 93.5, 92, 93)})
    assert (trades[0].exit_reason, trades[0].exit_price) == ("stop_gap", 93 - 0.25)


def test_time_stop_after_15_minutes():
    trades = run({181: (98, 98, 97.5, 97.75)})
    t = trades[0]
    assert t.exit_reason == "time_stop"
    assert t.exit_time - t.entry_time == timedelta(minutes=15)
    assert t.exit_price == QUIET - 0.25


def test_order_cancels_9_minutes_after_the_signal_close():
    assert run({720: (99, 99, 97, 97)}) == []  # 09:12:00 -- expired
    assert len(run({719: (99, 99, 97, 97)})) == 1


def test_flat_at_11():
    trades = run({181: (98, 98, 97.5, 97.75)}, cfg=replace(CFG, time_stop=timedelta(hours=5)), end=7300)
    t = trades[0]
    assert (t.exit_reason, t.exit_time.time(), t.exit_price) == ("session_flatten", time(11, 0), QUIET - 0.25)


def test_feed_gap_cancels_order_and_exits_position():
    assert run({241: (99, 99, 97, 97)}, missing=range(200, 241)) == []

    trades = run({181: (98, 98, 97.5, 97.75)}, missing=range(200, 241))
    assert (trades[0].exit_reason, trades[0].exit_time.second) == ("feed_gap_exit", 1)


def test_one_order_or_position_at_a_time():
    # Choppy tape that keeps re-arming: trades must never overlap.
    rng = np.random.default_rng(0)
    overrides = {}
    p = 100.0
    for s in range(180, 7000):
        p = round((p + rng.choice([-0.5, -0.25, 0, 0.25, 0.5])) * 4) / 4
        overrides[s] = (p, p + 0.25, p - 0.25, p)
    trades = run(overrides, end=7000)
    assert len(trades) > 3
    for a, b in zip(trades, trades[1:]):
        assert b.entry_time > a.exit_time
