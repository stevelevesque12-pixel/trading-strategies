from dataclasses import replace
from datetime import datetime, time, timedelta

import pandas as pd
import pytest

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS
from morning_dip.backtest import MorningDipBacktest
from morning_dip.strategy import CandleIndicators, MorningDipConfig, MorningDipStrategy

T0 = datetime(2026, 1, 5, 9, 0)
NQ = INSTRUMENTS["NQ"]

# Small lookbacks and 1-minute candles so a handful of bars sets up a signal.
# Three flat candles (100 +/- 1) give ATR=2, ER=0, so the candle closing at
# 09:03 arms: limit 98, stop 95, target 100, live from 09:03:01 to 09:12:01.
CFG = MorningDipConfig(
    candle_seconds=60,
    atr_length=2,
    er_length=2,
    feed_start=time(9, 0),
    signal_start=time(9, 0),
)
SETUP = [(0, 100, 101, 99, 100), (60, 100, 101, 99, 100), (120, 100, 101, 99, 100)]


def frame(rows):
    """rows: (seconds after 09:00 CT, open, high, low, close)."""
    idx = pd.DatetimeIndex([T0 + timedelta(seconds=s) for s, *_ in rows]).tz_localize("America/Chicago")
    data = [r[1:] for r in rows]
    df = pd.DataFrame(data, index=idx, columns=["open", "high", "low", "close"])
    df["volume"] = 1.0
    return df


def run(rows, cfg=CFG):
    return MorningDipBacktest(NQ, cfg, bar_seconds=1).run(frame(SETUP + rows))


def candle(o, h, l, c):
    return Bar(T0, o, h, l, c)


# -- indicators -----------------------------------------------------------

def test_atr_is_simple_mean_of_true_ranges():
    ind = CandleIndicators(atr_length=3, er_length=2)
    ind.update(candle(100, 102, 99, 101))  # TR 3 (no prior close)
    ind.update(candle(101, 101, 100, 100))  # TR 1
    assert ind.atr is None
    ind.update(candle(104, 106, 104, 105))  # gap up: TR = 106 - 100 = 6
    assert ind.atr == pytest.approx((3 + 1 + 6) / 3)


def test_efficiency_ratio():
    ind = CandleIndicators(atr_length=1, er_length=3)
    for close in (100, 102, 101):
        ind.update(candle(close, close, close, close))
    assert ind.er is None  # needs er_length + 1 closes
    ind.update(candle(104, 104, 104, 104))
    # net 100 -> 104 = 4; path 2 + 1 + 3 = 6
    assert ind.er == pytest.approx(4 / 6)


def test_trending_candles_do_not_arm():
    strat = MorningDipStrategy(config=CFG)
    orders = [strat.on_candle(candle(p, p + 1, p - 1, p), pd.Timestamp(T0, tz="America/Chicago") + timedelta(minutes=k))
              for k, p in enumerate((100, 105, 110, 115))]
    assert orders == [None] * 4  # ER = 1.0 > 0.35


def test_order_levels():
    strat = MorningDipStrategy(config=CFG)
    close_time = pd.Timestamp(T0, tz="America/Chicago")
    order = None
    for _ in range(3):
        order = strat.on_candle(candle(100, 101, 99, 100), close_time)
    assert (order.limit_price, order.stop_price, order.target_price) == (98.0, 95.0, 100.0)
    assert order.arm_time == close_time + timedelta(seconds=1)
    assert order.expire_time == order.arm_time + timedelta(minutes=9)


def test_no_signal_outside_window():
    strat = MorningDipStrategy(config=replace(CFG, signal_start=time(9, 30)))
    for _ in range(3):
        order = strat.on_candle(candle(100, 101, 99, 100), pd.Timestamp(T0, tz="America/Chicago"))
    assert order is None


# -- fill simulation --------------------------------------------------------

def test_fill_needs_one_tick_through_then_hits_target():
    trades = run([(181, 99, 99, 98.0, 98.5), (182, 98.5, 99, 97.75, 98), (200, 99, 100, 99, 100)])
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_time.second == 2  # 98.00 touch didn't fill; 97.75 did
    assert (t.entry_price, t.exit_price, t.exit_reason) == (98.0, 100.0, "target")
    assert t.pnl_dollars == pytest.approx(2 * 20 - 2 * 2.25)


def test_signal_bar_itself_cannot_fill():
    # The 09:03:00 bar closes the signal candle; the order only arms at 09:03:01.
    assert run([(180, 99, 99, 97, 98)]) == []


def test_fill_second_can_stop_but_not_target():
    trades = run([(181, 99, 100.5, 97.75, 100), (182, 100, 100.25, 99.5, 100)])
    assert trades[0].exit_time.second == 2
    assert trades[0].exit_reason == "target"

    trades = run([(181, 99, 99, 94, 94.5)])
    assert trades[0].exit_reason == "stop"
    assert trades[0].exit_price == 95.0 - 0.25  # 1 tick slippage


def test_adverse_move_first_within_a_bar():
    trades = run([(181, 98, 98, 97.5, 97.75), (190, 97, 100.5, 94.5, 99)])
    assert trades[0].exit_reason == "stop"


def test_stop_gap_fills_at_open():
    trades = run([(181, 98, 98, 97.5, 97.75), (190, 93, 93.5, 92, 93)])
    assert trades[0].exit_price == 93 - 0.25


def test_time_stop_after_15_minutes():
    quiet = [(181, 98, 98, 97.5, 97.75)] + [(181 + k * 60, 98, 99, 97, 98) for k in range(1, 16)]
    trades = run(quiet)
    t = trades[0]
    assert t.exit_reason == "time_stop"
    assert t.exit_time - t.entry_time == timedelta(minutes=15)
    assert t.exit_price == 98 - 0.25


def test_order_cancels_after_9_minutes():
    # 09:12:01 is exactly 9 minutes after arming -- too late.
    trades = run([(300 + k * 60, 99.5, 99.75, 99.25, 99.5) for k in range(7)] + [(721, 99, 99, 97, 97)])
    assert trades == []


def test_flat_at_11():
    rows = [(181, 98, 98, 97.5, 97.75), (3600 + 1, 98, 99, 97, 98), (7200, 98.5, 99, 98, 98.5)]
    cfg = replace(CFG, time_stop=timedelta(hours=5))
    trades = run(rows, cfg)
    assert trades[0].exit_reason == "session_flatten"
    assert trades[0].exit_time.time() == time(11, 0)
    assert trades[0].exit_price == 98.25


def test_one_order_or_position_at_a_time():
    # Lots of choppy bars that keep re-arming: trades must never overlap.
    rows = []
    for s in range(181, 3000, 7):
        p = 100 + (1 if (s // 60) % 2 else -1)
        rows.append((s, p, p + 1.5, p - 3, p))
    trades = run(rows)
    assert trades
    for a, b in zip(trades, trades[1:]):
        assert b.entry_time >= a.exit_time
