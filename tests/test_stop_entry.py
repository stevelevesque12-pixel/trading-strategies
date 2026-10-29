from datetime import datetime, timedelta

import pandas as pd

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS
from structure_scalp.stop_entry import StopEntryScalpStrategy
from backtest.scalp_engine import StopEntryBacktest

T0 = datetime(2026, 1, 5, 10, 0)


def bar(i, o, h, l, c, v=1.0):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c, v)


# Swing low at 95 (bar 2), swing high at 105 (bar 5), then price sits below 105.
SETUP = [
    bar(0, 100, 101, 98, 99),
    bar(1, 99, 100, 97, 98),
    bar(2, 98, 99, 95, 97),
    bar(3, 97, 101, 96, 100),
    bar(4, 100, 102, 99, 101),
    bar(5, 101, 105, 100, 103),
    bar(6, 103, 104, 101, 102),
    bar(7, 102, 103, 100, 101),
]


def _long_biased():
    strat = StopEntryScalpStrategy(tick_size=0.25, target_r=1.0)
    strat.on_bias_bar(bar(-60, 100, 101, 99, 100))  # sets the session date
    strat.bias.direction = "long"
    return strat


def test_buy_stop_rests_above_swing_high_with_stop_below_swing_low():
    strat = _long_biased()
    order = None
    for b in SETUP:
        order = strat.on_entry_bar(b)
    assert order is not None
    assert order.direction == "long"
    assert order.trigger == 105.25
    assert order.stop_price == 94.75
    assert order.target_for(105.25) == 115.75


def test_no_order_without_bias():
    strat = StopEntryScalpStrategy(tick_size=0.25)
    assert all(strat.on_entry_bar(b) is None for b in SETUP)


def test_each_swing_trades_once():
    strat = _long_biased()
    order = None
    for b in SETUP:
        order = strat.on_entry_bar(b)
    strat.mark_filled(order)
    assert strat.on_entry_bar(bar(8, 101, 102, 100, 101)) is None


def _frame(bars):
    idx = pd.DatetimeIndex([b.timestamp for b in bars]).tz_localize("America/New_York")
    return pd.DataFrame(
        {"open": [b.open for b in bars], "high": [b.high for b in bars], "low": [b.low for b in bars],
         "close": [b.close for b in bars], "volume": [b.volume for b in bars]},
        index=idx,
    )


def _engine(strat):
    return StopEntryBacktest(INSTRUMENTS["NQ"], entry_tf="1min", bias_tf="12min", strategy=strat)


def _tz_strat():
    strat = StopEntryScalpStrategy(tick_size=0.25, target_r=1.0)
    strat.bias.update = lambda b: None  # pin the bias for engine tests
    strat.bias.direction = "long"
    return strat


def test_engine_fills_on_next_bar_and_hits_target():
    bars = SETUP + [bar(8, 102, 106, 101.5, 105.5), bar(9, 105.5, 116, 105, 115)]
    trades = _engine(_tz_strat()).run_frame(_frame(bars))
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_price == 105.25 and t.exit_reason == "target" and t.exit_price == 115.75


def test_engine_same_bar_stop_and_target_counts_as_loss():
    bars = SETUP + [bar(8, 102, 120, 90, 100)]
    trades = _engine(_tz_strat()).run_frame(_frame(bars))
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop" and trades[0].pnl_points < 0


def test_engine_gap_through_trigger_fills_at_open():
    bars = SETUP + [bar(8, 107, 108, 106, 107)]
    trades = _engine(_tz_strat()).run_frame(_frame(bars))
    assert trades[0].entry_price == 107
