from datetime import time

import pandas as pd

from failed2s.bars import Bar
from ib_break.strategy import IBBreakStrategy


def _bar(hhmm: str, o, h, l, c, day="2025-03-03"):
    return Bar(pd.Timestamp(f"{day} {hhmm}", tz="America/New_York"), o, h, l, c, 100)


def _feed_ib(strat, high_last=True, close=None):
    """Feed 60 one-minute IB bars spanning 100..110; extreme order chosen by `high_last`."""
    for i in range(60):
        ts = (pd.Timestamp("09:30") + pd.Timedelta(minutes=i)).strftime("%H:%M")
        if i == 10:
            o, h, l, c = (102, 103, 100, 101) if high_last else (108, 110, 107, 109)
        elif i == 50:
            o, h, l, c = (108, 110, 107, 109) if high_last else (102, 103, 100, 101)
        else:
            o, h, l, c = 105, 106, 104, 105
        if i == 59 and close is not None:
            c = close
        assert strat.on_entry_bar(_bar(ts, o, h, l, c)) is None


def test_ib_levels_and_order():
    s = IBBreakStrategy()
    _feed_ib(s, high_last=True, close=107)
    assert (s.ib.high, s.ib.low, s.ib.mid, s.ib.order, s.ib.close) == (110, 100, 105, "high_last", 107)


def test_no_signal_during_ib_and_long_on_close_break():
    s = IBBreakStrategy(target_r=2.0)
    _feed_ib(s)
    assert s.on_entry_bar(_bar("10:30", 109, 111, 108, 109.5)) is None  # wick through, close inside
    sig = s.on_entry_bar(_bar("10:31", 109.5, 112, 109, 111))
    assert sig.direction == "long"
    assert (sig.entry_price, sig.stop_price, sig.target_price) == (111, 105, 123)


def test_one_trade_per_day_then_reset_next_day():
    s = IBBreakStrategy()
    _feed_ib(s)
    assert s.on_entry_bar(_bar("10:31", 109, 112, 109, 111)) is not None
    assert s.on_entry_bar(_bar("10:40", 101, 101, 98, 99)) is None
    assert s.ib is not None
    s.on_entry_bar(_bar("09:30", 105, 106, 104, 105, day="2025-03-04"))
    assert s.ib is None


def test_allow_reverse_gives_one_trade_each_way():
    s = IBBreakStrategy(allow_reverse=True)
    _feed_ib(s)
    assert s.on_entry_bar(_bar("10:31", 109, 112, 109, 111)).direction == "long"
    assert s.on_entry_bar(_bar("10:40", 101, 101, 98, 99)).direction == "short"
    assert s.on_entry_bar(_bar("10:50", 109, 113, 109, 112)) is None


def test_opposite_stop_and_ib_extension_target():
    s = IBBreakStrategy(stop_mode="opposite", target_mode="ib_ext", target_ib_ext=1.0)
    _feed_ib(s)
    sig = s.on_entry_bar(_bar("10:31", 101, 101, 98, 99))
    assert sig.direction == "short"
    assert (sig.stop_price, sig.target_price) == (110, 90)


def test_entry_cutoff():
    s = IBBreakStrategy(entry_cutoff=time(12, 0))
    _feed_ib(s)
    assert s.on_entry_bar(_bar("12:00", 109, 112, 109, 111)) is None


def test_bias_filter_both_requires_agreement():
    # close above mid + high printed last -> longs only
    s = IBBreakStrategy(bias_filter="both")
    _feed_ib(s, high_last=True, close=107)
    assert s.on_entry_bar(_bar("10:31", 101, 101, 98, 99)) is None
    assert s.on_entry_bar(_bar("10:32", 109, 112, 109, 111)).direction == "long"

    # close above mid but low printed last -> conflicting, no trade
    s = IBBreakStrategy(bias_filter="both")
    _feed_ib(s, high_last=False, close=107)
    assert s.on_entry_bar(_bar("10:31", 109, 112, 109, 111)) is None
    assert s.on_entry_bar(_bar("10:32", 101, 101, 98, 99)) is None


def test_bias_filter_order_shorts_when_low_last():
    s = IBBreakStrategy(bias_filter="order")
    _feed_ib(s, high_last=False)
    assert s.on_entry_bar(_bar("10:31", 109, 112, 109, 111)) is None
    assert s.on_entry_bar(_bar("10:32", 101, 101, 98, 99)).direction == "short"


def test_touch_entry_fills_at_level_or_gap_open():
    s = IBBreakStrategy(entry_mode="touch")
    _feed_ib(s)
    sig = s.on_entry_bar(_bar("10:31", 109, 111, 108, 109))  # wick through, close back inside
    assert sig.entry_price == 110

    s = IBBreakStrategy(entry_mode="touch")
    _feed_ib(s)
    assert s.on_entry_bar(_bar("10:31", 112, 113, 111.5, 112.5)).entry_price == 112
