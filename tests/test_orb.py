from datetime import datetime, time, timedelta

import pytest

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig
from orb.strategy import ORBStrategy

T0 = datetime(2026, 1, 5, 9, 30)  # session open


def bar(i, o, h, l, c, v=100.0):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c, v)


def _or_and_atr_warmup_bars():
    """5 one-minute bars forming a 5-minute OR (9:30-9:35), OR=[98,103],
    plus enough range to get the 3-period trailing ATR warmed up."""
    return [
        bar(0, 100, 101, 99, 100),
        bar(1, 100, 102, 99, 101),
        bar(2, 101, 103, 100, 102),
        bar(3, 102, 102, 100, 101),
        bar(4, 101, 101, 98, 99),
    ]


def test_rejects_or_minutes_that_would_leave_no_entry_window():
    # default no_entry_after=10:15 -- a 60-minute OR (ending 10:30) can never fire.
    with pytest.raises(ValueError, match="no time window left"):
        ORBStrategy(or_minutes=60, atr_period=3)

    # widening no_entry_after to accommodate it is fine.
    session = SessionConfig(no_entry_after=time(11, 0), flatten_at=time(15, 45))
    ORBStrategy(or_minutes=60, atr_period=3, session=session)


def test_opening_range_tracks_high_low_during_formation_and_no_signal_yet():
    strat = ORBStrategy(or_minutes=5, atr_period=3)
    for b in _or_and_atr_warmup_bars():
        assert strat.on_bar(b) is None
    assert strat._or_high == 103
    assert strat._or_low == 98


def test_no_signal_before_atr_warmup_even_past_the_or_window():
    # atr_period=3 but only 2 bars fed after the OR forms is a different story --
    # here we use an OR window itself too short to warm up ATR (need atr_period bars).
    strat = ORBStrategy(or_minutes=2, atr_period=10)
    bars = [bar(0, 100, 101, 99, 100), bar(1, 100, 102, 99, 101), bar(2, 100, 500, 1, 100)]
    for b in bars:
        assert strat.on_bar(b) is None  # atr not warmed up (only 3 TR samples, need 10)


def test_long_breakout_fires_at_trigger_level_with_or_low_stop():
    strat = ORBStrategy(or_minutes=5, atr_period=3, atr_mult=0.2, target_r=1.0, stop_buffer_ticks=0)
    for b in _or_and_atr_warmup_bars():
        strat.on_bar(b)
    strat.on_bar(bar(5, 99, 100, 98, 99))  # quiet bar, no trigger

    signal = strat.on_bar(bar(6, 99, 104, 99, 103))  # breaks out hard
    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == pytest.approx(103.66666667)
    assert signal.stop_price == 98.0  # OR low, no buffer
    assert signal.target_price == pytest.approx(109.33333333)


def test_short_breakout_fires_at_trigger_level_with_or_high_stop():
    strat = ORBStrategy(or_minutes=5, atr_period=3, atr_mult=0.2, target_r=1.0, stop_buffer_ticks=0)
    for b in _or_and_atr_warmup_bars():
        strat.on_bar(b)
    strat.on_bar(bar(5, 99, 100, 98, 99))

    signal = strat.on_bar(bar(6, 99, 99, 94, 95))  # breaks down hard
    assert signal is not None
    assert signal.direction == "short"
    assert signal.entry_price == pytest.approx(97.33333333)
    assert signal.stop_price == 103.0  # OR high, no buffer
    assert signal.target_price == pytest.approx(91.66666667)


def test_only_one_trade_per_day():
    strat = ORBStrategy(or_minutes=5, atr_period=3, atr_mult=0.2)
    for b in _or_and_atr_warmup_bars():
        strat.on_bar(b)
    strat.on_bar(bar(5, 99, 100, 98, 99))
    first = strat.on_bar(bar(6, 99, 104, 99, 103))
    assert first is not None

    # a further, even bigger breakout bar the same day must not fire again
    second = strat.on_bar(bar(7, 103, 200, 102, 150))
    assert second is None


def test_no_entry_after_cutoff_blocks_an_otherwise_valid_breakout():
    session = SessionConfig(no_entry_after=time(9, 37), flatten_at=time(15, 45))
    strat = ORBStrategy(or_minutes=5, atr_period=3, atr_mult=0.2, session=session)
    for b in _or_and_atr_warmup_bars():
        strat.on_bar(b)
    strat.on_bar(bar(5, 99, 100, 98, 99))  # 9:35, still before cutoff
    result = strat.on_bar(bar(6, 99, 100, 98, 99))  # 9:36, still before cutoff, no breakout
    assert result is None
    # 9:37 == cutoff -- this bar's huge range would otherwise trigger a long
    blocked = strat.on_bar(bar(7, 99, 104, 99, 103))
    assert blocked is None


def test_trailing_atr_persists_across_session_reset_but_opening_range_resets():
    strat = ORBStrategy(or_minutes=5, atr_period=3, atr_mult=0.2)
    for b in _or_and_atr_warmup_bars()[:3]:  # 3 bars -> ATR warmed up
        strat.on_bar(b)
    atr_before = strat._atr.value
    assert atr_before is not None
    assert strat._or_high == 103

    next_day = Bar(T0 + timedelta(days=1), 100, 100, 100, 100, 100.0)
    strat.on_bar(next_day)

    assert strat._atr.value == atr_before  # carried across the session boundary
    assert strat._or_high == 100  # opening range state reset for the new day
    assert strat._traded_today is False
