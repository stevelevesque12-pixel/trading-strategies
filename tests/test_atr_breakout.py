from datetime import datetime, time, timedelta

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig
from atr_breakout.strategy import ATRBreakoutStrategy, _TrueRangeATR

T0 = datetime(2026, 1, 5, 9, 30)  # within default session window


def bar(i, o, h, l, c, v=1.0):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c, v)


CONTRACTING_BARS = [
    bar(0, 100, 110, 90, 100),      # TR=20
    bar(1, 100, 105, 95, 100),      # TR=10
    bar(2, 100, 103, 97, 100),      # TR=6
    bar(3, 100, 101, 99, 100),      # TR=2
    bar(4, 100, 100.5, 99.5, 100),  # TR=1
]


def _warmed_up_strategy(**kwargs):
    strat = ATRBreakoutStrategy(atr_period_long=5, atr_period_short=2, **kwargs)
    for b in CONTRACTING_BARS:
        strat.on_bar(b)
    return strat


def test_true_range_atr_none_until_warmed_up_then_simple_average():
    atr = _TrueRangeATR(period=3)
    atr.update(Bar(T0, 100, 105, 95, 100))
    assert atr.value is None
    atr.update(Bar(T0 + timedelta(minutes=1), 100, 104, 98, 100))
    assert atr.value is None
    atr.update(Bar(T0 + timedelta(minutes=2), 100, 103, 99, 100))
    # TRs: bar0 (no prev close) = 10; bar1 = max(6, |104-100|=4, |98-100|=2) = 6; bar2 = max(4, |103-100|=3, |99-100|=1) = 4
    assert atr.value == (10 + 6 + 4) / 3


def test_no_signal_before_both_atrs_warmed_up():
    strat = ATRBreakoutStrategy(atr_period_long=5, atr_period_short=2)
    for b in CONTRACTING_BARS[:4]:  # only 4 bars -- atr_long(5) not yet warmed up
        assert strat.on_bar(b) is None


def test_long_fires_at_open_plus_entry_mult_times_atr_using_pre_bar_atr():
    strat = _warmed_up_strategy(entry_atr_mult=2.5, stop_atr_mult=0.5)
    atr_long = strat._atr_long.value  # 7.8, computed from bars 0-4 only

    b5 = bar(5, 100, 100 + 2.5 * atr_long + 1, 99, 100)  # opens at 100, spikes through the trigger
    signal = strat.on_bar(b5)

    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 100 + 2.5 * atr_long
    assert signal.stop_price == signal.entry_price - 0.5 * atr_long
    assert signal.stop_price < signal.entry_price


def test_short_fires_at_open_minus_entry_mult_times_atr():
    strat = _warmed_up_strategy(entry_atr_mult=2.5, stop_atr_mult=0.5)
    atr_long = strat._atr_long.value

    b5 = bar(5, 100, 101, 100 - 2.5 * atr_long - 1, 100)
    signal = strat.on_bar(b5)

    assert signal is not None
    assert signal.direction == "short"
    assert signal.entry_price == 100 - 2.5 * atr_long
    assert signal.stop_price == signal.entry_price + 0.5 * atr_long
    assert signal.stop_price > signal.entry_price


def test_trigger_is_not_polluted_by_the_current_bars_own_range():
    """Regression test for the no-lookahead ordering: a bar with an
    enormous range must not affect the ATR used to compute ITS OWN
    trigger, only the next bar's."""
    strat = _warmed_up_strategy()
    atr_long_before = strat._atr_long.value

    huge_bar = bar(5, 100, 500, 1, 100)
    strat.on_bar(huge_bar)  # regardless of whether it fires, this shouldn't crash or use post-update ATR

    assert strat._atr_long.value != atr_long_before  # now updated for the NEXT bar
    assert strat._atr_long.value > atr_long_before   # the huge range pushed it up


def test_filter_blocks_entry_when_volatility_is_expanding():
    strat = ATRBreakoutStrategy(atr_period_long=5, atr_period_short=2)
    expanding_bars = [
        bar(0, 100, 100.5, 99.5, 100),
        bar(1, 100, 101, 99, 100),
        bar(2, 100, 106, 94, 100),
        bar(3, 100, 108, 92, 100),
        bar(4, 100, 110, 90, 100),
    ]
    for b in expanding_bars:
        strat.on_bar(b)
    assert strat._atr_short.value >= strat._atr_long.value  # expanding, not contracting

    signal = strat.on_bar(bar(5, 100, 1000, 1, 100))  # would trivially breach any trigger
    assert signal is None


def test_no_entry_after_cutoff():
    session = SessionConfig(no_entry_after=time(9, 34))
    strat = _warmed_up_strategy(session=session)
    # bar index 5 is at T0+5min = 9:35, past the 9:34 cutoff
    signal = strat.on_bar(bar(5, 100, 200, 1, 100))
    assert signal is None


def test_degenerate_zero_stop_distance_does_not_fire():
    strat = _warmed_up_strategy(stop_atr_mult=0.0)
    atr_long = strat._atr_long.value
    signal = strat.on_bar(bar(5, 100, 100 + 2.5 * atr_long + 1, 99, 100))
    assert signal is None  # stop would equal entry -- zero risk, correctly rejected


def test_no_target_by_default():
    strat = _warmed_up_strategy()
    atr_long = strat._atr_long.value
    signal = strat.on_bar(bar(5, 100, 100 + 2.5 * atr_long + 1, 99, 100))
    assert signal is not None
    assert signal.target_price is None


def test_target_atr_mult_sets_a_target_price_on_both_sides():
    strat_long = _warmed_up_strategy(target_atr_mult=2.0)
    atr_long = strat_long._atr_long.value
    long_signal = strat_long.on_bar(bar(5, 100, 100 + 2.5 * atr_long + 1, 99, 100))
    assert long_signal is not None
    assert long_signal.target_price == long_signal.entry_price + 2.0 * atr_long
    assert long_signal.target_price > long_signal.entry_price

    strat_short = _warmed_up_strategy(target_atr_mult=2.0)
    atr_long_s = strat_short._atr_long.value
    short_signal = strat_short.on_bar(bar(5, 100, 101, 100 - 2.5 * atr_long_s - 1, 100))
    assert short_signal is not None
    assert short_signal.target_price == short_signal.entry_price - 2.0 * atr_long_s
    assert short_signal.target_price < short_signal.entry_price
