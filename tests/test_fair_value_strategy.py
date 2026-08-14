from datetime import datetime, timedelta

from failed2s.bars import Bar
from fair_value.strategy import ATR_BUCKETS, DEFAULT_WINDOWS, FairValueStrategy, atr_to_stop_target

D0 = datetime(2026, 1, 5)  # Monday


def t(hour, minute, second=0):
    return D0.replace(hour=hour, minute=minute, second=second)


def bar(ts, o, h, l, c, v=100):
    return Bar(ts, o, h, l, c, v)


def test_atr_bucket_boundaries():
    assert atr_to_stop_target(5.0) == (16.5, 24.75)      # below 7
    assert atr_to_stop_target(6.99) == (16.5, 24.75)
    assert atr_to_stop_target(7.0) == (25.0, 37.5)       # 7-20
    assert atr_to_stop_target(19.99) == (25.0, 37.5)
    assert atr_to_stop_target(20.0) == (50.0, 75.0)      # above 20
    assert atr_to_stop_target(100.0) == (50.0, 75.0)


def test_no_signal_outside_configured_windows():
    strategy = FairValueStrategy()
    signal = None
    for m in range(20):
        signal = strategy.on_bar(bar(t(12, m), 100, 101, 99, 100))  # noon: outside AM/PM windows
    assert signal is None


def test_no_signal_before_atr_warms_up():
    strategy = FairValueStrategy(atr_period=14)
    signal = strategy.on_bar(bar(t(9, 33), 100, 105, 95, 104))
    assert signal is None  # only 1 bar seen, ATR needs `atr_period` bars


def _warm_up_atr(strategy, start_ts, n, price=100.0):
    """Feed n quiet pre-window bars so ATR is seeded before the window opens."""
    ts = start_ts
    for i in range(n):
        strategy.on_bar(bar(ts, price, price + 1, price - 1, price))
        ts = ts + timedelta(minutes=1)
    return ts


def test_continuation_long_signal_fires_on_displacement_plus_mss():
    strategy = FairValueStrategy(atr_period=3, swing_strength=2)
    ts = _warm_up_atr(strategy, t(9, 20), 3, price=100.0)

    # 9:30 open bar sets fair value = 100 (open of first AM-window bar)
    ts = t(9, 30)
    strategy.on_bar(bar(ts, 100, 101, 99.5, 100.5))
    ts += timedelta(minutes=1)
    strategy.on_bar(bar(ts, 100.5, 102, 100, 101.5))
    ts += timedelta(minutes=1)
    # pivot high candidate at 110
    ts += timedelta(minutes=1)
    pivot_ts = ts
    strategy.on_bar(bar(pivot_ts, 101.5, 110, 101, 109))
    for _ in range(2):
        ts += timedelta(minutes=1)
        strategy.on_bar(bar(ts, 108, 109, 107, 108))

    ts += timedelta(minutes=1)
    # displacement candle breaking above the 110 swing high, within continuation window (<=15m after 9:30)
    signal = strategy.on_bar(bar(ts, 111, 121, 110.5, 120.5))

    assert signal is not None
    assert signal.direction == "long"
    assert signal.window == "AM"
    assert signal.phase == "continuation"
    assert signal.entry_price == 120.5


def test_reversion_short_signal_fires_after_continuation_window_when_price_above_fv():
    strategy = FairValueStrategy(atr_period=3, swing_strength=2)
    _warm_up_atr(strategy, t(9, 20), 3, price=100.0)

    ts = t(9, 30)
    strategy.on_bar(bar(ts, 100, 101, 99.5, 100.5))  # fair value = 100

    # push price up and away from FV, then build a swing low above FV for the
    # reversion leg to break, well past the 15-minute continuation cutoff.
    ts = t(9, 50)
    strategy.on_bar(bar(ts, 130, 131, 129, 130.5))
    ts += timedelta(minutes=1)
    strategy.on_bar(bar(ts, 130.5, 131, 129.5, 130))
    ts += timedelta(minutes=1)
    pivot_ts = ts
    strategy.on_bar(bar(pivot_ts, 130, 130.5, 120, 121))  # pivot low candidate at 120
    for _ in range(2):
        ts += timedelta(minutes=1)
        strategy.on_bar(bar(ts, 121, 123, 121, 122))

    ts += timedelta(minutes=1)
    # displacement candle breaking below the 120 swing low, back toward FV=100
    signal = strategy.on_bar(bar(ts, 121, 121.5, 110, 111))

    assert signal is not None
    assert signal.direction == "short"
    assert signal.window == "AM"
    assert signal.phase == "reversion"


def test_state_resets_on_new_session_day():
    strategy = FairValueStrategy()
    strategy.on_bar(bar(t(9, 30), 100, 101, 99, 100))
    assert strategy._window_fv["AM"] == 100

    next_day = D0 + timedelta(days=1)
    strategy.on_bar(bar(next_day.replace(hour=9, minute=30), 200, 201, 199, 200))
    assert strategy._window_fv["AM"] == 200  # fresh anchor, old one discarded


def test_default_windows_skip_first_three_minutes_after_open():
    am = DEFAULT_WINDOWS[0]
    assert am.name == "AM"
    assert am.entry_start.minute == 33
