from datetime import datetime, time, timedelta

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig
from volume_scalp.strategy import VolumeScalpStrategy

T0 = datetime(2026, 1, 5, 9, 30)  # within default session window


def bar(i, o, h, l, c, v=100):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c, v)


def _baseline_strategy(**kwargs):
    """window=3 everywhere so 3 warm-up bars are enough to fill every tracker."""
    return VolumeScalpStrategy(
        tick_size=0.25,
        stop_buffer_ticks=2,
        target_r=1.0,
        volume_window=3,
        breakout_window=3,
        delta_window=3,
        breakout_rvol_threshold=1.5,
        enable_climax_fade=False,
        **kwargs,
    )


def _warmup_bars():
    """3 flat, low-volume, zero-delta bars: channel high=103/low=99, rvol baseline avg=100."""
    return [
        bar(0, 100, 101, 99, 100, v=100),
        bar(1, 100, 102, 100, 101, v=100),
        bar(2, 101, 103, 101, 102, v=100),
    ]


def _breakout_bar():
    """Closes above the 103 channel high, rvol=2.0x, positive delta, close above VWAP."""
    return bar(3, 103, 106, 103, 105, v=200)


def test_breakout_long_fires_with_expected_prices():
    strat = _baseline_strategy()
    signal = None
    for b in _warmup_bars() + [_breakout_bar()]:
        signal = strat.on_bar(b)

    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 105
    assert signal.stop_price == 98.5  # min(bar.low=103, channel_low=99) - 2*0.25
    assert signal.target_price == 111.5  # entry + risk(6.5) * target_r(1.0)


def test_breakout_blocked_when_rvol_below_threshold():
    strat = _baseline_strategy()
    weak_volume_bar = bar(3, 103, 106, 103, 105, v=120)  # rvol = 1.2x < 1.5x threshold

    signal = None
    for b in _warmup_bars() + [weak_volume_bar]:
        signal = strat.on_bar(b)

    assert signal is None


def test_breakout_blocked_by_vwap_misalignment_and_unblocked_without_filter():
    """A huge earlier high-price/high-volume bar drags session VWAP well above
    where the breakout bar closes -- require_vwap_alignment=True should block
    the long; disabling the filter should let the same bars fire it."""
    inflate_vwap = Bar(T0 + timedelta(minutes=-1), 200, 205, 195, 200, 100_000)

    strat_blocked = _baseline_strategy(require_vwap_alignment=True)
    signal = None
    for b in [inflate_vwap] + _warmup_bars() + [_breakout_bar()]:
        signal = strat_blocked.on_bar(b)
    assert signal is None

    strat_unblocked = _baseline_strategy(require_vwap_alignment=False)
    signal = None
    for b in [inflate_vwap] + _warmup_bars() + [_breakout_bar()]:
        signal = strat_unblocked.on_bar(b)
    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 105
    assert signal.stop_price == 98.5
    assert signal.target_price == 111.5


def test_climax_fade_short_fires_on_rejected_buying_climax():
    strat = VolumeScalpStrategy(
        tick_size=0.25,
        stop_buffer_ticks=2,
        target_r=1.0,
        volume_window=3,
        enable_breakout=False,
        enable_climax_fade=True,
        climax_rvol_threshold=3.0,
        climax_wick_pct=0.5,
    )
    warmup = [
        bar(0, 100, 101, 99, 100, v=100),
        bar(1, 100, 102, 100, 101, v=100),
        bar(2, 101, 103, 101, 102, v=100),
    ]
    # range=11, upper_wick=9 (82% of range), close(106) below midpoint(109.5), rvol=3.5x
    climax_bar = bar(3, 105, 115, 104, 106, v=350)

    signal = None
    for b in warmup + [climax_bar]:
        signal = strat.on_bar(b)

    assert signal is not None
    assert signal.direction == "short"
    assert signal.entry_price == 106
    assert signal.stop_price == 115.5  # bar.high + 2*0.25
    assert signal.target_price == 96.5  # entry - risk(9.5) * target_r(1.0)


def test_climax_fade_long_fires_on_rejected_selling_climax():
    strat = VolumeScalpStrategy(
        tick_size=0.25,
        stop_buffer_ticks=2,
        target_r=1.0,
        volume_window=3,
        enable_breakout=False,
        enable_climax_fade=True,
        climax_rvol_threshold=3.0,
        climax_wick_pct=0.5,
    )
    warmup = [
        bar(0, 100, 101, 99, 100, v=100),
        bar(1, 100, 102, 100, 101, v=100),
        bar(2, 101, 103, 101, 102, v=100),
    ]
    # mirror of the short case: big down-push, rejected back up
    climax_bar = bar(3, 106, 107, 96, 105, v=350)

    signal = None
    for b in warmup + [climax_bar]:
        signal = strat.on_bar(b)

    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 105
    assert signal.stop_price == 95.5  # bar.low - 2*0.25
    assert signal.target_price == 114.5


def test_no_signal_outside_entry_window():
    strat = _baseline_strategy(session=SessionConfig(no_entry_after=time(9, 33)))
    signal = None
    for b in _warmup_bars() + [_breakout_bar()]:  # breakout bar lands at 9:33, excluded by `< no_entry_after`
        signal = strat.on_bar(b)
    assert signal is None


def test_state_resets_on_new_session_day():
    strat = _baseline_strategy()
    for b in _warmup_bars():
        strat.on_bar(b)
    assert strat.channel.high is not None
    assert strat.vwap.value is not None

    next_day = Bar(T0 + timedelta(days=1), 100, 101, 99, 100, 100)
    strat.on_bar(next_day)

    assert strat.channel.high is None  # only 1 bar into the new day, channel not yet full
    assert strat.vwap.value == 100.0  # VWAP re-anchored fresh from this bar alone
