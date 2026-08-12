from datetime import datetime, timedelta

from failed2s.bars import Bar
from failed2s.strategy import Failed2sStrategy

T0 = datetime(2026, 1, 5, 9, 30)  # within default session window


def ebar(i, o, h, l, c):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c)


def _bullish_failed_2_down_bias(strategy):
    """Feeds two bias bars forming an F2D -> sets a long bias, swept_level=190."""
    prev = Bar(T0, 200, 205, 195, 202)
    curr = Bar(T0 + timedelta(minutes=15), 193, 204, 190, 198)
    strategy.on_bias_bar(prev)
    strategy.on_bias_bar(curr)


def _swing_high_then_mss_fvg_break():
    """Entry-tf bars: confirms a swing high at 110, then an MSS+FVG break above it."""
    return [
        ebar(0, 100, 101, 99, 100),
        ebar(1, 100, 102, 100, 101),
        ebar(2, 101, 110, 101, 109),   # pivot high candidate
        ebar(3, 108, 109, 107, 108),
        ebar(4, 108, 108, 106, 107),
        ebar(5, 111, 120, 111, 119),   # MSS break: closes > 110, strong body, FVG vs bar3
    ]


def test_full_cascade_produces_long_signal():
    strategy = Failed2sStrategy(tick_size=0.25, stop_buffer_ticks=2, target_r=1.0)
    _bullish_failed_2_down_bias(strategy)

    signal = None
    for b in _swing_high_then_mss_fvg_break():
        signal = strategy.on_entry_bar(b)

    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 119
    assert signal.stop_price == 109.5  # min(swing 110, swept_level 190) - 2*0.25
    assert signal.target_price == 128.5  # entry + risk * target_r


def test_no_signal_without_bias():
    strategy = Failed2sStrategy()
    signal = None
    for b in _swing_high_then_mss_fvg_break():
        signal = strategy.on_entry_bar(b)
    assert signal is None


def test_no_signal_when_fvg_missing():
    strategy = Failed2sStrategy(require_fvg=True)
    _bullish_failed_2_down_bias(strategy)

    bars = _swing_high_then_mss_fvg_break()
    bars[-1] = ebar(5, 108, 118, 105, 117)  # breaks MSS but low overlaps bar3 -> no FVG

    signal = None
    for b in bars:
        signal = strategy.on_entry_bar(b)
    assert signal is None


def test_no_signal_outside_entry_window():
    from failed2s.strategy import SessionConfig
    from datetime import time

    strategy = Failed2sStrategy(session=SessionConfig(no_entry_after=time(9, 34)))
    _bullish_failed_2_down_bias(strategy)

    signal = None
    for b in _swing_high_then_mss_fvg_break():
        signal = strategy.on_entry_bar(b)  # break bar is at T0+5min, i.e. 9:35 -> past cutoff
    assert signal is None


def test_bias_expires_after_max_age():
    strategy = Failed2sStrategy(max_bias_age_bars=1)
    _bullish_failed_2_down_bias(strategy)
    assert strategy._pending_bias is not None

    # two more bias bars pass with no new Failed-2 -> bias should expire
    strategy.on_bias_bar(Bar(T0 + timedelta(minutes=30), 198, 200, 196, 199))
    strategy.on_bias_bar(Bar(T0 + timedelta(minutes=45), 199, 201, 197, 200))
    assert strategy._pending_bias is None


def test_state_resets_on_new_session_day():
    strategy = Failed2sStrategy()
    _bullish_failed_2_down_bias(strategy)
    assert strategy._pending_bias is not None

    next_day = Bar(T0 + timedelta(days=1), 199, 201, 197, 200)
    strategy.on_bias_bar(next_day)
    assert strategy._pending_bias is None
