from datetime import datetime, timedelta

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig
from structure_scalp.strategy import StructureScalpStrategy, StructureTracker

T0 = datetime(2026, 1, 5, 9, 30)  # within default session window


def bar(i, o, h, l, c):
    return Bar(T0 + timedelta(seconds=5 * i), o, h, l, c)


def test_structure_tracker_starts_none_and_flips_long_on_mss():
    st = StructureTracker(swing_strength=2, min_body_pct=0.5)
    setup = [
        bar(0, 100, 101, 99, 100),
        bar(1, 100, 102, 100, 101),
        bar(2, 101, 110, 101, 109),   # pivot high candidate
        bar(3, 108, 109, 107, 108),
        bar(4, 108, 108, 106, 107),   # confirms swing high = 110
    ]
    for b in setup:
        st.update(b)
    assert st.direction is None  # structure formed, but no break yet

    st.update(bar(5, 111, 120, 111, 119))  # strong close through 110
    assert st.direction == "long"


def test_structure_tracker_flips_short_on_bearish_mss():
    st = StructureTracker(swing_strength=2, min_body_pct=0.5)
    setup = [
        bar(0, 100, 101, 99, 100),
        bar(1, 99, 100, 97, 98),
        bar(2, 98, 99, 90, 91),   # pivot low candidate
        bar(3, 92, 94, 91, 93),
        bar(4, 93, 95, 92, 94),   # confirms swing low = 90
    ]
    for b in setup:
        st.update(b)
    assert st.direction is None

    st.update(bar(5, 89, 89, 80, 82))  # strong close through 90
    assert st.direction == "short"


def _aligned_long_strategy(**kwargs):
    strat = StructureScalpStrategy(tick_size=0.25, stop_buffer_ticks=2, target_r=1.0, **kwargs)
    strat.struct_15m.direction = "long"
    strat.struct_1m.direction = "long"
    return strat


def _swing_high_then_break_bars(start_i=0):
    """5s bars: confirms a swing high at 110, then breaks it with a strong close."""
    return [
        bar(start_i + 0, 100, 101, 99, 100),
        bar(start_i + 1, 100, 102, 100, 101),
        bar(start_i + 2, 101, 110, 101, 109),
        bar(start_i + 3, 108, 109, 107, 108),
        bar(start_i + 4, 108, 108, 106, 107),
        bar(start_i + 5, 111, 120, 111, 119),  # breaks 110, fires signal
    ]


def test_no_signal_when_structures_disagree():
    strat = StructureScalpStrategy()
    strat.struct_15m.direction = "long"
    strat.struct_1m.direction = "short"

    signal = None
    for b in _swing_high_then_break_bars():
        signal = strat.on_5s_bar(b)
    assert signal is None


def test_signal_fires_when_aligned_and_5s_mss_occurs():
    strat = _aligned_long_strategy()

    signal = None
    for b in _swing_high_then_break_bars():
        signal = strat.on_5s_bar(b)

    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 119
    assert signal.stop_price == 109.5   # swing (110) - 2*0.25 buffer
    assert signal.target_price == 128.5  # entry + risk * target_r


def test_no_signal_outside_entry_window():
    from datetime import time

    strat = _aligned_long_strategy(session=SessionConfig(no_entry_after=time(9, 30, 20)))
    signal = None
    for b in _swing_high_then_break_bars():
        signal = strat.on_5s_bar(b)  # break bar lands at 9:30:25, past the cutoff
    assert signal is None


def test_persisting_alignment_allows_repeated_entries():
    """Core behavior this strategy is built for: alignment isn't consumed by
    one trade -- a fresh 5s MSS in the same direction fires again."""
    strat = _aligned_long_strategy()

    signals = []
    for b in _swing_high_then_break_bars(start_i=0):
        s = strat.on_5s_bar(b)
        if s:
            signals.append(s)
    assert len(signals) == 1

    # a second swing-high-then-break cycle, far enough out that the first
    # breakout bar (index 5, high=120) has scrolled out of the 5-bar window
    second_cycle = [
        bar(6, 115, 116, 113, 114),
        bar(7, 114, 116, 113, 115),
        bar(8, 114, 118, 113, 117),   # new pivot high candidate (118)
        bar(9, 116, 117, 114, 115),
        bar(10, 115, 116, 112, 113),  # confirms swing high = 118
        bar(11, 119, 125, 119, 124),  # breaks 118, fires second signal
    ]
    for b in second_cycle:
        s = strat.on_5s_bar(b)
        if s:
            signals.append(s)

    assert len(signals) == 2
    assert signals[1].direction == "long"
    assert signals[1].entry_price == 124
    assert signals[1].stop_price == 117.5  # swing (118) - 0.5 buffer
    assert signals[1].target_price == 130.5


def test_state_resets_on_new_session_day():
    strat = StructureScalpStrategy()
    strat.on_15m_bar(bar(0, 100, 101, 99, 100))  # establishes _current_date on day 1
    strat.struct_15m.direction = "long"
    strat.struct_1m.direction = "long"
    assert strat.struct_15m.direction == "long"

    next_day = Bar(T0 + timedelta(days=1), 100, 101, 99, 100)
    strat.on_15m_bar(next_day)

    assert strat.struct_15m.direction is None
    assert strat.struct_1m.direction is None  # reset() clears both trackers
