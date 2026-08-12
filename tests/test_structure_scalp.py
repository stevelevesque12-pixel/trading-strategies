from datetime import datetime, timedelta

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig
from structure_scalp.strategy import StructureScalpStrategy, StructureTracker, _PendingLeg

T0 = datetime(2026, 1, 5, 9, 30)  # within default session window


def bar(i, o, h, l, c, v=1.0):
    return Bar(T0 + timedelta(seconds=5 * i), o, h, l, c, v)


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


def _strategy_with_long_bias(**kwargs) -> StructureScalpStrategy:
    strat = StructureScalpStrategy(tick_size=0.25, target_r=1.0, **kwargs)
    strat.on_1m_bar(bar(-1000, 100, 101, 99, 100))  # establishes _current_date
    strat.struct_1m.direction = "long"
    return strat


# Bars 0-4: forms a 5s swing low at 90 (no direction yet).
# Bar 5: breaks below 90 -> entry_5s flips "short" (pullback begins, low=80).
# Bars 6-10: pullback continues; bar 6 sets the pullback's actual low (75).
# Bar 11: breaks back above the swing high -> entry_5s flips "long", arming a
#         leg with extreme=75 (bar 6's low) and an anchored VWAP retroactively
#         summed from bar 6 through bar 11.
# Bars 12-13: VWAP accumulates forward; bar 13's low reaches down to the
#         (updated) VWAP value -> signal fires.
PULLBACK_AND_REVERSAL_BARS = [
    bar(0, 100, 101, 99, 100),
    bar(1, 99, 100, 97, 98),
    bar(2, 98, 99, 90, 91),
    bar(3, 92, 94, 91, 93),
    bar(4, 93, 95, 92, 94),
    bar(5, 89, 89, 80, 82),
    bar(6, 82, 84, 75, 78),
    bar(7, 78, 82, 76, 80),
    bar(8, 80, 88, 79, 87),
    bar(9, 86, 87, 83, 84),
    bar(10, 84, 85, 81, 82),
    bar(11, 83, 92, 82, 91),
    bar(12, 91, 93, 88, 89),
    bar(13, 89, 90, 83, 85),
]


def test_no_signal_without_a_pullback_and_reversal():
    strat = _strategy_with_long_bias()
    signal = None
    for b in PULLBACK_AND_REVERSAL_BARS[:11]:  # stop right before the arming bar
        signal = strat.on_5s_bar(b)
    assert signal is None
    assert strat._pending is None


def test_leg_arms_with_pullback_low_as_extreme():
    strat = _strategy_with_long_bias()
    for b in PULLBACK_AND_REVERSAL_BARS[:12]:  # through the arming bar (index 11)
        strat.on_5s_bar(b)
    assert strat._pending is not None
    assert strat._pending.direction == "long"
    assert strat._pending.extreme == 75  # bar 6's low, not bar 11's breakout level


def test_signal_fires_on_vwap_retest_with_correct_stop_and_target():
    strat = _strategy_with_long_bias()
    signal = None
    for b in PULLBACK_AND_REVERSAL_BARS:
        s = strat.on_5s_bar(b)
        if s:
            signal = s

    assert signal is not None
    assert signal.direction == "long"
    assert signal.stop_price == 75  # the leg's low
    assert round(signal.entry_price, 4) == round(84.33333333333334, 4)  # avwap at touch
    assert round(signal.target_price, 4) == round(93.66666666666669, 4)  # entry + risk * target_r
    assert strat._pending is None  # consumed after firing


def test_invalidated_if_price_breaks_leg_low_before_touch():
    strat = _strategy_with_long_bias()
    for b in PULLBACK_AND_REVERSAL_BARS[:12]:  # armed, extreme=75
        strat.on_5s_bar(b)
    assert strat._pending is not None

    breaks_leg_low = bar(12, 91, 93, 70, 72)  # low below the leg's low of 75
    signal = strat.on_5s_bar(breaks_leg_low)

    assert signal is None
    assert strat._pending is None


def test_fresh_pullback_supersedes_pending_leg():
    strat = _strategy_with_long_bias()
    # Simulate an already-armed leg (as if a prior reversal had just fired).
    strat.entry_5s.direction = "long"
    strat._pending = _PendingLeg(direction="long", extreme=50.0, sum_pv=100.0, sum_v=1.0)

    # A fresh, genuine bearish 5s MSS (known-good pattern, reused from
    # test_structure_tracker_flips_short_on_bearish_mss) should cancel it.
    fresh_bearish_break = [
        bar(0, 200, 201, 199, 200),
        bar(1, 199, 200, 197, 198),
        bar(2, 198, 199, 190, 191),   # pivot low candidate
        bar(3, 192, 194, 191, 193),
        bar(4, 193, 195, 192, 194),   # confirms swing low = 190
        bar(5, 189, 189, 180, 182),   # strong close through 190 -> flips "short"
    ]
    for b in fresh_bearish_break:
        strat.on_5s_bar(b)

    assert strat.entry_5s.direction == "short"
    assert strat._pending is None
    assert len(strat._pullback_bars) > 0  # now tracking the fresh pullback


def test_bias_flip_cancels_pending_leg():
    strat = _strategy_with_long_bias()
    for b in PULLBACK_AND_REVERSAL_BARS[:12]:  # armed, extreme=75
        strat.on_5s_bar(b)
    assert strat._pending is not None

    strat.struct_1m.direction = "short"
    strat.on_5s_bar(bar(12, 91, 93, 88, 89))

    assert strat._pending is None


def test_no_signal_outside_entry_window():
    from datetime import time

    strat = _strategy_with_long_bias(session=SessionConfig(no_entry_after=time(9, 30, 5)))
    signal = None
    for b in PULLBACK_AND_REVERSAL_BARS:
        s = strat.on_5s_bar(b)
        if s:
            signal = s
    assert signal is None


def test_state_resets_on_new_session_day():
    strat = _strategy_with_long_bias()
    for b in PULLBACK_AND_REVERSAL_BARS[:12]:
        strat.on_5s_bar(b)
    assert strat._pending is not None

    next_day = Bar(T0 + timedelta(days=1), 100, 101, 99, 100)
    strat.on_1m_bar(next_day)

    assert strat.struct_1m.direction is None
    assert strat._pending is None
    assert strat._pullback_bars == []
