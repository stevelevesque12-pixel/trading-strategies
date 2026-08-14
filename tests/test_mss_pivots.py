from datetime import datetime, timedelta

from failed2s.bars import Bar
from mss_pullback.pivots import PivotTracker, nearest_above, nearest_below, window_extreme

T0 = datetime(2026, 1, 5, 9, 30)


def bar(i, o, h, l, c):
    return Bar(T0 + timedelta(seconds=15 * i), o, h, l, c)


def test_confirms_only_on_close_below_opposite_wick():
    pt = PivotTracker()
    setup = [
        bar(0, 40, 45, 38, 41),
        bar(1, 41, 70, 40, 45),   # candidate high: high=70, low=40
        bar(2, 45, 55, 42, 46),  # touches lower but doesn't close below 40
        bar(3, 46, 48, 32, 36),  # closes below 40 -> confirms
    ]
    highs, lows = [], []
    for b in setup:
        ch, cl = pt.update(b)
        if ch:
            highs.append(ch)
        if cl:
            lows.append(cl)

    assert len(highs) == 1
    assert highs[0].price == 70
    assert highs[0].timestamp == setup[1].timestamp


def test_candidate_replaced_before_confirming():
    pt = PivotTracker()
    setup = [
        bar(0, 40, 50, 38, 42),   # candidate high: 50, low=38
        bar(1, 42, 60, 41, 55),   # higher high -> replaces candidate: 60, low=41
        bar(2, 55, 58, 39, 40),   # closes below 41 -> confirms the SECOND candidate (60), not 50
    ]
    highs = []
    for b in setup:
        ch, _ = pt.update(b)
        if ch:
            highs.append(ch)

    assert len(highs) == 1
    assert highs[0].price == 60


def test_tie_keeps_harder_to_break_high():
    pt = PivotTracker()
    setup = [
        bar(0, 40, 60, 45, 55),   # candidate high: 60, low=45
        bar(1, 55, 60, 40, 50),   # tied high (60), lower low (40) -> harder to break, replaces candidate
        bar(2, 50, 52, 46, 48),   # close 48 is above 45 -> would NOT confirm the first candidate...
    ]
    highs = []
    for b in setup:
        ch, _ = pt.update(b)
        if ch:
            highs.append(ch)
    assert highs == []  # candidate is now anchored to low=40, not confirmed by a close of 48

    ch, _ = pt.update(bar(3, 48, 49, 38, 39))  # closes below 40 -> confirms the tied (harder) candidate
    assert ch is not None
    assert ch.price == 60


def test_valid_highs_and_lows_tracked_independently_no_alternation_required():
    pt = PivotTracker()
    # Two consecutive valid highs confirm with no valid low in between.
    setup = [
        bar(0, 100, 105, 98, 102),   # candidate high #1: 105, low=98
        bar(1, 102, 96, 96, 97),     # closes below 98 -> confirms 105
        bar(2, 97, 110, 96, 108),    # candidate high #2: 110, low=96
        bar(3, 108, 109, 90, 92),    # closes below 96 -> confirms 110
    ]
    for b in setup:
        pt.update(b)

    assert [p.price for p in pt.valid_highs] == [105, 110]
    assert pt.valid_lows == []


def test_candidate_high_and_low_properties_track_running_extreme():
    pt = PivotTracker()
    pt.update(bar(0, 100, 105, 98, 101))
    assert pt.candidate_high == 105
    assert pt.candidate_low == 98
    # New high (112), higher low (99 > 98, so the low candidate doesn't move),
    # and a close (104) that doesn't clear the low candidate's opposite high
    # (105) -- so candidate_low is neither replaced nor confirmed away.
    pt.update(bar(1, 101, 112, 99, 104))
    assert pt.candidate_high == 112
    assert pt.candidate_low == 98


def test_window_extreme_finds_highest_between_last_two_lows():
    from mss_pullback.pivots import Pivot

    lows = [Pivot(0, 70), Pivot(10, 59)]
    # 92's pivot time (-5) is BEFORE the window -> excluded. 82's time (5) is
    # inside the window (0, 10) -> included. 75's time (20) is after -> excluded.
    highs = [Pivot(-5, 92), Pivot(5, 82), Pivot(20, 75)]
    assert window_extreme(lows, highs, want_max=True) == 82

    highs_bear = [Pivot(2, 90), Pivot(6, 65)]
    highs_anchor = [Pivot(1, 100), Pivot(10, 95)]
    assert window_extreme(highs_anchor, highs_bear, want_max=False) == 65


def test_nearest_above_and_below():
    from mss_pullback.pivots import Pivot

    highs = [Pivot(0, 90), Pivot(1, 108), Pivot(2, 130)]
    assert nearest_above(highs, 100) == 108
    assert nearest_above(highs, 200) is None

    lows = [Pivot(0, 70), Pivot(1, 55), Pivot(2, 40)]
    assert nearest_below(lows, 60) == 55
    assert nearest_below(lows, 30) is None
