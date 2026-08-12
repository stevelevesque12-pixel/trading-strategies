from datetime import datetime, timedelta

from failed2s.bars import Bar
from failed2s.structure import SwingTracker, detect_fvg, detect_mss

T0 = datetime(2026, 1, 5, 9, 30)


def bar(i, o, h, l, c):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c)


def test_swing_tracker_confirms_swing_high():
    tracker = SwingTracker(strength=2)
    bars = [
        bar(0, 100, 101, 99, 100),
        bar(1, 100, 102, 100, 101),
        bar(2, 101, 110, 101, 109),  # pivot high
        bar(3, 108, 109, 107, 108),
        bar(4, 108, 108, 106, 107),
    ]
    for b in bars:
        tracker.update(b)
    assert tracker.last_swing_high is not None
    assert tracker.last_swing_high.price == 110


def test_swing_tracker_confirms_swing_low():
    tracker = SwingTracker(strength=2)
    bars = [
        bar(0, 100, 101, 99, 100),
        bar(1, 99, 100, 97, 98),
        bar(2, 98, 99, 90, 91),  # pivot low
        bar(3, 92, 94, 91, 93),
        bar(4, 93, 95, 92, 94),
    ]
    for b in bars:
        tracker.update(b)
    assert tracker.last_swing_low is not None
    assert tracker.last_swing_low.price == 90


def test_detect_mss_long_requires_strong_bullish_close_above_swing_high():
    tracker = SwingTracker(strength=2)
    for b in [
        bar(0, 100, 101, 99, 100),
        bar(1, 100, 102, 100, 101),
        bar(2, 101, 110, 101, 109),
        bar(3, 108, 109, 107, 108),
        bar(4, 108, 108, 106, 107),
    ]:
        tracker.update(b)

    strong_break = bar(5, 109, 115, 109, 114)  # closes above swing high 110, strong body
    assert detect_mss(strong_break, tracker.last_swing_high, "long") is True

    weak_break = bar(6, 109, 111, 108.9, 109.5)  # barely breaks, weak body
    assert detect_mss(weak_break, tracker.last_swing_high, "long") is False


def test_detect_fvg_bullish_and_bearish():
    b1 = bar(0, 100, 101, 99, 100)
    b2 = bar(1, 101, 106, 101, 105)
    b3 = bar(2, 105, 108, 102, 107)
    assert detect_fvg(b1, b2, b3) == "bullish"

    b1d = bar(0, 100, 101, 99, 100)
    b2d = bar(1, 96, 96, 91, 92)
    b3d = bar(2, 92, 95, 90, 91)
    assert detect_fvg(b1d, b2d, b3d) == "bearish"

    b1n = bar(0, 100, 101, 99, 100)
    b2n = bar(1, 99, 101, 98, 100)
    b3n = bar(2, 100, 101, 99, 100)
    assert detect_fvg(b1n, b2n, b3n) is None
