from datetime import datetime

from failed2s.bars import Bar, BarType, classify_bar, detect_failed_2

T0 = datetime(2026, 1, 5, 9, 30)


def bar(o, h, l, c):
    return Bar(T0, o, h, l, c)


def test_classify_inside():
    prev = bar(100, 105, 95, 102)
    curr = bar(101, 104, 96, 103)
    assert classify_bar(prev, curr) == BarType.INSIDE


def test_classify_directional_up():
    prev = bar(100, 105, 95, 102)
    curr = bar(102, 107, 96, 106)
    assert classify_bar(prev, curr) == BarType.DIRECTIONAL_UP


def test_classify_directional_down():
    prev = bar(100, 105, 95, 102)
    curr = bar(98, 104, 90, 92)
    assert classify_bar(prev, curr) == BarType.DIRECTIONAL_DOWN


def test_classify_outside():
    prev = bar(100, 105, 95, 102)
    curr = bar(99, 108, 90, 95)
    assert classify_bar(prev, curr) == BarType.OUTSIDE


def test_failed_2_up():
    # breaks the high (2U) but closes bearish, back below prev.high -> F2U
    prev = bar(100, 105, 95, 102)
    curr = bar(103, 107, 100, 101)
    assert detect_failed_2(prev, curr) == "F2U"


def test_failed_2_down():
    # breaks the low (2D) but closes bullish, back above prev.low -> F2D
    prev = bar(100, 105, 95, 102)
    curr = bar(97, 100, 93, 99)
    assert detect_failed_2(prev, curr) == "F2D"


def test_no_failed_2_on_clean_directional_bar():
    prev = bar(100, 105, 95, 102)
    curr = bar(102, 108, 101, 107)  # 2U that closes bullish -- not failed
    assert detect_failed_2(prev, curr) is None


def test_no_failed_2_on_inside_bar():
    prev = bar(100, 105, 95, 102)
    curr = bar(101, 104, 96, 97)
    assert detect_failed_2(prev, curr) is None


def test_require_reclaim_gates_weak_failed_2():
    # 2U bar that closes red (bearish body) but does NOT reclaim back below prev.high
    prev = bar(100, 105, 95, 102)
    curr = bar(110, 112, 104, 106)  # close 106 >= prev.high 105 -> no reclaim
    assert detect_failed_2(prev, curr, require_reclaim=True) is None
    assert detect_failed_2(prev, curr, require_reclaim=False) == "F2U"
