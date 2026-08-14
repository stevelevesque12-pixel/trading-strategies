from datetime import datetime, timedelta

from failed2s.bars import Bar
from fair_value.structure import is_displacement_candle

T0 = datetime(2026, 1, 5, 9, 30)


def bar(o, h, l, c):
    return Bar(T0, o, h, l, c)


def test_bullish_displacement_candle_has_small_lower_wick():
    # range=20, lower wick = open-low = 1 (5%) -> well under 20% threshold
    b = bar(100, 120, 99, 119)
    assert is_displacement_candle(b, "long") is True


def test_bullish_candle_with_large_lower_wick_is_not_displacement():
    # range=20, lower wick = open-low = 8 (40%) -> over 20% threshold
    b = bar(100, 110, 92, 109)
    assert is_displacement_candle(b, "long") is False


def test_bearish_displacement_candle_has_small_upper_wick():
    b = bar(119, 120, 99, 100)
    assert is_displacement_candle(b, "short") is True


def test_bearish_candle_with_large_upper_wick_is_not_displacement():
    b = bar(109, 118, 90, 92)
    assert is_displacement_candle(b, "short") is False


def test_wrong_polarity_candle_fails_regardless_of_wick():
    bullish = bar(100, 120, 99, 119)
    assert is_displacement_candle(bullish, "short") is False

    bearish = bar(119, 120, 99, 100)
    assert is_displacement_candle(bearish, "long") is False


def test_zero_range_candle_is_never_displacement():
    flat = bar(100, 100, 100, 100)
    assert is_displacement_candle(flat, "long") is False
    assert is_displacement_candle(flat, "short") is False
