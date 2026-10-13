import pandas as pd
import pytest

from failed2s.bars import Bar
from volatility_break.strategy import VolatilityBreakStrategy, _Bracket, _Position

T0 = pd.Timestamp("2024-01-02 09:30", tz="America/New_York")


def bar(i, o, h, l, c):
    return Bar(T0 + pd.Timedelta(minutes=15 * i), o, h, l, c, 100.0)


def small_strategy(**kw):
    params = dict(bb_period=5, ratio_avg_period=5, atr_period=3, expiry_bars=4)
    params.update(kw)
    return VolatilityBreakStrategy(**params)


def wide_then_tight():
    """Volatile bars, then a quiet stretch -> BB width drops below its average."""
    bars, i = [], 0
    for k in range(12):
        c = 100 + (5 if k % 2 else -5)
        bars.append(bar(i, c, c + 2, c - 2, c)); i += 1
    for k in range(6):
        c = 100 + (0.5 if k % 2 else -0.5)
        bars.append(bar(i, c, c + 0.5, c - 0.5, c)); i += 1
    return bars, i


def test_squeeze_places_bracket_at_bands():
    s = small_strategy()
    bars, _ = wide_then_tight()
    for b in bars:
        s.on_bar(b)
        if s.bracket is not None:
            break
    assert s.bracket is not None
    assert s.bracket.buy_stop == pytest.approx(s.upper)
    assert s.bracket.sell_stop == pytest.approx(s.lower)
    assert s.bracket.buy_stop > 100 > s.bracket.sell_stop
    # the levels are fixed at placement while it stays pending
    placed = (s.bracket.buy_stop, s.bracket.sell_stop)
    s.on_bar(bar(99, 100, 100.2, 99.8, 100))
    assert (s.bracket.buy_stop, s.bracket.sell_stop) == placed


def test_no_bracket_while_volatility_is_expanding():
    s = small_strategy()
    for k in range(20):
        width = 1 + k  # every bar wider than the last
        c = 100 + (width if k % 2 else -width)
        s.on_bar(bar(k, c, c + width, c - width, c))
    assert s.bracket is None


def _with_bracket(buy=110.0, sell=90.0, atr=10.0, **kw):
    s = small_strategy(**kw)
    s.bracket = _Bracket(buy_stop=buy, sell_stop=sell, atr=atr, placed_index=-1)
    return s


def test_buy_stop_fill_sets_atr_stop_and_target_and_cancels_sell():
    s = _with_bracket()
    s.on_bar(bar(0, 105, 111, 104, 110.5))
    p = s.position
    assert p.direction == "long" and p.entry_price == 110
    assert p.stop == pytest.approx(110 - 3.5 * 10)
    assert p.target == pytest.approx(110 + 5.2 * 10)
    assert s.bracket is None  # OCO


def test_gap_through_stop_order_fills_at_open():
    s = _with_bracket()
    s.on_bar(bar(0, 85, 86, 80, 84))
    assert s.position.direction == "short" and s.position.entry_price == 85


def test_both_sides_hit_fills_side_nearer_open():
    s = _with_bracket()
    s.on_bar(bar(0, 108, 112, 89, 100))  # 2 from buy stop, 19 from sell stop
    assert s.position.direction == "long"


def test_pending_bracket_expires():
    s = _with_bracket(expiry_bars=3)  # placed at index -1
    for i in range(3):
        s.on_bar(bar(i, 100, 101, 99, 100))
    assert s.bracket is not None
    s.on_bar(bar(3, 100, 101, 99, 100))
    assert s.bracket is None
    # a later move through the old level must not fill
    s.on_bar(bar(4, 100, 120, 99, 119))
    assert s.position is None


def _long_position(s, entry=100.0, atr=10.0):
    s.position = _Position(
        "long", T0, entry, initial_stop=entry - 35, stop=entry - 35,
        target=entry + 52, atr=atr, best=entry,
    )


def test_target_exit():
    s = small_strategy()
    _long_position(s)
    fills = s.on_bar(bar(0, 120, 155, 119, 150))
    assert fills[0].exit_reason == "target" and fills[0].exit_price == 152


def test_stop_checked_before_target_when_both_in_range():
    s = small_strategy()
    _long_position(s)
    fills = s.on_bar(bar(0, 100, 160, 60, 100))
    assert fills[0].exit_reason == "stop" and fills[0].exit_price == 65


def test_trailing_activates_at_2_1_atr_and_trails_60_points():
    s = small_strategy(trail_points=10.0)
    _long_position(s)
    s.on_bar(bar(0, 100, 120, 99, 115))  # +20 < 21 activation
    assert not s.position.trailing and s.position.stop == 65
    s.on_bar(bar(1, 115, 125, 114, 124))  # +25 >= 21
    assert s.position.trailing and s.position.stop == 115
    s.on_bar(bar(2, 124, 130, 121, 129))
    assert s.position.stop == 120
    s.on_bar(bar(3, 129, 129, 126, 127))  # lower high doesn't loosen it
    assert s.position.stop == 120
    fills = s.on_bar(bar(4, 127, 128, 118, 119))
    assert fills[0].exit_reason == "trail_stop" and fills[0].exit_price == 120
    assert fills[0].stop_price == 65  # 1R still measured from the initial stop


def test_trail_never_loosens_initial_stop():
    s = small_strategy()  # 60-point trail, wider than the 35-point initial stop
    _long_position(s)
    s.on_bar(bar(0, 100, 125, 99, 120))
    assert s.position.trailing and s.position.stop == 65


def test_fill_bar_that_also_reaches_stop_exits_as_stop():
    s = _with_bracket(atr=1.0)  # stop 3.5 below a 110 fill
    fills = s.on_bar(bar(0, 105, 111, 100, 104))
    # buy stop (5 from the open) is nearer than the sell stop (15) -> long
    assert fills[0].direction == "long" and fills[0].exit_reason == "stop"
    assert fills[0].exit_price == pytest.approx(106.5)
