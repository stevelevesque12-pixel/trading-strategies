from datetime import datetime, time, timedelta

import pytest

from asian_sweep.session import SessionConfig, trading_date
from asian_sweep.strategy import AsianSweepStrategy
from failed2s.bars import Bar

# Box: Jan 5 20:00-20:30 ET, high=102 low=98. Belongs to the Jan 6 trading day.
BOX_START = datetime(2026, 1, 5, 20, 0)


def bar(dt, o, h, l, c):
    return Bar(dt, o, h, l, c)


def _box_bars():
    return [
        bar(BOX_START, 100, 101, 99, 100),
        bar(BOX_START + timedelta(minutes=15), 100, 102, 99, 101),
        bar(BOX_START + timedelta(minutes=30), 101, 101, 98, 99),
    ]


SWEEP_START = datetime(2026, 1, 6, 0, 0)


def _break_swing_mss_fvg_retest_bars():
    """
    Bar 0 (00:00): closes at 103, above the box high (102) -- sets a long
    break, swept_level=102.
    Bars 1-5: forms a swing high at 113 (confirmed once bar 5 pushes the
    5-bar fractal window), mirroring failed2s's swing-then-MSS test pattern.
    Bar 6 (01:30): MSS -- closes at 121, above the 113 swing, strong body,
    and a qualifying bullish FVG vs. bar 4 (01:00, high=112) -- retest_level
    becomes bar 6's low (113).
    Bar 7 (01:45): retests down into 113 -- fires the long signal.
    """
    return [
        bar(SWEEP_START, 101, 104, 100, 103),
        bar(SWEEP_START + timedelta(minutes=15), 103, 104, 102, 103),
        bar(SWEEP_START + timedelta(minutes=30), 103, 105, 103, 104),
        bar(SWEEP_START + timedelta(minutes=45), 104, 113, 104, 112),  # pivot high candidate = 113
        bar(SWEEP_START + timedelta(minutes=60), 111, 112, 110, 111),
        bar(SWEEP_START + timedelta(minutes=75), 111, 111, 109, 110),  # confirms swing high = 113
        bar(SWEEP_START + timedelta(minutes=90), 113, 122, 113, 121),  # MSS + FVG (gap vs bar[112 high])
        bar(SWEEP_START + timedelta(minutes=105), 120, 121, 112, 115),  # retest touch at 113
    ]


def _run(strategy, bars):
    signal = None
    for b in bars:
        s = strategy.on_bar(b)
        if s is not None:
            signal = s
    return signal


def test_full_cascade_produces_long_signal_on_retest():
    strategy = AsianSweepStrategy(tick_size=0.25, stop_buffer_ticks=4, target_r=2.0)
    for b in _box_bars():
        strategy.on_bar(b)

    signal = _run(strategy, _break_swing_mss_fvg_retest_bars())

    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 113  # FVG near edge (bar 6's low)
    assert signal.stop_price == 101  # min(swing 113, swept_level 102) - 4*0.25
    assert signal.target_price == 137  # entry + risk(12) * target_r(2.0)


def test_no_signal_without_overnight_box_data():
    strategy = AsianSweepStrategy()
    # No box bars fed at all -- the strategy should never trade this day.
    signal = _run(strategy, _break_swing_mss_fvg_retest_bars())
    assert signal is None


def test_no_break_no_signal_when_price_stays_inside_the_box():
    strategy = AsianSweepStrategy()
    for b in _box_bars():
        strategy.on_bar(b)

    inside_bars = [
        bar(SWEEP_START, 100, 101, 99, 100),
        bar(SWEEP_START + timedelta(minutes=15), 100, 101, 99, 100),
        bar(SWEEP_START + timedelta(minutes=30), 100, 101, 99, 100),
    ]
    signal = _run(strategy, inside_bars)
    assert signal is None
    assert strategy._pending_break is None


def test_pending_retest_invalidated_if_boundary_given_back_before_touch():
    strategy = AsianSweepStrategy(tick_size=0.25, stop_buffer_ticks=4)
    for b in _box_bars():
        strategy.on_bar(b)

    bars = _break_swing_mss_fvg_retest_bars()
    for b in bars[:7]:  # through the MSS+FVG confirmation bar -- retest armed
        strategy.on_bar(b)
    assert strategy._pending_retest is not None

    # Instead of retesting up near 113, price collapses straight back below
    # the broken box boundary (102) without ever touching the retest level.
    collapse = bar(SWEEP_START + timedelta(minutes=105), 100, 101, 95, 96)
    signal = strategy.on_bar(collapse)

    assert signal is None
    assert strategy._pending_retest is None


def test_no_signal_outside_entry_window():
    strategy = AsianSweepStrategy(session=SessionConfig(no_entry_after=time(0, 30)))
    for b in _box_bars():
        strategy.on_bar(b)

    signal = _run(strategy, _break_swing_mss_fvg_retest_bars())
    assert signal is None  # retest touch happens at 01:45, well past the 00:30 cutoff


def test_state_resets_on_new_trading_day():
    strategy = AsianSweepStrategy()
    for b in _box_bars():
        strategy.on_bar(b)
    for b in _break_swing_mss_fvg_retest_bars()[:7]:
        strategy.on_bar(b)
    assert strategy._pending_retest is not None

    next_day_rth = bar(datetime(2026, 1, 7, 9, 30), 120, 121, 119, 120)
    strategy.on_bar(next_day_rth)

    assert strategy._pending_retest is None
    assert strategy._box is None


@pytest.mark.parametrize(
    "ts,expected",
    [
        (datetime(2026, 1, 5, 19, 59), datetime(2026, 1, 5).date()),
        (datetime(2026, 1, 5, 20, 0), datetime(2026, 1, 6).date()),
        (datetime(2026, 1, 6, 0, 0), datetime(2026, 1, 6).date()),
        (datetime(2026, 1, 6, 9, 30), datetime(2026, 1, 6).date()),
    ],
)
def test_trading_date_rolls_forward_at_box_start(ts, expected):
    assert trading_date(ts, SessionConfig()) == expected
