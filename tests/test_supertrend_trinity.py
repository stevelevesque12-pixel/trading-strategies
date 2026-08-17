from datetime import datetime, timedelta

import pytest

from failed2s.bars import Bar
from supertrend_trinity.strategy import SuperTrendTracker, TrinitySuperTrendStrategy

T0 = datetime(2026, 1, 5, 9, 30)


def bar(i, h, l, c, o=None, v=1.0):
    return Bar(T0 + timedelta(minutes=i), o if o is not None else c, h, l, c, v)


TRENDING_BARS = [
    bar(0, 105, 95, 100),
    bar(1, 106, 96, 102),
    bar(2, 108, 98, 105),
    bar(3, 110, 100, 108),
    bar(4, 130, 115, 125),  # triggers a bullish flip
]


def test_supertrend_tracker_matches_hand_computed_values():
    st = SuperTrendTracker(atr_period=3, mult=1.0)
    results = [st.update(b) for b in TRENDING_BARS]

    assert results[0] == (None, 1)
    assert results[1] == (None, 1)
    assert results[2] == (113.0, 1)  # ATR seeds at 10.0 -> dn = hl2(103)+10 = 113, bearish (no prior band yet)
    assert results[3] == (113.0, 1)  # ratchets dn down from 115 to 113 (prior dn), still bearish
    assert results[4] == (108.5, -1)  # close(125) breaks above prior dn(113) -> bullish flip


def test_no_signal_before_st1_warms_up():
    strat = TrinitySuperTrendStrategy(atr_period1=3, mult1=1.0)
    for b in TRENDING_BARS[:2]:  # ATR(3) needs 3 bars
        assert strat.on_entry_bar(b) is None


def test_no_signal_on_the_bar_that_first_establishes_direction():
    """The first bar where ST1's ATR warms up sets an initial direction but
    can't be a 'flip' (there's no prior direction to flip from)."""
    strat = TrinitySuperTrendStrategy(atr_period1=3, mult1=1.0)
    for b in TRENDING_BARS[:3]:
        signal = strat.on_entry_bar(b)
    assert signal is None


def test_single_mode_fires_on_st1_flip_alone():
    strat = TrinitySuperTrendStrategy(
        atr_period1=3, mult1=1.0, entry_mode="single", use_htf_atr=False, sl_mult=1.0, tp_mult=1.0
    )
    signal = None
    for b in TRENDING_BARS:
        signal = strat.on_entry_bar(b)
    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 125  # bar4 close
    assert signal.stop_price == 125 - 1.0 * 14.0  # ST1's own ATR at bar4
    assert signal.target_price == 125 + 1.0 * 14.0


def test_triple_mode_requires_st2_and_st3_alignment():
    strat = TrinitySuperTrendStrategy(atr_period1=3, mult1=1.0, atr_period2=3, atr_period3=3, entry_mode="triple")
    for b in TRENDING_BARS[:3]:
        strat.on_tf2_bar(b)
        strat.on_tf3_bar(b)
    strat._st2.direction = 1  # bearish -- misaligned with the upcoming bullish ST1 flip
    strat._st3.direction = -1

    signal = None
    for b in TRENDING_BARS:
        signal = strat.on_entry_bar(b)
    assert signal is None  # blocked by misalignment despite ST1 flipping


def test_triple_mode_fires_when_all_three_aligned():
    strat = TrinitySuperTrendStrategy(
        atr_period1=3, mult1=1.0, atr_period2=3, atr_period3=3, entry_mode="triple", use_htf_atr=True, sl_mult=1.0, tp_mult=1.0
    )
    for b in TRENDING_BARS[:3]:
        strat.on_tf2_bar(b)
        strat.on_tf3_bar(b)
    strat._st2.direction = -1
    strat._st3.direction = -1

    signal = None
    for b in TRENDING_BARS:
        signal = strat.on_entry_bar(b)
    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 125
    assert signal.stop_price == 125 - 1.0 * strat._atr2_value  # HTF (ST2) ATR used for risk
    assert signal.target_price == 125 + 1.0 * strat._atr2_value


def test_double_mode_only_needs_st2_alignment():
    strat = TrinitySuperTrendStrategy(atr_period1=3, mult1=1.0, atr_period2=3, atr_period3=3, entry_mode="double", use_htf_atr=False)
    for b in TRENDING_BARS[:3]:
        strat.on_tf2_bar(b)
        strat.on_tf3_bar(b)
    strat._st2.direction = -1  # aligned
    strat._st3.direction = 1   # misaligned -- irrelevant in double mode

    signal = None
    for b in TRENDING_BARS:
        signal = strat.on_entry_bar(b)
    assert signal is not None
    assert signal.direction == "long"


def test_invalid_entry_mode_rejected():
    with pytest.raises(ValueError, match="entry_mode"):
        TrinitySuperTrendStrategy(entry_mode="quadruple")


def test_trailing_stop_ratchets_monotonically_and_never_loosens():
    strat = TrinitySuperTrendStrategy(trailing=True, trail_atr_period=3, trail_atr_mult=1.0, trail_source="atr")
    stop = 90.0
    stops = []
    for b in TRENDING_BARS:
        stop = strat.trail_stop(b, "long", stop)
        stops.append(stop)
    # never decreases (long side)
    assert all(stops[i] <= stops[i + 1] for i in range(len(stops) - 1))
    assert stops[-1] == 125 - 1.0 * 14.0  # close(125) - trail_atr(14.0) at bar4


def test_trailing_stop_percent_source():
    strat = TrinitySuperTrendStrategy(trailing=True, trail_source="percent", trail_pct=2.0)
    stop = strat.trail_stop(TRENDING_BARS[0], "long", 90.0)
    assert stop == max(90.0, 100 - 100 * 2.0 / 100.0)  # close=100, 2% offset -> candidate=98, stays at 90


def test_short_side_mirrors_long():
    # SuperTrend starts bearish by default -- to test a bearish FLIP (not just
    # "never left bearish"), first flip it bullish, then reverse hard.
    bars = TRENDING_BARS + [bar(5, 110, 90, 95)]  # sharp reversal down after the bullish flip
    strat = TrinitySuperTrendStrategy(atr_period1=3, mult1=1.0, entry_mode="single", use_htf_atr=False)
    signal = None
    for b in bars:
        signal = strat.on_entry_bar(b)
    assert signal is not None
    assert signal.direction == "short"
    assert signal.stop_price > signal.entry_price
    assert signal.target_price < signal.entry_price
