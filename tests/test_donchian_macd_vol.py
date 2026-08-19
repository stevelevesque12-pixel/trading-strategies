from collections import deque
from datetime import datetime, time, timedelta

import pytest

from failed2s.bars import Bar
from failed2s.strategy import SessionConfig
from donchian_macd_vol.strategy import DonchianMacdVolumeStrategy, _DonchianChannel, _MACD, _RollingAverage

T0 = datetime(2026, 1, 5, 9, 30)


def bar(i, o, h, l, c, v=100.0):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c, v)


FLAT_WARMUP = [bar(i, 100, 101, 99, 100, 100) for i in range(6)]


def _warmed_up_long(**kwargs):
    strat = DonchianMacdVolumeStrategy(
        tick_size=0.25, donchian_period=5, macd_fast=3, macd_slow=6, macd_signal=2,
        volume_period=5, volume_mult=1.5, target_r=1.5, stop_buffer_ticks=2, **kwargs
    )
    for b in FLAT_WARMUP:
        strat.on_bar(b)
    strat.on_bar(bar(6, 100, 100.5, 99.5, 100.5, 100))  # builds positive MACD momentum
    return strat


def test_donchian_channel_is_none_until_warmed_up_then_rolls():
    dc = _DonchianChannel(period=3)
    assert dc.upper is None and dc.lower is None
    dc.update(bar(0, 100, 105, 95, 100))
    assert dc.upper is None  # only 1 of 3 bars so far
    dc.update(bar(1, 100, 103, 97, 100))
    dc.update(bar(2, 100, 101, 99, 100))
    assert dc.upper == 105
    assert dc.lower == 95

    # Reading .upper/.lower must not itself mutate anything -- a caller
    # (the strategy) is responsible for reading before calling update() to
    # exclude the "current" bar from its own channel.
    assert dc.upper == 105 and dc.lower == 95

    huge = bar(3, 100, 200, 1, 100)
    dc.update(huge)  # window rolls: bar0 drops out, bars 1/2/3 remain
    assert dc.upper == 200  # huge bar is now IN the window
    assert dc.lower == 1


def test_macd_ema_seeds_directly_from_first_sample():
    macd = _MACD(fast=3, slow=6, signal=2)
    assert macd.histogram is None
    for c in [100, 101, 102, 103, 104, 105]:
        macd.update(c)
    assert macd.histogram is not None
    assert macd.histogram > 0  # rising prices -> fast EMA pulls ahead of slow EMA


def test_rolling_average_none_until_warmed_up():
    avg = _RollingAverage(period=3)
    avg.update(10)
    assert avg.value is None
    avg.update(20)
    assert avg.value is None
    avg.update(30)
    assert avg.value == 20.0


def test_no_signal_before_warmup():
    strat = DonchianMacdVolumeStrategy(tick_size=0.25, donchian_period=5, volume_period=5)
    for b in FLAT_WARMUP[:4]:  # donchian(5)/volume(5) not yet warmed up
        assert strat.on_bar(b) is None


def test_long_breakout_fires_at_channel_level_with_opposite_edge_stop():
    strat = _warmed_up_long()
    signal = strat.on_bar(bar(7, 100.5, 105, 100, 104, 300))
    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 101  # the donchian upper level, not bar close
    assert signal.stop_price == 98.5  # opposite edge (99) minus 2*0.25 buffer
    assert signal.target_price == pytest.approx(104.75)  # entry + 1.5*(101-98.5)


def test_low_volume_blocks_an_otherwise_valid_breakout():
    strat = _warmed_up_long()
    signal = strat.on_bar(bar(7, 100.5, 105, 100, 104, 120))  # avg*1.5 = 150, this is only 120
    assert signal is None


def test_momentum_disagreement_blocks_an_otherwise_valid_breakout():
    strat = _warmed_up_long()
    strat._macd.histogram = -0.5  # force disagreement with the upward breakout
    signal = strat.on_bar(bar(7, 100.5, 105, 100, 104, 300))
    assert signal is None


def test_short_breakout_mirrors_long():
    strat = DonchianMacdVolumeStrategy(
        tick_size=0.25, donchian_period=5, macd_fast=3, macd_slow=6, macd_signal=2,
        volume_period=5, volume_mult=1.5, target_r=1.5, stop_buffer_ticks=2,
    )
    for b in FLAT_WARMUP:
        strat.on_bar(b)
    strat.on_bar(bar(6, 100, 100.5, 99.5, 99.5, 100))  # builds negative MACD momentum

    signal = strat.on_bar(bar(7, 99.5, 100, 95, 96, 300))
    assert signal is not None
    assert signal.direction == "short"
    assert signal.entry_price == 99  # donchian lower
    assert signal.stop_price == 101.5  # opposite edge (101) plus buffer
    assert signal.target_price == pytest.approx(95.25)


def test_no_entry_after_cutoff():
    session = SessionConfig(no_entry_after=time(9, 36))
    strat = _warmed_up_long(session=session)  # last warmup bar lands at 9:36, cutoff excludes it
    signal = strat.on_bar(bar(7, 100.5, 105, 100, 104, 300))  # 9:37, past cutoff
    assert signal is None


def test_degenerate_zero_risk_does_not_fire():
    strat = _warmed_up_long()
    strat.stop_buffer = 0.0
    # force the opposite edge to equal the entry level itself -- zero risk
    strat._donchian._lows = deque([101, 101, 101, 101, 101], maxlen=5)
    signal = strat.on_bar(bar(7, 100.5, 105, 100, 104, 300))
    assert signal is None
