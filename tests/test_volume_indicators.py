from datetime import datetime, timedelta

from failed2s.bars import Bar
from volume_scalp.indicators import RollingChannel, RollingDelta, RollingVolume, SessionVWAP, bar_volume_delta, clv

T0 = datetime(2026, 1, 5, 9, 30)


def bar(i, o, h, l, c, v=100):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c, v)


def test_clv_at_high_low_and_mid():
    assert clv(bar(0, 100, 110, 100, 110)) == 1.0  # closed at the high
    assert clv(bar(0, 100, 110, 100, 100)) == -1.0  # closed at the low
    assert clv(bar(0, 100, 110, 100, 105)) == 0.0  # closed at the midpoint
    assert clv(bar(0, 100, 100, 100, 100)) == 0.0  # zero-range bar, no division by zero


def test_bar_volume_delta_scales_clv_by_volume():
    b = bar(0, 100, 110, 100, 110, v=200)  # clv = 1.0
    assert bar_volume_delta(b) == 200.0

    b2 = bar(0, 100, 110, 100, 100, v=50)  # clv = -1.0
    assert bar_volume_delta(b2) == -50.0


def test_session_vwap_is_volume_weighted_typical_price():
    vwap = SessionVWAP()
    assert vwap.value is None

    vwap.update(bar(0, 100, 102, 98, 100, v=100))  # typical = 100
    assert vwap.value == 100.0

    vwap.update(bar(1, 100, 106, 98, 102, v=300))  # typical = 102
    # cum_pv = 100*100 + 102*300 = 40600, cum_vol = 400 -> 101.5
    assert vwap.value == 101.5


def test_session_vwap_reset_clears_state():
    vwap = SessionVWAP()
    vwap.update(bar(0, 100, 102, 98, 100, v=100))
    assert vwap.value is not None
    vwap.reset()
    assert vwap.value is None


def test_rolling_volume_relvol_excludes_current_bar_until_updated():
    rv = RollingVolume(window=3)
    assert rv.relvol(999) == 0.0  # no baseline yet

    for v in (100, 100, 100):
        rv.update(v)

    assert rv.average == 100.0
    assert rv.relvol(250) == 2.5  # baseline unaffected by the bar being judged
    rv.update(250)
    assert rv.average == 150.0  # (100 + 100 + 250) / 3


def test_rolling_delta_sums_trailing_window_including_current():
    rd = RollingDelta(window=2)
    rd.update(10)
    assert rd.value == 10
    rd.update(-4)
    assert rd.value == 6
    rd.update(5)  # window=2, oldest (10) drops off
    assert rd.value == 1


def test_rolling_channel_none_until_full_then_excludes_current_bar():
    rc = RollingChannel(window=3)
    assert rc.high is None and rc.low is None

    rc.update(bar(0, 100, 101, 99, 100))
    rc.update(bar(1, 100, 102, 100, 101))
    assert rc.high is None  # still only 2 bars

    rc.update(bar(2, 101, 110, 101, 109))
    assert rc.high == 110
    assert rc.low == 99

    # Reading again before the next update() must not include a hypothetical current bar.
    assert rc.high == 110
    rc.update(bar(3, 108, 200, 107, 108))  # a huge new high
    assert rc.high == 200  # now included, since it was update()'d
