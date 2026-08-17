import math
from datetime import datetime, timedelta

from failed2s.bars import Bar
from overextension.strategy import OverextensionStrategy, SessionVWAP

T0 = datetime(2026, 1, 5, 9, 30)  # within default session window


def bar(i, o, h, l, c, v=1.0):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c, v)


def _flat_warmup(n=20, price=100.0):
    return [bar(i, price, price, price, price, 1.0) for i in range(n)]


def test_session_vwap_matches_manual_volume_weighted_calc():
    vw = SessionVWAP()
    bars = [
        bar(0, 100, 102, 98, 101, v=10),
        bar(1, 101, 103, 100, 102, v=20),
        bar(2, 102, 104, 101, 103, v=5),
    ]
    for b in bars:
        vw.update(b)

    tps = [(b.high + b.low + b.close) / 3.0 for b in bars]
    vols = [b.volume for b in bars]
    expected_vwap = sum(tp * v for tp, v in zip(tps, vols)) / sum(vols)
    expected_var = sum(v * tp * tp for tp, v in zip(tps, vols)) / sum(vols) - expected_vwap**2

    assert vw.vwap == expected_vwap
    assert math.isclose(vw.stdev, math.sqrt(expected_var))


def test_session_vwap_resets_and_zero_volume_bars_fall_back_to_equal_weight():
    vw = SessionVWAP()
    vw.update(bar(0, 100, 101, 99, 100, v=0.0))  # zero volume -> equal-weight fallback
    assert vw.vwap == 100.0
    vw.reset()
    assert vw.vwap is None
    assert vw.stdev is None


def test_no_signal_before_warmup_regardless_of_extension():
    strat = OverextensionStrategy(tick_size=0.25, warmup_bars=20)
    bars = _flat_warmup(19) + [bar(19, 100, 100, 50, 55, v=1.0)]  # violent drop, still bar #20
    for b in bars:
        assert strat.on_bar(b) is None
    # bar count is now 20 (== warmup_bars), so an extension may have just started
    # (not fired -- founding bar of an episode never fires on itself)
    assert strat._extension is not None or strat._bar_count == 20


def test_long_fade_does_not_fire_on_extending_bar_or_bearish_recovery_bar_and_fires_on_bullish_exhaustion():
    strat = OverextensionStrategy(tick_size=0.25, entry_z=2.0, confirm_z=0.5, max_z=10.0, warmup_bars=20)

    for b in _flat_warmup(20):
        assert strat.on_bar(b) is None

    # Bar 20: hard drop -- starts a new long-fade extension (z far below -2.0).
    # Founding bar never fires on itself.
    b20 = bar(20, 100, 100, 90, 91, v=1.0)
    assert strat.on_bar(b20) is None
    ext = strat._extension
    assert ext is not None and ext.direction == "long"
    assert ext.extreme == 90

    # Bar 21: further down AND bearish -- extension continues, extreme updates,
    # recovered-z alone is not enough without a bullish confirmation bar.
    b21 = bar(21, 91, 91, 85, 86, v=1.0)
    assert strat.on_bar(b21) is None
    assert strat._extension is not None
    assert strat._extension.extreme == 85  # pushed to the new low

    # Bar 22: bullish reversal bar with z recovered off the extreme -- fires long.
    b22 = bar(22, 86, 93, 85, 92, v=1.0)
    signal = strat.on_bar(b22)
    assert signal is not None
    assert signal.direction == "long"
    assert signal.entry_price == 92
    assert signal.stop_price == 85 - 2 * 0.25  # extreme minus stop buffer
    assert signal.target_price > signal.entry_price  # partial/full reversion back toward vwap
    assert strat._extension is None  # consumed


def test_extension_abandoned_when_extreme_z_exceeds_max_z_trend_day_guard():
    strat = OverextensionStrategy(tick_size=0.25, entry_z=2.0, confirm_z=0.5, max_z=4.0, warmup_bars=20)

    for b in _flat_warmup(20):
        strat.on_bar(b)

    # Same violent drop as above -- with the default-ish max_z=4.0 this extreme
    # is beyond the trend-day guard.
    strat.on_bar(bar(20, 100, 100, 90, 91, v=1.0))
    assert strat._extension is not None

    # Continuation bar re-checks the guard and drops the extension.
    result = strat.on_bar(bar(21, 91, 91, 85, 86, v=1.0))
    assert result is None
    assert strat._extension is None

    # A subsequent bullish bar with a mild z (not itself extended) does not fire.
    result = strat.on_bar(bar(22, 86, 93, 85, 92, v=1.0))
    assert result is None


def test_short_fade_fires_on_bearish_exhaustion_after_upward_extension():
    strat = OverextensionStrategy(tick_size=0.25, entry_z=2.0, confirm_z=0.5, max_z=10.0, warmup_bars=20)

    for b in _flat_warmup(20):
        strat.on_bar(b)

    # Hard rally -- starts a short-fade extension.
    assert strat.on_bar(bar(20, 100, 110, 100, 109, v=1.0)) is None
    ext = strat._extension
    assert ext is not None and ext.direction == "short"
    assert ext.extreme == 110

    # Further up AND bullish -- still extending.
    assert strat.on_bar(bar(21, 109, 115, 109, 114, v=1.0)) is None
    assert strat._extension.extreme == 115

    # Bearish reversal bar -- fires short.
    signal = strat.on_bar(bar(22, 114, 115, 107, 108, v=1.0))
    assert signal is not None
    assert signal.direction == "short"
    assert signal.entry_price == 108
    assert signal.stop_price == 115 + 2 * 0.25
    assert signal.target_price < signal.entry_price


def test_reset_clears_state_on_new_session_day():
    strat = OverextensionStrategy(tick_size=0.25, warmup_bars=20)
    for b in _flat_warmup(20):
        strat.on_bar(b)
    strat.on_bar(bar(20, 100, 100, 90, 91, v=1.0))
    assert strat._extension is not None
    assert strat._bar_count == 21

    next_day = Bar(T0 + timedelta(days=1), 100, 100, 100, 100, 1.0)
    strat.on_bar(next_day)
    assert strat._bar_count == 1  # reset then re-counted for this bar
    assert strat._extension is None
