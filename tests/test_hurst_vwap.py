import math
from datetime import datetime, timedelta

import pytest

from failed2s.bars import Bar
from hurst_vwap.strategy import HurstVWAPStrategy, RollingHurst

T0 = datetime(2026, 1, 5, 9, 30)  # within default session window


def bar(i, o, h, l, c, v=1.0):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c, v)


class _FixedHurst:
    """Test double: forces RollingHurst.value to a fixed reading, isolating
    the VWAP-band trigger logic from the Hurst estimator's own correctness."""

    def __init__(self, value):
        self._value = value

    def update(self, close):
        pass

    @property
    def value(self):
        return self._value


def _flat_warmup(n=5, price=100.0):
    return [bar(i, price, price, price, price, 1.0) for i in range(n)]


def test_rolling_hurst_manual_rs_calc_matches():
    h = RollingHurst(window=3)
    closes = [100.0, 101.0, 99.0, 102.0]  # 3 returns
    for c in closes:
        h.update(c)

    returns = [math.log(closes[i + 1] / closes[i]) for i in range(3)]
    mean_r = sum(returns) / 3
    deviations = [r - mean_r for r in returns]
    cumulative = []
    running = 0.0
    for d in deviations:
        running += d
        cumulative.append(running)
    expected_r = max(cumulative) - min(cumulative)
    expected_s = (sum(d * d for d in deviations) / 3) ** 0.5
    expected_h = math.log(expected_r / expected_s) / math.log(3)

    assert h.value == pytest.approx(expected_h)


def test_rolling_hurst_none_until_window_plus_one_closes():
    h = RollingHurst(window=5)
    for c in [100.0, 101.0, 99.0, 102.0, 98.0]:  # 5 closes -> only 4 returns, need 5
        assert h.value is None
        h.update(c)
    h.update(103.0)  # 6th close -> 5 returns, window is now full
    assert h.value is not None


def test_rolling_hurst_distinguishes_trend_from_oscillation():
    trend = RollingHurst(window=30)
    price = 100.0
    for i in range(31):
        price += 0.5
        trend.update(price)

    osc = RollingHurst(window=30)
    for i in range(31):
        osc.update(100 + 5 * math.sin(i * 1.1))

    assert trend.value > 0.5
    assert osc.value < trend.value  # oscillation reads meaningfully more mean-reverting


def test_constructor_rejects_stop_sd_not_greater_than_entry_sd():
    with pytest.raises(ValueError, match="must be greater than"):
        HurstVWAPStrategy(entry_sd=3.0, stop_sd=2.5)
    with pytest.raises(ValueError, match="must be greater than"):
        HurstVWAPStrategy(entry_sd=2.5, stop_sd=2.5)


def test_no_signal_while_regime_is_trending_even_if_band_is_touched():
    strat = HurstVWAPStrategy(tick_size=0.25, warmup_bars=5, min_stdev_ticks=2.0)
    strat.hurst = _FixedHurst(0.7)  # trending -- entries should be blocked
    for b in _flat_warmup():
        strat.on_bar(b)
    signal = strat.on_bar(bar(5, 100, 101.5, 99, 100, 1.0))  # would otherwise cross the short band
    assert signal is None


def test_short_fires_at_upper_band_touch_with_stop_beyond_third_band():
    strat = HurstVWAPStrategy(tick_size=0.25, entry_sd=2.5, stop_sd=3.0, stop_buffer_ticks=2, warmup_bars=5, min_stdev_ticks=2.0)
    strat.hurst = _FixedHurst(0.3)  # confirmed range-bound
    for b in _flat_warmup():
        strat.on_bar(b)

    signal = strat.on_bar(bar(5, 100, 101.5, 99, 100, 1.0))
    assert signal is not None
    assert signal.direction == "short"
    vwap = strat.vwap.vwap
    sd = 0.5  # floored: min_stdev_ticks(2.0) * tick_size(0.25)
    assert signal.entry_price == pytest.approx(vwap + 2.5 * sd)
    assert signal.stop_price == pytest.approx(vwap + 3.0 * sd + 2 * 0.25)
    assert signal.target_price == pytest.approx(vwap)  # frozen VWAP baseline
    assert signal.stop_price > signal.entry_price  # stop sits outside (beyond) the entry band


def test_long_fires_at_lower_band_touch_with_stop_beyond_third_band():
    strat = HurstVWAPStrategy(tick_size=0.25, entry_sd=2.5, stop_sd=3.0, stop_buffer_ticks=2, warmup_bars=5, min_stdev_ticks=2.0)
    strat.hurst = _FixedHurst(0.3)
    for b in _flat_warmup():
        strat.on_bar(b)

    signal = strat.on_bar(bar(5, 100, 101, 98.5, 100, 1.0))
    assert signal is not None
    assert signal.direction == "long"
    vwap = strat.vwap.vwap
    sd = 0.5
    assert signal.entry_price == pytest.approx(vwap - 2.5 * sd)
    assert signal.stop_price == pytest.approx(vwap - 3.0 * sd - 2 * 0.25)
    assert signal.target_price == pytest.approx(vwap)
    assert signal.stop_price < signal.entry_price


def test_no_signal_before_warmup_or_before_hurst_window_full():
    strat = HurstVWAPStrategy(tick_size=0.25, warmup_bars=5, min_stdev_ticks=2.0)
    strat.hurst = _FixedHurst(None)  # not enough history yet, like the real estimator pre-warmup
    for b in _flat_warmup():
        assert strat.on_bar(b) is None
    assert strat.on_bar(bar(5, 100, 101.5, 99, 100, 1.0)) is None


def test_vwap_resets_daily_but_hurst_window_persists():
    strat = HurstVWAPStrategy(tick_size=0.25, hurst_window=5, warmup_bars=3)
    for b in _flat_warmup(3):
        strat.on_bar(b)
    hurst_closes_before = len(strat.hurst._closes)
    assert strat.vwap.vwap == 100.0
    assert strat._bar_count == 3

    next_day = Bar(T0 + timedelta(days=1), 200, 200, 200, 200, 1.0)
    strat.on_bar(next_day)

    assert strat.vwap.vwap == 200.0  # VWAP reset for the new session
    assert strat._bar_count == 1
    assert len(strat.hurst._closes) == hurst_closes_before + 1  # Hurst window carried over, not reset
