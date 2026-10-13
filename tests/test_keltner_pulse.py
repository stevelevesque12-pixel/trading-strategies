import pandas as pd
import pytest

from failed2s.instruments import INSTRUMENTS
from keltner_pulse.strategy import KeltnerPulseBacktest, KeltnerPulseConfig, add_indicators

NQ = INSTRUMENTS["NQ"]
WARMUP = 40


def frame(extra):
    """WARMUP flat-ish bars around 1000 (each 2 points tall), then `extra` (o, h, l, c) bars."""
    rows = [(1000.0, 1001.0, 999.0, 1000.0)] * WARMUP + list(extra)
    idx = pd.date_range("2026-01-05 00:00", periods=len(rows), freq="15min", tz="America/New_York")
    return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"]).assign(volume=1.0)


def run(extra, **kw):
    return KeltnerPulseBacktest(NQ, KeltnerPulseConfig(**kw)).run(frame(extra))


def signal_levels(extra_upto_signal, **kw):
    cfg = KeltnerPulseConfig(**kw)
    d = add_indicators(frame(extra_upto_signal), cfg)
    sig = d.iloc[-1]
    assert sig.close > sig.kc_upper
    return sig, cfg


BREAKOUT = (1000.0, 1010.0, 1000.0, 1009.0)  # closes well above the ~1000 +/- 5.2 upper band


def test_no_trade_without_breakout():
    assert run([(1000.0, 1001.0, 999.0, 1000.0)] * 20) == []


def test_breakout_places_buy_stop_and_fills_then_stops_out():
    sig, cfg = signal_levels([BREAKOUT])
    entry = round((sig.high + 0.1 * sig.atr) / 0.25) * 0.25
    stop = round((entry - 2.5 * sig.atr) / 0.25) * 0.25
    trades = run([BREAKOUT, (1009.0, entry + 1, 1008.0, entry), (entry, entry, stop - 5, stop - 5)])
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "long"
    assert t.entry_price == pytest.approx(entry)
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(stop)
    assert t.r_multiple == pytest.approx(-1.0, abs=0.02)


def test_order_not_filled_on_signal_bar_itself():
    # Signal bar's own high is below the stop price by construction (high + buffer).
    trades = run([BREAKOUT] + [(1005.0, 1006.0, 1004.0, 1005.0)] * 5)
    assert trades == []


def test_pending_order_expires_after_46_bars():
    sig, _ = signal_levels([BREAKOUT])
    entry = sig.high + 0.1 * sig.atr
    quiet = [(1005.0, 1006.0, 1004.0, 1005.0)] * 46
    late_spike = [(1005.0, entry + 50, 1004.0, 1005.0)]
    # 46 bars after placement the order is still live: a spike on bar 46 fills.
    assert len(run([BREAKOUT] + quiet[:-1] + late_spike + [(1005.0, 1006.0, 1004.0, 1005.0)] * 30)) == 1
    # Bar 47 is too late.
    assert run([BREAKOUT] + quiet + late_spike) == []


def test_gap_through_entry_fills_at_open():
    sig, _ = signal_levels([BREAKOUT])
    gap_open = sig.high + 20
    t = run([BREAKOUT, (gap_open, gap_open + 1, gap_open - 1, gap_open)] + [(gap_open, gap_open + 1, gap_open - 1, gap_open)] * 40)[0]
    assert t.entry_price == pytest.approx(gap_open)


def test_profit_target_is_1_5_percent():
    sig, _ = signal_levels([BREAKOUT])
    entry = round((sig.high + 0.1 * sig.atr) / 0.25) * 0.25
    target = round(entry * 1.015 / 0.25) * 0.25
    # Fill bar, then a bar that runs to the target without the trail (activated
    # 1.8 ATR up, trailing 100 pts) getting anywhere near the low.
    t = run([BREAKOUT, (1009.0, entry, 1008.0, entry), (entry, target + 1, entry - 1, target)])[0]
    assert t.exit_reason == "target"
    assert t.exit_price == pytest.approx(target)


def test_time_exit_after_30_bars():
    sig, _ = signal_levels([BREAKOUT])
    entry = sig.high + 0.1 * sig.atr
    hold = (entry, entry + 1, entry - 1, entry)
    trades = run([BREAKOUT, hold] + [hold] * 40)
    t = trades[0]
    assert t.exit_reason == "time"
    held = (t.exit_time - t.entry_time) / pd.Timedelta("15min") + 1
    assert held == 30


def test_trailing_stop_activates_and_trails_100_points():
    sig, _ = signal_levels([BREAKOUT])
    entry = round((sig.high + 0.1 * sig.atr) / 0.25) * 0.25
    peak = entry + 120  # well past 1.8 ATR
    # 1.5% of ~1011 is only ~15 pts, so widen the target so only the trail can close the trade.
    trades = KeltnerPulseBacktest(NQ, KeltnerPulseConfig(profit_target_pct=50)).run(
        frame([BREAKOUT, (1009.0, entry, 1008.0, entry), (entry, peak, entry, peak - 5), (peak - 5, peak - 5, peak - 110, peak - 110)])
    )
    t = trades[0]
    assert t.exit_reason == "trail_stop"
    assert t.exit_price == pytest.approx(peak - 100)


def test_trailing_not_active_below_activation():
    sig, _ = signal_levels([BREAKOUT])
    entry = round((sig.high + 0.1 * sig.atr) / 0.25) * 0.25
    small = entry + 1.0 * sig.atr  # below 1.8 ATR
    stop = round((entry - 2.5 * sig.atr) / 0.25) * 0.25
    t = KeltnerPulseBacktest(NQ, KeltnerPulseConfig(profit_target_pct=50, trail_points=1)).run(
        frame([BREAKOUT, (1009.0, entry, 1008.0, entry), (entry, small, entry, small), (small, small, stop - 1, stop - 1)])
    )[0]
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(stop)


def test_short_breakdown_mirror():
    down = (1000.0, 1000.0, 990.0, 991.0)
    cfg = KeltnerPulseConfig()
    d = add_indicators(frame([down]), cfg)
    sig = d.iloc[-1]
    assert sig.close < sig.kc_lower
    entry = round((sig.low - 0.1 * sig.atr) / 0.25) * 0.25
    stop = round((entry + 2.5 * sig.atr) / 0.25) * 0.25
    t = run([down, (991.0, 992.0, entry, entry), (entry, stop + 5, entry, stop + 5)])[0]
    assert t.direction == "short"
    assert t.entry_price == pytest.approx(entry)
    assert t.exit_reason == "stop"
    assert t.pnl_points < 0


def test_costs_reduce_pnl():
    sig, _ = signal_levels([BREAKOUT])
    entry = sig.high + 0.1 * sig.atr
    hold = (entry, entry + 1, entry - 1, entry)
    bars = [BREAKOUT, hold] + [hold] * 40
    gross = run(bars)[0]
    net = run(bars, commission_rt=5.0, slippage_ticks=1)[0]
    assert net.entry_price == pytest.approx(gross.entry_price + 0.25)
    assert net.pnl_dollars == pytest.approx(gross.pnl_dollars - 0.25 * 20 - 5.0)
