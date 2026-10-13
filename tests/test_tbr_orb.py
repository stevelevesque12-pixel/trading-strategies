import numpy as np

from tbr_orb.research import Config, day_trades


def _day(bars):
    """bars: list of (minute_of_day, open, high, low, close)."""
    a = np.array(bars, dtype=float)
    m, o, h, l, c = a.T
    atr = np.full(len(a), 2.0)
    return (o, h, l, c, atr, m.astype(int))


def _range_bars():
    # 9:30-9:59 range: high 110, low 100
    out = []
    for k in range(30):
        out.append((570 + k, 105, 110 if k == 5 else 106, 100 if k == 10 else 104, 105))
    return out


def test_breakout_long_hits_3r_target():
    bars = _range_bars()
    bars.append((600, 109, 112, 109, 111))  # close above 110 -> long signal, entry 111.25
    bars.append((601, 111, 145, 110.5, 140))  # risk 11.25 -> target 145
    t = day_trades(_day(bars), Config(), "d")
    assert len(t) == 1
    assert t[0].direction == 1 and t[0].reason == "target"
    assert t[0].stop == 100 and t[0].exit == 111.25 + 3 * 11.25


def test_breakout_short_stopped_at_range_high():
    bars = _range_bars()
    bars.append((600, 101, 101, 98, 99))  # close below 100 -> short
    bars.append((601, 99, 111, 98, 110))  # through range high -> stop
    t = day_trades(_day(bars), Config(), "d")
    assert t[0].direction == -1 and t[0].reason == "stop"
    assert t[0].exit == 110 + 0.25  # stop slippage


def test_time_exit_and_no_signal_after_cutoff():
    bars = _range_bars()
    bars.append((870, 109, 112, 109, 111))  # 14:30 -> past entry cutoff
    assert day_trades(_day(bars), Config(), "d") == []
    bars = _range_bars() + [(600, 109, 112, 109, 111), (955, 112, 113, 111, 112)]
    t = day_trades(_day(bars), Config(), "d")
    assert t[0].reason == "time" and t[0].exit == 112


def test_ns_orb_limit_fill_and_target():
    bars = _range_bars()
    bars.append((600, 111, 113, 110.5, 112))  # full body above 110 -> setup
    # limit = 110 - 0.125*10 = 108.75, stop = 110 - 9.5 = 100.5, tp = 108.75 + 0.4*8.25 = 112.05
    bars.append((601, 111, 111, 108.0, 109))  # trades through limit -> fill
    bars.append((602, 109, 113, 108.5, 112.5))  # target
    t = day_trades(_day(bars), Config(style="ns"), "d")
    assert len(t) == 1 and t[0].reason == "target"
    assert abs(t[0].entry - 108.75) < 1e-9 and abs(t[0].exit - 112.05) < 1e-9


def test_max_range_filter_skips_wide_days():
    bars = _range_bars() + [(600, 109, 112, 109, 111)]
    # range 10 on price ~105 = 9.5% of price
    assert day_trades(_day(bars), Config(max_range_pct=5.0), "d") == []
