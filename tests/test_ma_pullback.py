import pandas as pd
import pytest

from ma_pullback.strategy import MAPullbackConfig, run

# Tiny MA lengths so the values can be checked by hand: SMA2 = fast, SMA3 = slow.
CFG = MAPullbackConfig(fast_len=2, slow_len=3, stop_mode="ticks", stop_ticks=30, target_r=1.5, tick_size=0.01, point_value=100.0)

# Uptrend closes 10..13 put SMA2 above SMA3. Bar 4 closes at 11.5:
#   SMA2 = (13 + 11.5) / 2 = 12.25, SMA3 = (12 + 13 + 11.5) / 3 = 12.1667
# -> fast > slow and close below both: setup armed, sell limits at 12.25 and 12.1667.
SETUP = [
    (10, 10.1, 9.9, 10),
    (10, 11.1, 9.9, 11),
    (11, 12.1, 10.9, 12),
    (12, 13.1, 11.9, 13),
    (13, 13.0, 11.4, 11.5),
]
SLOW_AT_SETUP = (12 + 13 + 11.5) / 3


def frame(rows):
    idx = pd.date_range("2026-01-05 09:00", periods=len(rows), freq="30min", tz="America/New_York")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx)


def test_fills_at_first_ma_touched_and_hits_target():
    trades = run(frame(SETUP + [
        (11.5, 12.2, 11.4, 11.6),   # reaches the 20-MA limit (12.1667) only
        (11.8, 11.8, 11.6, 11.65),  # target = 12.1667 - 0.45 = 11.7167
    ]), CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.direction == "short" and t.filled_at == "slow"
    assert t.entry_price == pytest.approx(SLOW_AT_SETUP)
    assert t.stop_price == pytest.approx(SLOW_AT_SETUP + 0.30)
    assert t.exit_reason == "target"
    assert t.exit_price == pytest.approx(SLOW_AT_SETUP - 0.45)
    assert t.r_multiple == pytest.approx(1.5)


def test_bar_touching_both_mas_takes_only_one_trade():
    trades = run(frame(SETUP + [
        (11.5, 12.3, 11.4, 12.0),   # reaches both limits -> lower one fills, other cancelled
        (12.0, 12.1, 11.6, 11.65),
    ]), CFG)
    assert len(trades) == 1
    assert trades[0].filled_at == "slow"


def test_gap_through_limit_fills_at_open():
    trades = run(frame(SETUP + [
        (12.20, 12.25, 12.0, 12.05),  # opens above the 20-MA limit
        (12.0, 12.0, 11.70, 11.72),
    ]), CFG)
    assert trades[0].entry_price == pytest.approx(12.20)


def test_stop_on_fill_bar():
    trades = run(frame(SETUP + [(11.5, 12.6, 11.4, 12.5)]), CFG)
    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"
    assert trades[0].r_multiple == pytest.approx(-1.0)


def test_bias_flip_cancels_setup():
    # Bar 5 doesn't reach either limit and its close drops SMA2 below SMA3,
    # so the setup is cancelled. Bar 6 rallies through the old levels: no trade.
    trades = run(frame(SETUP + [
        (11.5, 11.6, 10.0, 10.1),
        (10.1, 13.0, 10.1, 12.9),
    ]), CFG)
    assert trades == []


def test_no_setup_without_close_below_both():
    # Close at 12.6: SMA2 = 12.8, SMA3 = 12.53 -> close sits between the MAs, not below both.
    rows = SETUP[:4] + [(13, 13.0, 11.4, 12.6), (12.6, 13.5, 12.1, 13.4)]
    assert run(frame(rows), CFG) == []


def test_longs_off_by_default_and_mirror_when_enabled():
    rows = [(24 - o, 24 - l, 24 - h, 24 - c) for (o, h, l, c) in SETUP] + [
        (24 - 11.5, 24 - 11.4, 24 - 12.2, 24 - 11.6),
        (24 - 11.8, 24 - 11.6, 24 - 11.8, 24 - 11.65),
    ]
    assert run(frame(rows), CFG) == []
    longs = run(frame(rows), MAPullbackConfig(**{**CFG.__dict__, "enable_longs": True}))
    assert len(longs) == 1
    assert longs[0].direction == "long" and longs[0].exit_reason == "target"
    assert longs[0].entry_price == pytest.approx(24 - SLOW_AT_SETUP)


# Swing-stop mode. Bars 0-4 form a swing high of 12.6 at bar 2 (strength 2:
# two lower highs each side), confirmed at bar 4's close. Bar 6 then sets up
# the short.
SWING_CFG = MAPullbackConfig(fast_len=2, slow_len=3, stop_mode="swing", swing_strength=2,
                             target_r=1.0, tick_size=0.01, point_value=100.0)
SWING_SETUP = [
    (11.0, 11.2, 10.9, 11.0),
    (11.0, 11.5, 10.9, 11.4),
    (11.4, 12.6, 11.3, 11.6),   # swing high 12.6
    (11.6, 11.9, 11.5, 11.8),
    (11.8, 12.0, 11.7, 12.0),   # confirms the swing (two lower highs after it)
    (12.0, 12.4, 11.9, 12.3),
]


def test_swing_stop_at_recent_swing_high_with_1r_target():
    rows = SWING_SETUP + [(12.3, 12.3, 11.7, 11.8)]
    # Bar 6 close 11.8: SMA2 = (12.3 + 11.8) / 2 = 12.05, SMA3 = (12.0 + 12.3 + 11.8) / 3 = 12.0333
    # -> fast > slow, close below both: armed. Limits 12.05 and 12.0333, stop = 12.6.
    slow = (12.0 + 12.3 + 11.8) / 3
    trades = run(frame(rows + [
        (11.8, 12.1, 11.8, 12.0),           # fills the 20-MA (slow) limit
        (12.0, 12.0, 11.40, 11.45),         # 1R target = slow - (12.6 - slow) = 11.4667
    ]), SWING_CFG)
    assert len(trades) == 1
    t = trades[0]
    assert t.filled_at == "slow"
    assert t.stop_price == pytest.approx(12.6)
    assert t.target_price == pytest.approx(slow - (12.6 - slow))
    assert t.exit_reason == "target" and t.r_multiple == pytest.approx(1.0)


def test_swing_buffer_and_no_order_without_swing_above():
    rows = SWING_SETUP + [(12.3, 12.3, 11.7, 11.8), (11.8, 12.1, 11.8, 12.0)]
    buffered = run(frame(rows + [(12.0, 12.7, 12.0, 12.65)]),
                   MAPullbackConfig(**{**SWING_CFG.__dict__, "swing_buffer_ticks": 5}))
    assert buffered[0].stop_price == pytest.approx(12.65)
    # Without the bar-2 spike there is no confirmed swing high above the limits -> no orders.
    flat = [r if i != 2 else (11.4, 11.7, 11.3, 11.6) for i, r in enumerate(rows)]
    assert run(frame(flat), SWING_CFG) == []
