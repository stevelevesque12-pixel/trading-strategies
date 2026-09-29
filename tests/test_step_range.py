import numpy as np
import pandas as pd

from failed2s.instruments import INSTRUMENTS
from step_range.backtest import simulate
from step_range.strategy import StepRangeParams, find_breakouts, rma_atr

PARAMS = StepRangeParams(length=5, consolidation_bars=3, atr_length=3, atr_mult=1.0)


def _bars(closes, spread=0.5):
    closes = np.asarray(closes, float)
    idx = pd.date_range("2024-01-02 09:30", periods=len(closes), freq="15min", tz="America/New_York")
    return pd.DataFrame({"open": closes, "high": closes + spread, "low": closes - spread, "close": closes}, index=idx)


def _range_then(tail):
    # 12 bars oscillating inside 99.5..101.5, so the range midpoint goes flat.
    return _bars([100, 101] * 6 + list(tail))


def test_atr_matches_wilder_rma():
    df = _bars([10, 11, 12, 11, 13], spread=1.0)
    atr = rma_atr(df.high.values, df.low.values, df.close.values, 3)
    assert np.isnan(atr[:2]).all()
    assert atr[2] == 2.0  # SMA seed of true ranges (all 2.0)
    assert np.isclose(atr[4], (2 * atr[3] + 3.0) / 3)  # gap bar: TR = high 14 - prev close 11


def test_bullish_breakout_trails_and_exits_on_close():
    df = _range_then([103, 105, 107, 109, 104])
    (b,) = find_breakouts(df, PARAMS)
    assert b.direction == "long"
    assert b.signal_idx == 12 and b.zone_top == 101.5 and b.zone_bottom == 99.5
    assert (np.diff(b.trail) >= 0).all()  # long trail only ratchets up
    assert b.exit_signal_idx == 16         # 104 closes below the ratcheted trail


def test_bearish_breakout():
    (b,) = find_breakouts(_range_then([97, 95, 93, 91, 96]), PARAMS)
    assert b.direction == "short" and b.exit_signal_idx == 16
    assert (np.diff(b.trail) <= 0).all()


def test_simulate_fills_next_open_with_costs():
    df = _range_then([103, 105, 107, 109, 104, 104])
    es = INSTRUMENTS["ES"]
    (t,) = simulate(df, es, PARAMS, exit_mode="close", slip_ticks=1, commission=5)
    assert t.entry_price == 105 + 0.25   # next bar open + 1 tick
    assert t.exit_price == 104 - 0.25    # open after the exit-signal bar - 1 tick
    assert t.pnl_dollars == (103.75 - 105.25) * 50 - 5


def test_stop_mode_exits_intrabar_at_trail():
    df = _range_then([103, 105, 107, 109, 104, 104])
    (t,) = simulate(df, INSTRUMENTS["ES"], PARAMS, exit_mode="stop", slip_ticks=0, commission=0)
    assert t.exit_reason == "trail_stop"
    assert t.exit_time == df.index[16]
