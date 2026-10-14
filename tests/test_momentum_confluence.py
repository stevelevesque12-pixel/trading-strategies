from datetime import datetime, time, timedelta
from types import SimpleNamespace

import numpy as np
import pandas as pd

from failed2s.instruments import INSTRUMENTS
from momentum_confluence.backtest import fills_to_trades, run_strategy, summarize
from momentum_confluence.indicators import EMA, RSI, Momentum, Slope, UltimateOscillator
from momentum_confluence.strategy import (
    ConfluenceConfig,
    Fill,
    MomentumConfluenceStrategy,
    _last_occurrence,
)

RNG = np.random.default_rng(0)
PRICES = 100 + RNG.normal(0, 1, 400).cumsum()


def test_ema_matches_pandas_after_seed():
    e = EMA(10)
    out = []
    for x in PRICES:
        e.update(x)
        out.append(e.value)
    ref = pd.Series(PRICES).ewm(span=10, adjust=False).mean().to_numpy()
    assert np.allclose(out[200:], ref[200:])
    assert np.isclose(out[9], PRICES[:10].mean())  # SMA seed


def test_rsi_matches_wilder_reference():
    r = RSI(12)
    out = []
    for x in PRICES:
        r.update(x)
        out.append(r.value if r.is_ready else np.nan)
    d = pd.Series(PRICES).diff()
    ag = d.clip(lower=0).ewm(alpha=1 / 12, adjust=False).mean()
    al = (-d.clip(upper=0)).ewm(alpha=1 / 12, adjust=False).mean()
    ref = (100 - 100 / (1 + ag / al)).to_numpy()
    assert np.allclose(out[200:], ref[200:])
    assert not RSI(12).is_ready


def test_momentum_and_slope():
    m = Momentum(3)
    s = Slope(m)
    for x in [1, 2, 4, 7, 11, 16]:
        m.update(x)
        s.update()
    assert m.value == 16 - 4
    assert s.value == (16 - 4) - (11 - 2)
    assert s.previous == (11 - 2) - (7 - 1)


def test_ultimate_oscillator_all_closes_at_high_is_100():
    uo = UltimateOscillator(2, 3, 4)
    for i in range(10):
        uo.update(high=10 + i, low=9 + i, close=10 + i)
    assert uo.is_ready and abs(uo.value - 100.0) < 1e-9


def test_last_occurrence():
    t = datetime(2024, 3, 6, 10, 0)  # Wednesday
    assert _last_occurrence(t, time(23, 0)) == datetime(2024, 3, 5, 23, 0)
    assert _last_occurrence(t, time(16, 58), weekday=4) == datetime(2024, 3, 1, 16, 58)


def _stub(strat, **vals):
    """Replace slopes with fixed (value, previous) pairs."""
    strat.slopes = {k: SimpleNamespace(is_ready=True, value=v[0], previous=v[1]) for k, v in vals.items()}


BULL = dict(ema10=(1, 1), rsi_sma5=(1, 1), mom_ema_3=(1, 1), ou_ema_3=(1, 1), ema=(1, 1))
BEAR = {k: (-1, -1) for k in BULL}
T = datetime(2024, 3, 6, 10, 0)


def test_first_signal_of_day_is_skipped_then_opposite_trades():
    s = MomentumConfluenceStrategy()
    _stub(s, ema500=(1, 1), **BULL)
    s._decide(T, 100.0)
    assert s.position == 0  # first signal skipped
    s._decide(T + timedelta(minutes=1), 100.0)
    assert s.position == 0  # same direction stays blocked
    _stub(s, ema500=(1, 1), **BEAR)  # counter-trend short, confirmed
    s._decide(T + timedelta(minutes=2), 101.0)
    assert s.position == -1
    _stub(s, ema500=(1, 1), **BULL)
    s._decide(T + timedelta(minutes=3), 99.0)
    assert s.position == 1  # now reverses freely


def test_counter_trend_needs_confirmation():
    s = MomentumConfluenceStrategy(ConfluenceConfig(skip_first_signal=False))
    unconfirmed = {k: (-1, 1) for k in BULL}
    _stub(s, ema500=(1, 1), **unconfirmed)
    s._decide(T, 100.0)
    assert s.position == 0
    _stub(s, ema500=(-1, -1), **unconfirmed)  # with the trend: no confirmation needed
    s._decide(T, 100.0)
    assert s.position == -1


def test_no_entries_outside_window_but_after_hours_exit():
    s = MomentumConfluenceStrategy(ConfluenceConfig(skip_first_signal=False))
    _stub(s, ema500=(1, 1), **BULL)
    s._decide(datetime(2024, 3, 10, 10, 0), 100.0)  # Sunday
    s._decide(datetime(2024, 3, 6, 7, 59), 100.0)
    s._decide(datetime(2024, 3, 6, 20, 0), 100.0)
    assert s.position == 0
    s.position = -1
    s._decide(datetime(2024, 3, 6, 20, 0), 100.0)
    assert s.position == 0 and s.fills[-1].reason == "after_hours_exit"


def test_weekly_flatten_and_daily_reset_fire_once():
    s = MomentumConfluenceStrategy()
    s.position = 1
    s.first_trade_long = 2
    s._prev_end, s._prev_close = datetime(2024, 3, 8, 16, 57), 100.0
    s._scheduled_events(datetime(2024, 3, 8, 16, 58), 101.0)
    assert s.position == 0 and s.fills[-1].price == 101.0
    s._prev_end = datetime(2024, 3, 8, 22, 59)
    s._scheduled_events(datetime(2024, 3, 8, 23, 0), 101.0)
    assert s.first_trade_long == 0


def test_fills_to_trades_handles_reversals_and_costs():
    nq = INSTRUMENTS["NQ"]
    fills = [
        Fill(T, 0, 1, 100.0, "long"),
        Fill(T, 1, -1, 110.0, "short"),
        Fill(T, -1, 0, 105.0, "weekly_flatten"),
    ]
    trades = fills_to_trades(fills, nq, commission_per_side=2.0, slippage_ticks=1)
    assert [t.direction for t in trades] == ["long", "short"]
    assert trades[0].pnl_points == 10 and trades[1].pnl_points == 5
    assert trades[0].pnl_dollars == 10 * 20 - 2 * (2.0 + 5.0)


def test_end_to_end_on_synthetic_data_runs():
    idx = pd.date_range("2024-01-02 18:00", periods=6000, freq="1min", tz="America/New_York")
    close = 15000 + RNG.normal(0, 2, len(idx)).cumsum()
    df = pd.DataFrame({"open": close, "high": close + 1, "low": close - 1, "close": close}, index=idx)
    strat = run_strategy(df)
    assert all(s.is_ready for s in strat.slopes.values())
    trades = fills_to_trades(strat.fills, INSTRUMENTS["NQ"])
    assert summarize(trades)["trades"] == len(trades) > 0
