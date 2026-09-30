import numpy as np
import pandas as pd

from failed2s.instruments import INSTRUMENTS
from macd_cross.backtest import CostModel, SessionRules, run_backtest
from macd_cross.strategy import MACDParams, crossover_signals, macd


def _bars(closes, start="2026-08-03 09:30", freq="1min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="America/New_York")
    c = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame({"open": c.shift(1).fillna(c.iloc[0]), "high": c + 0.25, "low": c - 0.25,
                         "close": c, "volume": 1.0})


def test_macd_matches_manual_ema():
    close = pd.Series(np.linspace(100, 120, 60))
    m = macd(close)
    ema = lambda s, n: s.ewm(span=n, adjust=False).mean()
    expected = ema(close, 12) - ema(close, 26)
    assert np.allclose(m["macd"], expected)
    assert np.allclose(m["signal"], ema(expected, 9))


def test_cross_up_then_down_signals():
    # flat, then up-trend (bullish cross), then down-trend (bearish cross)
    closes = [100.0] * 40 + list(np.linspace(100, 110, 30)) + list(np.linspace(110, 95, 40))
    sig = crossover_signals(pd.Series(closes), MACDParams())
    ups, downs = np.flatnonzero(sig == 1), np.flatnonzero(sig == -1)
    assert len(ups) == 1 and 40 <= ups[0] < 45
    assert len(downs) == 1 and ups[0] < downs[0]


def test_signals_alternate():
    rng = np.random.default_rng(0)
    sig = crossover_signals(pd.Series(100 + rng.standard_normal(2000).cumsum()))
    nz = sig[sig != 0].to_numpy()
    assert len(nz) > 10
    assert (nz[1:] != nz[:-1]).all()


def test_stop_and_reverse_fills_next_open_and_costs():
    closes = [100.0] * 40 + list(np.linspace(100, 110, 30)) + list(np.linspace(110, 95, 40))
    bars = _bars(closes, start="2026-08-03 00:00")
    inst = INSTRUMENTS["MES"]
    sig = crossover_signals(bars["close"])
    up = np.flatnonzero(sig == 1)[0]
    trades = run_backtest(bars, pd.Timedelta("1min"), inst, costs=CostModel(0.5, 1.0))
    first = trades[0]
    assert first.direction == 1
    assert first.entry_time == bars.index[up + 1]
    assert first.entry_price == bars["open"].iloc[up + 1]
    assert first.exit_reason == "reverse"
    assert trades[1].direction == -1 and trades[1].entry_time == first.exit_time
    assert first.cost_dollars == 2 * (0.5 + 1.25)
    assert abs(first.gross_dollars - first.pnl_points * 5.0) < 1e-9


def test_rth_mode_flattens_same_day():
    rng = np.random.default_rng(1)
    bars = _bars(5000 + rng.standard_normal(3 * 1440).cumsum(), start="2026-08-03 00:00")
    trades = run_backtest(bars, pd.Timedelta("1min"), INSTRUMENTS["MES"], rules=SessionRules(mode="rth"))
    assert trades
    for t in trades:
        assert t.entry_time.date() == t.exit_time.date()
        assert pd.Timestamp("09:30").time() <= t.entry_time.time() < pd.Timestamp("15:55").time()
        assert t.exit_time.time() <= pd.Timestamp("15:55").time()


def test_trend_filter_blocks_countertrend_entries():
    rng = np.random.default_rng(2)
    bars = _bars(5000 + rng.standard_normal(3000).cumsum(), start="2026-08-03 00:00")
    inst = INSTRUMENTS["MES"]
    ema = bars["close"].ewm(span=200, adjust=False).mean()
    base = run_backtest(bars, pd.Timedelta("1min"), inst)
    filt = run_backtest(bars, pd.Timedelta("1min"), inst, trend_ema=200)
    assert 0 < len(filt) < len(base)
    for t in filt:
        sig_bar = bars.index.get_loc(t.entry_time) - 1
        c, e = bars["close"].iloc[sig_bar], ema.iloc[sig_bar]
        assert (c > e) if t.direction == 1 else (c < e)
