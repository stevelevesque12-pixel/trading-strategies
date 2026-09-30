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


def test_session_vwap_resets_daily_and_matches_manual():
    from macd_cross.backtest import session_vwap

    bars = _bars(np.arange(2 * 1440, dtype=float) % 50 + 100, start="2026-08-03 00:00")
    bars["volume"] = np.arange(len(bars)) % 7 + 1.0
    v = session_vwap(bars)
    day1 = bars[(bars.index.date == bars.index[0].date()) & (bars.index.time >= pd.Timestamp("09:30").time())]
    tp = (day1.high + day1.low + day1.close) / 3
    manual = (tp * day1.volume).cumsum() / day1.volume.cumsum()
    assert np.allclose(v[bars.index.get_indexer(day1.index)], manual)
    assert np.isnan(v[bars.index.get_loc(pd.Timestamp("2026-08-04 09:29", tz="America/New_York"))])
    first_day2 = bars.index.get_loc(pd.Timestamp("2026-08-04 09:30", tz="America/New_York"))
    assert np.isclose(v[first_day2], (bars.high + bars.low + bars.close).iloc[first_day2] / 3)


def test_vwap_filter_entries_on_correct_side():
    from macd_cross.backtest import session_vwap

    rng = np.random.default_rng(3)
    bars = _bars(5000 + rng.standard_normal(3 * 1440).cumsum(), start="2026-08-03 00:00")
    v = session_vwap(bars)
    trades = run_backtest(bars, pd.Timedelta("1min"), INSTRUMENTS["MES"], rules=SessionRules(mode="rth"), vwap=True)
    assert trades
    for t in trades:
        i = bars.index.get_loc(t.entry_time) - 1
        c = bars["close"].iloc[i]
        assert (c > v[i]) if t.direction == 1 else (c < v[i])


def test_zero_cross_matches_ema_cross():
    from macd_cross.strategy import zero_cross_signals

    rng = np.random.default_rng(4)
    close = pd.Series(100 + rng.standard_normal(3000).cumsum())
    sig = zero_cross_signals(close)
    diff = close.ewm(span=12, adjust=False).mean() - close.ewm(span=26, adjust=False).mean()
    up = (diff > 0) & (diff.shift(1) <= 0)
    dn = (diff < 0) & (diff.shift(1) >= 0)
    assert (sig.iloc[35:][up.iloc[35:]] == 1).all() and (sig.iloc[35:][dn.iloc[35:]] == -1).all()
    assert (sig != 0).sum() == (up | dn).iloc[35:].sum()


def test_zero_cross_hist_requires_both_conditions():
    from macd_cross.strategy import zero_cross_signals

    rng = np.random.default_rng(5)
    close = pd.Series(100 + rng.standard_normal(3000).cumsum())
    m = macd(close)
    sig = zero_cross_signals(close, confirm_hist=True)
    assert (sig != 0).sum() > 5
    assert ((m["macd"] > 0) & (m["hist"] > 0))[sig == 1].all()
    assert ((m["macd"] < 0) & (m["hist"] < 0))[sig == -1].all()


def _div_setup(seed=6, n=4000):
    rng = np.random.default_rng(seed)
    close = 5000 + rng.standard_normal(n).cumsum()
    bars = _bars(close, start="2026-08-03 00:00")
    bars["high"] = np.maximum(bars["open"], bars["close"]) + 0.5
    bars["low"] = np.minimum(bars["open"], bars["close"]) - 0.5
    return bars


def test_divergence_definition_and_no_lookahead():
    from macd_cross.divergence import DivergenceParams, find_divergences

    bars = _div_setup()
    div = DivergenceParams()
    sig, stop = find_divergences(bars, div=div)
    assert (sig == 1).sum() > 3 and (sig == -1).sum() > 3
    m = macd(bars["close"])["macd"].to_numpy()
    low, high = bars["low"].to_numpy(), bars["high"].to_numpy()
    for c in np.flatnonzero(sig):
        j = c - div.pivot_k  # swing bar
        window = slice(j - div.pivot_k, j + div.pivot_k + 1)
        if sig[c] == 1:
            assert low[j] == low[window].min() and stop[c] == low[j]
        else:
            assert high[j] == high[window].max() and stop[c] == high[j]
    # Truncating the future must not change past signals.
    cut = 3000
    sig_cut, _ = find_divergences(bars.iloc[:cut], div=div)
    assert (sig_cut[: cut - div.pivot_k] == sig[: cut - div.pivot_k]).all()


def test_divergence_backtest_respects_stops_and_session():
    from macd_cross.divergence import DivergenceParams, run_divergence_backtest

    bars = _div_setup(seed=7, n=5 * 1440)
    for confirm in (False, True):
        trades = run_divergence_backtest(bars, pd.Timedelta("1min"), INSTRUMENTS["MES"],
                                         div=DivergenceParams(confirm=confirm))
        assert trades
        for t in trades:
            assert t.entry_time.date() == t.exit_time.date()
            assert pd.Timestamp("09:30").time() <= t.entry_time.time() < pd.Timestamp("15:55").time()
            assert t.exit_reason in {"stop", "signal_exit", "session_flatten", "end_of_data"}
