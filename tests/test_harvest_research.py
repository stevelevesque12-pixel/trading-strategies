import numpy as np
import pandas as pd

from larsa.harvest_research import Rule, run_daily_rsi2, run_hourly, summarize


def _bars(closes, freq):
    idx = pd.date_range("2020-01-01", periods=len(closes), freq=freq, tz="America/New_York")
    c = pd.Series(closes, idx, dtype=float)
    o = c.shift(1).fillna(c.iloc[0]) + 0.25   # opens differ from prior close
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + 1, "low": np.minimum(o, c) - 1,
                         "close": c, "volume": 1.0})


def test_hourly_fade_fills_at_next_open():
    closes = [100.0, 101.0] * 12 + [90.0, 95.0, 100.5, 100.5, 100.5]
    h = _bars(closes, "1h")
    reg = pd.Series(2, h.index)  # always SIDEWAYS
    tr = run_hourly(h, reg, Rule("t", lookback=20, z_entry=1.5, z_exit=0.3, z_stop=10))
    assert len(tr) >= 1
    first = tr.iloc[0]
    i = h.index.get_loc(first["entry"])
    assert first["dir"] == 1
    assert closes[i - 1] == 90.0           # signal bar is the one before the fill
    assert first["entry"] == h.index[i]


def test_hourly_rearm_blocks_reentry_after_stop():
    closes = [100.0] * 20 + [97.0, 90.0, 89.0, 88.0, 87.0]
    h = _bars(closes, "1h")
    reg = pd.Series(2, h.index)
    base = run_hourly(h, reg, Rule("a", z_entry=1.5, z_stop=2.0))
    rearm = run_hourly(h, reg, Rule("b", z_entry=1.5, z_stop=2.0, rearm=True))
    assert len(rearm) <= len(base)


def test_daily_rsi2_requires_uptrend_and_exits_above_sma():
    up = list(np.linspace(100, 200, 210))
    closes = up + [195.0, 190.0, 192.0, 199.0, 205.0, 206.0]
    d = _bars(closes, "1D")
    tr = run_daily_rsi2(d, rsi_entry=10, exit_ma=5, trend_ma=200)
    assert len(tr) == 1
    t = tr.iloc[0]
    assert t["dir"] == 1 and t["entry"] > d.index[210]
    # downtrend: same dip never triggers
    d2 = _bars(list(np.linspace(200, 100, 210)) + [95.0, 90.0, 92.0, 99.0], "1D")
    assert run_daily_rsi2(d2).empty


def test_summarize_subtracts_costs():
    tz = "America/New_York"
    tr = pd.DataFrame({"entry": [pd.Timestamp("2020-06-01", tz=tz)],
                       "exit": [pd.Timestamp("2020-06-02", tz=tz)], "dir": [1], "pts": [2.0]})
    s = summarize(tr, "ES", "2020-01-01", "2021-01-01")
    assert s["n"] == 1 and s["net_pts"] == 1.5
