import numpy as np
import pandas as pd

from flintlock_orb import flintlock_orb as fo
from flintlock_orb.backtest import trades


def _day(break_close):
    """One RTH morning of 1m bars (UTC, bar-open): flat range, then a solid breakout candle."""
    idx = pd.date_range("2025-03-04 14:00", periods=120, freq="1min", tz="UTC")  # 09:00 ET
    o = np.full(len(idx), 100.0)
    h, l, c = o + 1, o - 1, o.copy()
    i = 40  # 09:40 ET
    o[i], c[i], h[i], l[i] = 100.0, break_close, max(100.0, break_close), min(100.0, break_close)
    o[i + 1:] = c[i + 1:] = break_close
    h[i + 1:], l[i + 1:] = break_close + 1, break_close - 1
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c, "volume": 1}, index=idx)


def test_long_breakout_on_solid_candle():
    df = _day(105.0)
    s = fo.signals(df)
    assert np.flatnonzero(s["long_entry"]).tolist() == [40]
    assert not s["short_entry"].any()
    assert s["stop_price"][40] < 105.0 < s["target_price"][40]


def test_short_breakout_and_trade_replay():
    df = _day(95.0)
    s = fo.signals(df)
    assert np.flatnonzero(s["short_entry"]).tolist() == [40]
    tr = trades(df, s)
    assert len(tr) == 1 and tr.side[0] == "short" and tr.entry[0] == 95.0


def test_no_signal_inside_range():
    s = fo.signals(_day(100.5))
    assert not (s["long_entry"] | s["short_entry"]).any()
