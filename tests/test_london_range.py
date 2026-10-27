from datetime import time

import pandas as pd

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS
from london_range.backtest import run_backtest
from london_range.strategy import LondonRangeConfig, LondonRangeStrategy, session_date

TZ = "America/New_York"


def _frame(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(ts, tz=TZ) for ts, *_ in rows])
    return pd.DataFrame(
        [r[1:] for r in rows], index=idx, columns=["open", "high", "low", "close", "volume"]
    ).astype(float)


def _flat(start, end, price, freq="5min"):
    return [(ts, price, price, price, price, 1) for ts in pd.date_range(start, end, freq=freq, inclusive="left")]


def _config(**kw):
    # short EMAs so a few bars of warmup is enough
    return LondonRangeConfig(ema_fast=3, ema_slow=5, **kw)


def test_session_date_rolls_at_range_start():
    assert str(session_date(pd.Timestamp("2025-03-03 19:55", tz=TZ))) == "2025-03-03"
    assert str(session_date(pd.Timestamp("2025-03-03 20:00", tz=TZ))) == "2025-03-04"
    assert str(session_date(pd.Timestamp("2025-03-04 01:55", tz=TZ))) == "2025-03-04"


def _long_setup_rows():
    rows = _flat("2025-03-03 18:00", "2025-03-03 20:00", 100.0)
    # range 20:00-02:00: high 110, low 90
    rows += [("2025-03-03 20:00", 100, 110, 90, 100, 1)]
    rows += _flat("2025-03-03 20:05", "2025-03-04 02:00", 100.0)
    rows += [("2025-03-04 02:00", 100, 112, 100, 111, 1)]  # breakout close above 110, above EMAs
    return rows


def test_breakout_arms_pullback_order_with_range_stop_and_half_r_target():
    s = LondonRangeStrategy(_config())
    orders = [s.on_5m_bar(Bar(pd.Timestamp(r[0], tz=TZ), *r[1:])) for r in _long_setup_rows()]
    orders = [o for o in orders if o is not None]
    assert len(orders) == 1
    o = orders[0]
    assert o.direction == "long"
    assert o.entry_price == 110 - 0.125 * 20  # 107.5
    assert o.stop_price == 90
    assert o.target_price == 107.5 + 0.5 * 17.5


def test_no_ema_alignment_no_entry():
    rows = _flat("2025-03-03 18:00", "2025-03-03 20:00", 200.0)  # EMAs far above the range
    rows += [("2025-03-03 20:00", 100, 110, 90, 100, 1)]
    rows += [("2025-03-03 20:05", 100, 100, 100, 100, 1)]
    rows += [("2025-03-04 02:00", 100, 112, 100, 111, 1)]
    s = LondonRangeStrategy(LondonRangeConfig(ema_fast=3, ema_slow=50))
    s.ema_slow.count = 50  # treat as warmed up
    assert all(s.on_5m_bar(Bar(pd.Timestamp(r[0], tz=TZ), *r[1:])) is None for r in rows)


def test_first_unaligned_breakout_skips_day_by_default():
    s = LondonRangeStrategy(_config())
    for r in _flat("2025-03-03 18:00", "2025-03-04 02:00", 100.0)[:-1]:
        s.on_5m_bar(Bar(pd.Timestamp(r[0], tz=TZ), *r[1:]))
    s.range_high, s.range_low = 101.0, 99.0
    s.ema_fast.value = s.ema_slow.value = 150.0  # far above -> upside break not aligned
    assert s.on_5m_bar(Bar(pd.Timestamp("2025-03-04 02:00", tz=TZ), 100, 102, 100, 101.5, 1)) is None
    s.ema_fast.value = s.ema_slow.value = 50.0
    assert s.on_5m_bar(Bar(pd.Timestamp("2025-03-04 02:05", tz=TZ), 101.5, 103, 101, 102.5, 1)) is None


def test_backtest_fills_on_pullback_and_hits_target():
    rows = _long_setup_rows()
    rows += [("2025-03-04 02:05", 111, 111, 107, 108, 1)]  # pulls back through 107.5 -> fill
    rows += [("2025-03-04 02:10", 108, 117, 108, 116, 1)]  # target 116.25? not yet
    rows += [("2025-03-04 02:15", 116, 117, 115, 116.5, 1)]  # 117 >= 116.25 -> target
    trades = run_backtest(_frame(rows), INSTRUMENTS["NQ"], _config())
    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "target"
    assert t.entry_price == 107.5
    assert abs(t.r_multiple - 0.5) < 1e-9


def test_backtest_stop_at_opposite_boundary():
    rows = _long_setup_rows()
    rows += [("2025-03-04 02:05", 111, 111, 107, 108, 1)]
    rows += [("2025-03-04 02:10", 108, 108, 89, 89, 1)]
    trades = run_backtest(_frame(rows), INSTRUMENTS["NQ"], _config())
    assert [t.exit_reason for t in trades] == ["stop"]
    assert trades[0].exit_price == 90
    assert trades[0].r_multiple == -1.0


def test_backtest_cancels_unfilled_order_at_cutoff():
    rows = _long_setup_rows()
    rows += _flat("2025-03-04 02:05", "2025-03-04 13:00", 111.0)
    trades = run_backtest(_frame(rows), INSTRUMENTS["NQ"], _config(no_entry_after=time(12, 0)))
    assert trades == []
