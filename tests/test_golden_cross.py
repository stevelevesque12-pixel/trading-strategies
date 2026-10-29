import datetime as dt

import pandas as pd

from golden_cross.strategy import run, target


def _daily(closes):
    idx = pd.Index([dt.date(2020, 1, 1) + dt.timedelta(days=i) for i in range(len(closes))])
    return pd.DataFrame({"close": closes}, index=idx), pd.Series(closes, index=idx)


def test_target_entry_needs_cross_and_close_above_fast():
    up = [float(i) for i in range(300)]
    assert target(up, 0) == 1
    assert target(up[:-1] + [0.0], 0) == 0  # last close below SMA50 -> no entry
    assert target(list(reversed(up)), 0) == 0


def test_exit_only_on_death_cross_and_signal_uses_prior_closes():
    closes = [100.0 + i for i in range(300)] + [399.0 - 3 * i for i in range(200)]
    daily, fills = _daily(closes)
    trades = run(daily, fills, point_value=1.0)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry_date == daily.index[260]  # first eligible day, decided from closes[:260]
    i_exit = list(daily.index).index(t.exit_date)
    sma = lambda xs, n: sum(xs[-n:]) / n
    assert sma(closes[:i_exit], 50) < sma(closes[:i_exit], 200)
    assert sma(closes[:i_exit - 1], 50) >= sma(closes[:i_exit - 1], 200)
