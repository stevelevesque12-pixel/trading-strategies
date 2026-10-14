import numpy as np
import pandas as pd

from ma_limit.strategy import Params, backtest, indicators, resample, signals


def _bars(closes, freq="15min"):
    idx = pd.date_range("2026-01-05 09:00", periods=len(closes), freq=freq, tz="America/New_York")
    c = np.asarray(closes, float)
    return pd.DataFrame({"open": c, "high": c + 0.05, "low": c - 0.05, "close": c}, index=idx)


def test_short_signal_when_close_drops_below_both_mas_in_uptrend():
    closes = list(np.linspace(60, 66, 40)) + [63.0]
    b = indicators(_bars(closes, "30min"), Params())
    s = signals(b, Params())
    assert s.iloc[-1] == -1
    assert (s.iloc[:-1] != -1).all()


def test_short_limit_fills_at_slow_ma_and_hits_target():
    # Uptrend on 30m, one bar closes below both MAs, then price pops back to the 20MA and sells off.
    up = list(np.repeat(np.linspace(60, 66, 40), 2))
    closes = up + [63.0, 63.0] + [63.6, 64.4, 64.9] + list(np.repeat(np.linspace(65, 55, 10), 2))
    fine = _bars(closes)
    p = Params(commission_rt=0, stop_slip_ticks=0)
    sb = indicators(resample(fine, "30min"), p)
    sig_bar = signals(sb, p)
    trades = backtest(fine, "30min", p)
    assert trades, "expected a fill"
    t = trades[0]
    # Consecutive signal bars re-anchor the pending order, so the fill belongs to the latest one.
    assert t.direction == -1 and sig_bar[t.signal_time] == -1
    assert t.entry >= sb.loc[t.signal_time, "slow"] - 1e-9  # sell limit fills at the 20 SMA or better
    assert t.reason == "target" and t.r_multiple > 0.99
