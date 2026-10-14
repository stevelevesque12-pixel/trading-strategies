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



def test_reverse_limit_closes_open_trade_at_the_flip_fill(monkeypatch):
    # Hand-built signal bars: a long setup, then a short setup while the long is still open.
    import ma_limit.strategy as st

    idx = pd.date_range("2026-01-05 09:00", periods=8, freq="30min", tz="America/New_York")
    fine = pd.DataFrame(
        {
            "open": [100, 100.2, 100.2, 100.5, 100.8, 100.8, 100.9, 100.8],
            "high": [100.1, 100.0, 100.4, 100.6, 101.0, 101.2, 101.0, 100.9],
            "low": [99.5, 99.9, 100.0, 100.3, 100.6, 100.7, 100.7, 100.6],
            "close": [99.6, 100.2, 100.3, 100.5, 101.0, 100.9, 100.8, 100.7],
        },
        index=idx,
    )
    ind = pd.DataFrame(index=idx, data={"fast": 0.0, "slow": 0.0, "atr": 10.0})
    ind.loc[idx[0], ["fast", "slow"]] = [99.0, 100.0]  # long setup: limit 100.0
    ind.loc[idx[4], ["fast", "slow"]] = [101.0, 100.9]  # short setup: limit 100.9
    sig = pd.Series(0, index=idx)
    sig[idx[0]], sig[idx[4]] = 1, -1
    monkeypatch.setattr(st, "indicators", lambda b, p: b.join(ind))
    monkeypatch.setattr(st, "signals", lambda b, p: sig)

    p = Params(commission_rt=0, stop_slip_ticks=0, reverse="limit", cancel_on_flip=False, live_ma=False)
    trades = backtest(fine, "30min", p)
    assert trades[0].direction == 1 and trades[0].entry == 100.0
    assert trades[0].reason == "reverse" and trades[0].exit == 100.9
    assert trades[0].exit_time == idx[5]

    # Without side swapping the short setup is ignored while the long is open.
    plain = backtest(fine, "30min", Params(commission_rt=0, stop_slip_ticks=0, cancel_on_flip=False, live_ma=False))
    assert all(t.reason != "reverse" for t in plain)


def test_replays_hand_marked_mcl_trades_jan_2025():
    # Trades marked on TradingView (MCL1! 30m, Jan 2-9 2025): direction, fill time, entry, stop distance.
    from pathlib import Path

    import pytest

    path = Path(__file__).parent.parent / "sample_data/real_multi_instrument/tv_mcl1_30m_2025-01-02_2026-10.parquet"
    pytest.importorskip("pyarrow")
    bars = pd.read_parquet(path)[["open", "high", "low", "close"]]
    # Every setup is taken, even a close 1 cent beyond the MA (Wed 8 22:00); the two Jan 9-10 shorts
    # are confirmed losers.
    trades = backtest(bars[:"2025-01-13"], "30min", Params())
    got = [(t.direction, t.entry_time.strftime("%d %H:%M"), round(t.entry, 2), round(abs(t.entry - t.stop), 2), t.reason) for t in trades]
    assert got[:8] == [
        (-1, "02 18:00", 73.21, 0.46, "target"),
        (-1, "05 20:00", 74.02, 0.42, "target"),
        (-1, "06 11:30", 74.13, 0.59, "target"),
        (1, "06 23:00", 73.42, 0.36, "target"),
        (-1, "08 07:30", 74.79, 0.49, "target"),
        (1, "08 22:30", 73.23, 0.39, "target"),
        (-1, "09 21:00", 74.17, 0.35, "stop"),
        (-1, "10 11:30", 75.99, 0.98, "stop"),
    ]
    assert trades[0].exit_time.strftime("%d %H:%M") == "03 04:00"
    assert trades[1].exit_time.strftime("%d %H:%M") == "06 00:00"


def test_hours_window_wraps_midnight():
    closes = list(np.linspace(60, 66, 40)) + [63.0]
    b = indicators(_bars(closes, "30min"), Params())
    last = b.index[-1]  # 2026-01-05 05:00 ET
    assert signals(b, Params(hours=("04:00", "06:00"))).iloc[-1] == -1
    assert signals(b, Params(hours=("09:00", "14:30"))).iloc[-1] == 0
    assert signals(b, Params(hours=("18:00", "05:30"))).iloc[-1] == -1
    assert last.hour == 5
