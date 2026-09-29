import numpy as np
import pandas as pd

from mesa_band.strategy import INSTRUMENTS, Params, run


def _synthetic(days=40, seed=0):
    rng = np.random.default_rng(seed)
    idx = []
    for d in pd.bdate_range("2024-01-02", periods=days):
        # 18:00 ET previous evening -> 17:00 ET, 15m bars
        start = pd.Timestamp(d - pd.Timedelta(days=1)).tz_localize("America/New_York") + pd.Timedelta(hours=18)
        idx.extend(pd.date_range(start, periods=92, freq="15min"))
    idx = pd.DatetimeIndex(idx).tz_convert("UTC")
    steps = rng.normal(0, 2.0, len(idx)) + np.sin(np.arange(len(idx)) / 40.0) * 1.5
    close = 4000 + np.cumsum(steps)
    open_ = np.r_[close[0], close[:-1]]
    high = np.maximum(open_, close) + rng.uniform(0, 2, len(idx))
    low = np.minimum(open_, close) - rng.uniform(0, 2, len(idx))
    return pd.DataFrame({"open": open_, "high": high, "low": low, "close": close}, index=idx)


def test_trades_respect_session_and_risk():
    df = _synthetic()
    p = Params()
    res = run(df, INSTRUMENTS["MES"], p, mode="continuous")
    assert res.trades, "expected some trades on trending synthetic data"
    for t in res.trades:
        ny_in = t.entry_time.tz_convert("America/New_York")
        ny_out = t.exit_time.tz_convert("America/New_York")
        assert ny_in.date() == ny_out.date(), "no position may be held overnight"
        assert ny_out.hour * 60 + ny_out.minute <= 15 * 60 + 55
        assert 1 <= t.qty <= p.max_contracts
        # sized at the signal close; allow one bar of slippage vs. the next-open fill
        assert t.risk_usd <= p.risk_per_trade * 1.5
    daily = res.daily_pnl
    assert daily.min() > -(p.daily_loss_limit + p.risk_per_trade * 1.5)


def test_evaluation_ends_with_a_result():
    df = _synthetic(days=60, seed=1)
    p = Params(profit_target=500, max_drawdown=500, dd_buffer=0)
    a = run(df, INSTRUMENTS["MES"], p, mode="evaluation", start=200, max_days=30).attempts[0]
    assert a.result in ("PASSED", "FAILED", "TIMEOUT")
    if a.result == "PASSED":
        assert a.pnl >= 500 - p.risk_per_trade
