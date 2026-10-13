import numpy as np
import pandas as pd

from risk_parity.strategy import (PropRules, RiskParityConfig, daily_bars,
                                  run_backtest, select_targets, simulate_eval)


def _row(close, vol, atr, sma):
    return pd.Series({"close": close, "vol": vol, "atr": atr, "sma": sma})


def test_sizing_is_dollar_risk_over_stop():
    cfg = RiskParityConfig(risk_per_market_usd=1000, max_total_risk_usd=5000, stop_atr=2.0)
    # MES: 2 * 40pt ATR * $5 = $400/contract -> 2 contracts for $1000
    t = select_targets({"MES": _row(6000, 0.01, 40, 5000)}, cfg, _instr())
    assert t == {"MES": 2}


def test_unaffordable_contract_is_skipped():
    cfg = RiskParityConfig(risk_per_market_usd=250, stop_atr=2.0)
    # MGC: 2 * 100 ATR * $10 = $2000/contract > $250 budget
    assert select_targets({"MGC": _row(4000, 0.01, 100, 3000)}, cfg, _instr()) == {}


def test_trend_filter_and_equity_slot():
    cfg = RiskParityConfig(risk_per_market_usd=1000, max_total_risk_usd=5000)
    rows = {
        "MES": _row(6000, 0.010, 40, 5000),   # uptrend, calmer equity -> kept
        "MNQ": _row(21000, 0.015, 100, 20000),  # uptrend but shares the equity slot
        "MCL": _row(60, 0.020, 1.5, 70),      # below SMA -> filtered
    }
    assert set(select_targets(rows, cfg, _instr())) == {"MES"}


def test_total_risk_cap_fills_calmest_first():
    cfg = RiskParityConfig(risk_per_market_usd=1000, max_total_risk_usd=1000)
    rows = {"MES": _row(6000, 0.010, 40, 5000),   # $400/contract
            "MCL": _row(60, 0.020, 2.0, 50)}      # $400/contract, but budget used up
    assert select_targets(rows, cfg, _instr()) == {"MES": 2}


def test_simulate_eval_outcomes():
    rules = PropRules(profit_target=1000, trailing_dd=500)
    assert simulate_eval(pd.Series([0, 300, 700, 1100.0]), 0, rules) == ("pass", 3)
    # peak 400 then down to -150: 550 off the peak -> fail
    assert simulate_eval(pd.Series([0, 400, 100, -150.0]), 0, rules) == ("fail", 3)


def test_daily_bars_use_cme_trade_date():
    idx = pd.DatetimeIndex(["2026-03-01 18:00", "2026-03-02 16:45", "2026-03-02 18:00"],
                           tz="America/New_York")
    bars = pd.DataFrame({"open": [1, 2, 3], "high": [1, 2, 3], "low": [1, 2, 3],
                         "close": [1, 2, 3.0]}, index=idx)
    d = daily_bars(bars)
    # Sunday 18:00 and Monday 16:45 are both Monday's session
    assert list(d.index.strftime("%Y-%m-%d")) == ["2026-03-02", "2026-03-03"]
    assert d["close"].iloc[0] == 2


def test_backtest_profits_on_steady_uptrend_and_respects_stop():
    days = pd.bdate_range("2026-01-05", periods=60)
    idx = pd.DatetimeIndex([d + pd.Timedelta(hours=h) for d in days for h in range(9, 16)],
                           tz="America/New_York")
    trend = np.arange(len(idx)) * 1.0 + 6000
    wiggle = np.where(np.arange(len(idx)) % 2, 8.0, -8.0)
    close = trend + wiggle
    bars = pd.DataFrame({"open": close - 1, "high": close + 10, "low": close - 10,
                         "close": close}, index=idx)
    cfg = RiskParityConfig(symbols=("MES",), vol_len=5, atr_len=5, trend_len=10,
                           risk_per_market_usd=1000, max_total_risk_usd=1000)
    res = run_backtest({"MES": bars}, cfg)
    assert res.eod_pnl.iloc[-1] > 0
    assert (res.rebalances["MES"] > 0).any()

    # A crash far below any stop fills at the gap open, not at the stop price.
    crash = bars.copy()
    crash.iloc[-20:] -= 2000
    res2 = run_backtest({"MES": crash}, cfg)
    stops = res2.trades[res2.trades["reason"] == "stop"]
    assert len(stops) >= 1
    assert stops["price"].iloc[0] < stops["entry"].iloc[0] - 1000


def _instr():
    from failed2s.instruments import INSTRUMENTS
    return INSTRUMENTS
