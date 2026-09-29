import datetime as dt

import numpy as np
import pandas as pd

from failed2s.instruments import INSTRUMENTS
from step_range.prop import EvalRules, PropConfig, PropTrade, run_prop, simulate_evals
from step_range.strategy import StepRangeParams

PARAMS = StepRangeParams(length=5, consolidation_bars=3, atr_length=3, atr_mult=1.0)
MNQ = INSTRUMENTS["MNQ"]


def _day(closes, start="2024-01-02 09:30"):
    closes = np.asarray(closes, float)
    idx = pd.date_range(start, periods=len(closes), freq="15min", tz="America/New_York")
    return pd.DataFrame({"open": closes, "high": closes + 0.5, "low": closes - 0.5, "close": closes}, index=idx)


def _cfg(**kw):
    base = dict(risk_per_trade=100.0, max_contracts=5, slip_ticks=0, commission=0)
    return PropConfig(**{**base, **kw})


def test_takes_long_and_sizes_from_risk():
    # Flat range, then a breakout that keeps trending until the flatten time.
    df = _day([100, 101] * 6 + [103, 105, 107, 109, 111, 113, 115, 117, 119, 121, 123, 125, 127, 129])
    stats = {}
    (t,) = run_prop(df, MNQ, PARAMS, _cfg(), stats)
    assert stats == {"signals": 1, "too_wide": 0}
    assert t.entry_time == df.index[13] and t.entry_price == 105
    assert t.contracts == min(5, int(100 // ((105 - t.stop_price) * MNQ.point_value)))
    assert t.exit_reason == "flatten" and t.exit_time.time() == dt.time(15, 45)


def test_ignores_short_breakouts():
    df = _day([100, 101] * 6 + [97, 95, 93, 91, 89])
    assert run_prop(df, MNQ, PARAMS, _cfg()) == []


def test_skips_entries_outside_window_and_too_wide_stops():
    df = _day([100, 101] * 6 + [103, 105, 107, 109, 104, 104])
    assert run_prop(df, MNQ, PARAMS, _cfg(entry_start=dt.time(13, 0))) == []
    stats = {}
    assert run_prop(df, MNQ, PARAMS, _cfg(risk_per_trade=1.0), stats) == []
    assert stats["too_wide"] == 1


def _trade(day, pnl, mae=0.0):
    ts = pd.Timestamp(day, tz="America/New_York")
    return PropTrade(ts, ts, 1, 0, 0, 0, "flatten", pnl, mae, 0)


def test_eval_pass_fail_and_trailing_lock():
    days = pd.Index([dt.date(2024, 1, d) for d in (2, 3, 4, 5)])
    rules = EvalRules(start_balance=50_000, profit_target=3_000, max_drawdown=2_000, min_days=2)

    passing = [_trade("2024-01-02", 3_500)]
    ev = simulate_evals(passing, days, rules)
    assert ev.iloc[0].outcome == "pass" and ev.iloc[0].days == 2  # needs min_days

    # +1,500 raises the line to 49,500; a -2,100 intraday MAE breaches it (it would not vs the original 48,000).
    failing = [_trade("2024-01-02", 1_500), _trade("2024-01-03", 0, mae=-2_100)]
    assert simulate_evals(failing, days, rules).iloc[0].outcome == "fail"

    # The line stops trailing at the starting balance.
    locked = [_trade("2024-01-02", 2_900), _trade("2024-01-03", -2_800)]
    assert simulate_evals(locked, days, rules).iloc[0].outcome == "unresolved"
