from datetime import date, time

import numpy as np
import pandas as pd

from crude_options import black76
from crude_options.expiry import expiries_between, front_expiry, lo_expiry
from crude_options.put30 import Put30Config, run_put30


def test_put_call_parity():
    f, k, t, v = 70.0, 65.0, 30 / 365, 0.35
    c = black76.price(f, k, t, v, "C")
    p = black76.price(f, k, t, v, "P")
    assert abs((c - p) - (f - k)) < 1e-9


def test_strike_for_delta_is_otm_put_near_30_delta():
    f, t, v = 70.0, 30 / 365, 0.35
    k = black76.strike_for_delta(f, t, v, "P", 0.30)
    assert k < f and k % 0.5 == 0
    assert abs(abs(black76.delta(f, k, t, v, "P")) - 0.30) < 0.03


def test_lo_expiry_rule():
    # Nov 2025: 25th is Tue -> CL LTD Thu Nov 20 -> option expiry 3 bdays earlier = Mon Nov 17
    assert lo_expiry(2025, 11) == date(2025, 11, 17)
    # Oct 2026: 25th is Sun -> CL LTD 4 bdays before = Tue Oct 20 -> option Thu Oct 15
    assert lo_expiry(2026, 10) == date(2026, 10, 15)
    exps = expiries_between(date(2025, 11, 18), date(2026, 1, 31))
    assert front_expiry(date(2025, 11, 18), exps) == lo_expiry(2025, 12)


def _bars(prices, start="2025-09-01 00:00"):
    idx = pd.date_range(start, periods=len(prices), freq="1h", tz="America/New_York")
    p = np.asarray(prices, dtype=float)
    return pd.DataFrame({"open": p, "high": p, "low": p, "close": p, "volume": 1.0}, index=idx)


def _wiggle(n, level=70.0, amp=0.6, seed=0):
    rng = np.random.default_rng(seed)
    return level + np.cumsum(rng.normal(0, amp / 5, n))


def test_one_entry_per_cycle_and_take_profit_in_flat_market():
    bars = _bars(_wiggle(24 * 120))
    trades = run_put30(bars, Put30Config(slippage=0.0, commission=0.0))
    assert trades
    assert len({t.expiry for t in trades}) == len(trades)  # never re-enter the same cycle
    for t in trades:
        assert t.entry_time.time() == time(10, 0)
        assert (t.expiry - t.entry_time.date()).days >= 15
        assert t.reason in {"take_profit", "pre_expiry", "end_of_data"}


def test_crash_hits_stop_only_when_enabled():
    quiet = _wiggle(24 * 60)
    crash = np.linspace(quiet[-1], quiet[-1] * 0.6, 24 * 30)
    bars = _bars(np.concatenate([quiet, crash]))
    no_stop = run_put30(bars, Put30Config())
    with_stop = run_put30(bars, Put30Config(stop_mult=3.0))
    assert any(t.reason == "stop" for t in with_stop)
    assert not any(t.reason == "stop" for t in no_stop)
    assert min(t.pnl_dollars for t in with_stop) > min(t.pnl_dollars for t in no_stop)
