import numpy as np
import pandas as pd

from trend_lab import indicators as ind
from trend_lab.metrics import lucid_eval
from trend_lab.sim import SimConfig, simulate


def _bars(closes, start="2026-08-03 09:00", freq="5min"):
    idx = pd.date_range(start, periods=len(closes), freq=freq, tz="America/New_York")
    c = np.asarray(closes, dtype=float)
    df = pd.DataFrame({"open": c, "high": c + 0.02, "low": c - 0.02, "close": c, "volume": 100.0}, index=idx)
    df["trade_day"] = (df.index + pd.Timedelta(hours=7)).date
    return df


def test_entry_fills_next_open_and_stop_hits():
    closes = [70.0] * 5 + [70.0, 70.10, 70.20, 69.50, 69.40]
    df = _bars(closes)
    long = np.zeros(len(df), bool)
    long[5] = True
    sig = {"long": long, "short": np.zeros(len(df), bool), "stop_dist": np.full(len(df), 0.30)}
    cfg = SimConfig(session="all", eia_filter=False, slippage_ticks=1)
    trades = simulate(df, sig, cfg)
    assert len(trades) == 1
    t = trades[0]
    assert t.entry == 70.10 + 0.01            # next bar's open + 1 tick
    assert t.reason == "stop"
    assert abs(t.stop0 - (70.11 - 0.30)) < 1e-9
    assert t.pnl < 0
    assert t.contracts == int(200 // (0.30 * 100 + 1.24 + 2))


def test_flatten_at_session_cutoff():
    df = _bars([70.0 + 0.01 * i for i in range(60)], start="2026-08-03 12:00")
    long = np.zeros(len(df), bool)
    long[2] = True
    sig = {"long": long, "short": np.zeros(len(df), bool), "stop_dist": np.full(len(df), 0.5)}
    trades = simulate(df, sig, SimConfig(session="ny", eia_filter=False))
    assert trades[0].reason == "flatten"
    assert trades[0].exit_time.hour * 60 + trades[0].exit_time.minute == 14 * 60 + 30


def test_htf_mapping_has_no_lookahead():
    df = _bars(list(range(1, 49)), start="2026-08-03 09:00")
    s = ind.htf(df, "60min", lambda h: h["close"])
    # the 09:00-10:00 hourly bar closes at 10:00 -> first visible on the 09:55 base bar (closes 10:00)
    assert np.isnan(s.iloc[10])
    assert s.iloc[11] == df["close"].iloc[11]
    assert s.iloc[12] == df["close"].iloc[11]


def test_lucid_eval_pass_and_fail():
    good = pd.Series([400.0] * 80)
    r = lucid_eval(good)
    assert r["lucid_pass_pct"] == 100.0 and r["lucid_median_days"] == 8
    bad = pd.Series([-300.0] * 80)
    assert lucid_eval(bad)["lucid_fail_pct"] == 100.0
    # one huge day breaks the 50% consistency rule until more profit accrues
    lumpy = pd.Series([3000.0] + [100.0] * 79)
    assert lucid_eval(lumpy)["lucid_median_days"] > 1


def test_montecarlo_eval_rules():
    from trend_lab.montecarlo import run_eval
    assert run_eval([500.0] * 10) == ("pass", 6)
    assert run_eval([-700.0] * 5) == ("fail", 3)
    # trailing: at a +1500 high-water mark the line is -500; it locks at 0 once the peak reaches +2000
    assert run_eval([1500.0, -1600.0])[0] == "timeout"
    assert run_eval([1500.0, -2000.0])[0] == "fail"
    assert run_eval([2500.0, -2400.0])[0] == "timeout"
    assert run_eval([2500.0, -2500.0])[0] == "fail"
    # consistency: one $3,000 day can't pass alone; needs total >= $6,000 or smaller days
    assert run_eval([3000.0, 0.0, 0.0])[0] == "timeout"
    assert run_eval([3000.0, 0.0], consistency=False) == ("pass", 1)


def test_combined_matches_merged_portfolio():
    """The Pine-parity single-position sim must agree with the merged-engine portfolio."""
    from trend_lab.combined import run
    from trend_lab.cross_market import ENGINE_A, ENGINE_B
    from trend_lab.data import load_bars
    from trend_lab.portfolio import merge
    from trend_lab.strategies import ALL_FAMILIES
    df = load_bars("15min")
    _, live = run(df)
    lists = [simulate(df, ALL_FAMILIES[n].generate(df, p), SimConfig(session=p["session"], risk_usd=300,
                                                                          daily_loss_stop=675))
             for n, p in (ENGINE_A, ENGINE_B)]
    merged = merge(lists, 675, exclusive=True)
    assert abs(len(live) - len(merged)) <= 3
    assert abs(sum(t.pnl for t in live) - sum(t.pnl for t in merged)) < 0.05 * abs(sum(t.pnl for t in merged))
