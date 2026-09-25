import pandas as pd
import pytest

from tr_breakout.strategy import Config, run_backtest, session_true_ranges

TZ = "America/New_York"


def frame(rows):
    """rows: (timestamp str, o, h, l, c)"""
    idx = pd.DatetimeIndex([pd.Timestamp(r[0], tz=TZ) for r in rows])
    return pd.DataFrame([r[1:] for r in rows], index=idx, columns=["open", "high", "low", "close"]).assign(volume=1.0)


# Day 1 sessions set up TR1 for day 2 (the traded day).
# Day 0 close 1000; day 1 RTH H=1040 L=1000 C=1020 -> TR1 = 40 -> k*TR1 = 10.
PRIOR = [
    ("2026-01-05 15:45", 1000, 1000, 1000, 1000),
    ("2026-01-06 09:30", 1000, 1040, 1000, 1020),
    ("2026-01-06 15:45", 1020, 1020, 1020, 1020),
]
# Day 2: open 1000 -> long level 1010, short level 990, 1R = 10 pts.
CFG = Config(start_equity=100_000, risk_pct=0.01)  # $1000 risk / (10 pts * $2) = 50 MNQ


def day2(*bars):
    return frame(PRIOR + [("2026-01-07 " + b[0],) + tuple(b[1:]) for b in bars])


def test_true_range_uses_prior_close():
    tr = session_true_ranges(frame(PRIOR), CFG)
    assert tr.iloc[-1] == 40


def test_long_stop_out_is_minus_one_r_plus_costs():
    df = day2(
        ("09:30", 1000, 1011, 999, 1011),   # up bar: O->L->H->C, fills long at 1010
        ("09:45", 1011, 1011, 999, 999),    # stop 1000 hit
        ("15:45", 999, 999, 999, 999),
    )
    (t,) = run_backtest(df, CFG)
    assert (t.side, t.contracts, t.stop_price) == ("long", 50, 1000)
    assert t.entry_price == 1010.25 and t.exit_price == 999.75  # 1 tick slippage each side
    assert t.exit_reason == "stop"
    assert t.pnl == pytest.approx(-10.5 * 2 * 50 - 0.95 * 50)


def test_breakeven_applies_from_next_bar_only():
    df = day2(
        ("09:30", 1000, 1030, 1000, 1029),  # entry 1010, reaches +20 = 2R this bar
        ("09:45", 1029, 1029, 1005, 1005),  # BE stop at 1010 now active -> out at entry
        ("15:45", 1005, 1005, 1005, 1005),
    )
    trades = run_backtest(df, CFG)
    assert trades[0].exit_reason == "breakeven" and trades[0].exit_price == 1009.75


def test_no_breakeven_below_two_r():
    df = day2(
        ("09:30", 1000, 1029.75, 1000, 1029),  # just short of +2R
        ("09:45", 1029, 1029, 1005, 1005),
        ("15:45", 1005, 1005, 1005, 1005),
    )
    (t,) = run_backtest(df, CFG)
    assert t.exit_reason == "time_exit"


def test_reentry_after_stop_and_daily_cap():
    # Whipsaw across the long level; each round trip stops out at 1000.
    bars = [("09:30", 1000, 1000, 1000, 1000)]
    minute = 45
    hour = 9
    for _ in range(5):
        bars.append((f"{hour:02d}:{minute:02d}", 1000, 1011, 1000, 1011))  # enter long
        minute += 15
        hour, minute = hour + minute // 60, minute % 60
        bars.append((f"{hour:02d}:{minute:02d}", 1011, 1011, 1000, 1000))  # stopped
        minute += 15
        hour, minute = hour + minute // 60, minute % 60
    bars.append(("15:45", 1000, 1000, 1000, 1000))
    trades = run_backtest(day2(*bars), CFG)
    assert len(trades) == 3
    assert all(t.side == "long" and t.exit_reason == "stop" for t in trades)


def test_short_side_and_no_entries_after_1300():
    df = day2(
        ("09:30", 1000, 1000, 1000, 1000),
        ("13:00", 1000, 1000, 980, 980),    # would trigger short, but window closed
        ("15:45", 980, 980, 980, 980),
    )
    assert run_backtest(df, CFG) == []

    df = day2(
        ("09:30", 1000, 1000, 985, 985),    # short at 990
        ("15:45", 985, 985, 985, 985),
    )
    (t,) = run_backtest(df, CFG)
    assert t.side == "short" and t.stop_price == 1000 and t.exit_reason == "time_exit"


def test_time_exit_at_1555_on_5m_bars():
    df = day2(
        ("09:30", 1000, 1015, 1000, 1015),
        ("15:50", 1015, 1016, 1014, 1016),
        ("15:55", 1017, 1018, 1017, 1018),
    )
    (t,) = run_backtest(df, CFG)
    assert t.exit_reason == "time_exit" and t.exit_price == 1016.75
    assert t.exit_time.strftime("%H:%M") == "15:55"


def test_sizing_is_fixed_by_default_and_compounds_when_asked():
    df = day2(
        ("09:30", 1000, 1011, 999, 1011),
        ("09:45", 1011, 1011, 999, 999),     # loss -> equity < 100k
        ("10:00", 999, 1011, 999, 1011),     # re-entry: smaller size
        ("15:45", 1011, 1011, 1011, 1011),
    )
    t1, t2 = run_backtest(df, CFG)
    assert t1.contracts == t2.contracts == 50  # $1000 of $100k, regardless of the loss

    t1, t2 = run_backtest(df, Config(compound=True))
    assert t2.contracts == int(0.01 * t1.equity_after / 20)
    assert t2.contracts < t1.contracts


def test_max_contracts_caps_size():
    df = day2(("09:30", 1000, 1011, 999, 1011), ("15:45", 1011, 1011, 1011, 1011))
    (t,) = run_backtest(df, Config(max_contracts=5))
    assert t.contracts == 5


def test_prop_eval_pass_and_fail():
    from tr_breakout.prop_sim import PropRules, run_eval

    days = [1, 2, 3]
    rules = PropRules()
    assert run_eval(days, {1: [1500], 2: [1600]}, 0, rules) == ("pass", 2)
    # EOD trail: +1500 lifts the floor to 49,500 -> a -1600 day survives
    # (49,900), a -2100 day breaches it.
    assert run_eval(days, {1: [1500], 2: [-1600]}, 0, rules)[0] == "open"
    assert run_eval(days, {1: [1500], 2: [-2100]}, 0, rules) == ("fail", 2)
