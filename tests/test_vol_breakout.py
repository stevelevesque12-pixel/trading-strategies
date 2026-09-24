from datetime import datetime, timedelta

import pandas as pd
import pytest

from vol_breakout.strategy import VolBreakoutConfig, run_backtest, simulate_day, true_ranges

TZ = "America/New_York"
T0 = pd.Timestamp(datetime(2026, 1, 6, 9, 30), tz=TZ)

# TR1 = 100, open = 1000 -> long level 1025, short level 975, 1R = 25 pts.
# MNQ: $2/pt, 0.25 tick. $100k * 1% / (25 * $2) = 20 contracts.
TR1 = 100.0
TICK, PV = 0.25, 2.0
CFG = VolBreakoutConfig()


def bars(*ohlc, start=T0, step=15):
    return [(start + timedelta(minutes=step * i), *b) for i, b in enumerate(ohlc)]


def run(bs, equity=100_000.0, cfg=CFG):
    return simulate_day(bs, TR1, equity, TICK, PV, cfg)


def test_long_breakout_held_to_eod_with_costs():
    trades, equity = run(bars((1000, 1030, 995, 1028), (1028, 1040, 1020, 1035)))
    assert len(trades) == 1
    t = trades[0]
    assert (t.direction, t.exit_reason, t.contracts) == ("long", "eod", 20)
    assert t.entry_price == 1025.25  # level + 1 tick
    assert t.exit_price == 1034.75  # close - 1 tick
    assert t.initial_stop == 1000
    assert t.pnl_dollars == pytest.approx((1034.75 - 1025.25) * PV * 20 - 0.95 * 20)
    assert equity == pytest.approx(100_000 + t.pnl_dollars)


def test_short_stop_is_the_open():
    trades, _ = run(bars((1000, 1005, 970, 972), (972, 1002, 971, 1001)))
    assert [(t.direction, t.exit_reason) for t in trades] == [("short", "stop")]
    assert trades[0].exit_price == 1000.25  # stop + 1 tick against us
    assert trades[0].r_multiple == pytest.approx(-1.0, abs=0.05)


def test_breakeven_applies_only_from_next_bar():
    trades, _ = run(
        bars(
            (1000, 1030, 995, 1028),  # enter long 1025
            (1070, 1080, 1024, 1030),  # hits +2R (1075) then dips below entry: same bar, stop still 1000
            (1030, 1032, 1020, 1022),  # now stop is at entry -> break-even exit
        )
    )
    assert len(trades) == 1
    assert trades[0].exit_reason == "breakeven"
    assert trades[0].exit_price == 1024.75
    assert trades[0].exit_time == T0 + timedelta(minutes=30)


def test_reentry_after_stop_and_max_three_entries():
    trades, _ = run(
        bars(
            (1000, 1030, 995, 1028),  # 1: long @1025
            (1028, 1029, 990, 992),  # long stopped @1000, re-armed below 1025
            (992, 1030, 991, 1029),  # 2: long again @1025
            (1029, 1029, 970, 972),  # long stopped, then 3: short @975
            (972, 1010, 971, 1005),  # short stopped @1000
            (1005, 1030, 1004, 1028),  # would be a 4th entry -- capped
        )
    )
    assert [(t.direction, t.exit_reason) for t in trades] == [
        ("long", "stop"),
        ("long", "stop"),
        ("short", "stop"),
    ]


def test_no_reentry_until_price_crosses_back_through_level():
    trades, _ = run(
        bars(
            (1000, 1030, 995, 1028),  # long @1025
            (1070, 1080, 1060, 1075),  # +2R reached -> BE from next bar
            (1075, 1076, 1025, 1026),  # BE exit exactly at 1025, never below it
            (1026, 1040, 1025.5, 1035),  # back up without trading below 1025: not re-armed
        )
    )
    assert [t.exit_reason for t in trades] == ["breakeven"]


def test_gap_through_level_fills_at_gap_price():
    trades, _ = run(bars((1000, 1010, 995, 1005), (1040, 1045, 1035, 1042)))
    assert trades[0].entry_price == 1040.25


def test_no_new_entries_at_or_after_1300():
    bs = [(T0, 1000, 1005, 995, 1002), (T0.replace(hour=13, minute=0), 1002, 1030, 1001, 1028)]
    trades, _ = run(bs)
    assert trades == []


def test_sizing_skips_trade_when_one_contract_exceeds_risk():
    trades, equity = run(bars((1000, 1030, 995, 1028)), equity=4_000.0)  # 1% = $40 < $50/contract
    assert trades == [] and equity == 4_000.0


def _frame(rows):
    idx = pd.DatetimeIndex([pd.Timestamp(r[0], tz=TZ) for r in rows])
    return pd.DataFrame([r[1:] for r in rows], index=idx, columns=["open", "high", "low", "close", "volume"])


def test_true_range_eth_and_rth():
    df = _frame(
        [
            ("2026-01-05 10:00", 100, 110, 90, 105, 1),
            ("2026-01-05 19:00", 105, 120, 104, 118, 1),  # evening: belongs to the 01-06 ETH session
            ("2026-01-06 10:00", 118, 119, 100, 101, 1),
        ]
    )
    eth = true_ranges(df, VolBreakoutConfig(tr_session="eth"))
    rth = true_ranges(df, VolBreakoutConfig(tr_session="rth"))
    assert eth[pd.Timestamp("2026-01-06").date()] == 20  # 120 - 100
    assert rth[pd.Timestamp("2026-01-06").date()] == 19  # 119 - 100 (prev close 105 inside)


def test_run_backtest_uses_previous_session_tr():
    rows = [("2026-01-05 09:30", 1000, 1050, 950, 1000, 1)]  # 01-05 TR = 100
    rows += [
        ("2026-01-06 09:30", 1000, 1030, 995, 1028, 1),
        ("2026-01-06 09:45", 1028, 1040, 1020, 1035, 1),
    ]
    trades = run_backtest(_frame(rows), TICK, PV, 100_000.0, VolBreakoutConfig(tr_session="rth"))
    assert len(trades) == 1
    assert trades[0].tr1 == 100 and trades[0].level == 1025
