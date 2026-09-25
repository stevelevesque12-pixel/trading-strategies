from datetime import datetime, timedelta

import pandas as pd
import pytest

from failed2s.bars import Bar
from failed2s.instruments import Instrument
from turtle_trading.backtest import load_daily, run_portfolio
from turtle_trading.strategy import Account, TurtleConfig, TurtleSystem

INST = Instrument("TEST", 0.01, 10.0)
D0 = datetime(2020, 1, 1)


class Feed:
    """Feeds sequential daily bars into a TurtleSystem."""

    def __init__(self, system: TurtleSystem):
        self.system = system
        self.i = 0

    def bar(self, o, h, l, c):
        b = Bar(D0 + timedelta(days=self.i), o, h, l, c)
        self.i += 1
        self.system.on_bar(b)
        return b

    def flat(self, n, mid=100.0):
        # Range 2 around mid, closes at mid -> true range 2, so N settles at 2.
        for _ in range(n):
            self.bar(mid, mid + 1, mid - 1, mid)


def make(system=2, **kw):
    cfg = TurtleConfig(system=system, **kw)
    s = TurtleSystem(INST, cfg, Account(100_000.0))
    return s, Feed(s)


def test_n_is_seeded_with_mean_then_wilder_smoothed():
    s, f = make()
    f.flat(19)
    assert s.n is None
    f.flat(1)
    assert s.n == pytest.approx(2.0)
    f.bar(100, 104, 98, 100)  # TR = 6
    assert s.n == pytest.approx((19 * 2.0 + 6) / 20)


def test_no_trading_until_longest_channel_is_filled():
    s, f = make()
    f.flat(54)
    f.bar(100, 120, 100, 119)  # would be a breakout, but only 54 bars of history
    assert s.position is None and not s.trades


def test_system2_breakout_entry_sizing_and_stop():
    s, f = make(system=2)
    f.flat(60)
    f.bar(100.5, 101.5, 100.2, 101.4)  # breaks the 55-day high of 101
    pos = s.position
    assert pos is not None and pos.direction == "long"
    assert pos.fills == [101.0]
    # 1% of 100k = $1000 per N; N=2 points * $10/pt = $20 -> 50 contracts per unit
    assert pos.unit_contracts == 50
    assert pos.stop == pytest.approx(101.0 - 2 * 2.0)


def test_gap_through_breakout_fills_at_open():
    s, f = make(system=2)
    f.flat(60)
    f.bar(101.8, 101.9, 101.6, 101.7)
    assert s.position.fills == [101.8]


def test_pyramiding_every_half_n_moves_all_stops():
    s, f = make(system=2)
    f.flat(60)
    f.bar(100.5, 104.5, 100.5, 104)  # entry 101, adds at 102, 103, 104
    pos = s.position
    assert pos.fills == [101.0, 102.0, 103.0, 104.0]
    assert pos.stop == pytest.approx(104.0 - 4.0)
    f.bar(104, 108, 104, 107)  # already at max units
    assert len(s.position.fills) == 4


def test_system2_exits_on_20_day_opposite_breakout():
    s, f = make(system=2)
    f.flat(60)
    f.bar(100.5, 101.5, 100.2, 101.4)  # long 1 unit @101, stop 97
    f.bar(101, 101.5, 98.5, 99)  # breaks 20-day low of 99, above the stop
    assert s.position is None
    t = s.trades[0]
    assert t.exit_reason == "exit_20"
    assert t.exit_price == 99.0
    assert t.pnl_dollars == pytest.approx((99 - 101) * 50 * 10)
    assert s.account.equity == pytest.approx(100_000 + t.pnl_dollars)


def test_gap_through_exit_fills_at_open():
    s, f = make(system=2)
    f.flat(60)
    f.bar(100.5, 101.5, 100.2, 101.4)  # long @101, stop 97, 20-day low 99
    f.bar(95, 95.5, 94, 94.5)  # gaps below both
    t = s.trades[0]
    assert t.exit_price == 95.0
    assert t.exit_reason == "exit_20"  # 99 was the tighter of the two levels


def test_stop_tighter_than_channel_after_pyramiding():
    s, f = make(system=2)
    f.flat(60)
    f.bar(100.5, 104.5, 100.5, 104)  # 4 units, stop raised to 100
    f.bar(103, 103.5, 99.8, 100.1)  # through the stop, not the 20-day low (99)
    t = s.trades[0]
    assert t.exit_reason == "stop"
    assert t.exit_price == pytest.approx(100.0)
    assert t.units == 4 and t.contracts == 200
    # Units lost 1, 2, 3, 4 points: 10 points * 50 contracts * $10
    assert t.pnl_dollars == pytest.approx(-10 * 50 * 10)
    assert t.r_multiple == pytest.approx(-2.5)


def test_short_side_mirrors():
    s, f = make(system=2)
    f.flat(60)
    f.bar(99.5, 99.8, 98.5, 98.6)  # breaks the 55-day low of 99
    pos = s.position
    assert pos.direction == "short" and pos.fills == [99.0]
    assert pos.stop == pytest.approx(103.0)


def test_same_bar_exit_after_entry_is_assumed_hit():
    s, f = make(system=2)
    f.flat(20, mid=90)  # puts the 55-day low far away (89)...
    f.flat(40, mid=100)  # ...while the 20-day exit low is 99
    f.bar(100.5, 101.5, 96.5, 97)  # breaks out at 101, then trades down through 99
    assert s.position is None
    t = s.trades[0]
    assert t.entry_reason == "breakout_55"
    assert t.exit_reason == "exit_20" and t.exit_price == 99.0


def test_bar_breaking_both_channels_is_ignored():
    s, f = make(system=2)
    f.flat(60)
    f.bar(100, 102, 98, 100)
    assert s.position is None and not s.trades


def test_system1_skips_breakout_after_winner_then_takes_failsafe():
    s, f = make(system=1)
    f.flat(60)
    # Breakout and trend up: 1 unit @101, pyramid, ride to ~115.
    f.bar(100.5, 101.5, 100.2, 101.4)
    for k in range(14):
        m = 102 + k
        f.bar(m - 0.5, m + 0.5, m - 0.8, m + 0.4)
    # Collapse through the 10-day low -> profitable exit (a winner).
    f.bar(110, 110, 105, 105)
    assert s.position is None
    assert s.trades[-1].exit_price > 101
    n_trades = len(s.trades)

    # Consolidate lower, then make a 20-day breakout that is still well
    # below the 55-day high (~115.5): skipped because the last one won.
    f.flat(25, mid=104)
    f.bar(104.5, 106, 104.2, 105.8)
    assert s.position is None
    assert s.skipped_breakouts == 1

    # Keep going; the skipped breakout's shadow trade is still running, and
    # once price clears the 55-day high the failsafe enters anyway.
    for k in range(12):
        m = 106 + k
        f.bar(m - 0.2, m + 0.6, m - 0.4, m + 0.5)
        if s.position is not None:
            break
    assert s.position is not None
    assert s.position.entry_reason == "failsafe_55"
    assert len(s.trades) == n_trades


def test_system1_takes_breakout_after_loser():
    s, f = make(system=1)
    f.flat(60)
    f.bar(100.5, 101.5, 100.2, 101.4)  # long @101
    f.bar(101, 101.2, 99, 99.5)  # 10-day exit at 99 -> loser (without a short breakout)
    assert s.trades[-1].pnl_dollars < 0
    f.flat(25)
    f.bar(100.5, 101.5, 100.2, 101.4)  # next 20-day breakout
    assert s.position is not None
    assert s.position.entry_reason == "breakout_20"


def test_unit_too_small_is_skipped():
    s = TurtleSystem(Instrument("BIG", 0.25, 100_000.0), TurtleConfig(system=2), Account(1_000.0))
    f = Feed(s)
    f.flat(60)
    f.bar(100.5, 101.5, 100.2, 101.4)
    assert s.position is None
    assert s.skipped_too_small == 1


def test_costs_reduce_pnl():
    s, f = make(system=2, slippage_ticks=1, commission_per_contract=2.0)
    f.flat(60)
    f.bar(100.5, 101.5, 100.2, 101.4)
    assert s.position.fills == [pytest.approx(101.01)]
    f.bar(101, 101.5, 98.5, 99)
    t = s.trades[0]
    assert t.exit_price == pytest.approx(98.99)
    assert t.pnl_dollars == pytest.approx((98.99 - 101.01) * 50 * 10 - 50 * 2.0)


def test_load_daily_groups_by_cme_session(tmp_path):
    idx = pd.date_range("2024-03-08 09:00", "2024-03-11 16:00", freq="1h", tz="America/New_York")
    idx = idx[(idx.dayofweek < 5) | ((idx.dayofweek == 6) & (idx.hour >= 18))]
    idx = idx[~((idx.dayofweek == 4) & (idx.hour >= 17))]
    df = pd.DataFrame({"timestamp": idx, "open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5})
    path = tmp_path / "bars.csv"
    df.to_csv(path, index=False)

    daily = load_daily(str(path))
    # Friday 09:00-16:00, then Sunday 18:00 through Monday 16:00 (across the
    # DST change on 2024-03-10) is all Monday's session.
    assert [d.strftime("%a %m-%d") for d in daily.index] == ["Fri 03-08", "Mon 03-11"]


def test_portfolio_run_books_pnl_to_shared_account(monkeypatch):
    def frame(start_mid):
        rows = [(start_mid, start_mid + 1, start_mid - 1, start_mid)] * 60
        rows.append((start_mid + 0.5, start_mid + 1.5, start_mid + 0.2, start_mid + 1.4))
        idx = pd.date_range("2020-01-01", periods=len(rows), freq="D", tz="America/New_York")
        return pd.DataFrame(rows, index=idx, columns=["open", "high", "low", "close"]).assign(volume=0.0)

    from turtle_trading import backtest

    monkeypatch.setitem(backtest.INSTRUMENTS, "TEST", INST)
    account, systems = run_portfolio([("TEST", frame(100.0))], TurtleConfig(system=2), 50_000.0)
    # Entered on the last bar and closed at end of data.
    assert len(systems["TEST"].trades) == 1
    assert account.equity == pytest.approx(50_000 + systems["TEST"].trades[0].pnl_dollars)
