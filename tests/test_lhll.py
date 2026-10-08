import pandas as pd
import pytest

from lhll.strategy import backtest_lhll, baseline_avg_points, lhll_signal, summarize


def make(rows, start="2026-01-05 09:30", freq="1min"):
    idx = pd.date_range(start, periods=len(rows), freq=freq)
    return pd.DataFrame(rows, columns=["high", "low", "close"], index=idx)


def test_signal_needs_both_lower_high_and_lower_low():
    df = make([
        (10, 5, 8),
        (9, 4, 6),    # LH + LL -> signal
        (11, 3, 7),   # higher high (outside bar) -> no
        (10, 3, 6),   # equal low -> no
        (9, 2, 4),    # LH + LL -> signal
    ])
    assert lhll_signal(df).tolist() == [False, True, False, False, True]


def test_enters_at_signal_close_and_exits_n_bars_later():
    df = make([(10, 5, 8), (9, 4, 6), (12, 6, 11), (13, 7, 12), (14, 8, 13)])
    t = backtest_lhll(df, hold_bars=2)
    assert len(t) == 1
    assert t[0].entry_price == 6 and t[0].exit_price == 12 and t[0].bars_held == 2
    assert t[0].pnl_points == 6

    short = backtest_lhll(df, hold_bars=2, direction="short", cost_points=0.5)
    assert short[0].pnl_points == -6.5


def test_ignores_signals_while_in_a_trade_but_allows_reentry_on_exit_bar():
    # every bar after the first is LH+LL
    df = make([(10 - i, 5 - i, 7 - i) for i in range(8)])
    t = backtest_lhll(df, hold_bars=3)
    assert [tr.entry_time for tr in t] == [df.index[1], df.index[4]]
    # the signal at bar 7 can't complete a 3-bar hold -> dropped
    assert t[-1].exit_time == df.index[7]


def test_session_flattens_at_session_end_and_no_cross_session_signal():
    df = make([(10, 5, 8), (9, 4, 6), (12, 6, 11), (8, 3, 5), (7, 2, 4)])
    session = pd.Series(["d1", "d1", "d1", "d2", "d2"], index=df.index)
    # bar 3 is LH/LL vs bar 2 but on a new session -> not a signal
    assert lhll_signal(df, session).tolist() == [False, True, False, False, True]
    t = backtest_lhll(df, hold_bars=5, session=session)
    assert len(t) == 1  # bar 4 is the last bar of its session, nothing to hold
    assert t[0].exit_time == df.index[2] and t[0].bars_held == 1


def test_summary_and_baseline():
    df = make([(10, 5, 8), (9, 4, 6), (12, 6, 11), (13, 7, 12), (14, 8, 13)])
    s = summarize(backtest_lhll(df, 1), point_value=20.0)
    assert s["trades"] == 1 and s["total_pts"] == 5 and s["total_$"] == 100
    assert baseline_avg_points(df, 1) == pytest.approx((-2 + 5 + 1 + 1) / 4)
    assert summarize([]) == {"trades": 0}


def test_baseline_uses_same_session_end_exit_as_trades():
    df = make([(10, 5, 8), (9, 4, 6), (12, 6, 11), (8, 3, 5), (7, 2, 4)])
    session = pd.Series(["d1", "d1", "d1", "d2", "d2"], index=df.index)
    # hold 5 -> every bar exits at its session's last close; last bars excluded
    assert baseline_avg_points(df, 5, session=session) == pytest.approx(((11 - 8) + (11 - 6) + (4 - 5)) / 3)


def test_build_bars_anchors_to_rth_open():
    from lhll.run import build_bars
    idx = pd.date_range("2026-01-05 09:30", "2026-01-05 15:59", freq="1min", tz="America/New_York")
    raw = pd.DataFrame({"open": 1.0, "high": 2.0, "low": 0.5, "close": 1.5, "volume": 1.0}, index=idx)
    bars, _ = build_bars(raw, "45min")
    assert [t.strftime("%H:%M") for t in bars.index[:3]] == ["09:30", "10:15", "11:00"]
    assert len(bars) == 9  # 390 / 45 = 8 full bars + a partial 15:30 bar


def test_consecutive_fires_only_on_third_bar_in_a_row():
    # bars 1-4 are LH/LL in a row, bar 5 breaks it, bars 6-8 are LH/LL again
    rows = [(20, 15, 18), (19, 14, 16), (18, 13, 15), (17, 12, 14), (16, 11, 13),
            (18, 12, 17), (17, 11, 15), (16, 10, 14), (15, 9, 13)]
    sig = lhll_signal(make(rows), consecutive=3)
    assert sig[sig].index.tolist() == [make(rows).index[3], make(rows).index[8]]
    t = backtest_lhll(make(rows), hold_bars=1, consecutive=3)
    assert [tr.entry_price for tr in t] == [14]  # bar 8 has no bar after it
