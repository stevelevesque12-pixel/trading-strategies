from dataclasses import replace
from datetime import time

import numpy as np
import pandas as pd

from donchian_breakout.strategy import Costs, Params, add_indicators, generate_signals, htf_trend, simulate, to_rth_bars
from failed2s.instruments import INSTRUMENTS

NQ = INSTRUMENTS["NQ"]
NO_COSTS = Costs(slippage_ticks=0.0, commission_per_contract=0.0)
P = Params(entry_len=3, exit_len=2, atr_len=2, stop_atr=2.0, vwap_filter=False, htf=None,
           entry_start=time(9, 30), entry_end=time(15, 30))


def make_bars(closes, day="2026-01-05", start="09:30", spread=0.5):
    """5m bars with open = prior close, high/low = max/min(open, close) +/- spread."""
    idx = pd.date_range(f"{day} {start}", periods=len(closes), freq="5min", tz="America/New_York")
    c = np.asarray(closes, dtype=float)
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"open": o, "high": np.maximum(o, c) + spread, "low": np.minimum(o, c) - spread,
                         "close": c, "volume": 1.0}, index=idx)


def test_channel_uses_previous_bars_of_same_session_only():
    bars = pd.concat([make_bars([100, 101, 102, 103]), make_bars([200, 201, 202, 203], day="2026-01-06")])
    ind = add_indicators(bars, P)
    # Day 1, 4th bar: previous 3 highs are 100.5, 101.5, 102.5.
    assert ind["entry_high"].iloc[3] == 102.5
    # Day 2 has only 3 bars of history by its last bar -> first 3 bars are NaN.
    assert ind["entry_high"].iloc[4:7].isna().all()
    assert ind["entry_high"].iloc[7] == 202.5


def test_breakout_close_signals_long_and_vwap_filter_can_block_it():
    bars = make_bars([100, 100, 100, 103])
    assert generate_signals(add_indicators(bars, P), P).iloc[3] == 1

    falling = make_bars([110, 108, 96, 95, 94, 93, 97])  # breaks above recent 3-bar high but below VWAP
    pf = replace(P, vwap_filter=True)
    ind = add_indicators(falling, pf)
    assert ind["close"].iloc[6] > ind["entry_high"].iloc[6]
    assert ind["close"].iloc[6] < ind["vwap"].iloc[6]
    assert generate_signals(ind, P).iloc[6] == 1
    assert generate_signals(ind, pf).iloc[6] == 0


def test_long_fills_next_open_and_holds_while_above_exit_channel():
    bars = make_bars([100, 100, 100, 103, 106, 109, 104, 104])
    t = simulate(bars, P, NQ, NO_COSTS)[0]
    assert t.direction == "long"
    assert t.entry_time == bars.index[4] and t.entry_price == bars["open"].iloc[4]
    # Closes of 104 stay above the previous-2-bar low (102.5), so it is held to the end.
    assert t.exit_reason == "end_of_data"


def test_trail_exit_fills_at_next_open():
    bars = make_bars([100, 100, 100, 103, 106, 109, 101, 100, 100])
    t = simulate(bars, P, NQ, NO_COSTS)[0]
    # Bar 6 closes 101 below previous-2 lows (105.5, 102.5) -> exit at bar 7 open.
    assert t.exit_reason == "trail_exit"
    assert t.exit_time == bars.index[7]
    assert t.exit_price == bars["open"].iloc[7]
    assert t.pnl_points == bars["open"].iloc[7] - bars["open"].iloc[4]


def test_stop_hit_intrabar_and_gap_through_fills_at_open():
    bars = make_bars([100, 100, 100, 103, 103])
    bars.iloc[4, bars.columns.get_loc("open")] = 90.0  # gaps far below the stop
    bars.iloc[4, bars.columns.get_loc("low")] = 89.5
    t = simulate(bars, P, NQ, NO_COSTS)[0]
    assert t.exit_reason == "stop"
    assert t.exit_price == 90.0
    assert t.stop_price > 90.0


def test_flatten_and_no_entries_outside_window():
    closes = [100] * 3 + [103 + i for i in range(80)]
    bars = make_bars(closes)
    trades = simulate(bars, P, NQ, NO_COSTS)
    assert trades and trades[-1].exit_reason == "session_flatten"
    assert all(t.entry_time.time() < time(15, 35) for t in trades)


def test_costs_reduce_pnl():
    bars = make_bars([100, 100, 100, 103, 106, 109, 101, 100, 100])
    free = simulate(bars, P, NQ, NO_COSTS)[0]
    paid = simulate(bars, P, NQ, Costs(slippage_ticks=1.0, commission_per_contract=4.5))[0]
    assert paid.pnl_points == free.pnl_points - 2 * NQ.tick_size
    assert paid.pnl_dollars == paid.pnl_points * NQ.point_value - 4.5


def test_htf_trend_is_only_known_after_the_htf_bar_closes():
    # 1m bars, all hours: steady rise, then a sharp drop starting at 10:00.
    idx = pd.date_range("2026-01-05 00:00", "2026-01-05 11:59", freq="1min", tz="America/New_York")
    close = np.where(idx.hour < 10, 100 + np.arange(len(idx)) * 0.01, 50.0)
    df = pd.DataFrame({"open": close, "high": close, "low": close, "close": close, "volume": 1.0}, index=idx)
    trend = htf_trend(df, "60min", ema_len=3)
    assert trend.loc["2026-01-05 10:00"] == 1   # 09:00 bar (closed 10:00) still rising
    assert trend.loc["2026-01-05 11:00"] == -1  # 10:00 bar is the first one to close below its EMA

    bars = to_rth_bars(df, replace(P, timeframe="5min", htf="60min", htf_ema_len=3))
    assert bars.loc["2026-01-05 10:55", "htf_trend"] == 1   # 10:00 HTF bar has not closed yet
    assert bars.loc["2026-01-05 11:00", "htf_trend"] == -1


def test_htf_filter_blocks_counter_trend_breakouts():
    bars = make_bars([100, 100, 100, 103])
    pf = replace(P, htf="60min")
    bars["htf_trend"] = -1
    assert generate_signals(add_indicators(bars, pf), pf).iloc[3] == 0
    bars["htf_trend"] = 1
    assert generate_signals(add_indicators(bars, pf), pf).iloc[3] == 1


def test_fixed_target_fills_at_limit_without_slippage():
    # Signal close 103, ATR(2) of the signal bar's last two TRs = 3.25 -> stop dist 6.5.
    bars = make_bars([100, 100, 100, 103, 104, 110, 120, 120])
    p = replace(P, target_r=1.0, use_trail=False)
    t = simulate(bars, p, NQ, Costs(slippage_ticks=1.0, commission_per_contract=0.0))[0]
    ind = add_indicators(bars, p)
    dist = p.stop_atr * ind["atr"].iloc[3]
    assert t.exit_reason == "target"
    assert t.target_price == 103 + dist
    assert t.exit_price == 103 + dist  # limit order: no slippage
    assert t.entry_price == bars["open"].iloc[4] + NQ.tick_size


def test_stop_wins_when_bar_touches_both_stop_and_target():
    bars = make_bars([100, 100, 100, 103, 103])
    bars.iloc[4, bars.columns.get_loc("high")] = 200.0
    bars.iloc[4, bars.columns.get_loc("low")] = 0.0
    t = simulate(bars, replace(P, target_r=1.0), NQ, NO_COSTS)[0]
    assert t.exit_reason == "stop"


def test_no_new_entry_while_in_a_target_only_trade():
    closes = [100, 100, 100] + [103 + 3 * i for i in range(10)]
    trades = simulate(make_bars(closes), replace(P, target_r=50.0, use_trail=False), NQ, NO_COSTS)
    assert len(trades) == 1 and trades[0].exit_reason == "end_of_data"
