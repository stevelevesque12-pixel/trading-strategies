import random
from datetime import datetime, time, timedelta

import pytest

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import TimeframePair
from rsi_8020.strategy import RSI8020Strategy, WilderRSI

from backtest.engine import BacktestEngine

T0 = datetime(2026, 1, 5, 9, 30)  # within default session window


def bar(i, c, h=None, l=None, o=None, day_offset=0):
    o = c if o is None else o
    h = max(o, c) + 0.25 if h is None else h
    l = min(o, c) - 0.25 if l is None else l
    return Bar(T0 + timedelta(days=day_offset, minutes=5 * i), o, h, l, c, 1.0)


def _reference_rsi(closes, n):
    """Straight textbook Wilder RSI, used to cross-check the incremental version."""
    changes = [b - a for a, b in zip(closes, closes[1:])]
    out = [None] * len(closes)
    if len(changes) < n:
        return out
    ag = sum(max(c, 0) for c in changes[:n]) / n
    al = sum(max(-c, 0) for c in changes[:n]) / n
    for i in range(n, len(changes) + 1):
        if i > n:
            c = changes[i - 1]
            ag = (ag * (n - 1) + max(c, 0)) / n
            al = (al * (n - 1) + max(-c, 0)) / n
        out[i] = 100.0 if al == 0 else 0.0 if ag == 0 else 100 - 100 / (1 + ag / al)
    return out


def test_wilder_rsi_matches_reference():
    rng = random.Random(3)
    closes = [100.0]
    for _ in range(200):
        closes.append(closes[-1] + rng.uniform(-1, 1))
    rsi = WilderRSI(14)
    got = [rsi.update(c) for c in closes]
    want = _reference_rsi(closes, 14)
    assert got[:14] == [None] * 14
    for g, w in zip(got[14:], want[14:]):
        assert g == pytest.approx(w)


def test_wilder_rsi_extremes():
    up = WilderRSI(3)
    for c in [1, 2, 3, 4, 5]:
        v = up.update(c)
    assert v == 100.0
    down = WilderRSI(3)
    for c in [5, 4, 3, 2, 1]:
        v = down.update(c)
    assert v == 0.0


def _feed(strat, bars):
    return [s for s in (strat.on_entry_bar(b) for b in bars) if s is not None]


def _selloff_then_bounce():
    # Flat-ish warmup, hard selloff (RSI < 20), then a bounce that takes RSI back over 20.
    closes = [100, 100.5, 100, 100.5, 100, 100.5]  # warmup (length=3)
    closes += [99, 97, 95, 93]  # selloff
    closes += [96]  # bounce: cross back above 20
    bars = [bar(i, c) for i, c in enumerate(closes)]
    bars[8] = bar(8, 95, l=94.0)  # wick sets the excursion low
    return bars


def test_long_signal_on_cross_back_above_oversold():
    strat = RSI8020Strategy(tick_size=0.25, rsi_length=3, stop_buffer_ticks=2, target_r=2.0)
    bars = _selloff_then_bounce()
    signals = _feed(strat, bars)

    assert len(signals) == 1
    s = signals[0]
    assert s.direction == "long"
    assert s.timestamp == bars[-1].timestamp
    assert s.entry_price == 96
    # lowest low during the excursion (bar 9, 92.75) minus 2 ticks
    assert s.stop_price == pytest.approx(92.75 - 0.5)
    risk = s.entry_price - s.stop_price
    assert s.target_price == pytest.approx(s.entry_price + 2 * risk)


def test_no_signal_while_still_oversold():
    strat = RSI8020Strategy(tick_size=0.25, rsi_length=3)
    assert _feed(strat, _selloff_then_bounce()[:-1]) == []
    assert strat.rsi.value < 20


def test_short_signal_is_mirror():
    closes = [100, 99.5, 100, 99.5, 100, 99.5, 101, 103, 105, 107, 104]
    bars = [bar(i, c) for i, c in enumerate(closes)]
    strat = RSI8020Strategy(tick_size=0.25, rsi_length=3, stop_buffer_ticks=2)
    signals = _feed(strat, bars)

    assert len(signals) == 1
    s = signals[0]
    assert s.direction == "short"
    assert s.entry_price == 104
    assert s.stop_price == pytest.approx(max(b.high for b in bars[6:]) + 0.5)
    assert s.target_price == pytest.approx(104 - (s.stop_price - 104))


def test_no_signal_outside_entry_window():
    # Same price path, but every bar after no_entry_after.
    bars = [Bar(datetime(2026, 1, 5, 15, 46) + timedelta(seconds=10 * i), b.open, b.high, b.low, b.close, b.volume)
            for i, b in enumerate(_selloff_then_bounce())]
    strat = RSI8020Strategy(tick_size=0.25, rsi_length=3)
    assert _feed(strat, bars) == []


def test_excursion_cleared_on_new_session_day():
    strat = RSI8020Strategy(tick_size=0.25, rsi_length=3)
    bars = _selloff_then_bounce()
    _feed(strat, bars[:-1])
    assert strat._excursion is not None

    # Next day's first bar clears the pending setup even though RSI is continuous.
    # RSI is still oversold, so a fresh excursion starts from this bar alone.
    strat.on_entry_bar(bar(0, 93, l=92.9, day_offset=1))
    assert strat.rsi.value < 20
    assert strat._excursion.extreme == 92.9


def test_invalid_levels_rejected():
    with pytest.raises(ValueError):
        RSI8020Strategy(overbought=20, oversold=80)


def test_runs_in_backtest_engine(tmp_path):
    rng = random.Random(11)
    rows = ["timestamp,open,high,low,close,volume"]
    price = 5000.0
    for d in range(5):
        t = datetime(2026, 1, 5 + d, 9, 30)
        for m in range(390):
            o = price
            price += rng.gauss(0, 1.5)
            h, l = max(o, price) + 0.5, min(o, price) - 0.5
            rows.append(f"{t + timedelta(minutes=m)},{o},{h},{l},{price},100")
    path = tmp_path / "bars.csv"
    path.write_text("\n".join(rows))

    instrument = INSTRUMENTS["MES"]
    engine = BacktestEngine(
        pair=TimeframePair("rsi-5min", "5min", "5min"),
        instrument=instrument,
        strategy=RSI8020Strategy(tick_size=instrument.tick_size),
        risk=RiskManager(daily_loss_limit=1e9, max_daily_trades=100),
    )
    trades = engine.run(str(path))
    for t in trades:
        assert t.entry_time.date() == t.exit_time.date()
        assert t.entry_time.time() < time(15, 45)
        if t.direction == "long":
            assert t.stop_price < t.entry_price < t.target_price
        else:
            assert t.target_price < t.entry_price < t.stop_price
