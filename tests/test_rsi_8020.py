import random
from datetime import datetime, time, timedelta

import pytest

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import TimeframePair
from rsi_8020.strategy import RSI8020Strategy, WilderRSI

from backtest.engine import BacktestEngine

T0 = datetime(2026, 1, 5, 10, 0)  # within default session window
LOOKBACK = 5


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


class _ScriptedRSI:
    """Replays a fixed RSI series so each test controls divergence exactly."""

    def __init__(self, values):
        self._values = iter(values)
        self.value = None

    def update(self, close):
        self.value = next(self._values)
        return self.value


def _run(rows, **kwargs):
    """rows: (high, low, close, rsi) per candle. Returns [(index, signal)]."""
    strat = RSI8020Strategy(tick_size=0.25, lookback=LOOKBACK, stop_buffer_ticks=2, target_r=3.0, **kwargs)
    strat.rsi = _ScriptedRSI([r[3] for r in rows])
    out = []
    for i, (h, l, c, _) in enumerate(rows):
        s = strat.on_entry_bar(Bar(T0 + timedelta(minutes=5 * i), c, h, l, c, 1.0))
        if s is not None:
            out.append((i, s))
    return out


WARMUP = [(102, 100, 101, 50)] * (LOOKBACK - 1)

# Long textbook case.
LONG_SETUP = WARMUP + [
    (100, 95, 96, 15),     # 4: 5-candle low + RSI < 20 -> first low (high=100)
    (98, 96, 97, 25),      # 5: bounce
    (97, 93, 94, 30),      # 6: lower low (93 < 95), higher RSI -> divergent second low
    (99, 94, 98, 40),      # 7: closes 98, not above first candle's high (100) yet
    (102, 97, 101, 55),    # 8: closes above 100 -> ENTRY
]


def test_long_divergence_entry():
    signals = _run(LONG_SETUP)
    assert [i for i, _ in signals] == [8]
    s = signals[0][1]
    assert s.direction == "long"
    assert s.entry_price == 101
    assert s.stop_price == pytest.approx(93 - 0.5)  # below the second low, 2-tick buffer
    assert s.target_price == pytest.approx(101 + 3 * (101 - 92.5))


def test_no_entry_without_divergence():
    rows = list(LONG_SETUP)
    rows[6] = (97, 93, 94, 10)  # lower low but *lower* RSI -> becomes the new first low (high 97)
    rows[8] = (99, 96, 98.5, 55)  # closes above the new first candle's high (97)...
    # ...but there has been no divergent second low since, so still no trade.
    assert _run(rows) == []


def test_no_entry_without_second_low():
    rows = WARMUP + [
        (100, 95, 96, 15),   # first low
        (99, 96, 98, 30),
        (103, 97, 102, 60),  # closes above first high, but price never undercut 95
    ]
    assert _run(rows) == []


def test_first_low_requires_rsi_below_20():
    rows = list(LONG_SETUP)
    rows[4] = (100, 95, 96, 25)  # 50-candle low but RSI not oversold
    assert _run(rows) == []


def test_first_low_requires_lookback_low():
    rows = list(LONG_SETUP)
    rows[0] = (102, 94, 101, 50)  # an earlier candle already went lower than candle 4
    assert _run(rows) == []


def test_second_low_and_entry_cannot_be_same_candle():
    rows = WARMUP + [
        (100, 95, 96, 15),
        (98, 96, 97, 25),
        (103, 93, 102, 40),  # undercuts 95 with higher RSI AND closes above 100 in one candle
        (104, 101, 103, 60),  # next candle also closes above 100 -> this is the entry
    ]
    assert [i for i, _ in _run(rows)] == [7]


def test_stop_uses_lowest_divergent_low():
    rows = list(LONG_SETUP)
    rows.insert(7, (95, 91, 92, 35))  # an even lower low, still divergent vs first (15)
    signals = _run(rows)
    assert len(signals) == 1
    assert signals[0][1].stop_price == pytest.approx(91 - 0.5)


def test_setup_expires():
    rows = LONG_SETUP[:7] + [(99, 94, 98, 40)] * 10 + [(102, 97, 101, 55)]
    assert _run(rows, max_setup_bars=5) == []
    assert len(_run(rows, max_setup_bars=50)) == 1


def test_short_is_mirror():
    warm = [(100, 98, 99, 50)] * (LOOKBACK - 1)
    rows = warm + [
        (105, 100, 104, 85),  # 5-candle high + RSI > 80 -> first high (low=100)
        (104, 102, 103, 75),
        (107, 103, 106, 70),  # higher high, lower RSI -> divergence
        (106, 101, 102, 60),  # 102 not below 100
        (103, 98, 99, 45),    # closes below 100 -> ENTRY
    ]
    signals = _run(rows)
    assert [i for i, _ in signals] == [8]
    s = signals[0][1]
    assert s.direction == "short"
    assert s.stop_price == pytest.approx(107 + 0.5)
    assert s.target_price == pytest.approx(99 - 3 * (107.5 - 99))


def test_no_signal_outside_entry_window():
    strat = RSI8020Strategy(tick_size=0.25, lookback=LOOKBACK)
    strat.rsi = _ScriptedRSI([r[3] for r in LONG_SETUP])
    late = datetime(2026, 1, 5, 15, 46)
    for i, (h, l, c, _) in enumerate(LONG_SETUP):
        assert strat.on_entry_bar(Bar(late + timedelta(seconds=10 * i), c, h, l, c, 1.0)) is None


def test_invalid_params_rejected():
    with pytest.raises(ValueError):
        RSI8020Strategy(overbought=20, oversold=80)
    with pytest.raises(ValueError):
        RSI8020Strategy(lookback=1)


def test_runs_in_backtest_engine(tmp_path):
    rng = random.Random(11)
    rows = ["timestamp,open,high,low,close,volume"]
    price = 5000.0
    for d in range(10):
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
        pair=TimeframePair("rsi-1min", "1min", "1min"),
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
