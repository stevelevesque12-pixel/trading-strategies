from datetime import time

import pandas as pd
import pytest

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import SessionConfig
from supertrend_trinity.strategy import Signal, TrinitySuperTrendStrategy

from backtest.supertrend_trinity_engine import TrinitySuperTrendBacktestEngine
from backtest.metrics import compute_metrics


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample_data"))
    from generate_sample import generate  # noqa: E402

    df = generate(days=30, seed=19, start_price=5000.0, tz="America/New_York")
    path = tmp_path_factory.mktemp("data") / "sample_1min.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_backtest_runs_end_to_end_no_forced_flatten(sample_csv):
    """Default (session=None) -- positions may legitimately span multiple
    days, unlike every other engine in this repo."""
    instrument = INSTRUMENTS["MES"]
    risk = RiskManager(daily_loss_limit=1000.0, max_daily_trades=6)
    strategy = TrinitySuperTrendStrategy(atr_period1=5, atr_period2=5, atr_period3=5, entry_mode="single")
    engine = TrinitySuperTrendBacktestEngine(tf1="5min", tf2="1h", tf3="4h", instrument=instrument, strategy=strategy, risk=risk)

    trades = engine.run(sample_csv)

    assert isinstance(trades, list)
    for t in trades:
        assert t.exit_time >= t.entry_time
        assert t.exit_reason in ("stop", "target", "end_of_data")  # never "session_flatten" -- no session set

    metrics = compute_metrics(trades)
    assert "num_trades" in metrics


def test_backtest_with_session_flatten_enabled(sample_csv):
    instrument = INSTRUMENTS["MES"]
    session = SessionConfig(flatten_at=time(15, 55))
    strategy = TrinitySuperTrendStrategy(atr_period1=5, atr_period2=5, atr_period3=5, entry_mode="single", session=session)
    engine = TrinitySuperTrendBacktestEngine(tf1="5min", tf2="1h", tf3="4h", instrument=instrument, strategy=strategy)

    trades = engine.run(sample_csv)

    for t in trades:
        assert t.entry_time.date() == t.exit_time.date() or t.exit_reason == "session_flatten"


class _FireOnceStrategy:
    """Test double: fires one long signal on the first bar, then stays quiet.

    Isolates the engine's position-management/gap-handling logic from the
    real multi-timeframe SuperTrend detection in TrinitySuperTrendStrategy.
    """

    def __init__(self, entry_price, stop_price, target_price, session=None, trailing=False):
        self._fired = False
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.session = session
        self.trailing = trailing

    def on_tf2_bar(self, bar):
        pass

    def on_tf3_bar(self, bar):
        pass

    def on_entry_bar(self, bar):
        if self._fired:
            return None
        self._fired = True
        return Signal(
            timestamp=bar.timestamp,
            direction="long",
            entry_price=self.entry_price,
            stop_price=self.stop_price,
            target_price=self.target_price,
            reason="test",
        )


def test_position_survives_a_multi_day_gap_when_no_session_is_set(tmp_path):
    """Regression test for the swing-by-default behavior: unlike every
    other engine in this repo, a position must NOT be force-closed just
    because a new calendar day started, when session=None."""
    rows = [
        ("2026-01-05 09:30:00-05:00", 100, 101, 99, 100, 10),  # signal fires here
        ("2026-01-06 09:30:00-05:00", 100, 101, 99, 100, 10),  # next day -- position should still be open
        ("2026-01-07 09:30:00-05:00", 100, 200, 99, 150, 10),  # target hit here
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    path = tmp_path / "swing.csv"
    df.to_csv(path, index=False)

    instrument = INSTRUMENTS["MES"]
    strategy = _FireOnceStrategy(entry_price=100, stop_price=50, target_price=150)
    engine = TrinitySuperTrendBacktestEngine(tf1="1min", tf2="1h", tf3="4h", instrument=instrument, strategy=strategy)

    trades = engine.run(str(path))

    assert len(trades) == 1
    assert trades[0].exit_reason == "target"
    assert trades[0].entry_time.date().isoformat() == "2026-01-05"
    assert trades[0].exit_time.date().isoformat() == "2026-01-07"  # spans multiple days, never force-flattened


def test_position_flattened_on_gap_when_session_is_set(tmp_path):
    """Same data, but WITH a session set -- should behave like the other
    engines' gap-flatten regression test."""
    rows = [
        ("2026-01-05 09:30:00-05:00", 100, 101, 99, 100, 10),
        ("2026-01-05 09:31:00-05:00", 100, 101, 99, 100, 10),
        ("2026-01-06 09:30:00-05:00", 100, 101, 99, 100, 10),
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    path = tmp_path / "gap.csv"
    df.to_csv(path, index=False)

    instrument = INSTRUMENTS["MES"]
    session = SessionConfig()
    strategy = _FireOnceStrategy(entry_price=100, stop_price=50, target_price=200, session=session)
    engine = TrinitySuperTrendBacktestEngine(tf1="1min", tf2="1h", tf3="4h", instrument=instrument, strategy=strategy)

    trades = engine.run(str(path))

    assert len(trades) == 1
    assert trades[0].exit_reason == "session_flatten"
    assert trades[0].entry_time.date().isoformat() == "2026-01-05"
    assert trades[0].exit_time.date().isoformat() == "2026-01-06"


def test_trailing_stop_is_applied_when_enabled(tmp_path):
    rows = [
        ("2026-01-05 09:30:00-05:00", 100, 101, 99, 100, 10),   # signal fires
        # bar1: close=118 ratchets the stop up to 113 BEFORE the stop-check
        # runs for this same bar (matching the original script's own
        # per-bar ordering) -- and this bar's own low(99) already
        # undercuts that freshly-ratcheted stop, so it exits right here,
        # not on some later pullback bar.
        ("2026-01-05 09:31:00-05:00", 100, 120, 99, 118, 10),
        ("2026-01-05 09:32:00-05:00", 118, 119, 105, 106, 10),  # never reached
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    path = tmp_path / "trail.csv"
    df.to_csv(path, index=False)

    instrument = INSTRUMENTS["MES"]
    strategy = _FireOnceStrategy(entry_price=100, stop_price=50, target_price=1000, trailing=True)
    strategy.trail_stop = lambda bar, direction, current_stop: max(current_stop, bar.close - 5)  # simple deterministic trail
    engine = TrinitySuperTrendBacktestEngine(tf1="1min", tf2="1h", tf3="4h", instrument=instrument, strategy=strategy)

    trades = engine.run(str(path))

    assert len(trades) == 1
    assert trades[0].exit_reason == "stop"
    assert trades[0].exit_time.isoformat() == "2026-01-05T09:31:00-05:00"
    assert trades[0].stop_price == 113  # ratcheted to 118-5, well above the original 50
