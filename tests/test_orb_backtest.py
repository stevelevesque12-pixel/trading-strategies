from datetime import time

import pandas as pd
import pytest

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import SessionConfig
from orb.strategy import Signal

from backtest.orb_engine import ORBBacktestEngine
from backtest.metrics import compute_metrics


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample_data"))
    from generate_sample import generate  # noqa: E402

    df = generate(days=15, seed=11, start_price=5000.0, tz="America/New_York")
    path = tmp_path_factory.mktemp("data") / "sample_1min.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("or_minutes", [5, 15])
def test_backtest_runs_end_to_end(sample_csv, or_minutes):
    instrument = INSTRUMENTS["MES"]
    risk = RiskManager(daily_loss_limit=1000.0, max_daily_trades=1)
    from orb.strategy import ORBStrategy

    strategy = ORBStrategy(or_minutes=or_minutes, tick_size=instrument.tick_size)
    engine = ORBBacktestEngine(timeframe="5min", instrument=instrument, strategy=strategy, risk=risk)

    trades = engine.run(sample_csv)

    assert isinstance(trades, list)
    for t in trades:
        assert t.exit_time >= t.entry_time
        assert t.exit_reason in ("stop", "target", "session_flatten", "end_of_data")
        assert t.entry_time.date() == t.exit_time.date()  # intraday only
        assert t.entry_time.time() < time(10, 15)  # never entered past the default breakout cutoff

    metrics = compute_metrics(trades)
    assert "num_trades" in metrics

    # ORB is one-attempt-per-day by design -- never more than one trade per date
    per_day = {}
    for t in trades:
        per_day[t.entry_time.date()] = per_day.get(t.entry_time.date(), 0) + 1
    assert all(count <= 1 for count in per_day.values())


class _FireOnceStrategy:
    """Test double: fires one long signal on the first bar, then stays quiet.

    Isolates the engine's position-management/gap-handling logic from the
    real OR/ATR/breakout detection in ORBStrategy.
    """

    def __init__(self, entry_price, stop_price, target_price):
        self._fired = False
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price
        self.session = SessionConfig(no_entry_after=time(10, 15), flatten_at=time(15, 45))

    def on_bar(self, bar):
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


def test_position_never_survives_a_data_gap_spanning_the_flatten_cutoff(tmp_path):
    """
    Regression test mirroring the other engines': a data gap spanning the
    flatten cutoff must not let a position carry into a later session day.
    """
    rows = [
        ("2026-01-05 09:30:00-05:00", 100, 101, 99, 100, 10),  # signal fires here
        ("2026-01-05 09:31:00-05:00", 100, 101, 99, 100, 10),  # still open, no gap yet
        ("2026-01-06 09:30:00-05:00", 100, 101, 99, 100, 10),  # next available bar, next day
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    path = tmp_path / "gap.csv"
    df.to_csv(path, index=False)

    instrument = INSTRUMENTS["MES"]
    strategy = _FireOnceStrategy(entry_price=100, stop_price=50, target_price=200)
    engine = ORBBacktestEngine(timeframe="1min", instrument=instrument, strategy=strategy, session=strategy.session)

    trades = engine.run(str(path))

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "session_flatten"
    assert trade.entry_time.date().isoformat() == "2026-01-05"
    assert trade.exit_time.date().isoformat() == "2026-01-06"
