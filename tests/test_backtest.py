from datetime import time

import pandas as pd
import pytest

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import PAIRS, Failed2sStrategy, Signal

from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample_data"))
    from generate_sample import generate  # noqa: E402

    df = generate(days=15, seed=7, start_price=5000.0, tz="America/New_York")
    path = tmp_path_factory.mktemp("data") / "sample_1min.csv"
    df.to_csv(path, index=False)
    return str(path)


@pytest.mark.parametrize("pair_name", list(PAIRS.keys()))
def test_backtest_runs_end_to_end(sample_csv, pair_name):
    pair = PAIRS[pair_name]
    instrument = INSTRUMENTS["MES"]
    strategy = Failed2sStrategy(tick_size=instrument.tick_size)
    risk = RiskManager(daily_loss_limit=1000.0, max_daily_trades=3)
    engine = BacktestEngine(pair=pair, instrument=instrument, strategy=strategy, risk=risk)

    trades = engine.run(sample_csv)

    assert isinstance(trades, list)
    for t in trades:
        assert t.exit_time >= t.entry_time
        assert t.exit_reason in ("stop", "target", "session_flatten", "end_of_data")
        # intraday only: never held overnight
        assert t.entry_time.date() == t.exit_time.date()
        # never entered/held past the flatten cutoff
        assert t.entry_time.time() < time(15, 45)

    metrics = compute_metrics(trades)
    assert "num_trades" in metrics


class _FireOnceStrategy:
    """Test double: fires one long signal on the first entry bar, then stays quiet.

    Isolates the engine's position-management/gap-handling logic from the
    real pattern-detection cascade in Failed2sStrategy.
    """

    def __init__(self, entry_price, stop_price, target_price):
        self._fired = False
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price

    def on_bias_bar(self, bar):
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


def test_position_never_survives_a_data_gap_spanning_the_flatten_cutoff(tmp_path):
    """
    Regression test: a data gap spanning the flatten cutoff (e.g. a thin
    holiday session with no bar exactly at/after flatten_at) must not let a
    position carry into a later session day. The engine should force-close
    on the first bar of the new date, however far off the flatten-at
    time-of-day that bar happens to be.
    """
    rows = [
        ("2026-01-05 09:30:00-05:00", 100, 101, 99, 100, 10),  # signal fires here
        ("2026-01-05 09:31:00-05:00", 100, 101, 99, 100, 10),  # still open, no gap yet
        # big gap: rest of 2026-01-05 and all of 2026-01-05 pre-close is missing
        ("2026-01-06 09:30:00-05:00", 100, 101, 99, 100, 10),  # next available bar, next day
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    path = tmp_path / "gap.csv"
    df.to_csv(path, index=False)

    pair = PAIRS["1m-15m"]
    instrument = INSTRUMENTS["MES"]
    # stop/target far from price so only the gap-handling path can close the trade
    strategy = _FireOnceStrategy(entry_price=100, stop_price=50, target_price=200)
    engine = BacktestEngine(pair=pair, instrument=instrument, strategy=strategy)

    trades = engine.run(str(path))

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "session_flatten"
    assert trade.entry_time.date().isoformat() == "2026-01-05"
    assert trade.exit_time.date().isoformat() == "2026-01-06"  # closed on the gap, not held further


def test_risk_manager_caps_daily_trade_count(sample_csv):
    pair = PAIRS["1m-15m"]
    instrument = INSTRUMENTS["MES"]
    strategy = Failed2sStrategy(tick_size=instrument.tick_size)
    risk = RiskManager(daily_loss_limit=1e9, max_daily_trades=2)
    engine = BacktestEngine(pair=pair, instrument=instrument, strategy=strategy, risk=risk)

    trades = engine.run(sample_csv)

    per_day = {}
    for t in trades:
        per_day[t.entry_time.date()] = per_day.get(t.entry_time.date(), 0) + 1
    assert all(count <= 2 for count in per_day.values())
