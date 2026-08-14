from datetime import time

import pandas as pd
import pytest

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager

from backtest.metrics import compute_metrics
from backtest.volume_engine import VolumeScalpBacktestEngine
from volume_scalp.strategy import Signal, VolumeScalpStrategy


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


@pytest.mark.parametrize("timeframe", ["1min", "5min"])
@pytest.mark.parametrize("symbol", ["MES", "MNQ"])
def test_backtest_runs_end_to_end(sample_csv, symbol, timeframe):
    instrument = INSTRUMENTS[symbol]
    strategy = VolumeScalpStrategy(tick_size=instrument.tick_size)
    risk = RiskManager(daily_loss_limit=1000.0, max_daily_trades=6)
    engine = VolumeScalpBacktestEngine(instrument=instrument, strategy=strategy, risk=risk, timeframe=timeframe)

    trades = engine.run(sample_csv)

    assert isinstance(trades, list)
    for t in trades:
        assert t.exit_time >= t.entry_time
        assert t.exit_reason in ("stop", "target", "session_flatten", "end_of_data")
        assert t.entry_time.date() == t.exit_time.date()  # intraday only: never held overnight
        assert t.entry_time.time() < time(15, 45)  # never entered past the flatten cutoff

    metrics = compute_metrics(trades)
    assert "num_trades" in metrics


class _FireOnceStrategy:
    """Test double: fires one long signal on the first bar, then stays quiet.

    Isolates the engine's position-management/gap-handling logic from the
    real breakout/climax-fade detection in VolumeScalpStrategy.
    """

    def __init__(self, entry_price, stop_price, target_price):
        self._fired = False
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price

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
    rows = [
        ("2026-01-05 09:30:00-05:00", 100, 101, 99, 100, 10),  # signal fires here
        ("2026-01-05 09:31:00-05:00", 100, 101, 99, 100, 10),  # still open, no gap yet
        # big gap: the rest of 2026-01-05 is missing
        ("2026-01-06 09:30:00-05:00", 100, 101, 99, 100, 10),  # next available bar, next day
    ]
    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    path = tmp_path / "gap.csv"
    df.to_csv(path, index=False)

    instrument = INSTRUMENTS["MES"]
    # stop/target far from price so only the gap-handling path can close the trade
    strategy = _FireOnceStrategy(entry_price=100, stop_price=50, target_price=200)
    engine = VolumeScalpBacktestEngine(instrument=instrument, strategy=strategy)

    trades = engine.run(str(path))

    assert len(trades) == 1
    trade = trades[0]
    assert trade.exit_reason == "session_flatten"
    assert trade.entry_time.date().isoformat() == "2026-01-05"
    assert trade.exit_time.date().isoformat() == "2026-01-06"


def test_risk_manager_caps_daily_trade_count(sample_csv):
    instrument = INSTRUMENTS["MES"]
    strategy = VolumeScalpStrategy(tick_size=instrument.tick_size)
    risk = RiskManager(daily_loss_limit=1e9, max_daily_trades=2)
    engine = VolumeScalpBacktestEngine(instrument=instrument, strategy=strategy, risk=risk)

    trades = engine.run(sample_csv)

    per_day = {}
    for t in trades:
        per_day[t.entry_time.date()] = per_day.get(t.entry_time.date(), 0) + 1
    assert all(count <= 2 for count in per_day.values())
