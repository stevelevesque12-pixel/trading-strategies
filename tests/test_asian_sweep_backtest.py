from datetime import time

import pytest

from asian_sweep.strategy import AsianSweepStrategy
from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager

from backtest.asian_sweep_engine import AsianSweepEngine
from backtest.metrics import compute_metrics


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample_data"))
    from generate_sample import generate  # noqa: E402

    df = generate(days=20, seed=11, start_price=18000.0, tz="America/New_York", full_day=True)
    path = tmp_path_factory.mktemp("data") / "sample_asian_sweep_1min.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_asian_sweep_backtest_runs_end_to_end(sample_csv):
    instrument = INSTRUMENTS["NQ"]
    strategy = AsianSweepStrategy(tick_size=instrument.tick_size)
    risk = RiskManager(daily_loss_limit=1000.0, max_daily_trades=2)
    engine = AsianSweepEngine(instrument=instrument, strategy=strategy, risk=risk)

    trades = engine.run(sample_csv)

    assert isinstance(trades, list)
    for t in trades:
        assert t.exit_time >= t.entry_time
        assert t.exit_reason in ("stop", "target", "session_flatten", "end_of_data")
        assert t.entry_time.date() == t.exit_time.date()  # intraday only
        assert t.entry_time.time() < time(15, 55)  # never entered past flatten

    metrics = compute_metrics(trades)
    assert "num_trades" in metrics


def test_risk_manager_caps_daily_trade_count(sample_csv):
    instrument = INSTRUMENTS["NQ"]
    strategy = AsianSweepStrategy(tick_size=instrument.tick_size)
    risk = RiskManager(daily_loss_limit=1e9, max_daily_trades=1)
    engine = AsianSweepEngine(instrument=instrument, strategy=strategy, risk=risk)

    trades = engine.run(sample_csv)

    per_day = {}
    for t in trades:
        per_day[t.entry_time.date()] = per_day.get(t.entry_time.date(), 0) + 1
    assert all(count <= 1 for count in per_day.values())
