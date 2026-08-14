from datetime import time

import pytest

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from fair_value.strategy import FairValueStrategy

from backtest.fair_value_engine import FairValueBacktestEngine
from backtest.metrics import compute_metrics


@pytest.fixture(scope="module")
def sample_csv(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample_data"))
    from generate_sample import generate  # noqa: E402

    df = generate(days=25, seed=11, start_price=15000.0, tz="America/New_York")
    path = tmp_path_factory.mktemp("data") / "sample_1min.csv"
    df.to_csv(path, index=False)
    return str(path)


def test_backtest_runs_end_to_end(sample_csv):
    instrument = INSTRUMENTS["NQ"]
    strategy = FairValueStrategy(tick_size=instrument.tick_size)
    risk = RiskManager(daily_loss_limit=1000.0, max_daily_trades=3)
    engine = FairValueBacktestEngine(instrument=instrument, strategy=strategy, risk=risk)

    trades = engine.run(sample_csv)

    assert isinstance(trades, list)
    for tr in trades:
        assert tr.exit_time >= tr.entry_time
        assert tr.exit_reason in ("stop", "target", "window_flatten", "session_flatten", "end_of_data")
        assert tr.entry_time.date() == tr.exit_time.date()  # strictly intraday
        assert 1 <= tr.contracts <= 3

    metrics = compute_metrics(trades)
    assert "num_trades" in metrics


def test_positions_never_open_outside_the_configured_windows(sample_csv):
    instrument = INSTRUMENTS["NQ"]
    engine = FairValueBacktestEngine(instrument=instrument)
    trades = engine.run(sample_csv)

    for tr in trades:
        et = tr.entry_time.time()
        in_am = time(9, 33) <= et < time(11, 0)
        in_pm = time(14, 0) <= et < time(15, 0)
        assert in_am or in_pm


def test_daily_trade_cap_is_enforced(sample_csv):
    instrument = INSTRUMENTS["NQ"]
    strategy = FairValueStrategy(tick_size=instrument.tick_size)
    risk = RiskManager(daily_loss_limit=1e9, max_daily_trades=2)
    engine = FairValueBacktestEngine(instrument=instrument, strategy=strategy, risk=risk)

    trades = engine.run(sample_csv)

    per_day = {}
    for tr in trades:
        per_day[tr.entry_time.date()] = per_day.get(tr.entry_time.date(), 0) + 1
    assert all(count <= 2 for count in per_day.values())
