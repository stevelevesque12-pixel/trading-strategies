from datetime import time

import pandas as pd
import pytest

from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import PAIRS, Failed2sStrategy

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
