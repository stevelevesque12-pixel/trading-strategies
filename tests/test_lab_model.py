from datetime import datetime, timedelta

import pytest

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS
from failed2s.structure import Swing

from backtest.lab_model_engine import LabModelEngine
from backtest.metrics import compute_metrics
from lab_model.fvg import InverseFVGTracker
from lab_model.smt import detect_smt
from lab_model.strategy import LabModelStrategy
from lab_model.zones import ZoneTracker

T0 = datetime(2026, 1, 5, 9, 0)


# ---------------------------------------------------------------------------
# zones.py
# ---------------------------------------------------------------------------

def swing(i, price, kind):
    return Swing(T0 + timedelta(hours=i), price, kind)


def zbar(i, o, h, l, c):
    return Bar(T0 + timedelta(hours=i) + timedelta(minutes=30), o, h, l, c)


def test_current_leg_pairs_latest_two_opposite_swings():
    zt = ZoneTracker()
    zt.history = [swing(0, 90, "low"), swing(1, 110, "high")]
    leg = zt.current_leg
    assert leg.direction == "up"
    assert leg.high == 110 and leg.low == 90 and leg.mid == 100


def test_balanced_flips_true_once_price_crosses_midpoint():
    zt = ZoneTracker()
    zt.history = [swing(0, 90, "low"), swing(1, 110, "high")]
    assert zt.is_balanced is False
    zt.update(zbar(2, 105, 106, 101, 102))  # stays above mid (100)
    assert zt.is_balanced is False
    zt.update(zbar(3, 101, 102, 99, 100))  # low touches mid=100
    assert zt.is_balanced is True


def test_new_leg_starts_unbalanced_again():
    zt = ZoneTracker()
    zt.history = [swing(0, 90, "low"), swing(1, 110, "high")]
    zt.update(zbar(2, 101, 102, 99, 100))  # balances the up-leg
    assert zt.is_balanced is True

    zt.history.append(swing(4, 80, "low"))  # forms a new down-leg (110 -> 80)
    assert zt.current_leg.direction == "down"
    assert zt.is_balanced is False  # fresh leg, not yet balanced


def test_llt_picks_nearest_prior_swing_beyond_midline():
    zt = ZoneTracker()
    zt.history = [
        swing(0, 90, "low"),
        swing(1, 97, "high"),   # a smaller earlier high
        swing(2, 90, "low"),
        swing(3, 110, "high"),  # leg high (start of the current down-leg)
        swing(4, 80, "low"),    # leg low (end of the current down-leg)
    ]
    leg = zt.current_leg
    assert leg.direction == "down"
    assert leg.mid == 95
    assert zt.llt == 97  # nearest prior high >= mid, not the leg's own extreme (110)


def test_llt_falls_back_to_leg_start_when_no_earlier_swing_qualifies():
    zt = ZoneTracker()
    zt.history = [swing(0, 110, "high"), swing(1, 80, "low")]
    assert zt.llt == 110


def test_no_leg_or_llt_with_fewer_than_two_swings():
    zt = ZoneTracker()
    assert zt.current_leg is None
    assert zt.llt is None
    assert zt.is_balanced is False


# ---------------------------------------------------------------------------
# smt.py
# ---------------------------------------------------------------------------

def sw(i, price, kind):
    return Swing(T0 + timedelta(minutes=i), price, kind)


def test_bullish_smt_when_primary_makes_lower_low_and_secondary_fails_to_confirm():
    primary = [sw(0, 100, "low"), sw(1, 95, "low")]
    secondary = [sw(0, 100, "low"), sw(1, 102, "low")]
    assert detect_smt(primary, secondary, "low") is True


def test_no_smt_when_both_assets_make_new_lows():
    primary = [sw(0, 100, "low"), sw(1, 95, "low")]
    secondary = [sw(0, 100, "low"), sw(1, 97, "low")]
    assert detect_smt(primary, secondary, "low") is False


def test_bearish_smt_when_primary_makes_higher_high_and_secondary_fails_to_confirm():
    primary = [sw(0, 100, "high"), sw(1, 105, "high")]
    secondary = [sw(0, 100, "high"), sw(1, 98, "high")]
    assert detect_smt(primary, secondary, "high") is True


def test_no_smt_with_insufficient_history():
    assert detect_smt([sw(0, 100, "low")], [sw(0, 100, "low"), sw(1, 95, "low")], "low") is False


# ---------------------------------------------------------------------------
# fvg.py
# ---------------------------------------------------------------------------

def fbar(i, o, h, l, c):
    return Bar(T0 + timedelta(minutes=i), o, h, l, c)


def test_bullish_fvg_forms_and_inverts_to_short_signal():
    tracker = InverseFVGTracker()
    bars = [
        fbar(0, 100, 101, 99, 100),
        fbar(1, 100, 108, 100, 107),
        fbar(2, 107, 109, 105, 108),  # bar1.high(101) < bar3.low(105) -> bullish FVG [101,105]
    ]
    for b in bars:
        assert tracker.update(b) is None
    zone = tracker.open_fvgs[0]
    assert zone.kind == "bullish" and zone.low == 101 and zone.high == 105

    inverted = tracker.update(fbar(3, 104, 104, 99, 99))  # closes below zone.low -> inversion
    assert inverted is not None
    assert inverted.inverted_direction == "short"


def test_bearish_fvg_forms_and_inverts_to_long_signal():
    tracker = InverseFVGTracker()
    bars = [
        fbar(0, 108, 109, 107, 108),
        fbar(1, 108, 108, 100, 101),
        fbar(2, 101, 103, 99, 100),  # bar1.low(107) > bar3.high(103) -> bearish FVG [103,107]
    ]
    for b in bars:
        assert tracker.update(b) is None
    zone = tracker.open_fvgs[0]
    assert zone.kind == "bearish" and zone.low == 103 and zone.high == 107

    inverted = tracker.update(fbar(3, 104, 109, 104, 108))  # closes above zone.high -> inversion
    assert inverted is not None
    assert inverted.inverted_direction == "long"


def test_no_inversion_without_a_prior_fvg():
    tracker = InverseFVGTracker()
    assert tracker.update(fbar(0, 100, 101, 99, 95)) is None


# ---------------------------------------------------------------------------
# strategy.py (integration)
# ---------------------------------------------------------------------------

REF_TS = datetime(2026, 1, 5, 6, 0)  # 4h candle closing at 10:00 local


def make_strategy(**kwargs):
    kwargs.setdefault("pending_expiry_bars", 50)
    strat = LabModelStrategy(tick_size=0.25, stop_buffer_ticks=4, **kwargs)
    ref_bar = Bar(REF_TS, 20900, 21000, 20800, 20900)
    strat.on_htf_bar("4h", ref_bar)
    return strat


def nqb(i, o, h, l, c):
    return Bar(REF_TS + timedelta(hours=4) + timedelta(minutes=i), o, h, l, c)


def esb(i, o, h, l, c):
    return Bar(REF_TS + timedelta(hours=4) + timedelta(minutes=i), o, h, l, c)


def test_reversal_fires_long_on_sweep_plus_smt_plus_ifvg():
    strat = make_strategy()

    # 1. Sweep the reference candle's low (20800) -> arms a long reversal.
    signal = strat.on_execution_bars(
        nqb(0, 20805, 20810, 20795, 20800),
        esb(0, 21005, 21010, 20995, 21000),
    )
    assert signal is None
    assert strat._pending is not None and strat._pending.direction == "long"
    armed_at = strat._pending.armed_at

    # 2. Inject SMT divergence: NQ prints a lower low after arming; ES fails to confirm.
    #    Also gives zone_exec a down-leg (high 21050 -> low 20795) to derive the LLT from.
    strat.zone_exec.history = [
        Swing(armed_at - timedelta(minutes=10), 21050, "high"),
        Swing(armed_at - timedelta(minutes=5), 20900, "low"),
        Swing(armed_at + timedelta(minutes=1), 20795, "low"),
    ]
    strat.smt_es.history = [
        Swing(armed_at - timedelta(minutes=5), 21010, "low"),
        Swing(armed_at + timedelta(minutes=1), 21015, "low"),
    ]

    # 3. Feed a 3-bar bearish FVG on NQ, then a candle closing back through it (bullish iFVG).
    assert strat.on_execution_bars(nqb(1, 20805, 20860, 20850, 20855), esb(1, 21000, 21005, 20998, 21001)) is None
    assert strat.on_execution_bars(nqb(2, 20855, 20856, 20800, 20805), esb(2, 21001, 21006, 20999, 21002)) is None
    # bar1.low(20850) > bar3.high(20830) -> bearish FVG zone [20830, 20850]
    assert strat.on_execution_bars(nqb(3, 20806, 20830, 20795, 20800), esb(3, 21002, 21007, 21000, 21003)) is None

    signal = strat.on_execution_bars(nqb(4, 20801, 20870, 20800, 20860), esb(4, 21003, 21008, 21001, 21004))

    assert signal is not None
    assert signal.direction == "long"
    assert signal.stop_price == 20800 - strat.stop_buffer  # frozen ref_low - buffer
    assert signal.target_price == 21050  # LLT: nearest prior high >= the down-leg's midpoint
    assert strat._pending is None  # consumed


def test_no_signal_with_smt_but_no_ifvg():
    strat = make_strategy()
    strat.on_execution_bars(nqb(0, 20805, 20810, 20795, 20800), esb(0, 21005, 21010, 20995, 21000))
    armed_at = strat._pending.armed_at

    strat.zone_exec.history = [
        Swing(armed_at - timedelta(minutes=5), 20900, "low"),
        Swing(armed_at + timedelta(minutes=1), 20795, "low"),
    ]
    strat.smt_es.history = [
        Swing(armed_at - timedelta(minutes=5), 21010, "low"),
        Swing(armed_at + timedelta(minutes=1), 21015, "low"),
    ]

    signal = None
    for i in range(1, 10):
        signal = strat.on_execution_bars(nqb(i, 20800, 20805, 20798, 20801), esb(i, 21000, 21005, 20998, 21001))
    assert signal is None
    assert strat._pending is not None
    assert strat._pending.smt_seen is True
    assert strat._pending.ifvg_seen is False


def test_pending_reversal_expires_after_max_bars():
    strat = make_strategy(pending_expiry_bars=2)
    strat.on_execution_bars(nqb(0, 20805, 20810, 20795, 20800), esb(0, 21005, 21010, 20995, 21000))
    assert strat._pending is not None

    # Bars stay well above ref_low (20800) so the pending setup expires instead of re-arming.
    for i in range(1, 5):
        strat.on_execution_bars(nqb(i, 20850, 20860, 20845, 20855), esb(i, 21000, 21005, 20998, 21001))
    assert strat._pending is None


def test_no_sweep_arm_without_a_reference_candle():
    strat = LabModelStrategy(tick_size=0.25)  # no on_htf_bar("4h", ...) call yet
    signal = strat.on_execution_bars(nqb(0, 20805, 20810, 20000, 20800), esb(0, 21005, 21010, 21000, 21000))
    assert signal is None
    assert strat._pending is None


def test_continuation_arms_when_ltf_balances_with_unbalanced_htf():
    strat = make_strategy()

    # HTF (1h) up-leg, not yet balanced -> continuation direction should be "short".
    strat.zone_1h.history = [
        Swing(REF_TS - timedelta(hours=10), 20800, "low"),
        Swing(REF_TS - timedelta(hours=5), 21000, "high"),
    ]
    # execution-tf leg to derive the continuation stop from (uses its high for a short).
    strat.zone_exec.history = [
        Swing(REF_TS - timedelta(hours=1), 20870, "low"),
        Swing(REF_TS - timedelta(minutes=30), 20950, "high"),
    ]
    # 5m LTF leg about to balance (up-leg, mid=20900) via a bar whose low reaches the midpoint.
    strat.zone_5m.history = [
        Swing(REF_TS - timedelta(hours=2), 20850, "low"),
        Swing(REF_TS - timedelta(hours=1), 20950, "high"),
    ]

    assert strat._pending is None
    strat.on_htf_bar("5m", Bar(REF_TS + timedelta(hours=1), 20905, 20910, 20890, 20895))
    assert strat._pending is not None
    assert strat._pending.direction == "short"
    assert strat._pending.kind == "continuation"
    assert strat._pending.stop_price == 20950 + strat.stop_buffer


# ---------------------------------------------------------------------------
# backtest/lab_model_engine.py (smoke test)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def synthetic_csvs(tmp_path_factory):
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "sample_data"))
    from generate_sample import generate  # noqa: E402

    data_dir = tmp_path_factory.mktemp("lab_model_data")
    nq_df = generate(days=20, seed=11, start_price=21000.0, tz="America/New_York")
    es_df = generate(days=20, seed=12, start_price=5000.0, tz="America/New_York")
    nq_path = data_dir / "nq_1min.csv"
    es_path = data_dir / "es_1min.csv"
    nq_df.to_csv(nq_path, index=False)
    es_df.to_csv(es_path, index=False)
    return str(nq_path), str(es_path)


def test_engine_runs_end_to_end_on_synthetic_data(synthetic_csvs):
    nq_csv, es_csv = synthetic_csvs
    instrument = INSTRUMENTS["NQ"]
    strategy = LabModelStrategy(tick_size=instrument.tick_size)
    engine = LabModelEngine(strategy=strategy, instrument=instrument)

    trades = engine.run(nq_csv, es_csv)

    assert isinstance(trades, list)
    for t in trades:
        assert t.exit_time >= t.entry_time
        assert t.exit_reason in ("stop", "target", "session_flatten", "end_of_data")
        assert t.entry_time.date() == t.exit_time.date()

    metrics = compute_metrics(trades)
    assert "num_trades" in metrics


@pytest.mark.parametrize("execution_tf", ["1min", "3min", "5min"])
def test_engine_runs_on_each_execution_timeframe(synthetic_csvs, execution_tf):
    nq_csv, es_csv = synthetic_csvs
    instrument = INSTRUMENTS["NQ"]
    engine = LabModelEngine(instrument=instrument, execution_tf=execution_tf)

    trades = engine.run(nq_csv, es_csv)
    assert isinstance(trades, list)


# ---------------------------------------------------------------------------
# backtest/lab_model_engine.py -- commission and slippage
# ---------------------------------------------------------------------------

class _FireOnceLabModelStrategy:
    """Test double: fires one signal on the first execution bar, then stays quiet.

    Isolates the engine's fill/commission/slippage logic from the real
    pattern-detection cascade in LabModelStrategy, mirroring
    `test_backtest.py`'s `_FireOnceStrategy`.
    """

    def __init__(self, direction, entry_price, stop_price, target_price):
        self._fired = False
        self.direction = direction
        self.entry_price = entry_price
        self.stop_price = stop_price
        self.target_price = target_price

    def on_htf_bar(self, tf, bar):
        pass

    def on_es_4h_bar(self, bar):
        pass

    def on_execution_bars(self, nq_bar, es_bar):
        if self._fired:
            return None
        self._fired = True
        from lab_model.strategy import Signal

        return Signal(
            timestamp=nq_bar.timestamp,
            direction=self.direction,
            entry_price=self.entry_price,
            stop_price=self.stop_price,
            target_price=self.target_price,
            reason="test",
        )


def _write_pair_csv(tmp_path, name, rows):
    import pandas as pd

    df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
    path = tmp_path / name
    df.to_csv(path, index=False)
    return str(path)


def _two_bar_pair(tmp_path, nq_bar1, es_bar1=(100, 101, 99, 100, 10)):
    """A 2-bar NQ+ES pair: bar0 (fires the signal) then bar1 (`nq_bar1`, drives the exit)."""
    base_nq = [
        ("2026-01-05 10:00:00-05:00", 100, 101, 99, 100, 10),
        ("2026-01-05 10:01:00-05:00",) + nq_bar1,
    ]
    base_es = [
        ("2026-01-05 10:00:00-05:00",) + es_bar1,
        ("2026-01-05 10:01:00-05:00",) + es_bar1,
    ]
    nq_path = _write_pair_csv(tmp_path, "nq.csv", base_nq)
    es_path = _write_pair_csv(tmp_path, "es.csv", base_es)
    return nq_path, es_path


def test_stop_exit_applies_slippage_and_commission(tmp_path):
    nq_path, es_path = _two_bar_pair(tmp_path, nq_bar1=(94, 96, 90, 91, 10))  # low 90 breaches stop=95

    instrument = INSTRUMENTS["NQ"]
    strategy = _FireOnceLabModelStrategy("long", entry_price=100, stop_price=95, target_price=110)
    engine = LabModelEngine(
        strategy=strategy, instrument=instrument, breakeven_at_r=None,
        commission_per_contract=4.60, slippage_ticks=1.0,
    )
    trades = engine.run(nq_path, es_path)

    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "stop"
    # entry fills worse (higher) than the signal price by 1 tick; stop exit fills worse (lower) by 1 tick
    assert t.entry_price == 100 + 0.25
    assert t.exit_price == 95 - 0.25
    expected_points = (t.exit_price - t.entry_price)
    expected_dollars = expected_points * instrument.point_value - 4.60
    assert round(t.pnl_dollars, 6) == round(expected_dollars, 6)


def test_target_exit_has_no_slippage(tmp_path):
    nq_path, es_path = _two_bar_pair(tmp_path, nq_bar1=(101, 115, 100, 110, 10))  # high 115 clears target=110

    instrument = INSTRUMENTS["NQ"]
    strategy = _FireOnceLabModelStrategy("long", entry_price=100, stop_price=95, target_price=110)
    engine = LabModelEngine(
        strategy=strategy, instrument=instrument, breakeven_at_r=None,
        commission_per_contract=0.0, slippage_ticks=1.0,
    )
    trades = engine.run(nq_path, es_path)

    assert len(trades) == 1
    t = trades[0]
    assert t.exit_reason == "target"
    assert t.entry_price == 100 + 0.25  # entry still slips
    assert t.exit_price == 110  # target fills exactly -- modeled as a limit order, no slippage


def test_zero_commission_and_slippage_matches_theoretical_prices(tmp_path):
    nq_path, es_path = _two_bar_pair(tmp_path, nq_bar1=(94, 96, 90, 91, 10))

    instrument = INSTRUMENTS["NQ"]
    strategy = _FireOnceLabModelStrategy("long", entry_price=100, stop_price=95, target_price=110)
    engine = LabModelEngine(
        strategy=strategy, instrument=instrument, breakeven_at_r=None,
        commission_per_contract=0.0, slippage_ticks=0.0,
    )
    trades = engine.run(nq_path, es_path)

    t = trades[0]
    assert t.entry_price == 100
    assert t.exit_price == 95
    assert t.pnl_dollars == (95 - 100) * instrument.point_value
