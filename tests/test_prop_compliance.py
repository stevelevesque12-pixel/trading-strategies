from datetime import datetime, timedelta

from backtest.prop_compliance import AccountProfile, compress_equity_curve, rolling_eval_pass_rate


def _minute_curve(daily_pnls, start=datetime(2024, 1, 2)):
    """
    Build a synthetic equity curve: two bars per trading day (day-open
    baseline, then day-close after that day's P&L), skipping weekends. Two
    points per day are the minimum needed for the daily-loss-limit check
    (which compares close against that day's *open*, not the prior day's
    close) to have anything to detect.
    """
    curve = []
    cum = 0.0
    d = start
    for pnl in daily_pnls:
        while d.weekday() >= 5:
            d += timedelta(days=1)
        curve.append((d.replace(hour=9, minute=30), cum))
        cum += pnl
        curve.append((d.replace(hour=16, minute=0), cum))
        d += timedelta(days=1)
    return curve


def test_clean_run_hits_target_without_breach():
    profile = AccountProfile(
        name="test", starting_balance=50_000, profit_target=3_000, max_drawdown=2_000,
        drawdown_type="static", daily_loss_limit=None,
    )
    # Steady grind up to +3000 over 10 days, never dips.
    curve = _minute_curve([300] * 10)

    report = rolling_eval_pass_rate(curve, profile, eval_trading_days=10)

    assert report["num_windows"] == 1
    r = report["results"][0]
    assert r.passed is True
    assert r.breached is False
    assert r.days_to_target == 10


def test_drawdown_breach_before_target_fails():
    profile = AccountProfile(
        name="test", starting_balance=50_000, profit_target=3_000, max_drawdown=2_000,
        drawdown_type="static", daily_loss_limit=None,
    )
    # Static floor = 48,000. Day 1 loses 2500 -> breaches before ever reaching target.
    curve = _minute_curve([-2500, 1000, 1000, 1000, 1000])

    report = rolling_eval_pass_rate(curve, profile, eval_trading_days=5)

    r = report["results"][0]
    assert r.breached is True
    assert r.breach_type == "drawdown"
    assert r.passed is False


def test_daily_loss_limit_breach_fails_even_if_within_max_drawdown():
    profile = AccountProfile(
        name="test", starting_balance=50_000, profit_target=3_000, max_drawdown=5_000,
        drawdown_type="static", daily_loss_limit=1_000,
    )
    # Well within the 5,000 max drawdown, but day 1 alone loses 1,200 > the 1,000 daily limit.
    curve = _minute_curve([-1200, 1000, 1000, 1000, 1000])

    report = rolling_eval_pass_rate(curve, profile, eval_trading_days=5)

    r = report["results"][0]
    assert r.breached is True
    assert r.breach_type == "daily_loss"


def test_trailing_drawdown_locks_at_breakeven_and_stops_trailing_up():
    profile = AccountProfile(
        name="test", starting_balance=50_000, profit_target=3_000, max_drawdown=2_000,
        drawdown_type="trailing_lock_at_breakeven", daily_loss_limit=None,
    )
    # Day 1 runs up to +2500 (floor would trail to peak-2000=500, i.e. above the
    # starting_balance ceiling -> locks at breakeven, floor pinned at 50,000 from
    # here on). Day 2 gives back 1,800 (equity 50,700) -- should NOT breach,
    # because an un-locked trailing floor would have used peak-2000=50,500 and
    # 50,700 still clears that too, but a *higher* peak later must not matter
    # once locked. Then climbs steadily to the 3,000 target by day 7.
    curve = _minute_curve([2500, -1800, 500, 500, 500, 500, 500, 500])

    report = rolling_eval_pass_rate(curve, profile, eval_trading_days=8)

    r = report["results"][0]
    assert r.breached is False
    assert r.passed is True
    assert r.days_to_target == 7


def test_multiple_windows_produce_aggregate_pass_rate():
    profile = AccountProfile(
        name="test", starting_balance=50_000, profit_target=1_000, max_drawdown=2_000,
        drawdown_type="static", daily_loss_limit=None,
    )
    # 6 trading days: windows of length 3 starting on day1..day4 (4 windows).
    # day pnls: +1200 (target hit day 1), -500, -500, -500, +1200, +1200
    curve = _minute_curve([1200, -500, -500, -500, 1200, 1200])

    report = rolling_eval_pass_rate(curve, profile, eval_trading_days=3)

    assert report["num_windows"] == 4
    assert report["passed"] >= 1
    assert 0.0 <= report["pass_rate_pct"] <= 100.0


def test_compression_is_lossless_for_pass_rate_computation():
    """
    Padding a curve with lots of redundant flat bars (the realistic case --
    equity only moves while a position is open, which is a small fraction
    of the trading day) must not change the compliance result at all.
    """
    profile = AccountProfile(
        name="test", starting_balance=50_000, profit_target=3_000, max_drawdown=2_000,
        drawdown_type="trailing_lock_at_breakeven", daily_loss_limit=1_000,
    )
    sparse_curve = _minute_curve([-800, 1200, -300, 900, 1100, 800, -200, 1500])

    padded_curve = []
    for ts, eq in sparse_curve:
        for extra_min in range(0, 120, 5):  # 24 redundant flat points between each real one
            padded_curve.append((ts + timedelta(minutes=extra_min), eq))

    sparse_report = rolling_eval_pass_rate(sparse_curve, profile, eval_trading_days=5)
    padded_report = rolling_eval_pass_rate(padded_curve, profile, eval_trading_days=5)

    assert padded_report["num_windows"] == sparse_report["num_windows"]
    assert padded_report["passed"] == sparse_report["passed"]
    assert padded_report["breached"] == sparse_report["breached"]
    for a, b in zip(sparse_report["results"], padded_report["results"]):
        assert a.passed == b.passed
        assert a.breached == b.breached
        assert a.days_to_target == b.days_to_target


def test_compress_equity_curve_keeps_one_point_per_flat_day_and_all_changes():
    curve = [
        (datetime(2024, 1, 2, 9, 30), 0.0),
        (datetime(2024, 1, 2, 9, 31), 0.0),
        (datetime(2024, 1, 2, 9, 32), 100.0),  # real change
        (datetime(2024, 1, 2, 9, 33), 100.0),
        (datetime(2024, 1, 3, 9, 30), 100.0),  # new day, still flat -- must be kept for day-open reference
        (datetime(2024, 1, 3, 9, 31), 100.0),
    ]
    compressed = compress_equity_curve(curve)

    assert compressed == [
        (datetime(2024, 1, 2, 9, 30), 0.0),
        (datetime(2024, 1, 2, 9, 32), 100.0),
        (datetime(2024, 1, 3, 9, 30), 100.0),
    ]
