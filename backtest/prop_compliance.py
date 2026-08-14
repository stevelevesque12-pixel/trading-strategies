"""
Prop-firm evaluation compliance simulation.

The question "is this strategy profitable" is the wrong one for a prop-firm
eval -- the real question is "does it hit the profit target before it
breaches the drawdown/daily-loss rule, within the eval's time budget."
That needs the account's actual intraday equity path (realized P&L so far
plus mark-to-market unrealized P&L of any open position), not just the
closed-trade P&L totals `backtest.metrics` reports.

`AccountProfile` rule sets below are **illustrative, generic profiles**
representative of common futures-prop-firm eval structures (a profit
target, a drawdown limit that's either static or trails the equity peak,
and often a daily loss limit) -- they are NOT verified current terms for
any specific real firm. Firms change these often; confirm your actual
target firm's current rule book before relying on this for a real
decision, and pass a custom AccountProfile with the real numbers.
"""

from dataclasses import dataclass
from typing import List, Optional

import pandas as pd


@dataclass(frozen=True)
class AccountProfile:
    name: str
    starting_balance: float
    profit_target: float
    max_drawdown: float
    drawdown_type: str = "trailing"  # "static" | "trailing" | "trailing_lock_at_breakeven"
    daily_loss_limit: Optional[float] = None
    daily_loss_fails_eval: bool = True  # conservative default: a breach ends the eval, not just locks the day
    min_trading_days: Optional[int] = None


# Illustrative only -- see module docstring. Roughly representative of a
# "small" and "larger" futures-prop combine shape (profit target ~ 6-8% of
# balance, drawdown ~ 4-6%, a daily loss limit on the smaller one).
GENERIC_PROFILES: List[AccountProfile] = [
    AccountProfile(
        name="Generic 50k (trailing DD, daily loss limit)",
        starting_balance=50_000,
        profit_target=3_000,
        max_drawdown=2_000,
        drawdown_type="trailing_lock_at_breakeven",
        daily_loss_limit=1_000,
    ),
    AccountProfile(
        name="Generic 150k (trailing DD, daily loss limit)",
        starting_balance=150_000,
        profit_target=9_000,
        max_drawdown=4_500,
        drawdown_type="trailing_lock_at_breakeven",
        daily_loss_limit=2_500,
    ),
    AccountProfile(
        name="Generic 50k (static DD, no daily loss limit)",
        starting_balance=50_000,
        profit_target=3_000,
        max_drawdown=2_000,
        drawdown_type="static",
        daily_loss_limit=None,
    ),
]


@dataclass
class WindowResult:
    window_start: object
    breached: bool
    breach_date: Optional[object]
    breach_type: Optional[str]  # "drawdown" | "daily_loss"
    target_hit: bool
    target_hit_date: Optional[object]
    days_to_target: Optional[int]
    passed: bool


def _drawdown_floor_series(equity: pd.Series, profile: AccountProfile) -> pd.Series:
    peak = equity.cummax().clip(lower=profile.starting_balance)
    if profile.drawdown_type == "static":
        return pd.Series(profile.starting_balance - profile.max_drawdown, index=equity.index)
    floor = peak - profile.max_drawdown
    if profile.drawdown_type == "trailing_lock_at_breakeven":
        locked = floor >= profile.starting_balance
        # Once locked, floor stays at starting_balance forever after, even if peak dips later.
        first_lock = locked.idxmax() if locked.any() else None
        if first_lock is not None and locked.loc[first_lock]:
            floor.loc[first_lock:] = profile.starting_balance
    return floor


def evaluate_window(day_equity: pd.Series, day_of: pd.Series, profile: AccountProfile) -> WindowResult:
    """
    day_equity: account equity (starting_balance-based) indexed by timestamp,
    covering exactly the candidate eval window (already sliced by the caller).
    day_of: same index, values = trading-day date for each bar (for daily-loss-limit resets).
    """
    floor = _drawdown_floor_series(day_equity, profile)
    dd_breach = day_equity <= floor

    daily_breach = pd.Series(False, index=day_equity.index)
    if profile.daily_loss_limit is not None:
        day_open_equity = day_equity.groupby(day_of).transform("first")
        daily_breach = day_equity <= (day_open_equity - profile.daily_loss_limit)

    target_hit = day_equity >= (profile.starting_balance + profile.profit_target)

    dd_breach_idx = dd_breach.idxmax() if dd_breach.any() else None
    daily_breach_idx = daily_breach.idxmax() if daily_breach.any() else None
    target_idx = target_hit.idxmax() if target_hit.any() else None

    breach_candidates = []
    if dd_breach_idx is not None:
        breach_candidates.append((dd_breach_idx, "drawdown"))
    if daily_breach_idx is not None and profile.daily_loss_fails_eval:
        breach_candidates.append((daily_breach_idx, "daily_loss"))
    breach_candidates.sort(key=lambda x: x[0])
    first_breach = breach_candidates[0] if breach_candidates else (None, None)

    breach_ts, breach_type = first_breach
    breached = breach_ts is not None
    target_hit_bool = target_idx is not None and (not breached or target_idx <= breach_ts)

    days_to_target = None
    if target_hit_bool:
        days_to_target = day_of.loc[:target_idx].nunique()

    window_start_day = day_of.iloc[0]
    passed = target_hit_bool and (
        profile.min_trading_days is None
        or day_of.loc[:target_idx].nunique() >= min(profile.min_trading_days, day_of.nunique())
    )

    return WindowResult(
        window_start=window_start_day,
        breached=breached and not target_hit_bool,
        breach_date=breach_ts if (breached and not target_hit_bool) else None,
        breach_type=breach_type if (breached and not target_hit_bool) else None,
        target_hit=target_hit_bool,
        target_hit_date=target_idx if target_hit_bool else None,
        days_to_target=days_to_target,
        passed=passed,
    )


def compress_equity_curve(equity_curve: List[tuple]) -> List[tuple]:
    """
    Collapses flat stretches (bars where equity hasn't moved since the last
    kept point -- true for the large majority of the day this strategy
    holds no position, e.g. overnight/outside the trading windows) down to
    their endpoints, while always keeping the first bar of every trading
    day (so a day-open equity reference for the daily-loss-limit check is
    never lost, even across a fully flat day). Lossless for the compliance
    checks in this module: peak/floor/target/breach and days-to-target only
    ever depend on the *values* equity takes and which day each value fell
    on, never on how many redundant bars a flat stretch lasted -- but it cuts
    a ~1,400-bars/trading-day raw curve down to just its handful of real
    change points per day, which is what makes the eval-day sweep below
    tractable (O(bars-with-an-open-position) instead of O(all bars) per
    window).
    """
    if not equity_curve:
        return []
    compressed = [equity_curve[0]]
    last_value = equity_curve[0][1]
    last_day = equity_curve[0][0].date()
    for ts, eq in equity_curve[1:]:
        day = ts.date()
        if day != last_day or eq != last_value:
            compressed.append((ts, eq))
            last_day = day
            last_value = eq
    return compressed


def rolling_eval_pass_rate(
    equity_curve: List[tuple],
    profile: AccountProfile,
    eval_trading_days: int = 10,
) -> dict:
    """
    Slides an `eval_trading_days`-long window over every possible start day
    in the equity curve and asks, for each: would this eval have passed,
    breached, or run out the clock? Returns aggregate stats plus the
    per-window results.
    """
    import numpy as np

    equity_curve = compress_equity_curve(equity_curve)
    ts, cum_pnl = zip(*equity_curve)
    s = pd.Series(cum_pnl, index=pd.DatetimeIndex(ts))
    day_of_arr = np.array([d for d in s.index.date])  # sorted, same order as s (s.index is time-sorted)
    trading_days = np.array(sorted(set(day_of_arr)))

    # Position (not label) boundaries per trading day, via searchsorted on the
    # sorted day array -- avoids an O(n) isin() mask over the full series per
    # window (750+ windows x 1M+ bars would otherwise dominate runtime).
    day_start_pos = np.searchsorted(day_of_arr, trading_days)

    results: List[WindowResult] = []
    n_days = len(trading_days)
    for i in range(n_days - eval_trading_days + 1):
        start_pos = day_start_pos[i]
        end_pos = day_start_pos[i + eval_trading_days] if i + eval_trading_days < n_days else len(s)

        window_pnl = s.iloc[start_pos:end_pos]
        window_day_of = pd.Series(day_of_arr[start_pos:end_pos], index=window_pnl.index)

        baseline = s.iloc[start_pos - 1] if start_pos > 0 else 0.0
        account_equity = profile.starting_balance + (window_pnl - baseline)

        results.append(evaluate_window(account_equity, window_day_of, profile))

    n = len(results)
    passed = sum(1 for r in results if r.passed)
    breached = sum(1 for r in results if r.breached)
    neither = n - passed - breached
    days_to_pass = [r.days_to_target for r in results if r.passed and r.days_to_target is not None]

    return {
        "profile": profile.name,
        "eval_trading_days": eval_trading_days,
        "num_windows": n,
        "passed": passed,
        "pass_rate_pct": round(passed / n * 100, 1) if n else 0.0,
        "breached": breached,
        "breach_rate_pct": round(breached / n * 100, 1) if n else 0.0,
        "neither_pct": round(neither / n * 100, 1) if n else 0.0,
        "avg_days_to_pass": round(sum(days_to_pass) / len(days_to_pass), 1) if days_to_pass else None,
        "results": results,
    }
