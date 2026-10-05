"""Performance metrics and a Lucid 50K Flex evaluation simulator."""

from typing import Dict, List

import numpy as np
import pandas as pd

from .sim import Trade

# LucidFlex 50K evaluation (checked Oct 2026 via public rule summaries --
# confirm on lucidtrading.com before relying on it):
LUCID_50K = dict(
    profit_target=3000.0,
    max_loss=2000.0,          # end-of-day trailing
    lock_at=0.0,              # trailing MLL stops rising once it reaches the starting balance
    daily_loss_limit=1200.0,  # optional on Flex; our sim's daily_loss_stop sits well inside it
    consistency=0.50,         # largest day <= 50% of total profit at time of pass
    max_days=60,
)


def daily_pnl(trades: List[Trade], all_days) -> pd.Series:
    s = pd.Series(0.0, index=pd.Index(sorted(set(all_days))))
    for t in trades:
        s[t.trade_day] += t.pnl
    return s


def lucid_eval(days: pd.Series, rules: Dict = LUCID_50K) -> Dict:
    """
    Start a fresh evaluation on every trading day and play forward on the
    real daily P&L sequence. Returns pass/fail/timeout rates and median
    days-to-pass. EOD trailing means only closing balances matter (our own
    intraday daily_loss_stop keeps us inside the daily loss limit).
    """
    v = days.to_numpy()
    passes, fails, timeouts, days_to_pass = 0, 0, 0, []
    for s in range(len(v)):
        bal, peak, best_day, outcome = 0.0, 0.0, 0.0, None
        for k in range(s, min(len(v), s + rules["max_days"])):
            bal += v[k]
            best_day = max(best_day, v[k])
            mll = min(peak - rules["max_loss"], rules["lock_at"])
            if bal <= mll:
                outcome = "fail"
                break
            peak = max(peak, bal)
            if bal >= rules["profit_target"] and best_day <= rules["consistency"] * bal:
                outcome = "pass"
                days_to_pass.append(k - s + 1)
                break
        if outcome is None:
            if s + rules["max_days"] > len(v):
                continue  # ran out of data, not a real timeout -- don't count it
            outcome = "timeout"
        passes += outcome == "pass"
        fails += outcome == "fail"
        timeouts += outcome == "timeout"
    total = passes + fails + timeouts
    return {
        "lucid_attempts": total,
        "lucid_pass_pct": round(100 * passes / total, 1) if total else None,
        "lucid_fail_pct": round(100 * fails / total, 1) if total else None,
        "lucid_median_days": float(np.median(days_to_pass)) if days_to_pass else None,
    }


def compute(trades: List[Trade], all_days) -> Dict:
    days = daily_pnl(trades, all_days)
    out = {"trades": len(trades), "days": len(days)}
    if not trades:
        return out | {"win_rate": 0.0, "profit_factor": 0.0, "net": 0.0, "max_dd": 0.0, "avg_r": 0.0,
                      "sharpe": 0.0, "trades_per_week": 0.0, "ret_dd": 0.0} | lucid_eval(days)
    pnl = np.array([t.pnl for t in trades])
    gp, gl = pnl[pnl > 0].sum(), -pnl[pnl <= 0].sum()
    eq = np.cumsum(pnl)
    max_dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], eq]))[1:] - eq))
    deq = days.cumsum().to_numpy()
    eod_dd = float(np.max(np.maximum.accumulate(np.concatenate([[0], deq]))[1:] - deq))
    sd = days.std()
    out.update({
        "win_rate": round(100 * (pnl > 0).mean(), 1),
        "profit_factor": round(gp / gl, 2) if gl > 0 else 99.0,
        "net": round(float(pnl.sum()), 0),
        "max_dd": round(max_dd, 0),
        "eod_dd": round(eod_dd, 0),
        "avg_r": round(float(np.mean([t.r for t in trades])), 3),
        "avg_trade": round(float(pnl.mean()), 1),
        "sharpe": round(float(days.mean() / sd * np.sqrt(252)), 2) if sd > 0 else 0.0,
        "trades_per_week": round(len(trades) / max(len(days) / 5, 1), 2),
        "ret_dd": round(float(pnl.sum()) / max(max_dd, 1.0), 2),
        "best_day": round(float(days.max()), 0),
        "worst_day": round(float(days.min()), 0),
    })
    out.update(lucid_eval(days))
    return out


def equity_points(trades: List[Trade], max_points: int = 400):
    """[(iso_time, cumulative_pnl)], downsampled for the dashboard."""
    pts, eq = [], 0.0
    for t in trades:
        eq += t.pnl
        pts.append((t.exit_time.isoformat(), round(eq, 1)))
    if len(pts) > max_points:
        step = len(pts) / max_points
        pts = [pts[int(k * step)] for k in range(max_points)] + [pts[-1]]
    return pts
