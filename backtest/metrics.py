"""Performance metrics for a list of closed backtest trades."""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from .engine import Trade


def compute_metrics(trades: "List[Trade]") -> dict:
    if not trades:
        return {"num_trades": 0}

    wins = [t for t in trades if t.pnl_dollars > 0]
    losses = [t for t in trades if t.pnl_dollars <= 0]

    total_pnl = sum(t.pnl_dollars for t in trades)
    gross_profit = sum(t.pnl_dollars for t in wins)
    gross_loss = abs(sum(t.pnl_dollars for t in losses))

    equity = 0.0
    peak = 0.0
    max_dd = 0.0
    for t in trades:
        equity += t.pnl_dollars
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)

    return {
        "num_trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "total_pnl": round(total_pnl, 2),
        "profit_factor": round(gross_profit / gross_loss, 2) if gross_loss else float("inf"),
        "avg_r": round(sum(t.r_multiple for t in trades) / len(trades), 3),
        "avg_win": round(gross_profit / len(wins), 2) if wins else 0.0,
        "avg_loss": round(-gross_loss / len(losses), 2) if losses else 0.0,
        "max_drawdown": round(max_dd, 2),
        "expectancy_per_trade": round(total_pnl / len(trades), 2),
    }
