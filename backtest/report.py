"""CSV output helpers shared by the single-pair and comparison backtest CLIs."""

import csv
from typing import List

from .engine import Trade

TRADE_FIELDS = [
    "entry_time", "exit_time", "direction", "entry_price", "stop_price",
    "target_price", "exit_price", "exit_reason", "contracts",
    "pnl_points", "pnl_dollars", "r_multiple",
]

METRIC_FIELDS = [
    "pair", "num_trades", "win_rate_pct", "total_pnl", "profit_factor",
    "avg_r", "avg_win", "avg_loss", "max_drawdown", "expectancy_per_trade",
]


def write_trades_csv(trades: List[Trade], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(TRADE_FIELDS)
        for t in trades:
            writer.writerow(
                [
                    t.entry_time, t.exit_time, t.direction, t.entry_price, t.stop_price,
                    t.target_price, t.exit_price, t.exit_reason, t.contracts,
                    round(t.pnl_points, 4), round(t.pnl_dollars, 2), round(t.r_multiple, 3),
                ]
            )


def write_comparison_csv(rows: List[dict], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRIC_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in METRIC_FIELDS})


def print_comparison_table(rows: List[dict]) -> None:
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in rows)) for k in METRIC_FIELDS}
    header = "  ".join(k.ljust(widths[k]) for k in METRIC_FIELDS)
    print(header)
    print("-" * len(header))
    for row in rows:
        print("  ".join(str(row.get(k, "")).ljust(widths[k]) for k in METRIC_FIELDS))
