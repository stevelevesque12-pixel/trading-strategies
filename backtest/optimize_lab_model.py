"""
CLI: grid-search the Lab Model's stop-loss and target-related parameters and
report the best combinations.

There's no free "target multiplier" to sweep here the way `failed2s` has
`--target-r` -- the Lab Model's target is always the LLT, a structural price
level, not an R-multiple. The parameters that actually shape stop/target
outcomes are:

  - `--stop-buffer-ticks`: extra room beyond the swept/recent H/L before the
    stop sits (too tight -> stopped out by noise on the same move that would
    have reversed; too wide -> gives back edge on losers).
  - `--exec-swing-strength`: the fractal window (in bars) used to confirm
    swing points on the execution timeframe. This directly controls the LLT
    (target) -- a smaller value confirms swings faster/closer, giving nearer
    (lower R, higher hit-rate) targets; a larger value waits for bigger
    swings, giving farther (higher R, lower hit-rate) targets. It also
    changes which swings feed the SMT divergence check.
  - `--breakeven-at-r`: trade management, not the initial stop/target, but
    it changes realized R on winners that pull back, so it's swept alongside
    the other two rather than left fixed.

Loads the NQ+ES data once (the expensive part) and reuses it across every
combination via `LabModelEngine.run_data`.

Usage:
    python -m backtest.optimize_lab_model --nq-data path/to/nq.csv --es-data path/to/es.csv
    python -m backtest.optimize_lab_model --nq-data ... --es-data ... --stop-buffer-ticks 0,4,8 --breakeven-at-r none,0.5,1.0
"""

import argparse
import math

from failed2s.instruments import INSTRUMENTS
from lab_model.strategy import LabModelStrategy

from .lab_model_engine import LabModelEngine
from .metrics import compute_metrics
from .report import write_comparison_csv

DEFAULT_STOP_BUFFER_TICKS = [0, 2, 4, 8, 12, 16]
DEFAULT_BREAKEVEN_AT_R = ["none", 0.25, 0.5, 0.75, 1.0]
DEFAULT_EXEC_SWING_STRENGTH = [1, 2, 3]

OPTIMIZE_METRIC_FIELDS = [
    "stop_buffer_ticks", "exec_swing_strength", "breakeven_at_r",
    "num_trades", "win_rate_pct", "total_pnl", "profit_factor",
    "avg_r", "avg_win", "avg_loss", "max_drawdown", "expectancy_per_trade",
]


def _parse_breakeven(raw: str):
    return None if raw.strip().lower() == "none" else float(raw)


def _sort_key(row: dict):
    # Profit factor is undefined (inf) with zero losing trades; treat that as
    # "best" rather than crashing the sort, but still rank by expectancy
    # underneath it so two undefined-profit-factor rows don't tie arbitrarily.
    pf = row.get("profit_factor", 0.0)
    pf = pf if math.isfinite(pf) else float("inf")
    return (pf, row.get("expectancy_per_trade", 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Grid-search the Lab Model's stop-loss/target-related parameters")
    parser.add_argument("--nq-data", required=True)
    parser.add_argument("--es-data", required=True)
    parser.add_argument("--execution-tf", default="1min", choices=["1min", "3min", "5min"])
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--stop-buffer-ticks", default=",".join(str(v) for v in DEFAULT_STOP_BUFFER_TICKS))
    parser.add_argument("--breakeven-at-r", default=",".join(str(v) for v in DEFAULT_BREAKEVEN_AT_R), help="'none' disables breakeven management")
    parser.add_argument("--exec-swing-strength", default=",".join(str(v) for v in DEFAULT_EXEC_SWING_STRENGTH))
    parser.add_argument("--commission-per-contract", type=float, default=4.60, help="Round-turn commission per contract, deducted per trade; pass 0 to disable")
    parser.add_argument("--slippage-ticks", type=float, default=1.0, help="Adverse slippage (in ticks) applied to entries, stop exits, and forced market closes -- not target exits; pass 0 to disable")
    parser.add_argument("--min-trades", type=int, default=10, help="Drop combinations with fewer trades than this (too noisy to rank)")
    parser.add_argument("--top", type=int, default=15, help="How many best combinations to print")
    parser.add_argument("--out", default="lab_model_stop_target_sweep.csv")
    args = parser.parse_args()

    stop_buffer_ticks = [int(v) for v in args.stop_buffer_ticks.split(",")]
    breakeven_at_rs = [_parse_breakeven(v) for v in args.breakeven_at_r.split(",")]
    exec_swing_strengths = [int(v) for v in args.exec_swing_strength.split(",")]

    instrument = INSTRUMENTS["NQ"]
    print(f"Loading data (execution TF: {args.execution_tf})...")
    data = LabModelEngine.load(args.nq_data, args.es_data, execution_tf=args.execution_tf)

    total = len(stop_buffer_ticks) * len(breakeven_at_rs) * len(exec_swing_strengths)
    print(f"Running {total} combinations...")

    rows = []
    for sbt in stop_buffer_ticks:
        for ess in exec_swing_strengths:
            for be in breakeven_at_rs:
                # A fresh strategy instance per combination -- LabModelStrategy is stateful
                # (zone/swing history, pending setups, reference candles), so reusing one
                # across runs would let a later run start from a prior run's leftover state.
                strategy = LabModelStrategy(tick_size=instrument.tick_size, stop_buffer_ticks=sbt, exec_swing_strength=ess)
                engine = LabModelEngine(
                    strategy=strategy, instrument=instrument, execution_tf=args.execution_tf, contracts=args.contracts,
                    breakeven_at_r=be, commission_per_contract=args.commission_per_contract, slippage_ticks=args.slippage_ticks,
                )
                trades = engine.run_data(data)

                metrics = compute_metrics(trades)
                metrics["stop_buffer_ticks"] = sbt
                metrics["exec_swing_strength"] = ess
                metrics["breakeven_at_r"] = "none" if be is None else be
                rows.append(metrics)

    write_comparison_csv(rows, args.out, fields=OPTIMIZE_METRIC_FIELDS)

    ranked = [r for r in rows if r.get("num_trades", 0) >= args.min_trades]
    ranked.sort(key=_sort_key, reverse=True)

    print(f"\nTop {min(args.top, len(ranked))} combinations by profit factor (min {args.min_trades} trades):\n")
    widths = {k: max(len(k), *(len(str(r.get(k, ""))) for r in ranked)) for k in OPTIMIZE_METRIC_FIELDS} if ranked else {}
    if ranked:
        header = "  ".join(k.ljust(widths[k]) for k in OPTIMIZE_METRIC_FIELDS)
        print(header)
        print("-" * len(header))
        for row in ranked[: args.top]:
            print("  ".join(str(row.get(k, "")).ljust(widths[k]) for k in OPTIMIZE_METRIC_FIELDS))
    else:
        print(f"(no combination reached --min-trades {args.min_trades})")

    dropped = len(rows) - len(ranked)
    if dropped:
        print(f"\n({dropped} combination(s) dropped for fewer than {args.min_trades} trades)")
    print(f"\nFull sweep written to {args.out}")


if __name__ == "__main__":
    main()
