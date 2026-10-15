"""CLI: backtest the IB breakout strategy, single config or a parameter sweep.

Usage:
    python -m ib_break.run --data sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet --symbol NQ
    python -m ib_break.run --data ... --symbol NQ --sweep

Costs: the engine fills at exact prices, so net figures here subtract
`--commission` dollars per contract round trip and `--slippage-ticks` ticks
per side (entry and exit) from every trade.
"""

import argparse
import itertools
from datetime import time

import pandas as pd

from backtest.data import load_1m_csv
from backtest.engine import BacktestEngine
from backtest.metrics import compute_metrics
from backtest.report import write_trades_csv
from failed2s.instruments import INSTRUMENTS
from failed2s.strategy import TimeframePair

from .strategy import CONFLUENCE_PRESET, IBBreakStrategy


def run_config(base: pd.DataFrame, instrument, entry_tf_minutes: int = 1, **kwargs):
    # The IB strategy ignores bias bars; a daily bias frame keeps that loop trivial.
    pair = TimeframePair(f"ib-{entry_tf_minutes}m", f"{entry_tf_minutes}min", "1D")
    strategy = IBBreakStrategy(tick_size=instrument.tick_size, entry_tf_minutes=entry_tf_minutes, **kwargs)
    engine = BacktestEngine(pair=pair, instrument=instrument, strategy=strategy)
    return engine.run(base)


def net_metrics(trades, instrument, commission: float, slippage_ticks: float) -> dict:
    m = compute_metrics(trades)
    if not trades:
        return m
    cost = commission + 2 * slippage_ticks * instrument.tick_size * instrument.point_value
    net = [t.pnl_dollars - cost * t.contracts for t in trades]
    equity = peak = max_dd = 0.0
    for p in net:
        equity += p
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    gross_win = sum(p for p in net if p > 0)
    gross_loss = -sum(p for p in net if p <= 0)
    m["net_pnl"] = round(sum(net), 2)
    m["net_pf"] = round(gross_win / gross_loss, 2) if gross_loss else float("inf")
    m["net_per_trade"] = round(sum(net) / len(net), 2)
    m["net_max_dd"] = round(max_dd, 2)
    by_year = {}
    for t, p in zip(trades, net):
        by_year[t.entry_time.year] = by_year.get(t.entry_time.year, 0.0) + p
    m["net_by_year"] = {y: round(v) for y, v in sorted(by_year.items())}
    return m


SWEEP = {
    "bias_filter": ["none", "mid", "order", "both"],
    "entry_mode": ["close", "touch"],
    "stop_mode": ["mid", "opposite"],
    "target_r": [1.0, 2.0],
    "entry_cutoff": [time(12, 0), time(15, 45)],
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Initial Balance breakout strategy")
    parser.add_argument("--data", required=True)
    parser.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--ib-minutes", type=int, default=60)
    parser.add_argument("--entry-tf", type=int, default=1, help="Entry timeframe in minutes")
    parser.add_argument("--stop-mode", default="mid", choices=["mid", "opposite", "ib_frac"])
    parser.add_argument("--target-mode", default="r", choices=["r", "ib_ext"])
    parser.add_argument("--target-r", type=float, default=1.0)
    parser.add_argument("--target-ib-ext", type=float, default=1.0)
    parser.add_argument("--entry-cutoff", default="12:00", help="HH:MM ET, no new entries at/after")
    parser.add_argument("--allow-reverse", action="store_true")
    parser.add_argument("--bias-filter", default="none", choices=["none", "mid", "order", "both"])
    parser.add_argument("--entry-mode", default="close", choices=["close", "touch"])
    parser.add_argument("--commission", type=float, default=4.5, help="$ per contract round trip")
    parser.add_argument("--slippage-ticks", type=float, default=1.0, help="ticks per side")
    parser.add_argument("--preset", action="store_true", help="Use CONFLUENCE_PRESET (overrides the strategy flags)")
    parser.add_argument("--sweep", action="store_true", help="Run the built-in parameter grid instead")
    parser.add_argument("--out", default="ib_break_trades.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    base = load_1m_csv(args.data)

    if not args.sweep:
        hh, mm = map(int, args.entry_cutoff.split(":"))
        params = dict(
            ib_minutes=args.ib_minutes, stop_mode=args.stop_mode, target_mode=args.target_mode,
            target_r=args.target_r, target_ib_ext=args.target_ib_ext,
            entry_cutoff=time(hh, mm), allow_reverse=args.allow_reverse,
            bias_filter=args.bias_filter, entry_mode=args.entry_mode,
        )
        if args.preset:
            params.update(CONFLUENCE_PRESET)
        trades = run_config(base, instrument, entry_tf_minutes=args.entry_tf, **params)
        write_trades_csv(trades, args.out)
        for k, v in net_metrics(trades, instrument, args.commission, args.slippage_ticks).items():
            print(f"  {k}: {v}")
        print(f"Trade log written to {args.out}")
        return

    keys = list(SWEEP)
    rows = []
    for combo in itertools.product(*SWEEP.values()):
        params = dict(zip(keys, combo))
        trades = run_config(base, instrument, entry_tf_minutes=args.entry_tf, **params)
        m = net_metrics(trades, instrument, args.commission, args.slippage_ticks)
        params["entry_cutoff"] = params["entry_cutoff"].strftime("%H:%M")
        rows.append({**params, **{k: m.get(k) for k in (
            "num_trades", "win_rate_pct", "avg_r", "total_pnl", "net_pnl", "net_pf",
            "net_per_trade", "net_max_dd", "net_by_year")}})
    df = pd.DataFrame(rows).sort_values("net_pnl", ascending=False)
    with pd.option_context("display.width", 250, "display.max_columns", 30, "display.max_rows", 200):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
