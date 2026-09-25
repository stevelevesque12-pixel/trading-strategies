"""CLI: backtest the Turtle rules on one or more markets as a shared-equity portfolio.

Usage:
    python -m turtle_trading.backtest \\
        --market MES=sample_data/real_multi_instrument/real_es_15m_2016-05-29_2026-08-25.parquet \\
        --market MGC=sample_data/real_multi_instrument/real_gc_15m_2016-05-26_2026-08-25.parquet \\
        --system both --equity 100000

Each --market is SYMBOL=PATH. The data can be any resolution the repo's
loader accepts (intraday is rolled up into daily bars on the CME 18:00 ET
session boundary). The symbol only chooses the contract spec, so full-size
ES data can be traded as MES -- same price series, 1/10 the point value,
which is what makes 1%-risk unit sizing possible on a small account.
"""

import argparse
import csv
from dataclasses import asdict
from typing import Dict, List, Tuple

import pandas as pd

from backtest.data import load_1m_csv
from backtest.metrics import compute_metrics
from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS

from .strategy import Account, TurtleConfig, TurtleSystem, TurtleTrade


def load_daily(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    """Load OHLCV data and roll it up to one bar per CME session (18:00 ET -> 18:00 ET)."""
    df = load_1m_csv(path, tz=tz)
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    # Label each bar by the session date it belongs to (anything from 18:00 ET
    # on rolls to the next day, so Sunday's evening open counts as Monday).
    # Done on local wall-clock time rather than a fixed-width resample, which
    # would drift an hour across DST changes.
    session = (df.index.tz_localize(None) + pd.Timedelta(hours=6)).normalize()
    daily = df.groupby(session).agg(agg).dropna(how="any")
    daily.index = daily.index.tz_localize(tz)
    daily.index.name = "session_date"
    return daily


def run_portfolio(
    markets: List[Tuple[str, pd.DataFrame]], config: TurtleConfig, starting_equity: float
) -> Tuple[Account, Dict[str, TurtleSystem]]:
    """Step every market's daily bars in date order against one shared account."""
    account = Account(starting_equity)
    systems = {sym: TurtleSystem(INSTRUMENTS[sym], config, account) for sym, _ in markets}

    events = []
    for sym, df in markets:
        for row in df.itertuples(index=True):
            events.append((row.Index, sym, Bar(row.Index, row.open, row.high, row.low, row.close, row.volume)))
    events.sort(key=lambda e: (e[0], e[1]))

    last_bar: Dict[str, Bar] = {}
    for _, sym, bar in events:
        systems[sym].on_bar(bar)
        last_bar[sym] = bar

    for sym, system in systems.items():
        if sym in last_bar:
            system.close_open_position(last_bar[sym].close, last_bar[sym].timestamp)
    return account, systems


def equity_metrics(trades: List[TurtleTrade], starting_equity: float, start, end) -> dict:
    trades = sorted(trades, key=lambda t: t.exit_time)
    equity, peak, max_dd_pct = starting_equity, starting_equity, 0.0
    for t in trades:
        equity += t.pnl_dollars
        peak = max(peak, equity)
        max_dd_pct = max(max_dd_pct, (peak - equity) / peak * 100 if peak > 0 else 0.0)
    years = max((end - start).days / 365.25, 1e-9)
    growth = equity / starting_equity
    cagr = (growth ** (1 / years) - 1) * 100 if growth > 0 else -100.0
    return {
        "final_equity": round(equity, 2),
        "return_pct": round((growth - 1) * 100, 2),
        "cagr_pct": round(cagr, 2),
        "max_dd_pct_closed": round(max_dd_pct, 2),
    }


TRADE_FIELDS = list(TurtleTrade.__dataclass_fields__.keys())


def write_trades(trades: List[TurtleTrade], path: str) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=TRADE_FIELDS)
        writer.writeheader()
        for t in sorted(trades, key=lambda t: t.entry_time):
            writer.writerow(asdict(t))


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest Richard Dennis's Turtle Trading rules")
    parser.add_argument("--market", action="append", required=True, metavar="SYMBOL=PATH",
                        help=f"Repeatable. SYMBOL one of {', '.join(INSTRUMENTS)}")
    parser.add_argument("--system", choices=["1", "2", "both"], default="both")
    parser.add_argument("--equity", type=float, default=100_000.0)
    parser.add_argument("--risk-pct", type=float, default=1.0, help="Percent of equity per 1N move per unit")
    parser.add_argument("--max-units", type=int, default=4)
    parser.add_argument("--no-skip-filter", action="store_true", help="System 1: take every 20-day breakout")
    parser.add_argument("--slippage-ticks", type=float, default=1.0, help="Per fill, per side")
    parser.add_argument("--commission", type=float, default=2.5, help="Round-trip $ per contract")
    parser.add_argument("--out", default="turtle_trades.csv")
    args = parser.parse_args()

    markets = []
    for spec in args.market:
        sym, _, path = spec.partition("=")
        sym = sym.upper()
        if sym not in INSTRUMENTS or not path:
            parser.error(f"bad --market {spec!r}; expected SYMBOL=PATH with SYMBOL in {list(INSTRUMENTS)}")
        markets.append((sym, load_daily(path)))

    start = min(df.index[0] for _, df in markets)
    end = max(df.index[-1] for _, df in markets)
    print(f"Markets: {', '.join(f'{s} ({len(df)} days)' for s, df in markets)}")
    print(f"Period: {start.date()} -> {end.date()}   Starting equity: ${args.equity:,.0f}\n")

    systems_to_run = [1, 2] if args.system == "both" else [int(args.system)]
    all_trades: List[TurtleTrade] = []
    for sysnum in systems_to_run:
        config = TurtleConfig(
            system=sysnum,
            risk_pct=args.risk_pct / 100,
            max_units=args.max_units,
            skip_after_winner=not args.no_skip_filter,
            slippage_ticks=args.slippage_ticks,
            commission_per_contract=args.commission,
        )
        account, systems = run_portfolio(markets, config, args.equity)
        trades = [t for s in systems.values() for t in s.trades]
        all_trades.extend(trades)

        print(f"=== System {sysnum} ({config.entry_len}-day entry / {config.exit_len}-day exit) ===")
        summary = {**compute_metrics(sorted(trades, key=lambda t: t.exit_time)),
                   **equity_metrics(trades, args.equity, start, end)}
        for k, v in summary.items():
            print(f"  {k}: {v}")
        print("  per market:")
        for sym, s in systems.items():
            pnl = sum(t.pnl_dollars for t in s.trades)
            extra = f", skipped after winner={s.skipped_breakouts}" if sysnum == 1 else ""
            print(f"    {sym:4s} trades={len(s.trades):3d}  pnl=${pnl:>12,.2f}  "
                  f"too-small-to-size={s.skipped_too_small}{extra}")
        print()

    write_trades(all_trades, args.out)
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
