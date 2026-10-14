"""Backtest the momentum-confluence port over 1-minute OHLCV data.

Usage:
    python -m momentum_confluence.backtest \\
        --data sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet

Fills are at the close of the bar that triggered them (what LEAN does for a
market order placed in `on_data` on minute data), plus optional per-side
commission and slippage. Prints the faithful port and the intraday RTH
variant side by side, with a per-year breakdown.
"""

import argparse
from dataclasses import dataclass
from datetime import datetime
from typing import List, Optional

import pandas as pd

from backtest.data import load_1m_csv
from failed2s.instruments import INSTRUMENTS, Instrument

from .strategy import ConfluenceConfig, Fill, MomentumConfluenceStrategy, intraday_rth_config


@dataclass
class RoundTrip:
    entry_time: datetime
    exit_time: datetime
    direction: str
    entry_price: float
    exit_price: float
    exit_reason: str
    pnl_points: float
    pnl_dollars: float  # net of costs


def run_strategy(df: pd.DataFrame, config: Optional[ConfluenceConfig] = None) -> MomentumConfluenceStrategy:
    """`df` is tz-aware 1m OHLC indexed by bar-open time."""
    strat = MomentumConfluenceStrategy(config)
    idx = df.index.tz_convert("America/New_York").tz_localize(None).to_pydatetime()
    o, h, l, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    for i in range(len(idx)):
        strat.on_bar(idx[i], o[i], h[i], l[i], c[i])
    return strat


def fills_to_trades(
    fills: List[Fill], instrument: Instrument, commission_per_side: float = 0.0, slippage_ticks: float = 0.0
) -> List[RoundTrip]:
    """Pair position changes into round trips. A reversal closes one trade and
    opens the next; each contract side costs commission + slippage."""
    side_cost = commission_per_side + slippage_ticks * instrument.tick_size * instrument.point_value
    trades: List[RoundTrip] = []
    open_fill: Optional[Fill] = None
    for f in fills:
        if f.position_before != 0 and open_fill is not None:
            pts = (f.price - open_fill.price) * f.position_before
            trades.append(
                RoundTrip(
                    open_fill.time, f.time, "long" if f.position_before > 0 else "short",
                    open_fill.price, f.price, f.reason, pts,
                    pts * instrument.point_value * abs(f.position_before) - 2 * side_cost * abs(f.position_before),
                )
            )
            open_fill = None
        if f.position_after != 0:
            open_fill = f
    return trades


def summarize(trades: List[RoundTrip]) -> dict:
    if not trades:
        return {"trades": 0}
    pnl = pd.Series([t.pnl_dollars for t in trades])
    wins, losses = pnl[pnl > 0], pnl[pnl <= 0]
    equity = pnl.cumsum()
    max_dd = (equity.cummax().clip(lower=0) - equity).max()
    daily = pd.Series(pnl.values, index=pd.DatetimeIndex([t.exit_time for t in trades]).normalize()).groupby(level=0).sum()
    sharpe = daily.mean() / daily.std() * (252 ** 0.5) if daily.std() else float("nan")
    hold = pd.Series([(t.exit_time - t.entry_time).total_seconds() / 60 for t in trades])
    return {
        "trades": len(trades),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 1),
        "total_pnl": round(pnl.sum(), 0),
        "profit_factor": round(wins.sum() / -losses.sum(), 2) if losses.sum() else float("inf"),
        "avg_trade": round(pnl.mean(), 1),
        "avg_win": round(wins.mean(), 0) if len(wins) else 0.0,
        "avg_loss": round(losses.mean(), 0) if len(losses) else 0.0,
        "max_drawdown": round(max_dd, 0),
        "daily_sharpe": round(sharpe, 2),
        "median_hold_min": round(hold.median(), 0),
    }


def by_year(trades: List[RoundTrip]) -> pd.DataFrame:
    years = sorted({t.exit_time.year for t in trades})
    rows = {y: summarize([t for t in trades if t.exit_time.year == y]) for y in years}
    return pd.DataFrame(rows).T[["trades", "win_rate_pct", "total_pnl", "profit_factor", "max_drawdown"]]


def write_trades(trades: List[RoundTrip], path: str) -> None:
    pd.DataFrame([t.__dict__ for t in trades]).to_csv(path, index=False)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", required=True)
    p.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    p.add_argument("--commission", type=float, default=2.25, help="$ per contract per side")
    p.add_argument("--slippage-ticks", type=float, default=1.0, help="ticks per contract per side")
    p.add_argument("--out-prefix", default="momentum_confluence_trades")
    args = p.parse_args()

    df = load_1m_csv(args.data)
    inst = INSTRUMENTS[args.symbol]
    variants = {"faithful": ConfluenceConfig(), "intraday_rth": intraday_rth_config()}

    rows = {}
    for name, cfg in variants.items():
        strat = run_strategy(df, cfg)
        gross = fills_to_trades(strat.fills, inst)
        net = fills_to_trades(strat.fills, inst, args.commission, args.slippage_ticks)
        rows[f"{name} (gross)"] = summarize(gross)
        rows[f"{name} (net)"] = summarize(net)
        print(f"\n== {name}: per year, net of ${args.commission}/side + {args.slippage_ticks} tick/side ==")
        print(by_year(net).to_string())
        if name == "faithful":
            print(f"weekend re-entries after the Friday 16:58 flatten: {strat.weekend_reentries}")
        write_trades(net, f"{args.out_prefix}_{name}.csv")

    print(f"\n== Summary ({args.symbol}, 1 contract, {df.index[0].date()} -> {df.index[-1].date()}) ==")
    print(pd.DataFrame(rows).T.to_string())


if __name__ == "__main__":
    main()
