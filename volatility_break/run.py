"""CLI entrypoint: backtest the Volatility Break strategy on M15 bars.

Usage:
    python -m volatility_break.run \
        --data sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet \
        --symbol NQ

Any finer-grained data (e.g. the NQ 1m file) is resampled to 15 minutes first.
Costs: `--commission` is dollars per contract per side; `--slippage-ticks` is
applied against you on every entry and on every stop/trailing/end-of-data
exit (targets are limit orders and fill at their price).
"""

import argparse
from typing import List, Optional

from backtest.data import load_1m_csv, resample_ohlc
from backtest.engine import Trade
from backtest.metrics import compute_metrics
from backtest.report import write_trades_csv
from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS, Instrument

from .strategy import Fill, VolatilityBreakStrategy


def to_trade(fill: Fill, instrument: Instrument, contracts: int, commission: float, slippage_ticks: float) -> Trade:
    sign = 1 if fill.direction == "long" else -1
    slip = slippage_ticks * instrument.tick_size
    entry = fill.entry_price + sign * slip
    exit_ = fill.exit_price if fill.exit_reason == "target" else fill.exit_price - sign * slip
    pnl_points = (exit_ - entry) * sign
    pnl_dollars = pnl_points * instrument.point_value * contracts - 2 * commission * contracts
    risk_points = abs(fill.entry_price - fill.stop_price)
    return Trade(
        entry_time=fill.entry_time,
        exit_time=fill.exit_time,
        direction=fill.direction,
        entry_price=entry,
        stop_price=fill.stop_price,
        target_price=fill.target_price,
        exit_price=exit_,
        exit_reason=fill.exit_reason,
        contracts=contracts,
        pnl_points=pnl_points,
        pnl_dollars=pnl_dollars,
        r_multiple=pnl_points / risk_points if risk_points else 0.0,
    )


def run_backtest(
    data_path: str,
    strategy: Optional[VolatilityBreakStrategy] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> List[Fill]:
    df = resample_ohlc(load_1m_csv(data_path), "15min")
    if start:
        df = df[df.index >= start]
    if end:
        df = df[df.index < end]
    strategy = strategy or VolatilityBreakStrategy()
    for row in df.itertuples(index=True):
        strategy.on_bar(Bar(row.Index, row.open, row.high, row.low, row.close, row.volume))
    strategy.finish()
    return strategy.fills


def main() -> None:
    parser = argparse.ArgumentParser(description="Backtest the Volatility Break (M15) strategy")
    parser.add_argument("--data", required=True, help="OHLCV CSV/parquet, 15-minute or finer")
    parser.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--contracts", type=int, default=1)
    parser.add_argument("--start", help="Inclusive start date, e.g. 2020-01-01")
    parser.add_argument("--end", help="Exclusive end date")
    parser.add_argument("--commission", type=float, default=0.0, help="USD per contract per side")
    parser.add_argument("--slippage-ticks", type=float, default=0.0)
    parser.add_argument("--trail-points", type=float, default=60.0)
    parser.add_argument("--expiry-bars", type=int, default=121)
    parser.add_argument("--out", default="volatility_break_trades.csv")
    args = parser.parse_args()

    instrument = INSTRUMENTS[args.symbol]
    strategy = VolatilityBreakStrategy(trail_points=args.trail_points, expiry_bars=args.expiry_bars)
    fills = run_backtest(args.data, strategy, args.start, args.end)
    trades = [to_trade(f, instrument, args.contracts, args.commission, args.slippage_ticks) for f in fills]

    write_trades_csv(trades, args.out)
    print(f"Symbol: {args.symbol}  Contracts: {args.contracts}  Bars: M15")
    for k, v in compute_metrics(trades).items():
        print(f"  {k}: {v}")
    print(f"Trade log written to {args.out}")


if __name__ == "__main__":
    main()
