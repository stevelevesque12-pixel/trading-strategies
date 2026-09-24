"""CLI: backtest the TR-scaled open breakout with compounding 1%-risk sizing.

Usage:
    python -m vol_breakout.backtest \
        --data sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet \
        --symbol MNQ

NQ and MNQ quote the same price, so the 10-year NQ file can be traded as
MNQ by passing --symbol MNQ (sizing/P&L use MNQ's $2/point).
"""

import argparse
import csv
from dataclasses import asdict, fields
from typing import List

import pandas as pd

from backtest.data import load_1m_csv
from failed2s.instruments import INSTRUMENTS

from .strategy import VBTrade, VolBreakoutConfig, run_backtest


def summarize(trades: List[VBTrade], start_equity: float) -> dict:
    if not trades:
        return {"num_trades": 0}
    pnl = [t.pnl_dollars for t in trades]
    wins = [p for p in pnl if p > 0]
    losses = [p for p in pnl if p <= 0]
    equity = [start_equity] + [t.equity_after for t in trades]
    peak, max_dd_pct = equity[0], 0.0
    for e in equity:
        peak = max(peak, e)
        max_dd_pct = max(max_dd_pct, (peak - e) / peak * 100)
    years = (pd.Timestamp(trades[-1].date) - pd.Timestamp(trades[0].date)).days / 365.25
    final = equity[-1]
    cagr = ((final / start_equity) ** (1 / years) - 1) * 100 if years > 0 and final > 0 else float("nan")
    reasons = pd.Series([t.exit_reason for t in trades]).value_counts().to_dict()
    return {
        "num_trades": len(trades),
        "days_traded": len({t.date for t in trades}),
        "win_rate_pct": round(len(wins) / len(trades) * 100, 2),
        "avg_r": round(sum(t.r_multiple for t in trades) / len(trades), 3),
        "profit_factor": round(sum(wins) / abs(sum(losses)), 2) if losses and sum(losses) else float("inf"),
        "final_equity": round(final, 2),
        "total_return_pct": round((final / start_equity - 1) * 100, 1),
        "cagr_pct": round(cagr, 2),
        "max_drawdown_pct": round(max_dd_pct, 2),
        "exit_reasons": reasons,
    }


def yearly_table(trades: List[VBTrade], start_equity: float) -> pd.DataFrame:
    df = pd.DataFrame([asdict(t) for t in trades])
    df["year"] = pd.to_datetime(df["date"]).dt.year
    df["equity_before"] = df["equity_after"].shift(1).fillna(start_equity)
    rows = []
    for y, g in df.groupby("year"):
        eq0, eq1 = g["equity_before"].iloc[0], g["equity_after"].iloc[-1]
        path = pd.concat([pd.Series([eq0]), g["equity_after"]]).reset_index(drop=True)
        dd = ((path.cummax() - path) / path.cummax()).max() * 100
        rows.append(
            {
                "year": y,
                "trades": len(g),
                "win_pct": round((g["pnl_dollars"] > 0).mean() * 100, 1),
                "avg_r": round(g["r_multiple"].mean(), 3),
                "avg_tr1_pts": round(g["tr1"].mean(), 1),
                "return_pct": round((eq1 / eq0 - 1) * 100, 1),
                "max_dd_pct": round(dd, 1),
            }
        )
    return pd.DataFrame(rows).set_index("year")


def write_trades(trades: List[VBTrade], path: str) -> None:
    names = [f.name for f in fields(VBTrade)]
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(names)
        for t in trades:
            row = asdict(t)
            w.writerow([round(v, 4) if isinstance(v, float) else v for v in (row[n] for n in names)])


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest the TR-scaled open breakout (vol_breakout)")
    p.add_argument("--data", required=True, help="Intraday OHLC CSV or parquet (1m/5m/15m)")
    p.add_argument("--symbol", default="MNQ", choices=list(INSTRUMENTS.keys()))
    p.add_argument("--start-equity", type=float, default=100_000.0)
    p.add_argument("--risk-pct", type=float, default=0.01)
    p.add_argument("--tr-session", choices=["eth", "rth"], default="eth")
    p.add_argument("--commission-rt", type=float, default=0.95)
    p.add_argument("--slippage-ticks", type=int, default=1)
    p.add_argument("--start", help="Only trade from this date (YYYY-MM-DD)")
    p.add_argument("--end", help="Only trade up to this date (YYYY-MM-DD)")
    p.add_argument("--out", default="vol_breakout_trades.csv")
    args = p.parse_args()

    cfg = VolBreakoutConfig(
        risk_pct=args.risk_pct,
        tr_session=args.tr_session,
        commission_rt=args.commission_rt,
        slippage_ticks=args.slippage_ticks,
    )
    inst = INSTRUMENTS[args.symbol]
    df = load_1m_csv(args.data, tz=cfg.tz)
    start = pd.Timestamp(args.start).date() if args.start else None
    end = pd.Timestamp(args.end).date() if args.end else None
    trades = run_backtest(df, inst.tick_size, inst.point_value, args.start_equity, cfg, start, end)

    write_trades(trades, args.out)
    print(f"Symbol: {args.symbol}  TR1 session: {args.tr_session}  Data: {args.data}")
    for k, v in summarize(trades, args.start_equity).items():
        print(f"  {k}: {v}")
    if trades:
        print()
        print(yearly_table(trades, args.start_equity).to_string())
    print(f"\nTrade log written to {args.out}")


if __name__ == "__main__":
    main()
