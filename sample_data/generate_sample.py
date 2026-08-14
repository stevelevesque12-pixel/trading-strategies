"""
Generates a SYNTHETIC 1-minute OHLCV CSV for smoke-testing the backtester.

This is random-walk data, not real market data -- it exists to exercise the
pipeline (loading, resampling, signal generation, fills) end to end. Do not
use it to draw any conclusion about the strategy's real edge; back-test
against real historical 1-minute futures data before trusting results.

Usage:
    python sample_data/generate_sample.py --days 10 --symbol MES --out sample_data/sample_1min.csv
"""

import argparse

import numpy as np
import pandas as pd


def generate(days: int, seed: int, start_price: float, tz: str, full_day: bool = False) -> pd.DataFrame:
    """
    full_day=False (default): RTH only, 9:30-16:00 (390 1-minute bars/day) --
    what the Failed-2s/structure-scalp backtests need.
    full_day=True: the full 24h clock, 00:00-23:59 (1440 bars/day) -- needed
    by strategies that trade off overnight/pre-market data (e.g. asian_sweep's
    20:00-00:00 ET box).
    """
    rng = np.random.default_rng(seed)
    rows = []
    price = start_price
    start_date = pd.Timestamp("2026-01-05", tz=tz)

    for d in range(days):
        session_date = start_date + pd.Timedelta(days=d)
        if session_date.dayofweek >= 5:  # skip weekends
            continue
        if full_day:
            session_start = session_date.replace(hour=0, minute=0)
            minutes = pd.date_range(session_start, periods=1440, freq="1min", tz=tz)  # 00:00-23:59
        else:
            session_start = session_date.replace(hour=9, minute=30)
            minutes = pd.date_range(session_start, periods=390, freq="1min", tz=tz)  # 9:30-16:00

        for ts in minutes:
            drift = rng.normal(0, 0.6)
            o = price
            c = price + drift
            h = max(o, c) + abs(rng.normal(0, 0.4))
            l = min(o, c) - abs(rng.normal(0, 0.4))
            v = rng.integers(50, 500)
            rows.append((ts, round(o, 2), round(h, 2), round(l, 2), round(c, 2), v))
            price = c

    return pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic 1-minute OHLCV data")
    parser.add_argument("--days", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--start-price", type=float, default=5000.0)
    parser.add_argument("--tz", default="America/New_York")
    parser.add_argument("--full-day", action="store_true", help="Generate the full 24h clock instead of RTH-only (needed for overnight-session strategies like asian_sweep)")
    parser.add_argument("--out", default="sample_data/sample_1min.csv")
    args = parser.parse_args()

    df = generate(args.days, args.seed, args.start_price, args.tz, full_day=args.full_day)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic 1-minute bars to {args.out}")


if __name__ == "__main__":
    main()
