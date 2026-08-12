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


def generate(days: int, seed: int, start_price: float, tz: str) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    price = start_price
    start_date = pd.Timestamp("2026-01-05", tz=tz)

    for d in range(days):
        session_date = start_date + pd.Timedelta(days=d)
        if session_date.dayofweek >= 5:  # skip weekends
            continue
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
    parser.add_argument("--out", default="sample_data/sample_1min.csv")
    args = parser.parse_args()

    df = generate(args.days, args.seed, args.start_price, args.tz)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df)} synthetic 1-minute bars to {args.out}")


if __name__ == "__main__":
    main()
