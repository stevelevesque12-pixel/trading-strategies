"""
Fetch recent real intraday data from Yahoo Finance (via yfinance).

This REQUIRES an environment with outbound access to Yahoo Finance --
it will NOT work from this sandboxed backtest environment (blocked by
egress policy). Built for Google Colab or your own machine; see
COLAB.md for copy-paste cells.

Yahoo's intraday history limits (enforced by Yahoo, not this script):
  - interval=1m            -> only the last 7 days
  - interval=2m/5m/15m/30m -> only the last 60 days
  - interval=60m/1h        -> up to ~730 days
This caps how far back each of the three timeframe pairs can be tested:
  - 1m-15m needs true 1-minute bars -> max ~7 days of real data
  - 5m-1h and 15m-4h only need 5-minute (or 15-minute) base bars -> up to
    ~60 days, fetched directly at that resolution (no need for 1-minute
    data for these two pairs).

Usage (in Colab, after `!pip install -q yfinance`):
    python sample_data/fetch_yfinance.py --symbol MES=F --interval 1m --period 7d --out sample_data/yf_mes_1m.csv
    python sample_data/fetch_yfinance.py --symbol MES=F --interval 5m --period 60d --out sample_data/yf_mes_5m.csv
"""

import argparse
import sys

import pandas as pd

MAX_DAYS_BY_INTERVAL = {
    "1m": 7,
    "2m": 60, "5m": 60, "15m": 60, "30m": 60,
    "60m": 730, "1h": 730,
}


def fetch(symbol: str, interval: str, period: str) -> pd.DataFrame:
    import yfinance as yf

    requested_days = int(period.rstrip("d")) if period.endswith("d") else None
    cap = MAX_DAYS_BY_INTERVAL.get(interval)
    if cap and requested_days and requested_days > cap:
        print(f"WARNING: Yahoo caps interval={interval} history at {cap}d; requested {requested_days}d "
              f"will likely be truncated by Yahoo itself.", file=sys.stderr)

    df = yf.download(symbol, interval=interval, period=period, progress=False, auto_adjust=False)
    if df.empty:
        raise SystemExit(f"No data returned for {symbol} interval={interval} period={period}")

    if isinstance(df.columns, pd.MultiIndex):
        # yfinance sometimes returns a (Field, Ticker) MultiIndex even for a single ticker.
        if "Close" in df.columns.get_level_values(0):
            df.columns = df.columns.get_level_values(0)
        else:
            df.columns = df.columns.get_level_values(-1)

    df = df.rename(columns={"Open": "open", "High": "high", "Low": "low", "Close": "close", "Volume": "volume"})
    df.index.name = "timestamp"
    df = df.reset_index()

    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    df["timestamp"] = df["timestamp"].dt.tz_convert("America/New_York")

    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real intraday OHLCV data from Yahoo Finance")
    parser.add_argument("--symbol", default="MES=F", help="Yahoo ticker, e.g. MES=F, ES=F, NQ=F, MNQ=F, CL=F, GC=F")
    parser.add_argument("--interval", default="5m", choices=list(MAX_DAYS_BY_INTERVAL.keys()))
    parser.add_argument("--period", default="60d", help="e.g. 7d, 60d (Yahoo will cap this per --interval)")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    df = fetch(args.symbol, args.interval, args.period)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} real bars ({df['timestamp'].min()} to {df['timestamp'].max()}) to {args.out}")


if __name__ == "__main__":
    main()
