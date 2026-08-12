"""
Fetch REAL historical 1-minute data (not synthetic) for backtesting.

Source: OANDA CFD minute bars for major indices, republished under GPL-3.0
by the FutureSharks/financial-data GitHub repo:
https://github.com/FutureSharks/financial-data

Important caveats, so you know exactly what you're backtesting against:
  - This is an index CFD (contract-for-difference) price series from a
    forex/CFD broker, NOT literal CME futures tick data for ES/MES/NQ/MNQ.
    It tracks the underlying index closely and is genuinely real market
    data, but basis, funding, and liquidity/volume characteristics differ
    from the actual futures contract you'll trade on Tradovate.
  - Coverage is 2005 through mid-2020 (this GitHub repo hasn't been updated
    since) -- it does NOT include recent years. Use it to sanity-check the
    strategy against real price action and market regimes (including the
    2018 selloff and the 2020 COVID crash), not as a substitute for a
    recent-history backtest before going live.
  - Volume is OANDA's own tick count, not real exchange contract volume.
  - Timestamps in the source are UTC; this script converts to
    America/New_York to match the session-window logic in failed2s/strategy.py.

For a real CME futures tick/minute dataset closer to present day, see
Databento (databento.com, has a free-credit trial) or FirstRate Data
(firstratedata.com) -- both require a paid/API-key account, which this
environment doesn't have configured.

Usage:
    python sample_data/fetch_real_data.py --instrument SPX500_USD --start-year 2018 --end-year 2019 --out sample_data/real_spx500_2018_2019.csv
    python sample_data/fetch_real_data.py --instrument NAS100_USD --start-year 2018 --end-year 2019 --out sample_data/real_nas100_2018_2019.csv
"""

import argparse
import subprocess
import sys
import tempfile
from pathlib import Path

import pandas as pd

REPO_URL = "https://github.com/FutureSharks/financial-data.git"


def clone_sparse(tmpdir: str, instrument: str) -> Path:
    repo_dir = Path(tmpdir) / "financial-data"
    subprocess.run(
        ["git", "clone", "--depth", "1", "--filter=blob:none", "--sparse", REPO_URL, str(repo_dir)],
        check=True,
    )
    data_path = f"pyfinancialdata/data/currencies/oanda/{instrument}"
    subprocess.run(["git", "-C", str(repo_dir), "sparse-checkout", "add", data_path], check=True)
    return repo_dir / data_path


def load_months(data_dir: Path, instrument: str, start_year: int, end_year: int) -> pd.DataFrame:
    frames = []
    for year in range(start_year, end_year + 1):
        year_dir = data_dir / str(year)
        if not year_dir.is_dir():
            print(f"  (no data for {year}, skipping)", file=sys.stderr)
            continue
        for month_file in sorted(year_dir.glob(f"oanda-{instrument}-{year}-*.csv"), key=lambda p: int(p.stem.split("-")[-1])):
            frames.append(pd.read_csv(month_file))

    if not frames:
        raise SystemExit(f"No data found for {instrument} between {start_year}-{end_year}")

    df = pd.concat(frames, ignore_index=True)
    df["timestamp"] = pd.to_datetime(df["time"], utc=True).dt.tz_convert("America/New_York")
    df = df.rename(columns={"open": "open", "high": "high", "low": "low", "close": "close", "volume": "volume"})
    df = df[["timestamp", "open", "high", "low", "close", "volume"]].sort_values("timestamp")
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch real historical 1-minute index CFD data for backtesting")
    parser.add_argument("--instrument", default="SPX500_USD", choices=["SPX500_USD", "NAS100_USD"])
    parser.add_argument("--start-year", type=int, default=2018)
    parser.add_argument("--end-year", type=int, default=2019)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory() as tmpdir:
        print(f"Cloning {REPO_URL} (sparse: {args.instrument} only)...")
        data_dir = clone_sparse(tmpdir, args.instrument)

        print(f"Loading {args.instrument} {args.start_year}-{args.end_year}...")
        df = load_months(data_dir, args.instrument, args.start_year, args.end_year)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"Wrote {len(df):,} real 1-minute bars ({df['timestamp'].min()} to {df['timestamp'].max()}) to {args.out}")


if __name__ == "__main__":
    main()
