"""Load 1-minute OHLCV data and resample to higher timeframes."""

from pathlib import Path

import pandas as pd

REQUIRED_COLS = {"open", "high", "low", "close"}


def load_1m_csv(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    """
    Load 1-minute-ish OHLCV data (CSV or parquet, despite the name - see
    `sample_data/real_multi_instrument/` for real parquet datasets, some
    of which are 5-minute/15-minute native rather than true 1-minute; the
    resolution just needs to be <= whatever pair you're running).

    CSV: expects a 'timestamp' column plus open/high/low/close (volume
    optional). Column names are case-insensitive.

    Parquet: expects a tz-aware DatetimeIndex (any name) plus
    Open/High/Low/Close/Volume columns (case-insensitive) - the shape
    `data/historical/external/*.parquet` uses in the
    AI-personal-hedge-fund repo, which is where these were converted
    from. Timestamp is already the index, so there's no 'timestamp
    column missing' check to run for this branch.
    """
    if Path(path).suffix == ".parquet":
        df = pd.read_parquet(path)
        df.columns = [c.strip().lower() for c in df.columns]
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"Parquet missing required columns: {sorted(missing)}")
        df.index = pd.DatetimeIndex(df.index)  # already real timestamps - just drop whatever name/dtype quirks
        df = df.sort_index()
    else:
        df = pd.read_csv(path)
        df.columns = [c.strip().lower() for c in df.columns]

        if "timestamp" not in df.columns:
            raise ValueError("CSV must contain a 'timestamp' column")
        missing = REQUIRED_COLS - set(df.columns)
        if missing:
            raise ValueError(f"CSV missing required columns: {sorted(missing)}")

        try:
            # Naive strings, or offset-aware strings that don't cross a DST
            # boundary (a single fixed offset throughout).
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except ValueError:
            # Offset-aware strings spanning a DST boundary (mixed UTC offsets,
            # e.g. real multi-year data) -- normalize through UTC first.
            df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)

        df = df.set_index("timestamp").sort_index()

    if df.index.tz is None:
        df.index = df.index.tz_localize(tz, ambiguous="NaT", nonexistent="shift_forward")
        df = df[df.index.notna()]  # drop the nonexistent-hour row on DST "spring forward"
    else:
        df.index = df.index.tz_convert(tz)

    if "volume" not in df.columns:
        df["volume"] = 0.0

    return df[["open", "high", "low", "close", "volume"]].astype(float)


def resample_ohlc(df: pd.DataFrame, rule: str) -> pd.DataFrame:
    """Resample 1-minute bars up to `rule` (e.g. '15min', '1h', '4h')."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left").agg(agg).dropna(how="any")
