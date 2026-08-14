"""Load 1-minute OHLCV data and resample to higher timeframes."""

import pandas as pd

REQUIRED_COLS = {"open", "high", "low", "close"}


def load_1m_csv(path: str, tz: str = "America/New_York") -> pd.DataFrame:
    """
    Load a 1-minute OHLCV CSV. Expects a 'timestamp' column plus
    open/high/low/close (volume optional). Column names are case-insensitive.
    """
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


def resample_ohlc(df: pd.DataFrame, rule: str, origin: str = "start_day") -> pd.DataFrame:
    """
    Resample 1-minute bars up to `rule` (e.g. '15min', '1h', '4h').

    `origin` is passed straight through to `DataFrame.resample` -- the
    default `"start_day"` floors to local-midnight-aligned boundaries (pandas
    default). Pass a tz-aware `Timestamp` to anchor bins elsewhere, e.g. to
    get 4-hour bars that close exactly at 10:00 in the index's tz instead of
    at UTC-clock-hour boundaries (see `lab_model.zones.four_hour_origin`).
    """
    agg = {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    return df.resample(rule, label="left", closed="left", origin=origin).agg(agg).dropna(how="any")
