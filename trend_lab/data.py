"""Load the repo's real MCL datasets into America/New_York bars with session metadata."""

from functools import lru_cache
from pathlib import Path

import pandas as pd

from backtest.data import load_1m_csv, resample_ohlc

DATA_DIR = Path(__file__).resolve().parent.parent / "sample_data" / "real_multi_instrument"

# Native files per timeframe. 1m covers ~3.5 weeks, 5m ~3.5 months, 15m ~11 months.
NATIVE_FILES = {
    "1min": "real_mcl_1m_2026-08-02_2026-08-25.parquet",
    "5min": "real_mcl_5m_2026-05-10_2026-08-25.parquet",
    "15min": "real_mcl_15m_2025-09-30_2026-08-25.parquet",
}


@lru_cache(maxsize=None)
def load_bars(tf: str, source_tf: str = None) -> pd.DataFrame:
    """
    Bars at `tf`, built from the native file `source_tf` (default: the native
    file for `tf`). E.g. load_bars("3min", "1min") or load_bars("10min", "5min").

    Adds `trade_day`: the CME trading day a bar belongs to (the session that
    opens 18:00 ET belongs to the next calendar date), which is also the day
    Lucid's end-of-day drawdown is evaluated on (17:00 ET close).
    """
    source_tf = source_tf or tf
    df = load_1m_csv(str(DATA_DIR / NATIVE_FILES[source_tf]))
    if tf != source_tf:
        df = resample_ohlc(df, tf)
    df = df.copy()
    df["trade_day"] = (df.index + pd.Timedelta(hours=7)).date
    return df
