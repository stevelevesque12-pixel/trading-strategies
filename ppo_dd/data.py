"""Load NQ 1m data and build the arrays the simulator needs."""

from functools import lru_cache

import numpy as np
import pandas as pd

NQ_1M = "sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet"


@lru_cache(maxsize=1)
def load(path: str = NQ_1M) -> pd.DataFrame:
    df = pd.read_parquet(path).tz_convert("America/New_York")
    df.columns = [c.lower() for c in df.columns]
    return df[["open", "high", "low", "close"]].astype(float)


def ema_chart(df: pd.DataFrame, length: int) -> np.ndarray:
    """ta.ema(close, length) on the chart (1m) timeframe."""
    return df["close"].ewm(span=length, adjust=False).mean().to_numpy()


def ema_htf(df: pd.DataFrame, minutes: int, length: int) -> np.ndarray:
    """The Pine f_ema() HTF value: EMA of the last COMPLETED `minutes` bar."""
    rule = f"{minutes}min"
    htf = df["close"].resample(rule, label="left", closed="left").last().dropna()
    e = htf.ewm(span=length, adjust=False).mean().shift(1)
    return e.reindex(df.index.floor(rule)).to_numpy()


def arrays(df: pd.DataFrame):
    t_ms = (df.index.asi8 // 1_000_000).astype(np.int64)
    day = (df.index.year * 10000 + df.index.month * 100 + df.index.day).to_numpy().astype(np.int64)
    return (df["open"].to_numpy(), df["high"].to_numpy(), df["low"].to_numpy(),
            df["close"].to_numpy(), t_ms, day)
