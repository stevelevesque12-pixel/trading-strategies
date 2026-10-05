"""Market data loading + session masks for the trend framework.

Datasets (all in sample_data/real_multi_instrument/):

  * es_15m  - ES 15m, 2016-05 .. 2026-08 (~10 years). MES is the same index at
              1/10th the multiplier and trades at the same price, so ES bars
              are used as the MES price series for the full-history 15m test
              (P&L is always computed at MES's $5/point). MES itself only
              launched May 2019 and the MES files here are much shorter.
  * mes_15m - MES 15m, 2025-09 .. 2026-08
  * mes_5m  - MES 5m, 2026-05 .. 2026-08
  * mes_1m  - MES 1m, 2026-08-02 .. 2026-08-25
"""

from dataclasses import dataclass
from datetime import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "sample_data" / "real_multi_instrument"

DATASETS = {
    "es_15m": ("real_es_15m_2016-05-29_2026-08-25.parquet", 15),
    "mes_15m": ("real_mes_15m_2025-09-30_2026-08-25.parquet", 15),
    "mes_5m": ("real_mes_5m_2026-05-10_2026-08-25.parquet", 5),
    "mes_1m": ("real_mes_1m_2026-08-02_2026-08-25.parquet", 1),
}

# Session windows (ET). Entries are only taken on signal bars whose *close*
# falls inside [entry_start, entry_end]; anything still open is flattened on
# the first bar whose close is >= flatten_at. Lucid requires flat by 16:45 ET;
# 15:55 leaves a margin and avoids the thin post-RTH tape.
WINDOWS = {
    "rth": (time(9, 45), time(15, 30)),
    "ny_am": (time(9, 45), time(12, 0)),
    "ext": (time(3, 0), time(15, 30)),  # London open through NY
}
FLATTEN_AT = time(15, 55)


@dataclass
class Market:
    name: str
    tf_min: int
    index: pd.DatetimeIndex  # bar open time, ET
    o: np.ndarray
    h: np.ndarray
    l: np.ndarray
    c: np.ndarray
    v: np.ndarray
    day_id: np.ndarray  # int trading-day id (session rolls at 18:00 ET)
    day_dates: np.ndarray  # date per day_id
    close_minute: np.ndarray  # minutes after midnight ET of each bar's close

    def __len__(self):
        return len(self.c)

    def entry_mask(self, window: str) -> np.ndarray:
        s, e = WINDOWS[window]
        lo, hi = s.hour * 60 + s.minute, e.hour * 60 + e.minute
        return (self.close_minute >= lo) & (self.close_minute <= hi)

    def flatten_mask(self) -> np.ndarray:
        f = FLATTEN_AT.hour * 60 + FLATTEN_AT.minute
        m = (self.close_minute >= f) & (self.close_minute < 18 * 60)
        # also flatten on the last bar of every session (data gaps, holidays)
        last = np.r_[self.day_id[1:] != self.day_id[:-1], True]
        return m | last

    def slice(self, start=None, end=None) -> "Market":
        idx = self.index
        m = np.ones(len(idx), bool)
        if start is not None:
            m &= idx >= pd.Timestamp(start, tz=idx.tz)
        if end is not None:
            m &= idx < pd.Timestamp(end, tz=idx.tz)
        return Market(self.name, self.tf_min, idx[m], self.o[m], self.h[m], self.l[m], self.c[m],
                      self.v[m], self.day_id[m], self.day_dates, self.close_minute[m])


@lru_cache(maxsize=None)
def load(name: str) -> Market:
    fname, tf = DATASETS[name]
    df = pd.read_parquet(ROOT / fname)
    df.columns = [c.lower() for c in df.columns]
    df.index = pd.DatetimeIndex(df.index).tz_convert("America/New_York")
    df = df.sort_index()
    df = df[~df.index.duplicated()]
    close_t = df.index + pd.Timedelta(minutes=tf)
    # trading day: bars opening at/after 18:00 ET belong to the next day's session
    tday = (df.index + pd.Timedelta(hours=6)).date
    codes, uniq = pd.factorize(pd.Index(tday), sort=True)
    cm = (close_t.hour * 60 + close_t.minute).to_numpy()
    return Market(
        name=name, tf_min=tf, index=df.index,
        o=df["open"].to_numpy(float), h=df["high"].to_numpy(float),
        l=df["low"].to_numpy(float), c=df["close"].to_numpy(float),
        v=df["volume"].to_numpy(float), day_id=codes.astype(np.int64),
        day_dates=np.asarray(uniq), close_minute=cm.astype(np.int64),
    )
