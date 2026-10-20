"""Run the Morning Dip Limit reference on the 3-year NQ 1-minute file.

The rule was specified on 1-second bars; no 1-second history is available here, so this
uses the reference's own resolution=60 path (no 1 s order latency, fills/exits resolved on
1-minute bars with the adverse move first). Treat the result as a coarse approximation.

Usage: python -m morning_dip.backtest_1m [--offsets 0 60 120] [--out DIR]
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from morning_dip.morning_dip_limit import CT, Config, candles, metrics, run

DATA = Path(__file__).resolve().parent.parent / 'sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet'


def load_days(path=DATA):
    df = pd.read_parquet(path).tz_convert(CT)
    df = df.between_time('08:25', '14:59')
    out = {}
    for date, g in df.groupby(df.index.date):
        if g.index.dayofweek[0] >= 5:
            continue
        t = ((g.index - pd.Timestamp(0, tz='UTC')) // pd.Timedelta('1s')).to_numpy(np.int64)
        out[str(date)] = pd.DataFrame({'t': t, 'o': g.open.values, 'h': g.high.values,
                                       'l': g.low.values, 'c': g.close.values})
    return out


def backtest(days, cfg=Config(), offset=0):
    rows = []
    for date, raw in days.items():
        b = candles(raw, offset, cfg, resolution=60)
        f, _ = run(raw, b, date, cfg, resolution=60, offset=offset)
        rows.append(f)
    return pd.concat(rows, ignore_index=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--offsets', type=int, nargs='+', default=[0, 60, 120])
    ap.add_argument('--out', default=None)
    a = ap.parse_args()
    days = load_days()
    dates = list(days)
    for off in a.offsets:
        T = backtest(days, offset=off)
        m = metrics(T, dates)
        print(f'offset={off}', {k: (round(v, 2) if isinstance(v, float) else v) for k, v in m.items()})
        if a.out:
            Path(a.out).mkdir(parents=True, exist_ok=True)
            T.to_csv(Path(a.out) / f'trades_offset{off}.csv', index=False)
