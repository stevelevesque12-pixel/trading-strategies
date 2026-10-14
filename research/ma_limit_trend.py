"""ma_limit on MCL 30m (TradingView export, Jan 2025 - Oct 2026): does an SMA trend filter help when flat by 16:55?
Longs only above the trend SMA, shorts only below. Compared on the window where every filter is warmed up.
  python -m research.ma_limit_trend
"""
import pandas as pd

from failed2s.instruments import INSTRUMENTS
from ma_limit.strategy import Params, backtest

PATH = "sample_data/real_multi_instrument/tv_mcl1_30m_2025-01-02_2026-10.parquet"
inst = INSTRUMENTS["MCL"]
bars = pd.read_parquet(PATH)[["open", "high", "low", "close"]]
BASE = dict(expiry_bars=2, rr=2.0, atr_mult=2.0, hours=("10:00", "15:00"), live_mode="realtime",
            tick_size=inst.tick_size, point_value=inst.point_value, commission_rt=inst.round_trip_fee)
FILTERS = {"none": {}, "200d": dict(trend_len=200), "100d": dict(trend_len=100), "50d": dict(trend_len=50),
           "20d": dict(trend_len=20), "200x30m": dict(trend_len=200, trend_tf=None)}
FLATS = {"overnight": None, "16:55": "16:55"}


def stats(df):
    if df.empty:
        return dict(n=0)
    pnl = df.pnl_dollars
    w, l = pnl[pnl > 0], pnl[pnl <= 0]
    eq = pnl.cumsum()
    return dict(n=len(df), win=round(100 * (pnl > 0).mean(), 1), net=round(pnl.sum()), per=round(pnl.mean(), 1),
                pf=round(w.sum() / -l.sum(), 2), dd=round((eq.cummax().clip(lower=0) - eq).max()),
                longs=int((df.direction == 1).sum()), shorts=int((df.direction == -1).sum()))


if __name__ == "__main__":
    start = pd.Timestamp("2025-10-20", tz="America/New_York")  # 200 sessions after the data starts
    rows = []
    for fk, fl in FLATS.items():
        for tk, tf in FILTERS.items():
            t = backtest(bars, "30min", Params(**BASE, flatten_at=fl, **tf))
            df = pd.DataFrame([vars(x) for x in t])
            full = stats(df)
            late = stats(df[df.entry_time >= start])
            rows.append(dict(flat=fk, trend=tk) | {f"all_{k}": v for k, v in full.items()}
                        | {f"oct25_{k}": v for k, v in late.items()})
    out = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(out.to_string(index=False))
