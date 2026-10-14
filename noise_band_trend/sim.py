"""Bar-by-bar Python mirror of the NinjaScript Lab_NoiseBandTrendV1 (noise-area breakout, VWAP stop).

Fills follow NT8 OrderFillResolution.Standard on a single 1m series: market orders fill at the next bar's
open; a working stop fills at its price, or at the open if the bar gaps through it. 1 contract.

    python -m noise_band_trend.sim [path.parquet] [--n-days 14]
"""

import argparse
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

TICK = 0.25
RTH_START, RTH_END = 570, 960
SLOTS = RTH_END - RTH_START
WINDOW_FROM, WINDOW_TO = 600, 930
FLATTEN_AT = 955
MAX_TRADES_PER_DAY = 3


@dataclass
class Trade:
    day: int
    side: int
    entry_time: pd.Timestamp
    entry: float
    exit_time: pd.Timestamp = None
    exit: float = math.nan
    reason: str = ""

    @property
    def points(self):
        return self.side * (self.exit - self.entry)


def run(df: pd.DataFrame, n_days: int = 14) -> list[Trade]:
    """df: 1m bars indexed by bar-open time in America/New_York, columns open/high/low/close/volume."""
    t = df.index
    mins = (t.hour * 60 + t.minute).to_numpy()
    tdays = (t + pd.Timedelta(hours=6)).normalize()
    day_ids = (tdays.year * 10000 + tdays.month * 100 + tdays.day).to_numpy()
    O, H, L, C = (np.round(df[k].to_numpy() / TICK) for k in ("open", "high", "low", "close"))
    V = df["volume"].to_numpy(dtype=float)
    n = len(df)
    # Bars.IsLastBarOfSession: the next bar belongs to another trading day.
    last_of_session = np.append(day_ids[1:] != day_ids[:-1], True)

    hist: list[tuple[float, np.ndarray]] = []
    closes = np.full(SLOTS, np.nan)
    cur_day = -1
    cur_open = None
    rth_day, rth_close, prev_rth_close = -1, None, None
    sum_pv = sum_v = 0.0
    vwap = None
    trades_today = 0
    flattened = False

    pos = 0  # +1 long, -1 short
    trades: list[Trade] = []
    pending = None  # ("enter", side) or ("exit", reason)
    stop = None  # ticks

    for i in range(n):
        # --- orders working into bar i ---
        filled_entry = False
        if pending is not None:
            kind, arg = pending
            pending = None
            if kind == "enter" and pos == 0:
                pos = arg
                trades.append(Trade(day_ids[i], arg, t[i], O[i] * TICK))
                filled_entry = True
            elif kind == "exit" and pos != 0:
                trades[-1].exit_time, trades[-1].exit, trades[-1].reason = t[i], O[i] * TICK, arg
                pos, stop = 0, None
        if pos != 0 and stop is not None:
            hit = (pos > 0 and L[i] <= stop) or (pos < 0 and H[i] >= stop)
            if hit:
                px = min(O[i], stop) if pos > 0 else max(O[i], stop)
                trades[-1].exit_time, trades[-1].exit, trades[-1].reason = t[i], px * TICK, "stop"
                pos, stop = 0, None

        # --- OnBarUpdate on the close of bar i ---
        m, day, c = mins[i], day_ids[i], C[i]
        if day != cur_day:
            if cur_open is not None:
                hist.append((cur_open, closes.copy()))
                if len(hist) > n_days:
                    hist.pop(0)
            closes[:] = np.nan
            cur_open = None
            cur_day = day
            sum_pv = sum_v = 0.0
            vwap = None
            trades_today = 0
            flattened = False
        if filled_entry:
            trades_today += 1  # NT8 reports the fill before OnBarUpdate of the fill bar

        sigma = None
        if RTH_START <= m < RTH_END:
            if rth_day != day:
                if rth_day != -1:
                    prev_rth_close = rth_close
                rth_day = day
            rth_close = c
            sum_pv += (H[i] + L[i] + c) / 3.0 * V[i]
            sum_v += V[i]
            if sum_v > 0:
                vwap = sum_pv / sum_v
            slot = m - RTH_START
            closes[slot] = c
            if cur_open is None:
                cur_open = O[i]
            if len(hist) >= n_days:
                vals = [hc[slot] for _, hc in hist]
                if not any(math.isnan(x) for x in vals):
                    sigma = sum(abs(x / ho - 1.0) for (ho, _), x in zip(hist, vals)) / n_days

        if not flattened and FLATTEN_AT - 1 <= m < 1020:
            flattened = True
            if pos != 0:
                pending = ("exit", "flatten")

        close_min = m + 1
        if close_min % 30 == 0 and not last_of_session[i]:
            if pos == 0 and pending is None:
                if (not flattened and WINDOW_FROM <= close_min < WINDOW_TO and trades_today < MAX_TRADES_PER_DAY
                        and sigma is not None and rth_day == day and prev_rth_close is not None):
                    upper = max(cur_open, prev_rth_close) * (1.0 + sigma)
                    lower = min(cur_open, prev_rth_close) * (1.0 - sigma)
                    if c > upper:
                        pending = ("enter", 1)
                    elif c < lower:
                        pending = ("enter", -1)
            elif pos != 0 and vwap is not None and pending is None:
                s = math.floor(vwap + 0.5)  # AwayFromZero for positive prices
                if (pos > 0 and s >= c) or (pos < 0 and s <= c):
                    pending, stop = ("exit", "vwap_mkt"), None
                else:
                    stop = s

        # IsExitOnSessionCloseStrategy: anything still open at the session's last bar goes out at its close.
        if last_of_session[i] and pos != 0:
            trades[-1].exit_time, trades[-1].exit, trades[-1].reason = t[i], c * TICK, "session_close"
            pos, stop, pending = 0, None, None
    return [tr for tr in trades if not math.isnan(tr.exit)]


def summarize(trades: list[Trade], point_value=20.0, cost_per_rt=4.5 + 2 * 0.25 * 20.0) -> pd.DataFrame:
    tdf = pd.DataFrame([{**tr.__dict__, "points": tr.points} for tr in trades])
    tdf["gross"] = tdf.points * point_value
    tdf["net"] = tdf.gross - cost_per_rt
    return tdf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?",
                    default="sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet")
    ap.add_argument("--n-days", type=int, default=14)
    a = ap.parse_args()
    df = pd.read_parquet(a.path).tz_convert("America/New_York")
    df.columns = [c.lower() for c in df.columns]
    tdf = summarize(run(df, a.n_days))
    eq = tdf.net.cumsum()
    yearly = tdf.groupby(tdf.entry_time.dt.year).agg(trades=("net", "size"), gross=("gross", "sum"), net=("net", "sum"))
    print(f"trades {len(tdf)}  long {int((tdf.side > 0).sum())}  short {int((tdf.side < 0).sum())}")
    print(f"win rate {(tdf.points > 0).mean():.1%}  avg pts {tdf.points.mean():.2f}  "
          f"gross ${tdf.gross.sum():,.0f}  net ${tdf.net.sum():,.0f}  max DD ${(eq.cummax() - eq).max():,.0f}")
    pf = tdf.net[tdf.net > 0].sum() / -tdf.net[tdf.net < 0].sum()
    print(f"profit factor (net) {pf:.2f}")
    print(tdf.groupby("reason").points.agg(["size", "mean", "sum"]).round(2))
    print(tdf.groupby("side").points.agg(["size", "mean", "sum"]).round(2))
    print(yearly.round(0))


if __name__ == "__main__":
    main()
