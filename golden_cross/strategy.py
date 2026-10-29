"""Golden-cross gold longs (port of the `GoldenCrossGoldm` platform strategy).

Rules, unchanged from the original:
- Once per day, look at completed daily closes *before* today.
- Flat -> long when SMA(50) > SMA(200) and the last close > SMA(50).
- Long -> flat when SMA(50) < SMA(200).
- Long only, one contract, no stop or target. Needs 260 prior daily closes.

The original decides at 15:20 IST on MCX GOLDM and fills at that moment's
price. GC on CME has no equivalent bar, so the fill time is a parameter
(`fill_time`, ET); default 05:45 ET is the 15m bar holding 15:20 IST in summer.
The original also rolls the future 3 days before expiry; this runs on a
continuous contract, so roll costs are not modelled beyond the per-trade cost.
"""

import datetime as dt
from dataclasses import dataclass
from typing import List, Optional

import pandas as pd

FAST = 50
SLOW = 200
MIN_HISTORY = 260


@dataclass
class Trade:
    entry_date: dt.date
    entry_price: float
    exit_date: Optional[dt.date] = None
    exit_price: Optional[float] = None
    points: Optional[float] = None
    pnl_dollars: Optional[float] = None
    days_held: Optional[int] = None


def daily_bars(intraday: pd.DataFrame, session_start_hour: int = 18) -> pd.DataFrame:
    """CME-style daily bars: a session that opens at 18:00 ET belongs to the next date."""
    ts = intraday.index.tz_convert("America/New_York")
    trade_date = (ts + pd.Timedelta(hours=24 - session_start_hour)).date
    g = intraday.groupby(trade_date)
    out = pd.DataFrame({
        "open": g["open"].first(),
        "high": g["high"].max(),
        "low": g["low"].min(),
        "close": g["close"].last(),
    })
    out.index = pd.Index(out.index, name="date")
    return out


def fill_prices(intraday: pd.DataFrame, fill_time: dt.time, session_start_hour: int = 18) -> pd.Series:
    """Per trade date, the close of the first bar at/after `fill_time` ET (falls back to the session's last close)."""
    ts = intraday.index.tz_convert("America/New_York")
    trade_date = (ts + pd.Timedelta(hours=24 - session_start_hour)).date
    df = pd.DataFrame({"close": intraday["close"].values, "date": trade_date, "t": ts.time}, index=ts)
    # Order bars inside a session from its 18:00 start, so "at/after fill_time" means within the same session.
    after = df[[_session_minutes(t, session_start_hour) >= _session_minutes(fill_time, session_start_hour) for t in df["t"]]]
    first_after = after.groupby("date")["close"].first()
    last = df.groupby("date")["close"].last()
    return first_after.reindex(last.index).fillna(last)


def _session_minutes(t: dt.time, session_start_hour: int) -> int:
    return ((t.hour - session_start_hour) % 24) * 60 + t.minute


def target(closes: List[float], pos: int) -> int:
    """The original `target()`: returns 1 (long) or 0 (flat)."""
    f = sum(closes[-FAST:]) / FAST
    s = sum(closes[-SLOW:]) / SLOW
    if pos == 0:
        return 1 if (f > s and closes[-1] > f) else 0
    return 0 if f < s else pos


def run(daily: pd.DataFrame, fills: pd.Series, point_value: float = 100.0,
        cost_per_round_turn: float = 0.0) -> List[Trade]:
    """Walk day by day; decide from closes before today, fill at today's fill price."""
    closes = daily["close"].tolist()
    dates = list(daily.index)
    trades: List[Trade] = []
    pos = 0
    for i, day in enumerate(dates):
        if i < MIN_HISTORY:
            continue
        want = target(closes[:i], pos)
        px = float(fills.loc[day])
        if want != pos:
            if pos != 0:
                _close(trades[-1], day, px, point_value, cost_per_round_turn, dates)
            if want != 0:
                trades.append(Trade(entry_date=day, entry_price=px))
            pos = want
    if pos != 0:  # mark the open trade at the last close
        _close(trades[-1], dates[-1], closes[-1], point_value, cost_per_round_turn, dates)
    return trades


def _close(t: Trade, day, px, point_value, cost, dates):
    t.exit_date = day
    t.exit_price = px
    t.points = px - t.entry_price
    t.pnl_dollars = t.points * point_value - cost
    t.days_held = dates.index(day) - dates.index(t.entry_date)
