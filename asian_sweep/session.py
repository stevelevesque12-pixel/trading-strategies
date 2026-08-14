"""
Asian-box session windows and trading-day bookkeeping for the overnight-range
liquidity-sweep strategy.

Trading-day convention: the Asian box (20:00-00:00 America/New_York, by
default) is built from the *previous* evening's bars but belongs to the
trading day that follows it -- the day its 00:00 close falls on. So a bar's
`trading_date` rolls forward to the next calendar date as soon as its
time-of-day reaches `box_start`, matching the date every other bar later
that same trading day (pre-market, RTH) already carries.
"""

from dataclasses import dataclass
from datetime import date as date_cls, time, timedelta


@dataclass
class SessionConfig:
    box_start: time = time(20, 0)  # Asian box open (prior evening)
    box_end: time = time(0, 0)  # Asian box close / sweep window opens
    sweep_end: time = time(9, 30)  # London/pre-market sweep window closes (RTH open)
    no_entry_after: time = time(11, 0)  # stop arming new retests after this
    flatten_at: time = time(15, 55)
    tz: str = "America/New_York"


@dataclass
class AsianBox:
    trading_date: date_cls
    high: float
    low: float


def trading_date(ts, session: SessionConfig) -> date_cls:
    d = ts.date()
    if ts.time() >= session.box_start:
        return d + timedelta(days=1)
    return d
