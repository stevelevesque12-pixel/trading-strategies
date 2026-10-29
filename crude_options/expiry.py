"""
Monthly WTI option expiry calendar (CME LO / Micro WTI monthly options).

CL futures stop trading 3 business days before the 25th of the month before
delivery (4 if the 25th isn't a business day). Monthly LO options expire
three business days before that (NYMEX rulebook ch. 310, 310101.E), i.e.
around the 17th-19th. Micro WTI monthly options use a different offset (two
days before MCL futures expiry) -- pass `option_offset` accordingly.
Business days here = weekdays; exchange holidays are NOT modeled, so a
date can be off by a day around holidays.
"""

from datetime import date, timedelta
from typing import List


def _prev_bday(d: date, n: int) -> date:
    while n > 0:
        d -= timedelta(days=1)
        if d.weekday() < 5:
            n -= 1
    return d


def lo_expiry(year: int, month: int, option_offset: int = 3) -> date:
    """Expiry date of the monthly WTI option that expires in (year, month)."""
    d25 = date(year, month, 25)
    futures_ltd = _prev_bday(d25, 3 if d25.weekday() < 5 else 4)
    return _prev_bday(futures_ltd, option_offset)


def expiries_between(start: date, end: date) -> List[date]:
    out = []
    y, m = start.year, start.month
    while date(y, m, 1) <= end + timedelta(days=62):
        out.append(lo_expiry(y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return sorted(out)


def front_expiry(today: date, expiries: List[date]) -> date:
    """`expNo: 0` -- the nearest monthly expiry on or after today."""
    return next(e for e in expiries if e >= today)
