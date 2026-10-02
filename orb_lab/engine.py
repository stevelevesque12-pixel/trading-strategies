"""Opening-range breakout replay on 5-minute bars.

Entry rules reverse-engineered from the PropQuantX MNQ trade list (they
reproduce 75 of its 84 May-Aug 2026 entries exactly):
  * opening range = first 20 minutes from the session open (09:30-09:50 ET
    for index futures),
  * a 5-minute CLOSE beyond the range high (long) / low (short) triggers,
  * fill at the next bar's open,
  * at most one long and one short per day, one position at a time,
  * no new entries after `last_entry` (12:00 ET for index futures).

Exits are modular. Within a bar the stop is assumed to hit before the target
(conservative: 5-minute bars cannot tell which came first).
"""
from dataclasses import dataclass

from pathlib import Path

import pandas as pd

BARS = Path(__file__).resolve().parent.parent / "sample_data" / "bars"


@dataclass
class Market:
    name: str
    path: Path
    point_value: float      # $ per point per micro contract
    tick: float
    open_hm: str = "09:30"  # session open used for the opening range (ET)
    last_entry: str = "12:00"
    flat_hm: str = "15:55"  # force exit at this bar's close


def load_bars(m: Market):
    b = pd.read_csv(m.path)
    b["t"] = pd.to_datetime(b.time, unit="s", utc=True).dt.tz_convert("America/New_York").dt.tz_localize(None)
    b["day"] = b.t.dt.date
    b["hm"] = b.t.dt.strftime("%H:%M")
    return b


def _add(hm, minutes):
    t = pd.Timestamp("2000-01-01 " + hm) + pd.Timedelta(minutes=minutes)
    return t.strftime("%H:%M")


def signals(b, m: Market, or_minutes=20):
    """Yield (day, side, entry_bar_index, or_high, or_low) using the reverse-engineered entry."""
    or_end = _add(m.open_hm, or_minutes)
    out = []
    for day, x in b.groupby("day"):
        if pd.Timestamp(day).weekday() >= 5:
            continue
        o = x[(x.hm >= m.open_hm) & (x.hm < or_end)]
        if len(o) < or_minutes // 5:
            continue
        hi, lo = o.high.max(), o.low.min()
        post = x[(x.hm >= or_end) & (x.hm < m.last_entry)]
        idx = list(x.index)
        for side in (1, -1):
            for i, r in post.iterrows():
                if (side == 1 and r.close > hi) or (side == -1 and r.close < lo):
                    j = idx.index(i) + 1
                    if j < len(idx) and x.loc[idx[j], "hm"] < m.last_entry:
                        out.append((day, side, idx[j], hi, lo))
                    break
    return sorted(out, key=lambda s: s[2])


def simulate(b, m: Market, exit_rule, or_minutes=20, sigs=None):
    """exit_rule(ctx) -> dict(stop=..., target=..., be_after=..., be_to=..., max_bars=...)
    in POINTS from entry. Returns DataFrame of trades with pts."""
    sigs = sigs if sigs is not None else signals(b, m, or_minutes)
    H, L, O, C, HM, DAY = (b[c].to_numpy() for c in ("high", "low", "open", "close", "hm", "day"))
    pos = {i: k for k, i in enumerate(b.index)}
    trades, busy_until = [], -1
    for day, side, i, hi, lo in sigs:
        k = pos[i]
        if k <= busy_until:
            continue
        entry = O[k]
        p = exit_rule(dict(side=side, entry=entry, or_high=hi, or_low=lo, rng=hi - lo, market=m))
        stop = entry - side * p["stop"]
        target = entry + side * p["target"] if p.get("target") else None
        be_after, be_to = p.get("be_after"), p.get("be_to", 0.0)
        max_bars = p.get("max_bars")
        best = 0.0
        exit_px, j = None, k
        while j < len(H) and DAY[j] == day:
            adverse_hit = (L[j] <= stop) if side == 1 else (H[j] >= stop)
            if adverse_hit:
                exit_px = stop if side * (O[j] - stop) > 0 else O[j]  # gap through stop fills at open
                break
            fav = (H[j] - entry) if side == 1 else (entry - L[j])
            if target is not None and fav >= p["target"]:
                exit_px = O[j] if side * (O[j] - target) > 0 else target
                break
            best = max(best, fav)
            if be_after is not None and best >= be_after:
                stop = entry + side * be_to if side * ((entry + side * be_to) - stop) > 0 else stop
            if (max_bars is not None and j - k + 1 >= max_bars) or HM[j] >= m.flat_hm:
                exit_px = C[j]
                break
            j += 1
        if exit_px is None:
            exit_px, j = C[j - 1], j - 1
        busy_until = j
        trades.append(dict(day=day, side=side, entry_hm=HM[k], entry=entry, exit=exit_px,
                           pts=side * (exit_px - entry), bars=j - k, rng=hi - lo))
    return pd.DataFrame(trades)
