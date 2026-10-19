"""Discover and parse Polymarket daily-temperature events via the Gamma API.

One event = one city/day/kind ("highest" or "lowest"), split into ~11 mutually
exclusive bucket markets ("51°F or below", "52-53°F", ..., "70°F or higher").
Exactly one bucket resolves YES. Events are neg-risk, so every bucket has its
own YES and NO token, each with its own CLOB order book.
"""

import json
import re
from dataclasses import dataclass, field
from datetime import date, datetime, timezone
from typing import Iterable, Optional

from . import http
from .stations import Station, station_for

GAMMA = "https://gamma-api.polymarket.com"

_SLUG_RE = re.compile(r"^(highest|lowest)-temperature-in-(.+?)-on-([a-z]+)-(\d{1,2})(?:-(\d{4}))?")
_LABEL_RE = re.compile(
    r"^\s*(-?\d+(?:\.\d+)?)\s*(?:-\s*(-?\d+(?:\.\d+)?))?\s*°\s*([CF])\s*(or below|or lower|or higher|or above)?\s*$",
    re.IGNORECASE,
)
_MONTHS = {m: i + 1 for i, m in enumerate(
    ["january", "february", "march", "april", "may", "june", "july", "august",
     "september", "october", "november", "december"])}
_MONTHS.update({k[:3]: v for k, v in list(_MONTHS.items())})


@dataclass
class Bucket:
    label: str
    lo: Optional[float]  # inclusive lower bound in resolved units, None = open-ended
    hi: Optional[float]  # inclusive upper bound, None = open-ended
    yes_token: str
    no_token: str
    market_id: str = ""
    condition_id: str = ""
    best_bid: Optional[float] = None
    best_ask: Optional[float] = None
    last_price: Optional[float] = None
    outcome: Optional[int] = None  # 1 = resolved YES, 0 = resolved NO, None = open

    def contains(self, value: float, precision: float = 1.0) -> bool:
        """Does a resolved reading land in this bucket?

        Buckets are written in whole degrees. With whole-degree resolution a
        reading of 62 is in "62-63". With 0.1° resolution (Hong Kong) "31°C"
        covers 31.0..31.9.
        """
        lo_ok = self.lo is None or value >= self.lo - 1e-9
        if self.hi is None:
            hi_ok = True
        elif precision < 1:
            hi_ok = value < self.hi + 1.0 - 1e-9
        else:
            hi_ok = value <= self.hi + 1e-9
        return lo_ok and hi_ok

    @property
    def mid(self) -> Optional[float]:
        if self.best_bid is not None and self.best_ask is not None:
            return 0.5 * (self.best_bid + self.best_ask)
        return self.last_price


@dataclass
class WeatherEvent:
    slug: str
    title: str
    city: str
    kind: str  # "high" or "low"
    target_date: date
    unit: str
    station: Optional[Station]
    station_code: Optional[str]
    buckets: list[Bucket]
    end_time: Optional[datetime] = None
    neg_risk: bool = True
    neg_risk_market_id: str = ""
    tick_size: float = 0.001
    min_order_size: float = 5.0
    fee_rate: float = 0.0
    fee_exponent: float = 1.0
    hourly_only: bool = False  # rules say "Show Hourly Data": routine METARs only, no SPECIs
    precision: float = 1.0
    closed: bool = False
    accepting_orders: bool = True
    volume: float = 0.0
    liquidity: float = 0.0
    description: str = field(default="", repr=False)

    @property
    def tradeable(self) -> bool:
        return (not self.closed and self.accepting_orders and self.station is not None
                and self.station.source == "metar" and len(self.buckets) >= 2)

    def winner(self) -> Optional[int]:
        for i, b in enumerate(self.buckets):
            if b.outcome == 1:
                return i
        return None

    def bucket_index_for(self, value: float) -> Optional[int]:
        for i, b in enumerate(self.buckets):
            if b.contains(value, self.precision):
                return i
        return None


def parse_bucket_label(label: str) -> tuple[Optional[float], Optional[float], str]:
    m = _LABEL_RE.match(label.replace("–", "-"))
    if not m:
        raise ValueError(f"unrecognised bucket label {label!r}")
    a, b, unit, tail = m.group(1), m.group(2), m.group(3).upper(), (m.group(4) or "").lower()
    lo, hi = float(a), float(b) if b is not None else float(a)
    if tail in ("or below", "or lower"):
        return None, hi, unit
    if tail in ("or higher", "or above"):
        return lo, None, unit
    return lo, hi, unit


def _loads(x, default):
    if x is None:
        return default
    if isinstance(x, str):
        try:
            return json.loads(x)
        except json.JSONDecodeError:
            return default
    return x


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def _target_date(raw: dict, m: re.Match) -> Optional[date]:
    if raw.get("eventDate"):
        try:
            return date.fromisoformat(raw["eventDate"][:10])
        except ValueError:
            pass
    month = _MONTHS.get(m.group(3)) or _MONTHS.get(m.group(3)[:3])
    end = _parse_dt(raw.get("endDate"))
    year = int(m.group(5)) if m.group(5) else (end.year if end else None)
    if month is None or year is None:
        return None
    return date(year, month, int(m.group(4)))


def parse_event(raw: dict) -> Optional[WeatherEvent]:
    """Gamma event JSON -> WeatherEvent, or None if it isn't a daily-temperature event."""
    m = _SLUG_RE.match(raw.get("slug", ""))
    if not m:
        return None
    kind = "high" if m.group(1) == "highest" else "low"
    city = m.group(2)
    tdate = _target_date(raw, m)
    if tdate is None:
        return None

    desc = raw.get("description") or ""
    codes = re.findall(r"site=(\w{4})\b", desc) or re.findall(r"/([A-Z]{4})\.?(?:\s|$|\))", desc)
    station_code = codes[0].upper() if codes else None
    station = station_for(city, station_code)

    buckets: list[Bucket] = []
    unit = None
    first_market: dict = {}
    for mk in raw.get("markets", []):
        label = mk.get("groupItemTitle") or ""
        try:
            lo, hi, unit = parse_bucket_label(label)
        except ValueError:
            continue
        tokens = _loads(mk.get("clobTokenIds"), [])
        if len(tokens) != 2:
            continue
        prices = [float(p) for p in _loads(mk.get("outcomePrices"), [])] or [None, None]
        outcome = None
        if mk.get("closed") and prices[0] is not None and (prices[0] >= 0.99 or prices[0] <= 0.01):
            outcome = 1 if prices[0] >= 0.99 else 0
        first_market = first_market or mk
        buckets.append(Bucket(
            label=label, lo=lo, hi=hi, yes_token=str(tokens[0]), no_token=str(tokens[1]),
            market_id=str(mk.get("id", "")), condition_id=mk.get("conditionId", ""),
            best_bid=mk.get("bestBid"), best_ask=mk.get("bestAsk"),
            last_price=mk.get("lastTradePrice") if mk.get("lastTradePrice") is not None else prices[0],
            outcome=outcome,
        ))
    if not buckets:
        return None
    buckets.sort(key=lambda b: (b.lo if b.lo is not None else -1e9))

    fee = first_market.get("feeSchedule") or {}
    fees_on = bool(first_market.get("feesEnabled"))
    return WeatherEvent(
        slug=raw["slug"], title=raw.get("title", ""), city=city, kind=kind, target_date=tdate,
        unit=unit or (station.unit if station else "C"), station=station, station_code=station_code,
        buckets=buckets, end_time=_parse_dt(raw.get("endDate")),
        neg_risk=bool(raw.get("negRisk", True)), neg_risk_market_id=raw.get("negRiskMarketID", ""),
        tick_size=float(first_market.get("orderPriceMinTickSize") or 0.001),
        min_order_size=float(first_market.get("orderMinSize") or 5),
        fee_rate=float(fee.get("rate", 0.0)) if fees_on else 0.0,
        fee_exponent=float(fee.get("exponent", 1.0)),
        hourly_only="hourly data" in desc.lower(),
        precision=0.1 if "one decimal place" in desc.lower() else 1.0,
        closed=bool(raw.get("closed")),
        accepting_orders=all(mk.get("acceptingOrders", True) for mk in raw.get("markets", [])),
        volume=float(raw.get("volume") or 0.0), liquidity=float(raw.get("liquidity") or 0.0),
        description=desc,
    )


def fetch_open_events(max_pages: int = 20) -> list[WeatherEvent]:
    """All currently open daily-temperature events."""
    out: list[WeatherEvent] = []
    for page in range(max_pages):
        raw = http.get(f"{GAMMA}/events", params={
            "tag_slug": "weather", "closed": "false", "limit": 100, "offset": page * 100})
        if not raw:
            break
        out.extend(e for e in (parse_event(r) for r in raw) if e is not None)
        if len(raw) < 100:
            break
    return out


def event_slug(kind: str, city: str, d: date) -> str:
    month = ["january", "february", "march", "april", "may", "june", "july", "august",
             "september", "october", "november", "december"][d.month - 1]
    prefix = "highest" if kind == "high" else "lowest"
    return f"{prefix}-temperature-in-{city}-on-{month}-{d.day}-{d.year}"


def fetch_event(slug: str, cache: bool = False) -> Optional[WeatherEvent]:
    raw = http.get(f"{GAMMA}/events", params={"slug": slug}, cache=cache)
    if not raw:
        return None
    return parse_event(raw[0])


def fetch_events_for(kind: str, city: str, dates: Iterable[date]) -> list[WeatherEvent]:
    """Historical lookups by slug. Resolved events are cached (they never change)."""
    out = []
    for d in dates:
        ev = fetch_event(event_slug(kind, city, d), cache=d < datetime.now(timezone.utc).date())
        if ev is not None and ev.closed:
            out.append(ev)
        elif ev is not None:
            # Don't let an unresolved snapshot stick in the cache.
            http._cache_path(f"{GAMMA}/events", {"slug": ev.slug}).unlink(missing_ok=True)
    return out
