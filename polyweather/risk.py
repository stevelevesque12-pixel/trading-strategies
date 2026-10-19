"""Portfolio limits that sit above per-event Kelly sizing.

Weather outcomes are correlated in ways one event can't see: the high and
low for the same city/day share a weather regime, neighbouring cities
(Dallas/Austin/Houston, the Chinese cluster) share synoptic errors, and the
same model bias hits every day of a heat wave. So cap exposure per city-day
and in total, and stop trading for the day after a loss limit.
"""

from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone

from .ledger import Position
from .markets import WeatherEvent


@dataclass
class RiskConfig:
    max_total_frac: float = 0.50  # open cost basis across everything
    max_city_day_frac: float = 0.06  # high + low events for one city and date
    max_region_day_frac: float = 0.15  # e.g. all of Texas on one date
    daily_loss_limit_frac: float = 0.10
    max_open_events: int = 60


REGIONS = {
    "dallas": "texas", "austin": "texas", "houston": "texas",
    "los-angeles": "california", "san-francisco": "california",
    "shenzhen": "pearl-delta", "guangzhou": "pearl-delta", "hong-kong": "pearl-delta",
    "beijing": "north-china", "jinan": "north-china", "zhengzhou": "north-china", "qingdao": "north-china",
    "chengdu": "sichuan", "chongqing": "sichuan",
    "london": "nw-europe", "paris": "nw-europe", "amsterdam": "nw-europe",
    "seoul": "korea", "busan": "korea", "istanbul": "turkey", "ankara": "turkey",
}


def region(city: str) -> str:
    return REGIONS.get(city, city)


class RiskManager:
    def __init__(self, cfg: RiskConfig, bankroll: float):
        self.cfg = cfg
        self.bankroll = bankroll
        self.day_pnl = 0.0
        self._day = datetime.now(timezone.utc).date()

    def record_pnl(self, pnl: float) -> None:
        self._roll()
        self.day_pnl += pnl

    def _roll(self) -> None:
        today = datetime.now(timezone.utc).date()
        if today != self._day:
            self._day = today
            self.day_pnl = 0.0

    def locked(self) -> bool:
        self._roll()
        return self.day_pnl <= -self.cfg.daily_loss_limit_frac * self.bankroll

    def budget_for(self, ev: WeatherEvent, positions: list[Position]) -> float:
        if self.locked():
            return 0.0
        total = sum(p.cost for p in positions)
        city_day = defaultdict(float)
        region_day = defaultdict(float)
        events = set()
        for p in positions:
            city_day[(p.city, p.target_date)] += p.cost
            region_day[(region(p.city), p.target_date)] += p.cost
            events.add(p.slug)
        if ev.slug not in events and len(events) >= self.cfg.max_open_events:
            return 0.0
        d = str(ev.target_date)
        b = self.bankroll
        return max(0.0, min(
            self.cfg.max_total_frac * b - total,
            self.cfg.max_city_day_frac * b - city_day[(ev.city, d)],
            self.cfg.max_region_day_frac * b - region_day[(region(ev.city), d)],
        ))
