"""Market making on temperature buckets: quote around model fair value, earn spread + rewards.

Only BUY orders are used. A YES bid at b and a NO bid at a on the same
bucket are a two-sided quote (the NO bid is a YES ask at 1 - a); if both
fill, the pair redeems for $1 and we keep 1 - b - a. Makers pay no fee.

Per bucket and cycle:
    f      = fair YES probability (model/market blend), skewed against inventory
    YES bid = min(f - min_edge, mid), rounded down to tick, if it lands within the
              reward band (mid - max_spread) -- closer to mid scores more
    NO bid  = same on the complement
The main risk is adverse selection: a taker who has just seen a reading we
haven't. So quotes come down around each station's METAR minutes on the
target day, near-certain buckets aren't quoted, and inventory per bucket and
per event is capped.
"""

import math
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from zoneinfo import ZoneInfo

from .markets import WeatherEvent
from .rewards import RewardConfig


@dataclass
class MakerConfig:
    quote_shares: float = 50.0  # per side per bucket (reward min size is usually 20)
    min_edge: float = 0.01  # bid at least this far below fair value
    model_weight: float = 0.5  # fair = w*model + (1-w)*mid
    min_fair: float = 0.04  # don't quote buckets this close to decided
    max_fair: float = 0.96
    # Halving these from 40/150 halved the backtested trading loss (-$365 -> -$140/day, 50 cities)
    # while keeping most of the quoting time.
    max_side_inventory_usd: float = 15.0  # cost held on one side of one bucket
    max_event_inventory_usd: float = 60.0
    max_total_inventory_usd: float = float("inf")  # across all events; above it, only risk-reducing quotes
    skew_per_10_shares: float = 0.002  # shift fair against net inventory
    # Stop quoting an event at this local hour on its target day (None = quote to the end).
    # Highs: the afternoon is when the day's max forms and informed takers pick quotes off.
    # Lows: the minimum usually sets before dawn, so stop as the target day starts.
    stop_hour_high: Optional[int] = 12
    stop_hour_low: Optional[int] = 0
    blackout_before_min: float = 5.0  # pull quotes this long before an expected reading ...
    blackout_after_min: float = 8.0  # ... until this long after it (publication + reaction)
    rewarded_only: bool = True


@dataclass
class Quote:
    bucket: int
    side: str  # "YES" / "NO" (always a BUY)
    token_id: str
    price: float
    shares: float
    fair: float  # fair value of the token being bought


@dataclass
class Inventory:
    yes: dict = field(default_factory=dict)  # bucket -> (shares, cost)
    no: dict = field(default_factory=dict)

    def add(self, side: str, bucket: int, shares: float, cost: float) -> None:
        d = self.yes if side == "YES" else self.no
        s, c = d.get(bucket, (0.0, 0.0))
        d[bucket] = (s + shares, c + cost)

    def cost(self, side: str, bucket: int) -> float:
        return (self.yes if side == "YES" else self.no).get(bucket, (0.0, 0.0))[1]

    def net_yes(self, bucket: int) -> float:
        return self.yes.get(bucket, (0.0, 0.0))[0] - self.no.get(bucket, (0.0, 0.0))[0]

    def total_cost(self) -> float:
        return sum(c for _, c in self.yes.values()) + sum(c for _, c in self.no.values())

    def payout(self, winner: int) -> float:
        out = sum(s for b, (s, _) in self.yes.items() if b == winner)
        out += sum(s for b, (s, _) in self.no.items() if b != winner)
        return out


def in_blackout(now: datetime, tz: str, obs_minutes: list[int], cfg: MakerConfig) -> bool:
    """Is `now` within the window around one of the station's usual reporting minutes?"""
    if not obs_minutes:
        return False
    m = now.astimezone(ZoneInfo(tz)).minute + now.second / 60.0
    for om in obs_minutes:
        d = (m - om) % 60.0  # minutes since the reading (0..60)
        if d <= cfg.blackout_after_min or d >= 60.0 - cfg.blackout_before_min:
            return True
    return False


def quoting_allowed(ev: WeatherEvent, now: datetime, cfg: MakerConfig) -> bool:
    stop = cfg.stop_hour_high if ev.kind == "high" else cfg.stop_hour_low
    if stop is None:
        return True
    local = now.astimezone(ZoneInfo(ev.station.tz))
    days = (ev.target_date - local.date()).days
    return days > 0 or (days == 0 and local.hour < stop)


def usual_obs_minutes(times: list[datetime], min_count: int = 3) -> list[int]:
    counts: dict[int, int] = {}
    for t in times:
        counts[t.minute] = counts.get(t.minute, 0) + 1
    return sorted(m for m, c in counts.items() if c >= min_count)


def _floor_tick(x: float, tick: float) -> float:
    return round(math.floor(x / tick + 1e-9) * tick, 6)


def make_quotes(ev: WeatherEvent, fair_yes: list[Optional[float]], mids: list[Optional[float]],
                rewards: dict[str, RewardConfig], inv: Inventory, cfg: MakerConfig,
                best_asks: Optional[list[tuple[Optional[float], Optional[float]]]] = None,
                total_inventory: float = 0.0) -> list[Quote]:
    """fair_yes/mids per bucket (None = don't quote). best_asks: per bucket (yes_ask, no_ask).

    When this event or the whole book is at its inventory cap, only quotes that
    *reduce* risk are made: the side opposite a bucket's net position (a filled
    NO against held YES pairs off into a locked $1).
    """
    quotes = []
    tick = ev.tick_size
    global_room = cfg.max_total_inventory_usd - total_inventory
    reduce_only = inv.total_cost() >= cfg.max_event_inventory_usd or global_room <= 0
    for i, b in enumerate(ev.buckets):
        f, mid = fair_yes[i], mids[i]
        if f is None or mid is None:
            continue
        rc = rewards.get(b.condition_id)
        if cfg.rewarded_only and rc is None:
            continue
        if not (cfg.min_fair <= f <= cfg.max_fair):
            continue
        band = rc.max_spread if rc else 0.045
        f_adj = f - cfg.skew_per_10_shares * inv.net_yes(i) / 10.0
        for side, token, fair, m in (("YES", b.yes_token, f_adj, mid), ("NO", b.no_token, 1 - f_adj, 1 - mid)):
            net = inv.net_yes(i)
            reduces = (side == "NO" and net > 0) or (side == "YES" and net < 0)
            if reduce_only and not reduces:
                continue
            if inv.cost(side, i) >= cfg.max_side_inventory_usd and not reduces:
                continue
            price = min(fair - cfg.min_edge, m)
            if best_asks is not None:
                ask = best_asks[i][0 if side == "YES" else 1]
                if ask is not None:
                    price = min(price, ask - tick)
            price = _floor_tick(price, tick)
            if price < tick or m - price > band:
                continue  # outside the reward band: not worth the adverse-selection risk
            shares = cfg.quote_shares
            if reduces:
                shares = min(shares, abs(net))  # just enough to pair off
            else:
                room = min(cfg.max_side_inventory_usd - inv.cost(side, i), global_room)
                shares = min(shares, math.floor(room / price))
            if rc is not None and shares < rc.min_size:
                continue
            quotes.append(Quote(i, side, token, price, float(shares), fair))
    return quotes


def fills_from_prints(q: Quote, ev: WeatherEvent, trades: list[dict], t0: float, t1: float,
                      at_price_capture: float = 0.5) -> float:
    """Shares of a resting BUY that real taker trades in (t0, t1] would have filled.

    Our bid at p is hit by a taker SELL of our token at <= p, or (neg-risk
    matching) a taker BUY of the complement at >= 1 - p. Prints strictly
    through our price would have hit us first; prints exactly at our price
    are shared with the queue.
    """
    b = ev.buckets[q.bucket]
    comp = b.no_token if q.token_id == b.yes_token else b.yes_token
    filled = 0.0
    for t in trades:
        if t["ts"] <= t0:
            continue
        if t["ts"] > t1 or filled >= q.shares - 1e-9:
            break
        if t["asset"] == q.token_id and t["side"] == "SELL":
            px = t["price"]
        elif t["asset"] == comp and t["side"] == "BUY":
            px = 1 - t["price"]
        else:
            continue
        if px > q.price + 1e-9:
            continue
        cap = 1.0 if px < q.price - 1e-9 else at_price_capture
        filled += min(t["size"] * cap, q.shares - filled)
    return filled
