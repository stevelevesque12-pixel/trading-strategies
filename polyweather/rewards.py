"""Polymarket liquidity rewards: pool discovery and score estimation.

Weather markets carry daily reward pools on their near-the-money buckets
(about $15.9k/day across all temperature markets on 2026-10-19; $100-200/day
on a single bucket is common). Rewards go to resting orders within
`max_spread` of the midpoint with at least `min_size` shares, scored each
minute (docs.polymarket.com/market-makers/liquidity-rewards):

    S(v, s) = ((v - s) / v)^2                       per order, v and s in cents
    Q_one   = sum S * size  (YES bids + NO asks)
    Q_two   = sum S * size  (YES asks + NO bids)
    Q_min   = max(min(Q_one, Q_two), max(Q_one, Q_two) / 3)   if 0.10 <= mid <= 0.90
            = min(Q_one, Q_two)                                  otherwise
    share   = my Q_min / everyone's Q_min, averaged over the day's samples

A bot that only BUYs still quotes both sides: a NO bid at x is a YES ask
at 1 - x, so a YES bid counts toward Q_one and a NO bid toward Q_two.
"""

import json
import math
from dataclasses import dataclass
from typing import Optional

from . import http
from .orderbook import OrderBook

GAMMA = "https://gamma-api.polymarket.com"
SINGLE_SIDED_DIVISOR = 3.0


@dataclass
class RewardConfig:
    condition_id: str
    daily_rate: float  # USDC per day for this market
    max_spread: float  # in price units (docs quote it in cents: 4.5 -> 0.045)
    min_size: float  # shares


def parse_reward_config(market: dict) -> Optional[RewardConfig]:
    rr = market.get("clobRewards") or []
    rate = sum(float(x.get("rewardsDailyRate") or 0.0) for x in rr)
    if rate <= 0:
        return None
    return RewardConfig(
        condition_id=market.get("conditionId", ""), daily_rate=rate,
        max_spread=float(market.get("rewardsMaxSpread") or 4.5) / 100.0,
        min_size=float(market.get("rewardsMinSize") or 20.0))


def fetch_reward_configs(max_pages: int = 10) -> dict[str, RewardConfig]:
    """{condition_id: RewardConfig} for every open temperature bucket with a pool."""
    out = {}
    for page in range(max_pages):
        raw = http.get(f"{GAMMA}/events", params={
            "tag_slug": "weather", "closed": "false", "limit": 100, "offset": page * 100})
        for e in raw or []:
            if "temperature-in" not in e.get("slug", ""):
                continue
            for m in e.get("markets", []):
                cfg = parse_reward_config(m)
                if cfg:
                    out[cfg.condition_id] = cfg
        if not raw or len(raw) < 100:
            break
    return out


def order_score(distance: float, max_spread: float) -> float:
    """distance and max_spread in price units."""
    if distance < 0 or distance > max_spread:
        return 0.0
    return ((max_spread - distance) / max_spread) ** 2


def adjusted_mid(yes: OrderBook, min_size: float) -> Optional[float]:
    """Midpoint using only levels at least `min_size` deep (the docs' size-cutoff mid)."""
    bid = next((p for p, s in yes.bids if s >= min_size), None)
    ask = next((p for p, s in yes.asks if s >= min_size), None)
    if bid is None or ask is None:
        bid, ask = yes.best_bid, yes.best_ask
    if bid is None or ask is None:
        return None
    return 0.5 * (bid + ask)


def q_sides(yes_bids, yes_asks, mid: float, cfg: RewardConfig) -> tuple[float, float]:
    """(Q_one, Q_two) for orders given in YES-price terms: lists of (price, size)."""
    q1 = sum(order_score(mid - p, cfg.max_spread) * s for p, s in yes_bids if s >= cfg.min_size)
    q2 = sum(order_score(p - mid, cfg.max_spread) * s for p, s in yes_asks if s >= cfg.min_size)
    return q1, q2


def q_min(q1: float, q2: float, mid: float) -> float:
    if 0.10 <= mid <= 0.90:
        return max(min(q1, q2), max(q1, q2) / SINGLE_SIDED_DIVISOR)
    return min(q1, q2)


def book_q(yes: OrderBook, no: Optional[OrderBook], mid: float, cfg: RewardConfig) -> float:
    """Everyone's resting liquidity, scored as one maker (an upper bound on competition)."""
    bids = list(yes.bids) + ([(1 - p, s) for p, s in no.asks] if no else [])
    asks = list(yes.asks) + ([(1 - p, s) for p, s in no.bids] if no else [])
    q1, q2 = q_sides(bids, asks, mid, cfg)
    # We can't see which maker owns which order. Each maker's Q_min is at most its larger
    # side, so summing both sides overstates the competition: our share estimate is conservative.
    return q1 + q2


def our_q(yes_bid: Optional[tuple[float, float]], no_bid: Optional[tuple[float, float]], mid: float,
          cfg: RewardConfig) -> float:
    bids = [yes_bid] if yes_bid else []
    asks = [(1 - no_bid[0], no_bid[1])] if no_bid else []
    q1, q2 = q_sides(bids, asks, mid, cfg)
    return q_min(q1, q2, mid)


def expected_daily_reward(ours: float, competition: float, cfg: RewardConfig) -> float:
    """Steady-state $/day if the book stayed as it is now."""
    if ours <= 0:
        return 0.0
    return cfg.daily_rate * ours / (ours + max(competition, 0.0))


def snapshot_estimate(events, books: dict, configs: dict[str, RewardConfig], quote_shares: float = 50.0,
                      edge: float = 0.01, tick: float = 0.001):
    """Expected $/day per rewarded bucket if we quoted mid -/+ edge on both sides right now.

    Rows: (event, bucket label, pool $/day, mid, our Q, competing Q, est $/day).
    """
    rows = []
    for ev in events:
        for b in ev.buckets:
            rc = configs.get(b.condition_id)
            if rc is None:
                continue
            yb, nb = books.get(b.yes_token), books.get(b.no_token)
            if yb is None:
                continue
            mid = adjusted_mid(yb, rc.min_size)
            if mid is None:
                continue
            bid = round(math.floor((mid - edge) / tick + 1e-9) * tick, 6)
            no_bid = round(math.floor((1 - mid - edge) / tick + 1e-9) * tick, 6)
            ours = our_q((bid, quote_shares) if bid > 0 else None,
                         (no_bid, quote_shares) if no_bid > 0 else None, mid, rc)
            comp = book_q(yb, nb, mid, rc)
            rows.append((ev, b.label, rc.daily_rate, mid, ours, comp, expected_daily_reward(ours, comp, rc)))
    return rows
