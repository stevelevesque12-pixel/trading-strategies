"""Turn bucket probabilities + order books into orders.

Sizing is fractional Kelly over the *whole event*, not per bucket. The
buckets are mutually exclusive, so a YES on "62-63" and a NO on "64-65" are
correlated bets; sizing them independently over-bets. We maximise expected
log wealth across the event's outcomes:

    max  sum_k q_k * log(W_k)      W_k = wealth if bucket k wins

greedily, a few dollars at a time: each step buys the contract (YES_i or
NO_i, at its *current* order-book level incl. taker fee) with the highest
marginal log-growth, until nothing clears the edge filters. Walking real
depth means size falls out of liquidity rather than being assumed.

Fractional Kelly is approximated by running the optimiser with
`kelly_fraction * bankroll` as the wealth base.
"""

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .fees import fee_per_share
from .ledger import Position
from .markets import WeatherEvent
from .orderbook import OrderBook


@dataclass
class StrategyConfig:
    model_weight: float = 0.75  # blend: q = w*model + (1-w)*market, for calibrated sources
    model_weight_uncal: float = 0.40  # ... and for sources with no calibration history yet
    kelly_fraction: float = 0.25
    min_edge: float = 0.04  # expected $/share after fees, on the blended probability
    min_edge_frac: float = 0.10  # ... and at least this fraction of the all-in price
    min_price: float = 0.03  # don't buy below (fee/longshot trap, model tails least reliable)
    min_price_yes: float = 0.03  # separate floor for YES legs (cheap YES lost money in backtests)
    max_price: float = 0.97
    max_event_frac: float = 0.04  # max new $ at risk per event, fraction of bankroll
    max_order_usd: float = 250.0
    step_frac: float = 0.002  # greedy step size, fraction of bankroll
    prob_floor: float = 0.003  # never treat any bucket as impossible unless obs rule it out
    # Settlement can disagree with the METAR feed (late corrections, missing obs, a reading we
    # didn't see): ~0.7% of events in validation. Buckets the obs rule out keep this much mass.
    dead_adjacent_prob: float = 0.01
    dead_other_prob: float = 0.001
    exit_edge: float = 0.05  # sell a held contract when its bid exceeds fair value by this much
    min_arb_profit: float = 0.01  # $/set for neg-risk arbitrage
    allow_no: bool = True


@dataclass
class OrderIntent:
    slug: str
    bucket: int
    label: str
    side: str  # "YES" / "NO"
    action: str  # "BUY" / "SELL"
    token_id: str
    shares: float
    limit_price: float  # worst level we're willing to touch
    est_cost: float  # USDC incl. fees (BUY), or est. proceeds net of fees (SELL)
    est_fee: float
    prob: float  # blended probability the contract pays out
    model_prob: float
    edge: float  # expected profit per share at the average fill
    reason: str = "kelly"


@dataclass
class Plan:
    q: np.ndarray  # blended probabilities used
    orders: list[OrderIntent] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


def market_probs(ev: WeatherEvent, books: dict[str, OrderBook], max_spread: float = 0.15) -> np.ndarray:
    """Normalised YES mids; NaN where the book is too thin to say anything.

    Last-trade prices are deliberately ignored: on dead buckets they are
    often hours stale and would leak phantom probability into the blend.
    """
    out = []
    for b in ev.buckets:
        ob = books.get(b.yes_token)
        bid = ob.best_bid if ob else b.best_bid
        ask = ob.best_ask if ob else b.best_ask
        if bid is not None and ask is not None and ask - bid <= max_spread:
            out.append(0.5 * (bid + ask))
        elif ask is not None and ask <= 0.02:
            out.append(0.5 * ask)  # one-sided but pinned near zero: informative
        else:
            out.append(np.nan)
    return np.array(out, dtype=float)


def blend(model_p: np.ndarray, market_p: Optional[np.ndarray], cfg: StrategyConfig,
          observed_extreme=None, ev: Optional[WeatherEvent] = None, final: bool = False,
          calibrated_frac: float = 1.0) -> np.ndarray:
    """q = w*model + (1-w)*market, bucket by bucket where the market is informative.

    w falls toward `model_weight_uncal` when the sources haven't been
    calibrated for this station: an uncalibrated raw ensemble should move us
    off the market price, not override it.
    """
    w = cfg.model_weight_uncal + (cfg.model_weight - cfg.model_weight_uncal) * calibrated_frac
    if market_p is None or final:
        q = model_p.copy()
    else:
        m = np.where(np.isnan(market_p), np.nan, market_p)
        informative = ~np.isnan(m)
        if informative.any():
            # Scale market mids to the probability mass the model gives the same buckets.
            m_sum = m[informative].sum()
            m[informative] *= model_p[informative].sum() / m_sum if m_sum > 0 else 0.0
        m = np.where(informative, m, model_p)
        q = w * model_p + (1 - w) * m
    if not final:
        q = np.maximum(q, cfg.prob_floor)
    if ev is not None and observed_extreme is not None:
        # Buckets the observations rule out get only settlement-mismatch risk, whatever the market says.
        live = ev.bucket_index_for(observed_extreme)
        for i, b in enumerate(ev.buckets):
            dead = (ev.kind == "high" and b.hi is not None and b.hi < observed_extreme) or \
                   (ev.kind == "low" and b.lo is not None and b.lo > observed_extreme)
            if dead:
                q[i] = cfg.dead_adjacent_prob if live is not None and abs(i - live) == 1 else cfg.dead_other_prob
    elif final:
        q = np.maximum(q, cfg.dead_other_prob)
    return q / q.sum()


@dataclass
class _Contract:
    bucket: int
    side: str
    token_id: str
    payoff: np.ndarray  # 1 in outcomes where it pays
    levels: list  # [(price, size, all_in)] ascending
    cursor: int = 0
    used_at_level: float = 0.0
    shares: float = 0.0
    cost: float = 0.0
    fee: float = 0.0
    worst: float = 0.0

    def current(self):
        while self.cursor < len(self.levels) and self.used_at_level >= self.levels[self.cursor][1] - 1e-9:
            self.cursor += 1
            self.used_at_level = 0.0
        return self.levels[self.cursor] if self.cursor < len(self.levels) else None


def plan_event(ev: WeatherEvent, model_p: np.ndarray, books: dict[str, OrderBook], bankroll: float,
               positions: list[Position], cfg: StrategyConfig, observed_extreme=None,
               final: bool = False, risk_budget: Optional[float] = None,
               calibrated_frac: float = 1.0) -> Plan:
    n = len(ev.buckets)
    mkt = market_probs(ev, books)
    q = blend(model_p, mkt, cfg, observed_extreme, ev, final, calibrated_frac)
    plan = Plan(q=q)

    # Wealth per outcome, from the Kelly base plus what we already hold in this event.
    base = cfg.kelly_fraction * bankroll
    W = np.full(n, base, dtype=float)
    held = {}
    for p in positions:
        if p.slug != ev.slug:
            continue
        pay = np.zeros(n)
        if p.side == "YES":
            pay[p.bucket] = 1.0
        else:
            pay[:] = 1.0
            pay[p.bucket] = 0.0
        W += pay * p.shares - p.cost
        held[(p.bucket, p.side)] = p

    contracts: list[_Contract] = []
    for i, b in enumerate(ev.buckets):
        for side, tok in (("YES", b.yes_token), ("NO", b.no_token)):
            if side == "NO" and not cfg.allow_no:
                continue
            ob = books.get(tok)
            if ob is None or not ob.asks:
                continue
            pay = np.zeros(n)
            if side == "YES":
                pay[i] = 1.0
            else:
                pay[:] = 1.0
                pay[i] = 0.0
            contracts.append(_Contract(i, side, tok, pay, ob.buy_levels(ev.fee_rate, ev.fee_exponent)))

    budget = min(cfg.max_event_frac * bankroll, risk_budget if risk_budget is not None else np.inf)
    step = max(cfg.step_frac * bankroll, 1.0)
    spent = 0.0
    while spent < budget - 1e-6 and np.all(W > 0.05 * base):
        best, best_g = None, 0.0
        for c in contracts:
            lvl = c.current()
            if lvl is None or c.cost >= cfg.max_order_usd:
                continue
            price, _, all_in = lvl
            if not (cfg.min_price <= price <= cfg.max_price):
                continue
            if c.side == "YES" and price < cfg.min_price_yes:
                continue
            pw = float(q @ c.payoff)
            edge = pw - all_in
            if edge < cfg.min_edge or edge < cfg.min_edge_frac * all_in:
                continue
            g = float(np.sum(q * (c.payoff / all_in - 1.0) / W))
            if g > best_g:
                best, best_g = c, g
        if best is None:
            break
        price, size, all_in = best.current()
        dollars = min(step, budget - spent, cfg.max_order_usd - best.cost)
        sh = min(size - best.used_at_level, dollars / all_in)
        if sh <= 1e-6:
            break
        best.used_at_level += sh
        best.shares += sh
        best.cost += sh * all_in
        best.fee += sh * (all_in - price)
        best.worst = max(best.worst, price)
        W += best.payoff * sh - sh * all_in
        spent += sh * all_in

    for c in contracts:
        if c.shares <= 0:
            continue
        shares = float(np.floor(c.shares * 100) / 100)
        if shares < ev.min_order_size:
            plan.notes.append(f"{ev.buckets[c.bucket].label} {c.side}: {shares:.2f} sh below min size")
            continue
        avg = c.cost / c.shares
        pw = float(q @ c.payoff)
        plan.orders.append(OrderIntent(
            slug=ev.slug, bucket=c.bucket, label=ev.buckets[c.bucket].label, side=c.side, action="BUY",
            token_id=c.token_id, shares=shares, limit_price=round(c.worst, 4),
            est_cost=round(avg * shares, 4), est_fee=round(c.fee * shares / c.shares, 4),
            prob=pw, model_prob=float(model_p @ c.payoff), edge=pw - avg))

    plan.orders.extend(plan_exits(ev, q, model_p, books, list(held.values()), cfg))
    return plan


def plan_exits(ev: WeatherEvent, q, model_p, books, held: list[Position], cfg: StrategyConfig):
    """Sell into a market that has overshot our fair value (no fee edge needed to hold)."""
    out = []
    n = len(ev.buckets)
    for p in held:
        ob = books.get(p.token_id)
        if ob is None or not ob.bids or p.shares < ev.min_order_size:
            continue
        pay = np.zeros(n)
        if p.side == "YES":
            pay[p.bucket] = 1.0
        else:
            pay[:] = 1.0
            pay[p.bucket] = 0.0
        fair = float(q @ pay)
        shares, proceeds, fee, worst = 0.0, 0.0, 0.0, None
        for price, size in ob.bids:
            net = price - fee_per_share(price, ev.fee_rate, ev.fee_exponent)
            if net < fair + cfg.exit_edge:
                break
            take = min(size, p.shares - shares)
            shares += take
            proceeds += take * net
            fee += take * (price - net)
            worst = price
            if shares >= p.shares - 1e-9:
                break
        shares = float(np.floor(shares * 100) / 100)
        if shares >= ev.min_order_size and worst is not None:
            out.append(OrderIntent(ev.slug, p.bucket, p.label, p.side, "SELL", p.token_id, shares, worst,
                                   round(proceeds, 4), round(fee, 4), fair, float(model_p @ pay),
                                   proceeds / max(shares, 1e-9) - fair, reason="exit"))
    return out


def find_arbitrage(ev: WeatherEvent, books: dict[str, OrderBook], cfg: StrategyConfig) -> list[OrderIntent]:
    """Neg-risk set arbitrage: exactly one bucket pays, so

    * sum of YES all-in asks < 1   -> buy one YES of every bucket, locked $1 payout
    * sum of NO  all-in asks < n-1 -> buy one NO of every bucket, locked $(n-1) payout
    """
    n = len(ev.buckets)
    out = []
    for side, payout in (("YES", 1.0), ("NO", n - 1.0)):
        legs = []
        for i, b in enumerate(ev.buckets):
            ob = books.get(b.yes_token if side == "YES" else b.no_token)
            if ob is None or not ob.asks:
                legs = []
                break
            price, size = ob.asks[0]
            legs.append((i, b, price, size, price + fee_per_share(price, ev.fee_rate, ev.fee_exponent)))
        if len(legs) != n:
            continue
        cost = sum(l[4] for l in legs)
        if payout - cost < cfg.min_arb_profit:
            continue
        shares = min(min(l[3] for l in legs), cfg.max_order_usd / cost)
        shares = float(np.floor(shares * 100) / 100)
        if shares < ev.min_order_size:
            continue
        for i, b, price, _, all_in in legs:
            out.append(OrderIntent(ev.slug, i, b.label, side, "BUY",
                                   b.yes_token if side == "YES" else b.no_token, shares, price,
                                   round(all_in * shares, 4), round((all_in - price) * shares, 4),
                                   prob=1.0, model_prob=1.0, edge=(payout - cost) / n, reason="arb"))
    return out
