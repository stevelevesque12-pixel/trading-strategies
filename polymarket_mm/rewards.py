"""
Liquidity-reward scoring, following Polymarket's published formula:

    S(v, s) = ((v - s) / v)^2 * size          per resting order
    v = market's max_spread (cents), s = distance from midpoint (cents)

Q_one is a maker's score on the bid side, Q_two on the ask side (NO asks
count as YES bids and vice versa; the unified book already folds them in).
Then
    mid in [0.10, 0.90]: Q_min = max(min(Q1, Q2), max(Q1, Q2) / c)   (one-sided quoting earns, penalised)
    otherwise:           Q_min = min(Q1, Q2)                          (must be two-sided)
and a maker's payout is daily_rate * Q_min / sum of everyone's Q_min.

We only see aggregate levels, not other makers' individual orders, so the
competition's Q_min is approximated by applying the formula to the book's
aggregate sides. That overstates the competition (min of sums >= sum of
mins), so the estimated share is conservative. Treat it as a rough estimate.
"""

C_SINGLE_SIDED = 3.0


def order_score(distance: float, max_spread_cents: float, size: float) -> float:
    """distance in price units (0.01 = 1 cent)."""
    v = max_spread_cents
    s = abs(distance) * 100
    if v <= 0 or s >= v:
        return 0.0
    return ((v - s) / v) ** 2 * size


def side_score(levels: dict[float, float], mid: float, max_spread_cents: float) -> float:
    return sum(order_score(p - mid, max_spread_cents, sz) for p, sz in levels.items())


def q_min(q_one: float, q_two: float, mid: float) -> float:
    if 0.10 <= mid <= 0.90:
        return max(min(q_one, q_two), max(q_one, q_two) / C_SINGLE_SIDED)
    return min(q_one, q_two)


def our_share(book, our_bids, our_asks, max_spread_cents: float, min_size: float) -> float:
    """Fraction of the market's reward pool our resting orders would earn
    right now. our_bids / our_asks: iterables of (price, size). Orders
    smaller than min_size don't score."""
    mid = book.mid
    if mid is None:
        return 0.0
    ours_one = sum(order_score(p - mid, max_spread_cents, s) for p, s in our_bids if s >= min_size)
    ours_two = sum(order_score(p - mid, max_spread_cents, s) for p, s in our_asks if s >= min_size)
    our_q = q_min(ours_one, ours_two, mid)
    if our_q <= 0:
        return 0.0
    others_q = q_min(side_score(book.bids, mid, max_spread_cents),
                     side_score(book.asks, mid, max_spread_cents), mid)
    return our_q / (our_q + others_q)
