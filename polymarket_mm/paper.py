"""
Simulated broker: fills our virtual resting orders against the real feed.

Fill model (deliberately on the pessimistic side for fill *count*, honest
about adverse selection):
  * A new order joins the back of the queue: queue_ahead = the size already
    resting at that price when we place it.
  * A real trade at our price first eats queue_ahead, then us.
  * A real trade *through* our price (a sell printing below our bid) means
    the whole level was swept, so we fill in full.
  * If others' best ask moves to or through our bid (or best bid to or
    through our ask), the level we sit in was consumed: full fill.
  * Cancels by others ahead of us are NOT credited (we can't tell them from
    trades in the aggregate feed), so passive fills are under-counted.
  * Orders go live, and cancels take effect, only after `latency` seconds.
    A stale quote can still be picked off while its cancel is in flight,
    which is exactly the adverse selection we need to measure.
Fills are at our limit price (we're always the maker).
"""

import itertools
from dataclasses import dataclass

EPS = 1e-9
_ids = itertools.count(1)


@dataclass
class PaperOrder:
    side: str            # "BUY" or "SELL" (YES token)
    price: float
    size: float
    queue_ahead: float
    active_from: float
    dead_at: float = float("inf")
    filled: float = 0.0
    id: int = 0

    @property
    def remaining(self) -> float:
        return self.size - self.filled

    def is_live(self, now: float) -> bool:
        return self.active_from <= now < self.dead_at and self.remaining > EPS


@dataclass
class Fill:
    ts: float
    order_id: int
    side: str
    price: float
    size: float
    fee: float


class PaperBroker:
    def __init__(self, latency: float = 0.25, maker_fee_bps: float = 0.0):
        self.latency = latency
        self.maker_fee_bps = maker_fee_bps
        self.orders: list[PaperOrder] = []
        self.position = 0.0   # YES shares; negative = short YES (economically long NO)
        self.cash = 0.0
        self.fees = 0.0

    def place(self, side: str, price: float, size: float, queue_ahead: float, now: float) -> PaperOrder:
        o = PaperOrder(side, price, size, queue_ahead, active_from=now + self.latency, id=next(_ids))
        self.orders.append(o)
        return o

    def cancel(self, order: PaperOrder, now: float) -> None:
        order.dead_at = min(order.dead_at, now + self.latency)

    def cancel_all(self, now: float) -> None:
        for o in self.orders:
            self.cancel(o, now)

    def working(self, now: float) -> list[PaperOrder]:
        """Orders not yet cancelled (live or still in flight to the venue)."""
        self.orders = [o for o in self.orders if o.dead_at > now and o.remaining > EPS]
        return [o for o in self.orders if o.dead_at == float("inf")]

    def live(self, now: float) -> list[PaperOrder]:
        return [o for o in self.orders if o.is_live(now)]

    def on_trade(self, taker_side: str, price: float, size: float, now: float) -> list[Fill]:
        """A real trade printed on the YES book. taker_side SELL hits bids."""
        maker_side = "BUY" if taker_side == "SELL" else "SELL"
        fills = []
        for o in self.live(now):
            if o.side != maker_side:
                continue
            through = o.price > price + EPS if maker_side == "BUY" else o.price < price - EPS
            if through:
                fills.append(self._fill(o, o.remaining, now))
            elif abs(o.price - price) < EPS:
                excess = size - o.queue_ahead
                o.queue_ahead = max(0.0, o.queue_ahead - size)
                if excess > EPS:
                    fills.append(self._fill(o, min(excess, o.remaining), now))
        return fills

    def on_book(self, best_bid, best_ask, now: float) -> list[Fill]:
        """Others' book crossed our resting price -> our level was consumed."""
        fills = []
        for o in self.live(now):
            if o.side == "BUY" and best_ask is not None and best_ask <= o.price + EPS:
                fills.append(self._fill(o, o.remaining, now))
            elif o.side == "SELL" and best_bid is not None and best_bid >= o.price - EPS:
                fills.append(self._fill(o, o.remaining, now))
        return fills

    def _fill(self, o: PaperOrder, qty: float, now: float) -> Fill:
        o.filled += qty
        fee = qty * o.price * self.maker_fee_bps / 10_000
        sign = 1 if o.side == "BUY" else -1
        self.position += sign * qty
        self.cash -= sign * qty * o.price + fee
        self.fees += fee
        return Fill(now, o.id, o.side, o.price, qty, fee)

    def pnl(self, mid: float | None) -> float:
        return self.cash + (self.position * mid if mid is not None else 0.0)
