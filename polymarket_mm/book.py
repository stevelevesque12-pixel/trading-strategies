"""
Order book for one outcome token.

Polymarket's book is unified: a resting NO bid at q shows up on the YES book
as an ask at 1-q (the feed mirrors every price_change onto both tokens), so
tracking the YES token's book alone gives the full picture.
"""

import math


def _key(price) -> float:
    return round(float(price), 6)


def floor_tick(price: float, tick: float) -> float:
    return round(math.floor(price / tick + 1e-9) * tick, 6)


def ceil_tick(price: float, tick: float) -> float:
    return round(math.ceil(price / tick - 1e-9) * tick, 6)


class OrderBook:
    """Aggregate price levels: {price: size}. Sides use the feed's names,
    BUY for bids and SELL for asks."""

    def __init__(self):
        self.bids: dict[float, float] = {}
        self.asks: dict[float, float] = {}

    def load_snapshot(self, bids, asks) -> None:
        self.bids = {_key(l["price"]): float(l["size"]) for l in bids if float(l["size"]) > 0}
        self.asks = {_key(l["price"]): float(l["size"]) for l in asks if float(l["size"]) > 0}

    def set_level(self, side: str, price, size) -> None:
        levels = self.bids if side == "BUY" else self.asks
        p, s = _key(price), float(size)
        if s > 0:
            levels[p] = s
        else:
            levels.pop(p, None)

    def size_at(self, side: str, price: float) -> float:
        levels = self.bids if side == "BUY" else self.asks
        return levels.get(_key(price), 0.0)

    @property
    def best_bid(self) -> float | None:
        return max(self.bids) if self.bids else None

    @property
    def best_ask(self) -> float | None:
        return min(self.asks) if self.asks else None

    @property
    def mid(self) -> float | None:
        bb, ba = self.best_bid, self.best_ask
        if bb is None or ba is None:
            return None
        return round((bb + ba) / 2, 6)
