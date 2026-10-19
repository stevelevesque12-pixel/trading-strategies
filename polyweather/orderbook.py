"""CLOB order books: fetch, and walk the book to price a fill of a given size."""

from dataclasses import dataclass, field
from typing import Iterable

from . import http
from .fees import fee_per_share

CLOB = "https://clob.polymarket.com"


@dataclass
class OrderBook:
    token_id: str
    bids: list[tuple[float, float]] = field(default_factory=list)  # (price, size) best first
    asks: list[tuple[float, float]] = field(default_factory=list)

    @property
    def best_bid(self):
        return self.bids[0][0] if self.bids else None

    @property
    def best_ask(self):
        return self.asks[0][0] if self.asks else None

    @classmethod
    def from_json(cls, d: dict) -> "OrderBook":
        bids = sorted(((float(x["price"]), float(x["size"])) for x in d.get("bids", [])), reverse=True)
        asks = sorted((float(x["price"]), float(x["size"])) for x in d.get("asks", []))
        return cls(token_id=str(d.get("asset_id", "")), bids=bids, asks=asks)

    def buy_levels(self, fee_rate: float = 0.0, fee_exponent: float = 1.0) -> list[tuple[float, float, float]]:
        """Asks as (price, size, all-in cost per share incl. taker fee)."""
        return [(p, s, p + fee_per_share(p, fee_rate, fee_exponent)) for p, s in self.asks]

    def sweep_cost(self, shares: float, fee_rate: float = 0.0, fee_exponent: float = 1.0):
        """(filled shares, total USDC incl. fees, worst price touched) for a market buy."""
        filled, cost, worst = 0.0, 0.0, None
        for p, s, c in self.buy_levels(fee_rate, fee_exponent):
            take = min(s, shares - filled)
            if take <= 0:
                break
            filled += take
            cost += take * c
            worst = p
        return filled, cost, worst


def fetch_books(token_ids: Iterable[str]) -> dict[str, OrderBook]:
    ids = [t for t in token_ids if t]
    out: dict[str, OrderBook] = {}
    for i in range(0, len(ids), 50):
        chunk = ids[i:i + 50]
        resp = http.post(f"{CLOB}/books", [{"token_id": t} for t in chunk])
        for d in resp or []:
            ob = OrderBook.from_json(d)
            out[ob.token_id] = ob
    return out


def fetch_price_history(token_id: str, start_ts: int, end_ts: int, fidelity_min: int = 60,
                        cache: bool = False) -> list[tuple[int, float]]:
    """Trade-price history (not executable quotes) for one token."""
    d = http.get(f"{CLOB}/prices-history", params={
        "market": token_id, "startTs": int(start_ts), "endTs": int(end_ts), "fidelity": fidelity_min},
        cache=cache)
    return [(int(x["t"]), float(x["p"])) for x in (d or {}).get("history", [])]


DATA_API = "https://data-api.polymarket.com"


def fetch_trades(condition_ids: list[str], cache: bool = False, page: int = 500, max_pages: int = 8) -> list[dict]:
    """Executed taker trades (Polymarket data API) for a set of markets, oldest first.

    Each: {"ts", "asset" (token id), "side" (taker side, BUY/SELL), "price", "size"}.
    """
    out = []
    for i in range(max_pages):
        d = http.get(f"{DATA_API}/trades", params={
            "market": ",".join(condition_ids), "limit": page, "offset": i * page, "takerOnly": "true"},
            cache=cache)
        if not isinstance(d, list) or not d:
            break
        out.extend({"ts": int(t["timestamp"]), "asset": str(t["asset"]), "side": t["side"],
                    "price": float(t["price"]), "size": float(t["size"])} for t in d)
        if len(d) < page:
            break
    return sorted(out, key=lambda t: t["ts"])
