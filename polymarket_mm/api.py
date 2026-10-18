"""
Public, unauthenticated Polymarket REST calls + a reward-market scanner.

The scanner ranks reward-eligible markets by the reward our default quote
would earn right now (see rewards.py), after skipping markets that are
near resolution, priced near 0/1, or live sports (game_start_time set):
those are where a passive maker gets run over.
"""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import requests

from .book import OrderBook
from .engine import MarketInfo
from .quoting import QuoteParams, compute_quotes
from .rewards import our_share

CLOB = "https://clob.polymarket.com"
GAMMA = "https://gamma-api.polymarket.com"
_END_CURSOR = "LTE="


def _get(url, **params):
    r = requests.get(url, params=params or None, timeout=20)
    r.raise_for_status()
    return r.json()


def fetch_reward_markets(max_pages: int = 20) -> list[dict]:
    """CLOB markets currently paying liquidity rewards."""
    out, cursor = [], ""
    for _ in range(max_pages):
        d = _get(f"{CLOB}/sampling-markets", **({"next_cursor": cursor} if cursor else {}))
        out += d["data"]
        cursor = d.get("next_cursor")
        if not cursor or cursor == _END_CURSOR:
            break
    return out


def fetch_book(token_id: str) -> dict:
    return _get(f"{CLOB}/book", token_id=token_id)


def fetch_market(ref: str) -> dict:
    """ref: a condition id (0x...) or a market slug."""
    if not ref.startswith("0x"):
        g = _get(f"{GAMMA}/markets", slug=ref)
        if not g:
            raise ValueError(f"no market with slug {ref!r}")
        ref = g[0]["conditionId"]
    return _get(f"{CLOB}/markets/{ref}")


def daily_rate(m: dict) -> float:
    return sum(r.get("rewards_daily_rate", 0) for r in ((m.get("rewards") or {}).get("rates") or []))


def market_info(m: dict) -> MarketInfo:
    yes, no = m["tokens"][0], m["tokens"][1]
    rw = m.get("rewards") or {}
    return MarketInfo(
        condition_id=m["condition_id"], question=m["question"], slug=m.get("market_slug", ""),
        yes_token=yes["token_id"], no_token=no["token_id"],
        tick=float(m.get("minimum_tick_size") or 0.01),
        min_size=float(rw.get("min_size") or 0), max_spread=float(rw.get("max_spread") or 0),
        daily_rate=daily_rate(m), end_date=m.get("end_date_iso") or "",
    )


def _eligible(m: dict, now: datetime, min_days_to_end: float, include_sports: bool) -> bool:
    if not (m.get("active") and m.get("accepting_orders") and not m.get("closed")):
        return False
    if len(m.get("tokens") or []) != 2 or daily_rate(m) <= 0:
        return False
    if (m.get("rewards") or {}).get("max_spread", 0) <= 0:
        return False
    if m.get("game_start_time") and not include_sports:
        return False
    end = m.get("end_date_iso")
    if end:
        end_dt = datetime.fromisoformat(end.replace("Z", "+00:00"))
        if end_dt - now < timedelta(days=min_days_to_end):
            return False
    return True


def scan(markets: list[dict], qp: QuoteParams, top_n: int = 10, prefilter: int = 60,
         min_days_to_end: float = 2.0, include_sports: bool = False, max_capital: float = 500.0):
    """Returns [(MarketInfo, est_daily_reward, capital_needed)] best first."""
    now = datetime.now(timezone.utc)
    cands = [m for m in markets if _eligible(m, now, min_days_to_end, include_sports)]
    cands = sorted(cands, key=daily_rate, reverse=True)[:prefilter]

    def evaluate(m):
        info = market_info(m)
        try:
            snap = fetch_book(info.yes_token)
        except requests.RequestException:
            return None
        book = OrderBook()
        book.load_snapshot(snap.get("bids", []), snap.get("asks", []))
        info.tick = float(snap.get("tick_size") or info.tick)
        q = compute_quotes(book.best_bid, book.best_ask, 0.0, info.max_spread, info.min_size, info.tick, qp)
        if q.bid is None or q.ask is None:
            return None
        capital = q.bid_size * q.bid + q.ask_size * (1 - q.ask)
        if capital > max_capital:
            return None
        share = our_share(book, [(q.bid, q.bid_size)], [(q.ask, q.ask_size)], info.max_spread, info.min_size)
        return info, info.daily_rate * share, capital

    with ThreadPoolExecutor(8) as ex:
        results = [r for r in ex.map(evaluate, cands) if r is not None]
    return sorted(results, key=lambda r: r[1], reverse=True)[:top_n]


def summarize(results) -> str:
    lines = [f"{'est $/day':>9} {'capital':>8} {'pool/day':>8} {'maxspr':>6} {'minsz':>5}  question"]
    for info, est, cap in results:
        lines.append(f"{est:9.2f} {cap:8.0f} {info.daily_rate:8.0f} {info.max_spread:6.1f} "
                     f"{info.min_size:5.0f}  {info.question[:70]}")
    return "\n".join(lines)


if __name__ == "__main__":
    print(summarize(scan(fetch_reward_markets(), QuoteParams())))
