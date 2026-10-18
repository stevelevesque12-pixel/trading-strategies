"""
Pure quote decision: book top + inventory + market reward params -> bid/ask.

Quotes sit `band_fraction` of the way out to the reward max_spread (closer
earns more reward per share but gets hit more), shifted against inventory so
a long position leans toward selling. By default we only join existing price
levels rather than stepping ahead of the best bid/ask: being alone at the
top of the book is where informed flow picks you off first.
"""

from dataclasses import dataclass

from .book import ceil_tick, floor_tick


@dataclass
class QuoteParams:
    size: float = 0.0              # shares per side; raised to the market's reward min_size
    band_fraction: float = 0.5     # distance from mid as a fraction of max_spread
    min_edge_ticks: int = 1        # never quote closer to mid than this
    skew_per_share: float = 0.00005  # price shift per share of inventory (1000 sh -> 5c)
    max_position: float = 1000.0   # stop adding to a side beyond this many shares
    min_mid: float = 0.10          # outside [min_mid, max_mid] rewards need two-sided size; also tail risk
    max_mid: float = 0.90
    max_book_spread: float = 0.05  # others' spread wider than this -> market is thin/stressed, stand aside
    join_only: bool = True


@dataclass
class Quote:
    bid: float | None = None
    bid_size: float = 0.0
    ask: float | None = None
    ask_size: float = 0.0


def compute_quotes(best_bid, best_ask, position: float, max_spread_cents: float,
                   min_size: float, tick: float, p: QuoteParams) -> Quote:
    if best_bid is None or best_ask is None or best_ask <= best_bid:
        return Quote()
    mid = (best_bid + best_ask) / 2
    if not (p.min_mid <= mid <= p.max_mid) or best_ask - best_bid > p.max_book_spread + 1e-9:
        return Quote()

    offset = max(p.band_fraction * max_spread_cents / 100, p.min_edge_ticks * tick)
    skew = position * p.skew_per_share
    bid = floor_tick(mid - offset - skew, tick)
    ask = ceil_tick(mid + offset - skew, tick)
    if p.join_only:
        bid, ask = min(bid, best_bid), max(ask, best_ask)
    bid = min(bid, round(best_ask - tick, 6))  # never cross / take
    ask = max(ask, round(best_bid + tick, 6))

    size = max(p.size, min_size, 5.0)  # 5 shares is the CLOB's minimum order
    q = Quote()
    if position < p.max_position and bid >= tick:
        q.bid, q.bid_size = bid, size
    if position > -p.max_position and ask <= 1 - tick:
        q.ask, q.ask_size = ask, size
    return q
