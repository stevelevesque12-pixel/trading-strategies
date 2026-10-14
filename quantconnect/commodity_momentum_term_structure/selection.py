"""Pure selection logic for the commodity momentum + term structure strategy.

No QuantConnect imports here so it can be unit-tested locally; ``main.py``
feeds it plain numbers pulled from the futures chains.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Hashable, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ContractQuote:
    key: Hashable  # whatever identifies the tradable contract (a QC Symbol in main.py)
    expiry: datetime
    price: float
    open_interest: float


def annualized_roll_yield(near_price: float, far_price: float, days_apart: int) -> float:
    """ln(near/far) annualized. Positive = backwardation, negative = contango.

    Uses a log price ratio rather than the raw price difference so commodities
    with very different price levels (cocoa ~8000 vs natural gas ~3) rank on the
    same scale.
    """
    if near_price <= 0 or far_price <= 0 or days_apart <= 0:
        raise ValueError("prices and days_apart must be positive")
    return math.log(near_price / far_price) * 365.0 / days_apart


def pick_roll_pair(
    contracts: Iterable[ContractQuote],
    now: datetime,
    min_days_to_expiry: int = 0,
) -> Optional[Tuple[ContractQuote, ContractQuote]]:
    """Return (nearest, deferred) for one commodity, or None.

    nearest  = earliest-expiring contract with a price.
    deferred = next contract after it with a price, open interest, and at
               least ``min_days_to_expiry`` days left (so the position doesn't
               expire before the next monthly rebalance). This is the one traded.
    """
    priced = sorted((c for c in contracts if c.price > 0), key=lambda c: c.expiry)
    if len(priced) < 2:
        return None
    nearest = priced[0]
    for c in priced[1:]:
        if c.expiry <= nearest.expiry or c.open_interest <= 0:
            continue
        if (c.expiry - now).days < min_days_to_expiry:
            continue
        return nearest, c
    return None


def select_targets(
    roll_yield: Dict[Hashable, float],
    momentum: Dict[Hashable, float],
    gross_per_side: float = 0.05,
) -> Dict[Hashable, float]:
    """Double sort: tertiles on roll yield, then halves on momentum.

    Long  = top-roll-yield tertile, top half by momentum.
    Short = bottom-roll-yield tertile, bottom half by momentum.
    Each leg is equal weighted and sums to ``gross_per_side`` in absolute terms.
    Only keys present in both dicts are considered. Returns {} if there are
    fewer than 3 eligible commodities (no meaningful tertile split).
    """
    keys = [k for k in roll_yield if k in momentum]
    ranked = sorted(keys, key=lambda k: roll_yield[k], reverse=True)
    tertile = len(ranked) // 3
    if tertile == 0:
        return {}
    half = max(1, tertile // 2)

    high_ry = ranked[:tertile]
    low_ry = ranked[len(ranked) - tertile:]
    longs: List[Hashable] = sorted(high_ry, key=lambda k: momentum[k], reverse=True)[:half]
    shorts: List[Hashable] = sorted(low_ry, key=lambda k: momentum[k])[:half]

    targets = {k: gross_per_side / len(longs) for k in longs}
    targets.update({k: -gross_per_side / len(shorts) for k in shorts})
    return targets

