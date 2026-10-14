"""
Tradeify hedging / correlated-product guard, kept pure and broker-agnostic.

Tradeify (all account types) prohibits holding opposing positions on the same
instrument or on two products from the same Product Group, in one account OR
across any accounts you control. Contract size is irrelevant (long MES + short
NQ is a hedge). Long Equity Index + long Volatility (FVS) also counts as
offsetting, since volatility moves inversely to equities. Same-direction
positions across products/accounts are fine (that's how copy trading works).

Separately, minis and micros may be held together as long as an account's
combined position stays within its contract limit, counted at 10 micros = 1 mini.

`check_entry` is the single entry point: give it every open position you can
see (across all accounts) plus the order you're about to send, and it returns
a human-readable reason to block, or None if the entry is allowed.

Group table copied from Tradeify's "Hedging & Correlated Products" rules page
(Aug 19 2026). Tradeify says the list isn't exhaustive -- an unknown root is
treated as its own group, so only same-instrument hedges are caught for it.
"""

import re
from dataclasses import dataclass
from typing import Iterable, Optional

PRODUCT_GROUPS = {
    "Equity Index": [
        "ES", "MES", "NQ", "MNQ", "YM", "MYM", "RTY", "M2K", "EMD", "NKD",
        "FDAX", "FDXM", "FDXS", "FESX", "FSXE", "FXXP",
        # Volatility: its own group in Tradeify's table, but long equities +
        # long FVS is treated as offsetting, so it sits here with inverted
        # polarity (see _POLARITY) rather than as a separate group.
        "FVS",
    ],
    "Energy": ["CL", "QM", "MCL", "NG", "QG"],
    # SIL (micro silver) isn't in Tradeify's table, but mini/micro pairs
    # always share a group, so it goes with SI.
    "Metals": ["GC", "QO", "MGC", "SI", "SIL", "HG", "PL", "PA"],
    "Currencies": ["6E", "M6E", "6B", "6J", "6A", "M6A", "6C", "6S"],
    "Interest Rates": ["FGBL", "FGBM", "FGBS", "FGBX"],
    "Grains": ["ZS", "ZL", "ZM", "ZC", "ZW"],
    "Livestock": ["HE", "LE", "GF"],
}

_GROUP_OF = {root: group for group, roots in PRODUCT_GROUPS.items() for root in roots}
_POLARITY = {"FVS": -1}  # long FVS behaves like short equities
MICROS = {"MES", "MNQ", "MYM", "M2K", "MCL", "MGC", "SIL", "M6E", "M6A"}
_ROOTS_LONGEST_FIRST = sorted(_GROUP_OF, key=len, reverse=True)
# What can follow a root in a contract name: Tradovate "MESZ6"/"MESZ2026",
# TradingView continuous "MES1!", or nothing at all.
_SUFFIX = re.compile(r"^([FGHJKMNQUVXZ]\d{1,4}|\d!)?$")


@dataclass(frozen=True)
class Position:
    account_id: object
    symbol: str  # contract name or bare root, e.g. "MESZ6", "MNQ1!", "NQ"
    net_qty: int  # positive=long, negative=short


def root_symbol(symbol: str) -> str:
    """'MESZ6' / 'MESZ2026' / 'MES1!' / 'MES' -> 'MES'. Unknown roots pass through as-is."""
    s = symbol.strip().upper()
    for root in _ROOTS_LONGEST_FIRST:
        if s.startswith(root) and _SUFFIX.match(s[len(root):]):
            return root
    return s


def product_group(symbol: str) -> str:
    root = root_symbol(symbol)
    return _GROUP_OF.get(root, root)


def mini_equivalent(symbol: str, qty: int) -> float:
    return abs(qty) * (0.1 if root_symbol(symbol) in MICROS else 1.0)


def _exposure_sign(symbol: str, qty: int) -> int:
    """Direction of the position's exposure to its product group (+1 long, -1 short)."""
    direction = 1 if qty > 0 else -1
    return direction * _POLARITY.get(root_symbol(symbol), 1)


def check_entry(
    symbol: str,
    qty: int,
    open_positions: Iterable[Position],
    account_id: object = None,
    contract_limit_minis: Optional[float] = None,
) -> Optional[str]:
    """
    Reason to block a new entry of `qty` contracts of `symbol` (positive=buy,
    negative=sell), or None if it's allowed.

    `open_positions` should cover every account you control that you can see --
    the hedge check is cross-account. The contract-limit check (if
    `contract_limit_minis` is given) only counts positions in `account_id`.
    """
    if qty == 0:
        raise ValueError("qty must be non-zero")

    positions = [p for p in open_positions if p.net_qty != 0]
    group = product_group(symbol)
    new_sign = _exposure_sign(symbol, qty)

    for p in positions:
        if product_group(p.symbol) == group and _exposure_sign(p.symbol, p.net_qty) != new_sign:
            side = "long" if qty > 0 else "short"
            held = "long" if p.net_qty > 0 else "short"
            return (
                f"hedge: {side} {root_symbol(symbol)} would oppose {held} {abs(p.net_qty)} "
                f"{root_symbol(p.symbol)} on account {p.account_id} ({group})"
            )

    if contract_limit_minis is not None:
        held = sum(mini_equivalent(p.symbol, p.net_qty) for p in positions if p.account_id == account_id)
        total = held + mini_equivalent(symbol, qty)
        if total > contract_limit_minis + 1e-9:
            return (
                f"contract limit: {total:g} mini-equivalents on account {account_id} "
                f"would exceed {contract_limit_minis:g} (10 micros = 1 mini)"
            )

    return None
