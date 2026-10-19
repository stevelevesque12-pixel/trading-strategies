"""Polymarket taker fees.

    fee (USDC) = shares * rate * (p * (1 - p)) ** exponent

Weather markets (Oct 2026): rate 0.05, exponent 1, taker-only, makers pay
nothing and get a 25% rebate share. At p=0.50 that's 1.25c/share (2.5% of
notional); at p=0.10 it's 0.45c/share (4.5% of notional!). The fee is largest
relative to price on cheap longshots, which is exactly where naive
"model says 15%, market says 8%" trades live -- always net it out.
"""


def taker_fee(shares: float, price: float, rate: float, exponent: float = 1.0) -> float:
    if rate <= 0 or shares <= 0:
        return 0.0
    p = min(max(price, 0.0), 1.0)
    return round(shares * rate * (p * (1.0 - p)) ** exponent, 5)


def fee_per_share(price: float, rate: float, exponent: float = 1.0) -> float:
    if rate <= 0:
        return 0.0
    p = min(max(price, 0.0), 1.0)
    return rate * (p * (1.0 - p)) ** exponent
