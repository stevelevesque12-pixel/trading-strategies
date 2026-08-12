"""Risk-based contract sizing, kept pure/testable and separate from the HTTP/broker I/O."""

from failed2s.instruments import Instrument


def contracts_for_risk(
    instrument: Instrument,
    entry_price: float,
    stop_price: float,
    risk_usd: float,
    max_contracts: int,
) -> int:
    """
    Number of contracts such that a stop-out risks approximately `risk_usd`,
    floored to a whole contract and clamped to [1, max_contracts].

    A floored risk that comes out below one contract's worth still trades
    1 contract (the smallest size Tradovate allows) rather than 0 -- if
    that's not acceptable for a given stop distance, reject the signal
    before calling this rather than silently skip it here.
    """
    risk_points = abs(entry_price - stop_price)
    if risk_points <= 0:
        raise ValueError("stop_price must differ from entry_price")

    risk_per_contract = risk_points * instrument.point_value
    if risk_per_contract <= 0:
        raise ValueError("computed risk per contract is not positive")

    contracts = int(risk_usd // risk_per_contract)
    return max(1, min(contracts, max_contracts))
