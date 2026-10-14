"""Futures instrument specs (tick size, point value, fees) for common prop-firm symbols."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    point_value: float  # dollars per 1.00 price move, per contract
    round_trip_fee: float  # $ per contract, both sides: exchange + NFA + clearing + commission


# Fees: Tradeify schedule (Apr 28 2026, free membership), total round-trip cost per contract.
INSTRUMENTS = {
    "MES": Instrument("MES", 0.25, 5.0, 1.82),
    "ES": Instrument("ES", 0.25, 50.0, 5.76),
    "MNQ": Instrument("MNQ", 0.25, 2.0, 1.82),
    "NQ": Instrument("NQ", 0.25, 20.0, 5.76),
    "MYM": Instrument("MYM", 1.0, 0.5, 1.82),
    "YM": Instrument("YM", 1.0, 5.0, 5.76),
    "M2K": Instrument("M2K", 0.1, 5.0, 1.82),
    "RTY": Instrument("RTY", 0.1, 50.0, 5.76),
    "MCL": Instrument("MCL", 0.01, 100.0, 2.12),
    "CL": Instrument("CL", 0.01, 1000.0, 6.00),
    "MGC": Instrument("MGC", 0.1, 10.0, 2.12),
    "GC": Instrument("GC", 0.1, 100.0, 6.20),
}
