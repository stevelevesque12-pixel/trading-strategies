"""Futures instrument specs (tick size, point value) for common prop-firm symbols."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Instrument:
    symbol: str
    tick_size: float
    point_value: float  # dollars per 1.00 price move, per contract


INSTRUMENTS = {
    "MES": Instrument("MES", 0.25, 5.0),
    "ES": Instrument("ES", 0.25, 50.0),
    "MNQ": Instrument("MNQ", 0.25, 2.0),
    "NQ": Instrument("NQ", 0.25, 20.0),
    "MCL": Instrument("MCL", 0.01, 100.0),
    "CL": Instrument("CL", 0.01, 1000.0),
    "MGC": Instrument("MGC", 0.1, 10.0),
    "GC": Instrument("GC", 0.1, 100.0),
}
