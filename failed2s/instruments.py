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
    "SIL": Instrument("SIL", 0.005, 1000.0),
    "SI": Instrument("SI", 0.005, 5000.0),
    "MYM": Instrument("MYM", 1.0, 0.5),
    "YM": Instrument("YM", 1.0, 5.0),
    "M2K": Instrument("M2K", 0.1, 5.0),
    "RTY": Instrument("RTY", 0.1, 50.0),
}
