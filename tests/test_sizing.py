import pytest

from failed2s.instruments import INSTRUMENTS
from webhook.sizing import contracts_for_risk

MES = INSTRUMENTS["MES"]  # tick_size=0.25, point_value=5.0


def test_basic_sizing():
    # stop distance 10 pts * $5/pt = $50 risk/contract; $200 risk -> 4 contracts
    assert contracts_for_risk(MES, entry_price=6000, stop_price=5990, risk_usd=200, max_contracts=10) == 4


def test_clamped_to_minimum_one_contract():
    # stop distance 1000 pts -> risk/contract $5000, way more than $10 budget -> still 1
    assert contracts_for_risk(MES, entry_price=6000, stop_price=5000, risk_usd=10, max_contracts=10) == 1


def test_clamped_to_max_contracts():
    # tiny stop distance would compute a huge size -> capped
    assert contracts_for_risk(MES, entry_price=6000, stop_price=5999, risk_usd=100_000, max_contracts=5) == 5


def test_direction_agnostic():
    # short-side stop above entry should size identically to the long-side mirror
    long_size = contracts_for_risk(MES, entry_price=6000, stop_price=5990, risk_usd=200, max_contracts=10)
    short_size = contracts_for_risk(MES, entry_price=6000, stop_price=6010, risk_usd=200, max_contracts=10)
    assert long_size == short_size == 4


def test_rejects_zero_distance():
    with pytest.raises(ValueError):
        contracts_for_risk(MES, entry_price=6000, stop_price=6000, risk_usd=200, max_contracts=5)
