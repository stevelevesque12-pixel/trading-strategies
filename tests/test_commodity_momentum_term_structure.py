import math
from datetime import datetime, timedelta

import pytest

from quantconnect.commodity_momentum_term_structure.selection import (
    ContractQuote,
    annualized_roll_yield,
    pick_roll_pair,
    select_targets,
)

NOW = datetime(2026, 1, 2)


def q(key, days, price, oi=100):
    return ContractQuote(key, NOW + timedelta(days=days), price, oi)


def test_roll_yield_sign_and_scale_invariance():
    assert annualized_roll_yield(102, 100, 30) > 0  # backwardation
    assert annualized_roll_yield(98, 100, 30) < 0  # contango
    # Same percentage spread at very different price levels ranks the same.
    assert math.isclose(annualized_roll_yield(8160, 8000, 60), annualized_roll_yield(3.06, 3.0, 60))


def test_roll_yield_rejects_bad_inputs():
    with pytest.raises(ValueError):
        annualized_roll_yield(0, 100, 30)
    with pytest.raises(ValueError):
        annualized_roll_yield(100, 100, 0)


def test_pick_roll_pair_skips_unpriced_no_oi_and_soon_expiring():
    quotes = [
        q("front_unpriced", 5, 0),
        q("front", 10, 50),
        q("too_soon", 20, 51),
        q("no_oi", 50, 52, oi=0),
        q("target", 80, 53),
        q("later", 110, 54),
    ]
    near, far = pick_roll_pair(quotes, NOW, min_days_to_expiry=35)
    assert near.key == "front"
    assert far.key == "target"


def test_pick_roll_pair_none_when_no_deferred():
    assert pick_roll_pair([q("a", 10, 50)], NOW) is None
    assert pick_roll_pair([q("a", 10, 50), q("b", 20, 51)], NOW, min_days_to_expiry=35) is None


def test_select_targets_double_sort():
    # 6 commodities -> tertile of 2, half of 1 per leg.
    ry = {"A": 0.30, "B": 0.20, "C": 0.05, "D": 0.0, "E": -0.10, "F": -0.20}
    mom = {"A": 0.001, "B": 0.003, "C": 0.009, "D": -0.009, "E": -0.002, "F": 0.004}
    targets = select_targets(ry, mom, gross_per_side=0.05)
    assert targets == {"B": 0.05, "E": -0.05}


def test_select_targets_equal_weights_per_leg():
    ry = {k: -i for i, k in enumerate("ABCDEFGHIJKL")}  # 12 -> tertile 4, half 2
    mom = {k: i for i, k in enumerate("ABCDEFGHIJKL")}
    targets = select_targets(ry, mom, gross_per_side=0.06)
    assert targets == {"D": 0.03, "C": 0.03, "I": -0.03, "J": -0.03}


def test_select_targets_small_universe():
    # Original code: tertile 0 shorted everything, tertile 1 divided by zero.
    assert select_targets({"A": 1, "B": 0}, {"A": 0, "B": 0}) == {}
    targets = select_targets({"A": 1, "B": 0, "C": -1}, {"A": 0, "B": 0, "C": 0})
    assert targets == {"A": 0.05, "C": -0.05}


def test_select_targets_ignores_keys_without_momentum():
    ry = {"A": 1, "B": 0, "C": -1, "X": 5}
    mom = {"A": 0, "B": 0, "C": 0}
    assert "X" not in select_targets(ry, mom)
