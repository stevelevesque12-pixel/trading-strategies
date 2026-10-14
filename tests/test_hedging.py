import pytest

from failed2s.hedging import Position, check_entry, mini_equivalent, product_group, root_symbol


@pytest.mark.parametrize("symbol,root", [
    ("MESZ6", "MES"), ("MESZ2026", "MES"), ("MES1!", "MES"), ("ES", "ES"), ("ESH7", "ES"),
    ("M2KZ6", "M2K"), ("6EZ6", "6E"), ("M6EZ6", "M6E"), ("SILZ6", "SIL"), ("SIZ6", "SI"),
    ("mnq1!", "MNQ"), ("XYZ", "XYZ"),
])
def test_root_symbol(symbol, root):
    assert root_symbol(symbol) == root


def test_mini_and_micro_share_a_group():
    assert product_group("MES") == product_group("NQ") == product_group("M2KZ6") == "Equity Index"
    assert product_group("GC") == product_group("MGC") == "Metals"


# Examples straight from Tradeify's rules page.
@pytest.mark.parametrize("held,new_symbol,new_qty", [
    (Position("A", "ESZ6", 1), "ESZ6", -1),     # same instrument
    (Position("A", "ESZ6", 1), "NQZ6", -1),     # same group
    (Position("A", "MESZ6", 3), "NQZ6", -1),    # size irrelevant
    (Position("B", "MNQZ6", -2), "MESZ6", 1),   # across accounts
    (Position("A", "ESZ6", 1), "FVS", 1),       # long equities + long volatility
])
def test_hedges_blocked(held, new_symbol, new_qty):
    assert check_entry(new_symbol, new_qty, [held], account_id="A").startswith("hedge:")


@pytest.mark.parametrize("held,new_symbol,new_qty", [
    (Position("A", "ESZ6", 1), "NQZ6", 1),      # same direction
    (Position("A", "ESZ6", 1), "MNQZ6", 1),     # mini + micro, same direction
    (Position("A", "ESZ6", 1), "CLZ6", -1),     # different groups
    (Position("A", "GCZ6", 1), "ZCZ6", -1),
    (Position("B", "ESZ6", 1), "ESZ6", 1),      # copy trading
    (Position("A", "ESZ6", 0), "ESZ6", -1),     # flat isn't a position
    (Position("A", "ESZ6", 1), "FVS", -1),      # long equities + short volatility: same exposure
])
def test_allowed(held, new_symbol, new_qty):
    assert check_entry(new_symbol, new_qty, [held], account_id="A") is None


def test_contract_limit_counts_ten_micros_per_mini():
    assert mini_equivalent("MESZ6", 10) == pytest.approx(1.0)
    held = [Position("A", "ESZ6", 2), Position("B", "ESZ6", 5)]  # B doesn't count toward A's limit
    assert check_entry("MESZ6", 10, held, account_id="A", contract_limit_minis=3) is None
    reason = check_entry("MESZ6", 11, held, account_id="A", contract_limit_minis=3)
    assert reason.startswith("contract limit:")


def test_zero_qty_rejected():
    with pytest.raises(ValueError):
        check_entry("ES", 0, [])
