"""Position sizing and prop-firm-style daily risk gating for the FVT strategy."""

from failed2s.risk import RiskManager  # noqa: F401 (re-exported, same daily-loss-limit/trade-cap logic)


def contracts_for_target_risk(
    stop_points: float,
    point_value: float,
    target_risk: float = 1000.0,
    min_contracts: int = 1,
    max_contracts: int = 3,
) -> int:
    """
    Size contracts so stop-loss distance * point_value * contracts lands near
    `target_risk` dollars, matching the PDF's "$1k risk per trade with 1, 2,
    or 3 contracts" guidance. Rounds to the nearest contract count and clamps
    to [min_contracts, max_contracts].
    """
    if stop_points <= 0:
        return min_contracts
    raw = target_risk / (stop_points * point_value)
    return max(min_contracts, min(max_contracts, round(raw)))
