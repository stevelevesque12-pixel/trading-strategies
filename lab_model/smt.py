"""
SMT (Smart Money Technique) divergence: a "crack in correlation" between two
highly correlated assets (NQ traded, ES used as the correlation check).

Operationalized as: at the most recent two confirmed swing points of a given
kind, the primary asset (NQ) prints a new extreme while the secondary asset
(ES) fails to confirm it -- e.g. NQ makes a lower low while ES's matching
swing low is higher than its own prior one. That failure-to-confirm is the
divergence.
"""

from typing import List, Literal

from failed2s.structure import Swing


def detect_smt(primary_history: List[Swing], secondary_history: List[Swing], kind: Literal["high", "low"]) -> bool:
    """
    kind="low": bullish SMT -- primary prints a new lower swing low while
      secondary's latest swing low is higher than its prior one.
    kind="high": bearish SMT -- primary prints a new higher swing high while
      secondary's latest swing high is lower than its prior one.
    """
    primary = [s for s in primary_history if s.kind == kind]
    secondary = [s for s in secondary_history if s.kind == kind]
    if len(primary) < 2 or len(secondary) < 2:
        return False

    p_new, p_prev = primary[-1], primary[-2]
    s_new, s_prev = secondary[-1], secondary[-2]

    if kind == "low":
        return p_new.price < p_prev.price and s_new.price > s_prev.price
    return p_new.price > p_prev.price and s_new.price < s_prev.price
