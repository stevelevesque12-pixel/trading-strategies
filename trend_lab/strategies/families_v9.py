"""
Batch 9: refinements of Engine A (trend_dip_atr), the walk-forward winner.
The core entry stays in a narrow band around the validated settings; the new
dimensions are a volatility/trend-strength regime gate and extra exits
(time stop, exit when the higher-timeframe trend flips). Judged ONLY by the
anchored walk-forward against the baseline engine.
"""

import numpy as np

from .base import common_exits, htf_ema
from .families_v3 import TrendDipATR
from .families_v4 import REGIME_SPACE, _regime
from .base import atr


class TrendDipATRPlus:
    name = "trend_dip_atr_plus"
    description = ("Engine A (ATR dip in a higher-TF trend) plus an optional regime gate (ADX / efficiency ratio / "
                   "ATR percentile), an optional time stop, and an optional exit when the 1h trend flips.")
    space = {
        "session": ["us"],
        "trail_k": [2.0, 2.5, 3.0],
        "target_r": [1.5, 2.0, 3.0],
        "be_r": [None, 1.0],
        "htf_rule": ["60min"],
        "htf_len": [50],
        "slope_bars": [8, 12, 16],
        "base_ema": [100],
        "fast": [13],
        "dip_k": [0.75, 1.0, 1.25],
        "dip_bars": [4, 6, 8],
        "swing_lb": [5, 8, 10],
        "max_bars": [None, 12, 24],
        "exit_htf_flip": [False, True],
    } | REGIME_SPACE

    def generate(self, df, p):
        a = atr(df)
        sig = TrendDipATR().generate(df, p)
        reg = _regime(df, p, a)
        sig["long"] = sig["long"] & reg
        sig["short"] = sig["short"] & reg
        sig["max_bars"] = p["max_bars"]
        if p["exit_htf_flip"]:
            c = df["close"].to_numpy()
            he = htf_ema(df, p["htf_rule"], p["htf_len"])
            sig["exit_long"], sig["exit_short"] = c < he, c > he
        return sig


FAMILIES = [TrendDipATRPlus()]
