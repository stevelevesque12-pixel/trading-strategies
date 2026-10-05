"""Frozen Engine A settings (the walk-forward pick, identical to tradingview/mcl_trend_dip.pine)."""

ENGINE_A_PARAMS = {"session": "us", "trail_k": 2.5, "target_r": 2.0, "be_r": 1.0, "htf_rule": "60min",
                   "htf_len": 50, "slope_bars": 12, "base_ema": 100, "fast": 13, "dip_k": 1.0,
                   "dip_bars": 6, "swing_lb": 8}
