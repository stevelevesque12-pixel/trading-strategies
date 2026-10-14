import numpy as np
import pandas as pd

from carver_trend.strategy import (
    Params,
    blended_sigma,
    buffer_width,
    buffered_target,
    capped_forecast,
    instrument_diversification_multiplier,
    optimal_position,
)


def test_buffer_holds_inside_band_and_trades_to_nearest_edge():
    assert buffered_target(current=5, optimal=5.4, buffer=1.0) == 5
    assert buffered_target(current=0, optimal=5.0, buffer=1.0) == 4
    assert buffered_target(current=10, optimal=5.0, buffer=1.0) == 6


def test_position_sizing_matches_carver_formula():
    p = Params()
    # forecast 10, $1M, IDM 1, weight 1, mult 50, price 4000, sigma 16% -> 1M*0.2/(50*4000*0.16) = 6.25
    assert np.isclose(optimal_position(10, 1e6, 1.0, 1.0, 50, 4000, 0.16, p), 6.25)
    assert np.isclose(buffer_width(1e6, 1.0, 1.0, 50, 4000, 0.16, p), 0.625)


def test_forecast_is_capped_and_signed_with_trend():
    p = Params()
    up = pd.Series(100 * 1.002 ** np.arange(400))
    sig = blended_sigma(up * (1 + 0.01 * np.sin(np.arange(400))), p)
    f = capped_forecast(up, sig, p)
    assert (f.iloc[100:] > 0).all()
    assert f.abs().max() <= p.abs_forecast_cap
    assert (capped_forecast(up[::-1].reset_index(drop=True), sig, p).iloc[100:] < 0).all()


def test_idm_table():
    assert instrument_diversification_multiplier(1) == 1.0
    assert instrument_diversification_multiplier(4) == 1.56
    assert instrument_diversification_multiplier(10) == 2.2
