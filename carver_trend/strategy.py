"""Local port of the QuantConnect "Futures Fast Trend Following Long And Short
With Trend Strength" algorithm (Carver, *Advanced Futures Trading Strategies*,
strategy 7: EWMAC fast/slow with a continuous, capped forecast).

Pieces mirrored from the QC algorithm's parameters:
  slow_ema_span  = 2 ** 7 = 128, fast span = slow / 4 = 32
  (QC default is exponent 6 = EWMAC 16/64; 32/128 tested better here)
  abs_forecast_cap = 20, sigma_span = 32, target_risk (tau) = 0.2,
  blend_years = 3, buffer_scaler = 0.1

Everything runs on daily bars (session close). A target computed from the
close of day t is filled at the open of day t+1.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

BUSINESS_DAYS_PER_YEAR = 256

# Forecast scalars keyed by fast EMA span (Carver, Table 29).
FORECAST_SCALAR_BY_FAST_SPAN = {2: 12.1, 4: 8.53, 8: 5.95, 16: 4.10, 32: 2.79, 64: 1.91}


def instrument_diversification_multiplier(n: int) -> float:
    """Carver, Table 16."""
    if n <= 0:
        return 1.0
    table = {1: 1.00, 2: 1.20, 3: 1.48, 4: 1.56, 5: 1.70, 6: 1.90, 7: 2.10}
    if n in table:
        return table[n]
    if n <= 14:
        return 2.20
    if n <= 24:
        return 2.30
    if n <= 29:
        return 2.40
    return 2.50


@dataclass(frozen=True)
class Params:
    slow_ema_span: int = 128  # QC slow_ema_span_exponent = 7
    abs_forecast_cap: float = 20.0
    sigma_span: int = 32
    target_risk: float = 0.2
    blend_years: int = 3
    buffer_scaler: float = 0.1

    @property
    def fast_ema_span(self) -> int:
        return self.slow_ema_span // 4

    @property
    def forecast_scalar(self) -> float:
        return FORECAST_SCALAR_BY_FAST_SPAN[self.fast_ema_span]


def blended_sigma(close: pd.Series, p: Params) -> pd.Series:
    """Annualised % volatility: 70% EWMA(sigma_span) + 30% long-run average
    of that EWMA over `blend_years` (expanding until enough history)."""
    returns = close.pct_change()
    ewma_sigma = returns.ewm(span=p.sigma_span, adjust=False).std() * np.sqrt(BUSINESS_DAYS_PER_YEAR)
    window = p.blend_years * BUSINESS_DAYS_PER_YEAR
    long_run = ewma_sigma.rolling(window, min_periods=1).mean()
    return 0.3 * long_run + 0.7 * ewma_sigma


def capped_forecast(close: pd.Series, sigma_pct: pd.Series, p: Params) -> pd.Series:
    """Risk-adjusted EWMAC crossover, scaled to average |forecast| ~10, capped."""
    fast = close.ewm(span=p.fast_ema_span, adjust=False).mean()
    slow = close.ewm(span=p.slow_ema_span, adjust=False).mean()
    daily_price_vol = close * sigma_pct / np.sqrt(BUSINESS_DAYS_PER_YEAR)
    raw = (fast - slow) / daily_price_vol
    return (raw * p.forecast_scalar).clip(-p.abs_forecast_cap, p.abs_forecast_cap)


def optimal_position(forecast, capital, idm, weight, multiplier, price, sigma_pct, p: Params):
    """N = forecast * capital * IDM * weight * tau / (10 * multiplier * price * sigma%)."""
    return forecast * capital * idm * weight * p.target_risk / (10 * multiplier * price * sigma_pct)


def buffer_width(capital, idm, weight, multiplier, price, sigma_pct, p: Params):
    """B = buffer_scaler * capital * IDM * weight * tau / (multiplier * price * sigma%)."""
    return p.buffer_scaler * capital * idm * weight * p.target_risk / (multiplier * price * sigma_pct)


def buffered_target(current: int, optimal: float, buffer: float) -> int:
    """Trade only when outside [N-B, N+B], and then only to the nearest edge."""
    lower, upper = optimal - buffer, optimal + buffer
    if current < lower:
        return int(round(lower))
    if current > upper:
        return int(round(upper))
    return current
