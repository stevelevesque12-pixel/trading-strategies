"""Forecast paths + observations -> probability of each bucket.

For each source, each member m gives a continuous forecast R_m of the
extreme over the *remaining* hours of the local day (after bias correction,
spread scaling, and nudging toward the latest observation). The settlement
value is

    V = max(E_obs, round(X))   (highest-temperature markets; min for lowest)

with X ~ N(R_m, sd^2) per member (kernel dressing) and E_obs the extreme
already observed today. That gives a closed-form CDF over the resolved grid,
so no Monte Carlo noise:

    P(V <= v) = 1[v >= E_obs] * mean_m Phi((v + step/2 - R_m) / sd)

Sources are mixed with weights. Everything is in the market's unit.
"""

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from .forecast import DayForecast
from .markets import WeatherEvent
from .observations import ObsState

_SQRT2 = math.sqrt(2.0)
_erf = np.frompyfunc(math.erf, 1, 1)


def norm_cdf(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _erf(x / _SQRT2).astype(float))


@dataclass
class SourceCalibration:
    bias: float = 0.0  # added to the forecast, market units
    spread_scale: float = 1.0  # multiplies each member's deviation from the ensemble mean
    kernel_sd: float = 1.5  # dressing kernel, market units
    weight: float = 1.0
    n: int = 0  # samples behind this calibration (0 = defaults)
    # Intraday mode (same day, obs in hand): bias is applied to the remaining extreme *after*
    # nudging to the latest reading, and kernel_sd is the empirical error of that remaining
    # extreme at this local hour -- no further shrinking.
    intraday: bool = False


# Uncalibrated defaults (°C). Typical 2m-temperature extreme errors of a raw
# global ensemble at an airport point; the kernel adds what the members miss.
_KERNEL_C = {0: 0.7, 1: 1.0, 2: 1.3, 3: 1.6, 4: 1.9}
_NBM_KERNEL_C = {0: 0.25, 1: 0.25, 2: 0.35, 3: 0.5, 4: 0.6}  # NBM's own XND/TSD already ~= its RMSE
DEFAULT_WEIGHTS = {"nbm": 6.0, "ecmwf_ens": 2.0, "gefs": 1.0, "icon_ens": 1.0, "ukmo_ens": 1.0, "gem_ens": 0.5}


def default_calibration(source: str, lead: int, unit: str, kind: str, hourly_only: bool) -> SourceCalibration:
    scale = 1.8 if unit == "F" else 1.0
    table = _NBM_KERNEL_C if source == "nbm" else _KERNEL_C
    k = table[min(lead, max(table))] * scale
    # Routine-only settlement samples the curve once an hour, so it under-reads
    # the true max (over-reads the min) by a few tenths.
    bias = 0.0
    if hourly_only:
        bias = (-0.3 if kind == "high" else 0.3) * scale
    return SourceCalibration(bias=bias, kernel_sd=k, weight=DEFAULT_WEIGHTS.get(source, 1.0))


@dataclass
class SourceView:
    source: str
    center: float  # mean of R_m, the forecast extreme before obs-floor
    spread: float  # std of R_m
    kernel_sd: float
    weight: float


@dataclass
class ModelOutput:
    probs: np.ndarray  # one per bucket, sums to 1
    observed_extreme: Optional[float]
    expected: float  # mean resolved value
    sources: list[SourceView] = field(default_factory=list)
    final: bool = False  # no forecast hours left: the outcome is (nearly) known
    calibrated_frac: float = 0.0  # weight share of sources with calibration history


def remaining_extreme(fc: DayForecast, cal: SourceCalibration, kind: str,
                      obs: Optional[ObsState], asof_ts: float, tau_hours: float = 8.0):
    """Per-member continuous extreme over the hours still to come, and hours-to-extreme."""
    t = fc.times
    paths = fc.paths + (0.0 if cal.intraday else cal.bias)
    future = t > asof_ts
    if not future.any():
        return None, 0.0
    if obs is not None and obs.last_time is not None:
        lt = obs.last_time.timestamp()
        if asof_ts - lt <= 3 * 3600:
            # Each member's current error vs the latest reading, decaying ahead.
            model_now = np.array([np.interp(lt, t, row) if not np.isnan(row).all() else np.nan
                                  for row in paths])
            err = np.nan_to_num(obs.last_value - model_now)
            decay = np.where(t > lt, np.exp(-(t - lt) / (tau_hours * 3600.0)), 0.0)
            paths = paths + err[:, None] * decay[None, :]
    sub = paths[:, future]
    good = ~np.all(np.isnan(sub), axis=0)
    if not good.any():
        return None, 0.0
    sub = sub[:, good]
    tf = t[future][good]
    sub_f = np.where(np.isnan(sub), -np.inf if kind == "high" else np.inf, sub)
    idx = np.argmax(sub_f, axis=1) if kind == "high" else np.argmin(sub_f, axis=1)
    R = sub_f[np.arange(len(sub_f)), idx]
    R = R[np.isfinite(R)]
    if not len(R):
        return None, 0.0
    c = R.mean()
    R = c + cal.spread_scale * (R - c)
    if cal.intraday:
        R = R + cal.bias
    hours_to_ext = float(np.mean((tf[idx] - asof_ts) / 3600.0))
    return R, max(hours_to_ext, 0.0)


def _bucket_edges(ev: WeatherEvent):
    step = ev.precision
    width_pad = (1.0 - step) if step < 1 else 0.0
    for b in ev.buckets:
        lo = -np.inf if b.lo is None else b.lo
        hi = np.inf if b.hi is None else b.hi + width_pad
        yield lo, hi


def source_bucket_probs(ev: WeatherEvent, R: np.ndarray, sd: float, floor: Optional[float]) -> np.ndarray:
    step, kind = ev.precision, ev.kind
    sd = max(sd, 1e-3)
    out = []
    for lo, hi in _bucket_edges(ev):
        if kind == "high":
            def F(v):  # P(V <= v)
                if v == np.inf:
                    return 1.0
                if v == -np.inf or (floor is not None and v < floor - 1e-9):
                    return 0.0
                return float(norm_cdf((v + step / 2 - R) / sd).mean())
            out.append(F(hi) - F(lo - step))
        else:
            def S(v):  # P(V >= v)
                if v == -np.inf:
                    return 1.0
                if v == np.inf or (floor is not None and v > floor + 1e-9):
                    return 0.0
                return float(norm_cdf((R - (v - step / 2)) / sd).mean())
            out.append(S(lo) - S(hi + step))
    p = np.clip(np.array(out), 0.0, 1.0)
    return p / p.sum() if p.sum() > 0 else p


def bucket_probabilities(ev: WeatherEvent, forecasts: list[tuple[DayForecast, SourceCalibration]],
                         obs: Optional[ObsState], asof_ts: float, tz: str,
                         min_sd: float = 0.25, tau_hours: float = 8.0) -> Optional[ModelOutput]:
    floor = obs.extreme if obs is not None else None
    mix = np.zeros(len(ev.buckets))
    total_w = 0.0
    cal_w = 0.0
    views = []
    any_future = False
    for fc, cal in forecasts:
        R, h_ext = remaining_extreme(fc, cal, ev.kind, obs, asof_ts, tau_hours)
        if R is None:
            continue
        any_future = True
        sd = cal.kernel_sd
        if obs is not None and obs.n > 0 and not cal.intraday:
            # Intraday: the closer the remaining extreme, the less can still go wrong.
            sd = sd * math.sqrt(min(1.0, max(h_ext, 2.0) / 12.0))
        sd = max(sd, min_sd * (1.8 if ev.unit == "F" else 1.0))
        p = source_bucket_probs(ev, R, sd, floor)
        if p.sum() <= 0:
            continue
        mix += cal.weight * p
        total_w += cal.weight
        views.append(SourceView(fc.source, float(R.mean()), float(R.std()), sd, cal.weight))
        cal_w += cal.weight if cal.n > 0 else 0.0

    if total_w == 0:
        if floor is None or any_future:
            return None
        # Day is over (or no forecast hours left): outcome is the observed extreme.
        probs = np.array([1.0 if b.contains(floor, ev.precision) else 0.0 for b in ev.buckets])
        return ModelOutput(probs, floor, float(floor), [], final=True)

    probs = mix / total_w
    probs = probs / probs.sum()
    mids = []
    for lo, hi in _bucket_edges(ev):
        lo_f = hi - 1 if lo == -np.inf else lo
        hi_f = lo + 1 if hi == np.inf else hi
        mids.append(0.5 * (lo_f + hi_f))
    return ModelOutput(probs, floor, float(np.dot(probs, mids)), views, final=False,
                       calibrated_frac=cal_w / total_w)
