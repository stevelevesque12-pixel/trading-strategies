"""Per-station, per-source, per-lead calibration (bias, spread, weight).

The fit is deliberately simple and heavily shrunk toward defaults -- with
60-120 samples per station a richer EMOS regression mostly fits noise:

    residual_i = observed_i - center_i
    bias       = shrink(mean(residual))
    total_var  = var(residual) - step^2/12        (rounding is modelled separately)
    kernel^2   = total_var - spread_scale^2 * mean(spread^2)
    weight     = default_weight * default_mse / mse   (sources that miss more count less)

Samples come from three places:
* NBM archive (US): exact as-issued forecasts -> full calibration.
* Open-Meteo previous-runs (deterministic, any station): station bias only.
* The bot's own forecast log (`Ledger.forecast_samples`): full calibration
  of whatever the bot actually traded on, including live ensemble spreads.
"""

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from .model import SourceCalibration, default_calibration

SHRINK_N = 15  # pseudo-samples of prior weight

# Same-day error of the remaining-day extreme vs the obs-anchored NBM forecast, by local hour
# (°F, pooled over 11 US stations, 2026-06-19..08-17 -- the backtest's training half).
# (bias, sd): realised remaining extreme minus forecast.
INTRADAY_PRIOR_F = {
    "high": {8: (-0.89, 2.01), 9: (-0.88, 1.96), 10: (-0.72, 1.97), 11: (-0.46, 2.03), 12: (-0.37, 1.99),
             13: (0.12, 1.85), 14: (0.76, 1.91), 15: (0.95, 1.91), 16: (1.08, 1.70), 17: (1.56, 1.33),
             18: (1.64, 1.20), 19: (2.01, 1.13), 20: (1.81, 1.13), 21: (1.75, 1.25)},
    "low": {8: (-1.56, 2.17), 9: (-1.25, 2.50), 10: (-0.85, 2.68), 11: (-0.60, 2.76), 12: (-0.63, 2.72),
            13: (-0.68, 2.76), 14: (-0.36, 2.64), 15: (-0.29, 2.71), 16: (-0.53, 2.58), 17: (-0.30, 2.33),
            18: (-0.44, 2.29), 19: (-0.36, 2.22), 20: (-0.31, 2.10), 21: (-0.08, 1.81)},
}


def intraday_prior(kind: str, hour: int, unit: str, source: str) -> SourceCalibration:
    table = INTRADAY_PRIOR_F[kind]
    h = min(max(hour, min(table)), max(table))
    bias, sd = table[h]
    scale = 1.0 if unit == "F" else 1 / 1.8
    if source != "nbm":
        bias = 0.0  # the bias pattern is NBM's diurnal shape error; don't transplant it
    return SourceCalibration(bias=bias * scale, kernel_sd=sd * scale, intraday=True,
                             weight=default_calibration(source, 0, unit, kind, False).weight)


def fit_intraday(residuals, prior: SourceCalibration) -> SourceCalibration:
    r = np.asarray([x for x in residuals if x is not None], dtype=float)
    if len(r) < 10:
        return prior
    med = np.median(r)
    mad = np.median(np.abs(r - med)) * 1.4826 + 1e-6
    r = r[np.abs(r - med) <= 5 * mad]
    n = len(r)
    w = n / (n + 3 * SHRINK_N)
    bias = w * float(r.mean()) + (1 - w) * prior.bias
    sd = math.sqrt(w * float(r.var()) + (1 - w) * prior.kernel_sd ** 2)
    return SourceCalibration(bias=bias, kernel_sd=sd, intraday=True, weight=prior.weight, n=n)


def fit(samples: Iterable[tuple[float, float, float]], prior: SourceCalibration,
        step: float = 1.0, bias_only: bool = False) -> SourceCalibration:
    """samples: (center, spread, observed)."""
    arr = np.array([s for s in samples if s[0] is not None and s[2] is not None], dtype=float)
    if len(arr) < 5:
        return prior
    c, spread, obs = arr[:, 0], np.nan_to_num(arr[:, 1]), arr[:, 2]
    resid = obs - c
    # Trim gross outliers (bad obs, station outages) before fitting.
    med = np.median(resid)
    mad = np.median(np.abs(resid - med)) * 1.4826 + 1e-6
    keep = np.abs(resid - med) <= 5 * mad
    resid, spread = resid[keep], spread[keep]
    n = len(resid)
    w = n / (n + SHRINK_N)
    bias = w * float(resid.mean()) + (1 - w) * prior.bias
    if bias_only:
        return SourceCalibration(bias=bias, spread_scale=prior.spread_scale, kernel_sd=prior.kernel_sd,
                                 weight=prior.weight, n=n)

    mse = float(np.mean((resid - bias) ** 2))
    total_var = max(mse - step ** 2 / 12.0, 0.05)
    ens_var = float(np.mean(spread ** 2))
    scale = 1.0
    if ens_var > 0.8 * total_var:
        # Over-dispersed ensemble: shrink members, keep a small kernel.
        scale = math.sqrt(0.8 * total_var / ens_var)
    k2 = max(total_var - scale ** 2 * ens_var, 0.04)
    kernel = math.sqrt(w * k2 + (1 - w) * prior.kernel_sd ** 2)
    prior_mse = prior.kernel_sd ** 2 + ens_var + step ** 2 / 12.0
    weight = prior.weight * min(3.0, max(0.25, prior_mse / max(mse, 1e-3)))
    return SourceCalibration(bias=bias, spread_scale=w * scale + (1 - w) * prior.spread_scale,
                             kernel_sd=kernel, weight=w * weight + (1 - w) * prior.weight, n=n)


def _key(city: str, kind: str, source: str, lead: int) -> str:
    return f"{city}|{kind}|{source}|{min(lead, 4)}"


class Calibrator:
    """Lookup with fallbacks: exact lead -> nearest calibrated lead's bias -> defaults."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path) if path else None
        self.table: dict[str, dict] = {}
        if self.path and self.path.exists():
            self.table = json.loads(self.path.read_text())

    def get(self, city: str, kind: str, source: str, lead: int, unit: str,
            hourly_only: bool) -> SourceCalibration:
        prior = default_calibration(source, lead, unit, kind, hourly_only)
        hit = self.table.get(_key(city, kind, source, lead))
        if hit:
            return SourceCalibration(**hit)
        # Same-day (lead 0) has no clean pre-obs samples: borrow lead 1's bias.
        for alt in (1, 2) if lead == 0 else (lead - 1, lead + 1):
            near = self.table.get(_key(city, kind, source, alt))
            if near:
                prior.bias = near["bias"]
                prior.weight = near["weight"]
                break
        return prior

    def intraday(self, city: str, kind: str, source: str, hour: int, unit: str) -> SourceCalibration:
        for key in (f"{city}|{kind}|{source}|0|h{hour:02d}", f"*|{kind}|{source}|0|h{hour:02d}"):
            hit = self.table.get(key)
            if hit:
                return SourceCalibration(**hit)
        return intraday_prior(kind, hour, unit, source)

    def set_intraday(self, city: str, kind: str, source: str, hour: int, cal: SourceCalibration) -> None:
        self.table[f"{city}|{kind}|{source}|0|h{hour:02d}"] = asdict(cal)

    def set(self, city, kind, source, lead, cal: SourceCalibration) -> None:
        self.table[_key(city, kind, source, lead)] = asdict(cal)

    def save(self, path: Optional[str] = None) -> None:
        p = Path(path) if path else self.path
        if p is None:
            raise ValueError("no calibration path")
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.table, indent=1, sort_keys=True))

    def fit_group(self, city, kind, source, lead, samples, unit, hourly_only, step=1.0, bias_only=False):
        prior = self.get(city, kind, source, lead, unit, hourly_only) if bias_only else \
            default_calibration(source, lead, unit, kind, hourly_only)
        cal = fit(samples, prior, step=step, bias_only=bias_only)
        if cal.n:
            self.set(city, kind, source, lead, cal)
        return cal
