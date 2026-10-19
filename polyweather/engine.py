"""Assemble forecasts + observations for an event and run the model."""

import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from . import forecast as fc
from .calibration import Calibrator
from .markets import WeatherEvent
from .model import ModelOutput, bucket_probabilities
from .observations import ObsState, day_obs, fetch_hf_nws


@dataclass
class Evaluation:
    event: WeatherEvent
    asof: datetime
    lead: int
    obs: Optional[ObsState]
    output: Optional[ModelOutput]
    forecasts: list = field(default_factory=list)


class ForecastCache:
    """Live-mode fetch cache so a polling loop doesn't hammer the APIs."""

    def __init__(self, ensemble_ttl: float = 45 * 60, nbm_ttl: float = 30 * 60, obs_ttl: float = 4 * 60):
        self.ttl = {"ens": ensemble_ttl, "nbm": nbm_ttl, "obs": obs_ttl}
        self._store: dict = {}

    def get(self, kind: str, key, loader):
        hit = self._store.get((kind, key))
        if hit and time.time() - hit[0] < self.ttl[kind]:
            return hit[1]
        val = loader()
        self._store[(kind, key)] = (time.time(), val)
        return val


def local_lead(ev: WeatherEvent, asof: datetime) -> int:
    return (ev.target_date - asof.astimezone(ZoneInfo(ev.station.tz)).date()).days


def evaluate_event(ev: WeatherEvent, asof: datetime, calibrator: Calibrator, cache: ForecastCache,
                   use_nbm: bool = True, use_ensembles: bool = True,
                   ensemble_models: Optional[list[str]] = None) -> Evaluation:
    st = ev.station
    lead = local_lead(ev, asof)
    obs_state = None
    if lead <= 0:
        include_specials = not ev.hourly_only
        obs = cache.get("obs", (st.icao, ev.target_date, include_specials),
                        lambda: day_obs(st.icao, ev.target_date, st.tz, ev.unit, include_specials, live=True))
        hf = None
        if st.is_us:
            try:
                hf = cache.get("obs", (st.icao, "hf"), lambda: fetch_hf_nws(st.icao, ev.unit))
            except Exception:
                hf = None
        obs_state = ObsState.from_obs(obs, ev.kind, asof, hf)

    forecasts = []
    if use_nbm and st.nbm:
        runtime, rows = cache.get("nbm", st.icao, lambda: fc.latest_nbm(st.icao, asof))
        if rows:
            d = fc.nbm_day_forecast(rows, runtime, st, ev.target_date, ev.kind)
            if d is not None:
                forecasts.append(d)
    if use_ensembles:
        try:
            payload = cache.get("ens", st.icao, lambda: fc.fetch_open_meteo_ensemble(st, ensemble_models))
            forecasts.extend(fc.ensemble_day_forecasts(payload, st, ev.target_date))
        except Exception as exc:  # rate limits, outages: trade on what we have
            print(f"  ensemble fetch failed for {st.icao}: {exc}")

    lead_c = max(lead, 0)
    if obs_state is not None and obs_state.last_value is not None:
        hour = asof.astimezone(ZoneInfo(st.tz)).hour
        pairs = [(f, calibrator.intraday(ev.city, ev.kind, f.source, hour, ev.unit)) for f in forecasts]
    else:
        pairs = [(f, calibrator.get(ev.city, ev.kind, f.source, lead_c, ev.unit, ev.hourly_only)) for f in forecasts]
    out = bucket_probabilities(ev, pairs, obs_state, asof.timestamp(), st.tz)
    return Evaluation(ev, asof, lead, obs_state, out, pairs)


def now_utc() -> datetime:
    return datetime.now(timezone.utc)
