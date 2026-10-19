"""Forecast sources, normalised to one shape: hourly temperature *paths*.

A `DayForecast` holds M member paths over the hours of the target local day
(in the market's unit). Keeping whole paths -- not just each member's daily
max -- is what lets the model condition on today's observations: once 2pm is
in, only the remaining hours of each member can still set a new extreme.

Sources:
* Open-Meteo ensembles (global): ECMWF IFS 0.25 (51 members), NCEP GEFS
  (31), DWD ICON (40), CMC GEM (21), UKMO MOGREPS-G (18). Downscaled to the
  station elevation by Open-Meteo's lapse-rate correction.
* NOAA National Blend of Models (US sites), via IEM's archive of the NBS text
  bulletin. NBM is already bias-corrected to the station and ships its own
  standard deviation (TXN/XND), so it is usually the sharpest US source. It
  is also archived, which is what makes an honest US backtest possible.
"""

import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from statistics import NormalDist
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from . import http
from .observations import day_window_utc
from .stations import Station

ENSEMBLE_MODELS = ["ecmwf_ifs025", "gfs025", "icon_seamless", "gem_global", "ukmo_global_ensemble_20km"]
# Open-Meteo echoes models back under these names.
_SOURCE_NAMES = {
    "ecmwf_ifs025_ensemble": "ecmwf_ens", "ncep_gefs025": "gefs", "icon_seamless_eps": "icon_ens",
    "gem_global_ensemble": "gem_ens", "ukmo_global_ensemble_20km": "ukmo_ens",
}
IEM_MOS = "https://mesonet.agron.iastate.edu/api/1/mos.json"


@dataclass
class DayForecast:
    source: str
    times: np.ndarray  # (H,) epoch seconds UTC, hourly, covering the target local day
    paths: np.ndarray  # (M, H) in market unit; NaN where the source has no value
    issued: datetime  # when this forecast became available
    target_date: date

    @property
    def n_members(self) -> int:
        return self.paths.shape[0]

    def lead_days(self, asof: datetime, tz: str) -> int:
        """0 = trading on the target day itself, 1 = the day before, ..."""
        local = asof.astimezone(ZoneInfo(tz)).date()
        return max(0, (self.target_date - local).days)


def _hour_grid(d: date, tz: str) -> np.ndarray:
    start, end = day_window_utc(d, tz)
    n = int((end - start).total_seconds() // 3600)  # 23/24/25 with DST
    return np.array([start.timestamp() + 3600 * i for i in range(n)])


def _om_host(name: str) -> tuple[str, dict]:
    key = os.environ.get("OPEN_METEO_API_KEY")
    if key:
        return f"https://customer-{name}.open-meteo.com", {"apikey": key}
    return f"https://{name}.open-meteo.com", {}


# ---------------------------------------------------------------- Open-Meteo


def parse_open_meteo_ensemble(payload: dict) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Ensemble JSON -> {source: (times_epoch (H,), members (M, H))}."""
    hourly = payload["hourly"]
    times = np.array([datetime.fromisoformat(t).replace(tzinfo=timezone.utc).timestamp()
                      for t in hourly["time"]])
    groups: dict[str, dict[int, list]] = {}
    for key, vals in hourly.items():
        if not key.startswith("temperature_2m"):
            continue
        rest = key[len("temperature_2m"):].lstrip("_")
        m = re.match(r"member(\d+)_?(.*)$", rest)
        member, suffix = (int(m.group(1)), m.group(2)) if m else (0, rest)
        groups.setdefault(suffix or "default", {})[member] = vals
    out = {}
    for suffix, members in groups.items():
        arr = np.array([[np.nan if v is None else float(v) for v in members[k]]
                        for k in sorted(members)], dtype=float)
        out[_SOURCE_NAMES.get(suffix, suffix)] = (times, arr)
    return out


def fetch_open_meteo_ensemble(st: Station, models: Optional[list[str]] = None,
                              forecast_days: int = 4, past_days: int = 1) -> dict:
    host, extra = _om_host("ensemble-api")
    params = {
        "latitude": st.lat, "longitude": st.lon, "elevation": st.elevation_m,
        "hourly": "temperature_2m", "models": ",".join(models or ENSEMBLE_MODELS),
        "timezone": "GMT", "forecast_days": forecast_days, "past_days": past_days,
        "temperature_unit": "fahrenheit" if st.unit == "F" else "celsius", **extra,
    }
    return http.get(f"{host}/v1/ensemble", params=params)


def ensemble_day_forecasts(payload: dict, st: Station, target: date,
                           issued: Optional[datetime] = None) -> list[DayForecast]:
    grid = _hour_grid(target, st.tz)
    out = []
    for source, (times, members) in parse_open_meteo_ensemble(payload).items():
        idx = {int(t): i for i, t in enumerate(times)}
        cols = [idx.get(int(t)) for t in grid]
        if sum(c is not None for c in cols) < len(grid) // 2:
            continue
        paths = np.full((members.shape[0], len(grid)), np.nan)
        for j, c in enumerate(cols):
            if c is not None:
                paths[:, j] = members[:, c]
        paths = paths[~np.all(np.isnan(paths), axis=1)]
        if len(paths):
            out.append(DayForecast(source, grid, paths, issued or datetime.now(timezone.utc), target))
    return out


def fetch_open_meteo_previous_runs(st: Station, start: date, end: date, models: list[str],
                                   lead_days: list[int]) -> dict:
    """Deterministic forecasts as issued N days earlier (for calibration)."""
    host, extra = _om_host("previous-runs-api")
    vars_ = ["temperature_2m"] + [f"temperature_2m_previous_day{n}" for n in lead_days if n > 0]
    params = {
        "latitude": st.lat, "longitude": st.lon, "elevation": st.elevation_m,
        "hourly": ",".join(vars_), "models": ",".join(models), "timezone": "GMT",
        "start_date": start.isoformat(), "end_date": end.isoformat(),
        "temperature_unit": "fahrenheit" if st.unit == "F" else "celsius", **extra,
    }
    return http.get(f"{host}/v1/forecast", params=params, cache=end < datetime.now(timezone.utc).date())


# ---------------------------------------------------------------- NBM (US)


def fetch_nbm(icao: str, runtime: datetime, cache: bool = False) -> list[dict]:
    try:
        d = http.get(IEM_MOS, params={"station": icao, "model": "NBS",
                                      "runtime": runtime.strftime("%Y-%m-%dT%H:%MZ")}, cache=cache)
    except Exception:
        return []
    return d.get("data", []) if isinstance(d, dict) else []


# NBS bulletins appear ~1h after cycle time; IEM archives the 00/06/12/18Z cycles reliably.
NBM_AVAILABLE_AFTER = timedelta(hours=1, minutes=45)


def latest_nbm(icao: str, asof: datetime, cache: bool = False, max_back: int = 4, fetch=None):
    """Most recent NBS run public at `asof`, backfilled from the run before it.

    An NBS bulletin's first projection is 6h after its cycle time, so on its
    own the newest run is blind to the next few hours -- exactly the
    afternoon hours that decide a same-day "highest temperature" market.
    Rows from the previous run fill those hours. Returns (runtime, rows).
    """
    fetch = fetch or (lambda cyc: fetch_nbm(icao, cyc, cache=cache))
    t = asof.astimezone(timezone.utc) - NBM_AVAILABLE_AFTER
    cycle = t.replace(hour=(t.hour // 6) * 6, minute=0, second=0, microsecond=0)
    runs = []
    for _ in range(max_back):
        rows = fetch(cycle)
        if rows:
            runs.append((cycle, rows))
            if len(runs) == 2:
                break
        cycle -= timedelta(hours=6)
    if not runs:
        return None, []
    runtime, rows = runs[0]
    if len(runs) > 1:
        have = {r.get("ftime_utc") or r.get("ftime") for r in rows}
        first = min(_utc(r.get("ftime_utc") or r["ftime"]) for r in rows)
        older = [r for r in runs[1][1] if (r.get("ftime_utc") or r.get("ftime")) not in have
                 and _utc(r.get("ftime_utc") or r["ftime"]) < first]
        rows = sorted(older + list(rows), key=lambda r: _utc(r.get("ftime_utc") or r["ftime"]))
    return runtime, rows


def _utc(s: str) -> float:
    return datetime.fromisoformat(s.replace("Z", "")).replace(tzinfo=timezone.utc).timestamp()


def nbm_day_forecast(rows: list[dict], runtime: datetime, st: Station, target: date, kind: str,
                     n_members: int = 41) -> Optional[DayForecast]:
    """Turn NBS rows into quantile-spaced pseudo-member paths for the target day.

    Path shape comes from the 3-hourly TMP guidance; its peak (trough) is then
    pulled to TXN, the NBM daytime max (overnight min), and the per-hour spread
    blends TSD into XND around that peak/trough. Members are perfectly
    correlated quantiles (z_i = Phi^-1((i+.5)/M)), so the day's extreme is
    ~N(TXN, XND) on normal days while a midnight-high frontal day still comes
    out right because the night-time hours are in the path.
    """
    grid = _hour_grid(target, st.tz)
    ft = np.array([_utc(r["ftime_utc"] if r.get("ftime_utc") else r["ftime"]) for r in rows])
    tmp = np.array([np.nan if r.get("tmp") is None else float(r["tmp"]) for r in rows])
    tsd = np.array([np.nan if r.get("tsd") is None else float(r["tsd"]) for r in rows])
    ok = ~np.isnan(tmp)
    if ok.sum() < 2:
        return None
    base = np.interp(grid, ft[ok], tmp[ok], left=np.nan, right=np.nan)
    ok_sd = ~np.isnan(tsd)
    sd = np.interp(grid, ft[ok_sd], tsd[ok_sd]) if ok_sd.any() else np.full(len(grid), 2.0)

    start = grid[0]
    if kind == "high":
        lo_t, hi_t = start + 12 * 3600, start + 36 * 3600  # 00Z after the local day, US zones
    else:
        lo_t, hi_t = start - 6 * 3600, start + 18 * 3600
    # TXN at 00Z is the daytime max, at 12Z the overnight min.
    want_hour = 0 if kind == "high" else 12
    ext = [(r, ft[i]) for i, r in enumerate(rows)
           if r.get("txn") is not None and lo_t <= ft[i] <= hi_t
           and datetime.fromtimestamp(ft[i], timezone.utc).hour == want_hour]
    if ext:
        r, _ = ext[0]
        txn, xnd = float(r["txn"]), float(r.get("xnd") or 3.0)
        local_hour = np.array([(datetime.fromtimestamp(t, ZoneInfo(st.tz)).hour) for t in grid])
        window = (local_hour >= 7) & (local_hour <= 19) if kind == "high" else (local_hour <= 9)
        valid = window & ~np.isnan(base)
        if valid.any():
            seg = np.where(valid, base, -np.inf if kind == "high" else np.inf)
            k = int(np.argmax(seg) if kind == "high" else np.argmin(seg))
            delta = txn - base[k]
            w = np.clip(1.0 - np.abs(np.arange(len(grid)) - k) / 5.0, 0.0, 1.0)
            base = base + delta * w
            sd = sd + (xnd - sd) * w
    if np.all(np.isnan(base)):
        return None
    z = np.array([NormalDist().inv_cdf((i + 0.5) / n_members) for i in range(n_members)])
    paths = base[None, :] + z[:, None] * sd[None, :]
    return DayForecast("nbm", grid, paths, runtime + NBM_AVAILABLE_AFTER, target)
