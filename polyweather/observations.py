"""Station observations: what the settlement source will see.

Markets settle on the max/min of the station's METAR temperatures over the
local calendar day, as displayed (whole °F for US sites, whole °C elsewhere).
Two consequences the model must respect:

* Only reported readings count. A routine-only ("Show Hourly Data") market
  sees ~24 readings at :51 past each hour, so its max sits ~0.5-1°F below the
  true continuous max. Calibration absorbs this as a bias.
* Once a reading is in, the day's extreme can only move one way. By mid
  afternoon a "highest" market's lower buckets are dead -- often before the
  order book has caught up.

History comes from Iowa Environmental Mesonet (IEM); live obs from
aviationweather.gov (lower latency).

US ASOS sites also report every 5 minutes (MADIS "high-frequency" METARs).
Those don't count for settlement, but traders watch them on the same
weather.gov page, so a model that only sees the hourly METAR is always
behind the market. They're used as the *nowcast* anchor: the latest
temperature the forecast paths are nudged toward.
"""

import csv
import io
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from . import http

IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
AWC_METAR = "https://aviationweather.gov/api/data/metar"


def round_half_up(x: float) -> int:
    return int(math.floor(x + 0.5))


def c_to_f(c: float) -> float:
    return c * 9.0 / 5.0 + 32.0


def day_window_utc(d: date, tz: str) -> tuple[datetime, datetime]:
    z = ZoneInfo(tz)
    start = datetime.combine(d, time(0), tzinfo=z)
    end = datetime.combine(d + timedelta(days=1), time(0), tzinfo=z)
    return start.astimezone(timezone.utc), end.astimezone(timezone.utc)


@dataclass
class Obs:
    times: list[datetime] = field(default_factory=list)  # UTC
    values: list[float] = field(default_factory=list)  # in market unit, as displayed (rounded)

    def window(self, start: datetime, end: datetime) -> "Obs":
        pairs = [(t, v) for t, v in zip(self.times, self.values) if start <= t < end]
        return Obs([t for t, _ in pairs], [v for _, v in pairs])

    def __len__(self) -> int:
        return len(self.values)


@dataclass
class ObsState:
    """Observations for the target local day up to `asof`."""
    extreme: Optional[float]  # max (high market) / min (low market) so far, settlement readings only
    last_value: Optional[float]  # freshest temperature (5-minute feed if available), market unit
    last_time: Optional[datetime]
    n: int
    asof: datetime

    @classmethod
    def from_obs(cls, obs: Obs, kind: str, asof: datetime, hf: Optional[Obs] = None) -> "ObsState":
        pairs = [(t, v) for t, v in zip(obs.times, obs.values) if t <= asof]
        last = pairs[-1] if pairs else None
        if hf is not None:
            hf_pairs = [(t, v) for t, v in zip(hf.times, hf.values) if t <= asof]
            if hf_pairs and (last is None or hf_pairs[-1][0] > last[0]):
                last = hf_pairs[-1]
        if not pairs:
            return cls(None, last[1] if last else None, last[0] if last else None, 0, asof)
        vals = [v for _, v in pairs]
        ext = max(vals) if kind == "high" else min(vals)
        return cls(ext, last[1], last[0], len(pairs), asof)


def _iem_id(icao: str) -> str:
    # IEM's ASOS networks key US sites by FAA id (KLGA -> LGA), everything else by ICAO.
    return icao[1:] if len(icao) == 4 and icao.startswith("K") else icao


def fetch_obs_iem(icao: str, start_utc: datetime, end_utc: datetime, unit: str,
                  include_specials: bool = True, cache: bool = False) -> Obs:
    field_name = "tmpf" if unit == "F" else "tmpc"
    s, e = start_utc.astimezone(timezone.utc), end_utc.astimezone(timezone.utc) + timedelta(minutes=1)
    params = {
        "station": _iem_id(icao), "data": field_name, "tz": "Etc/UTC", "format": "onlycomma",
        "year1": s.year, "month1": s.month, "day1": s.day, "hour1": s.hour, "minute1": s.minute,
        "year2": e.year, "month2": e.month, "day2": e.day, "hour2": e.hour, "minute2": e.minute,
        "report_type": [3, 4] if include_specials else [3],
    }
    text = http.get(IEM_ASOS, params=params, as_json=False, cache=cache, timeout=90)
    obs = Obs()
    for row in csv.DictReader(io.StringIO(text)):
        v = row.get(field_name)
        if v in (None, "", "M"):
            continue
        t = datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc)
        obs.times.append(t)
        obs.values.append(float(round_half_up(float(v))))
    return obs


_TGROUP = re.compile(r"\bT([01])(\d{3})([01])(\d{3})\b")
_BODY_T = re.compile(r"\s(M?\d{2})/(M?\d{2})?\s")


def metar_temp_c(metar: str) -> Optional[float]:
    """Temperature from a METAR: T-group tenths if present, else the whole-degree body group."""
    m = _TGROUP.search(metar or "")
    if m:
        v = int(m.group(2)) / 10.0
        return -v if m.group(1) == "1" else v
    m = _BODY_T.search(metar or "")
    if m:
        t = m.group(1)
        return -float(t[1:]) if t.startswith("M") else float(t)
    return None


def fetch_hf_iem(icao: str, start_utc: datetime, end_utc: datetime, unit: str, cache: bool = False) -> Obs:
    """5-minute MADIS high-frequency METARs (US ASOS), unrounded, from the IEM archive."""
    s, e = start_utc.astimezone(timezone.utc), end_utc.astimezone(timezone.utc)
    params = {
        "station": _iem_id(icao), "data": "metar", "tz": "Etc/UTC", "format": "onlycomma",
        "year1": s.year, "month1": s.month, "day1": s.day, "hour1": s.hour,
        "year2": e.year, "month2": e.month, "day2": e.day, "hour2": e.hour, "report_type": 1,
    }
    text = http.get(IEM_ASOS, params=params, as_json=False, cache=cache, timeout=180)
    obs = Obs()
    for row in csv.DictReader(io.StringIO(text)):
        c = metar_temp_c(row.get("metar", ""))
        if c is None:
            continue
        obs.times.append(datetime.strptime(row["valid"], "%Y-%m-%d %H:%M").replace(tzinfo=timezone.utc))
        obs.values.append(c_to_f(c) if unit == "F" else c)
    return obs


def fetch_hf_nws(icao: str, unit: str, limit: int = 60) -> Obs:
    """Latest 5-minute observations from api.weather.gov (live)."""
    d = http.get(f"https://api.weather.gov/stations/{icao}/observations", params={"limit": limit})
    obs = Obs()
    for f in reversed((d or {}).get("features", [])):
        p = f.get("properties", {})
        c = (p.get("temperature") or {}).get("value")
        if c is None:
            continue
        obs.times.append(datetime.fromisoformat(p["timestamp"]))
        obs.values.append(c_to_f(float(c)) if unit == "F" else float(c))
    return obs


def fetch_obs_awc(icao: str, unit: str, hours: int = 30, include_specials: bool = True) -> Obs:
    data = http.get(AWC_METAR, params={"ids": icao, "hours": hours, "format": "json"})
    rows = sorted(data or [], key=lambda r: r.get("obsTime", 0))
    obs = Obs()
    for r in rows:
        if r.get("temp") is None:
            continue
        if not include_specials and str(r.get("metarType", "METAR")).upper() == "SPECI":
            continue
        c = float(r["temp"])
        v = c_to_f(c) if unit == "F" else c
        obs.times.append(datetime.fromtimestamp(int(r["obsTime"]), tz=timezone.utc))
        obs.values.append(float(round_half_up(v)))
    return obs


def fetch_awc_many(icaos: list[str], hours: int = 30) -> dict[str, list[tuple[datetime, float, str]]]:
    """Latest METARs for many stations in one call: {icao: [(time, temp_c, METAR|SPECI)]}."""
    out: dict[str, list] = {i: [] for i in icaos}
    for k in range(0, len(icaos), 40):
        data = http.get(AWC_METAR, params={"ids": ",".join(icaos[k:k + 40]), "hours": hours, "format": "json"})
        for r in data or []:
            if r.get("temp") is None or r.get("icaoId") not in out:
                continue
            out[r["icaoId"]].append((datetime.fromtimestamp(int(r["obsTime"]), tz=timezone.utc), float(r["temp"]),
                                     str(r.get("metarType", "METAR")).upper()))
    for v in out.values():
        v.sort()
    return out


def obs_from_awc(rows, unit: str, include_specials: bool) -> Obs:
    obs = Obs()
    for t, c, typ in rows:
        if not include_specials and typ == "SPECI":
            continue
        obs.times.append(t)
        obs.values.append(float(round_half_up(c_to_f(c) if unit == "F" else c)))
    return obs


def day_obs(icao: str, d: date, tz: str, unit: str, include_specials: bool,
            live: bool = False) -> Obs:
    start, end = day_window_utc(d, tz)
    if live:
        try:
            obs = fetch_obs_awc(icao, unit, hours=30, include_specials=include_specials)
            if len(obs):
                return obs.window(start, end)
        except Exception:
            pass  # fall through to IEM
    past = end < datetime.now(timezone.utc) - timedelta(hours=6)
    return fetch_obs_iem(icao, start, end, unit, include_specials, cache=past).window(start, end)


def resolved_extreme(obs: Obs, kind: str) -> Optional[float]:
    if not len(obs):
        return None
    return max(obs.values) if kind == "high" else min(obs.values)
