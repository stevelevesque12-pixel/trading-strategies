"""Shared HTTP session: retries with backoff, plus an optional on-disk cache.

Only cache responses that can never change (archived forecasts, past
observations, resolved markets) -- pass `cache=True` for those.
"""

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Optional

import requests

CACHE_DIR = Path(os.environ.get("POLYWEATHER_CACHE", Path.home() / ".cache" / "polyweather"))
USER_AGENT = "polyweather/0.1 (research; github.com/stevelevesque12-pixel/trading-strategies)"

_session: Optional[requests.Session] = None


def session() -> requests.Session:
    global _session
    if _session is None:
        _session = requests.Session()
        _session.headers["User-Agent"] = USER_AGENT
    return _session


class RateLimited(RuntimeError):
    pass


def _cache_path(url: str, params: Optional[dict]) -> Path:
    key = url + "?" + json.dumps(params or {}, sort_keys=True)
    return CACHE_DIR / hashlib.sha256(key.encode()).hexdigest()[:32]


def get(url: str, params: Optional[dict] = None, *, cache: bool = False, as_json: bool = True,
        retries: int = 4, timeout: float = 30.0) -> Any:
    path = _cache_path(url, params) if cache else None
    if path is not None and path.exists():
        text = path.read_text()
        return json.loads(text) if as_json else text

    delay = 2.0
    last_exc: Optional[Exception] = None
    for attempt in range(retries + 1):
        try:
            r = session().get(url, params=params, timeout=timeout)
            if r.status_code == 429 or (r.status_code >= 500 and attempt < retries):
                raise requests.HTTPError(f"{r.status_code} from {url}", response=r)
            if r.status_code == 400 and "limit exceeded" in r.text.lower():
                # Open-Meteo's free tier answers its daily quota with a 400 + JSON reason.
                raise RateLimited(r.text[:200])
            r.raise_for_status()
            text = r.text
            if path is not None:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(text)
            return json.loads(text) if as_json else text
        except RateLimited:
            raise
        except (requests.ConnectionError, requests.Timeout, requests.HTTPError) as exc:
            last_exc = exc
            resp = getattr(exc, "response", None)
            if resp is not None and 400 <= resp.status_code < 500 and resp.status_code != 429:
                raise
            if attempt == retries:
                break
            time.sleep(delay)
            delay *= 2
    raise last_exc  # type: ignore[misc]


def post(url: str, payload: Any, timeout: float = 30.0, retries: int = 3) -> Any:
    delay = 1.0
    for attempt in range(retries + 1):
        try:
            r = session().post(url, json=payload, timeout=timeout)
            r.raise_for_status()
            return r.json()
        except (requests.ConnectionError, requests.Timeout):
            if attempt == retries:
                raise
            time.sleep(delay)
            delay *= 2
