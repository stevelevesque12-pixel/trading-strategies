"""Download and cache NBA regular-season box scores from ESPN's public API.

Each finished game is reduced to one small JSON file in ``cache/``.
``load_games`` turns the cache into a games table and a player-game table.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import pandas as pd
import requests

API = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba"
CACHE = Path(__file__).resolve().parent / "cache"
SEASONS = {  # start year -> (first day, last day) of the regular season
    2024: (date(2024, 10, 22), date(2025, 4, 13)),
    2025: (date(2025, 10, 21), date(2026, 4, 12)),
    2026: (date(2026, 10, 20), date(2027, 4, 11)),
}
STATS = ("reb", "ast", "threes", "pts")

_session = requests.Session()


def get_json(path: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        try:
            r = _session.get(f"{API}/{path}", timeout=25)
            if r.status_code != 429 and r.status_code < 500:
                break
        except requests.RequestException:
            pass
        time.sleep(2 ** attempt)
    r.raise_for_status()
    return r.json()


def _minutes(value: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _extract(summary: dict, event: dict) -> dict:
    comp = event["competitions"][0]
    side = {c["homeAway"]: c["team"]["abbreviation"] for c in comp["competitors"]}
    local_day = pd.Timestamp(event["date"]).tz_convert("America/New_York").strftime("%Y-%m-%d")
    game = {"game_id": int(event["id"]), "date": local_day, "home": side["home"],
            "away": side["away"], "players": []}
    for team_box in summary["boxscore"]["players"]:
        team = team_box["team"]["abbreviation"]
        opp = side["away"] if team == side["home"] else side["home"]
        block = team_box["statistics"][0]
        idx = {name: i for i, name in enumerate(block["names"])}
        for a in block["athletes"]:
            s = a.get("stats") or []
            if a.get("didNotPlay") or not s:
                continue
            ath = a["athlete"]
            game["players"].append({
                "player_id": int(ath["id"]), "name": ath["displayName"],
                "pos": (ath.get("position") or {}).get("abbreviation", ""),
                "team": team, "opp": opp, "is_home": team == side["home"], "starter": bool(a.get("starter")),
                "min": _minutes(s[idx["MIN"]]), "pts": int(s[idx["PTS"]]), "reb": int(s[idx["REB"]]),
                "ast": int(s[idx["AST"]]), "threes": int(s[idx["3PT"]].split("-")[0]),
            })
    return game


def _events(day: date) -> list[dict]:
    d = get_json(f"scoreboard?dates={day:%Y%m%d}")
    return [e for e in d.get("events", []) if e.get("season", {}).get("type") == 2]


def _fetch_game(event: dict) -> str:
    path = CACHE / f"{event['id']}.json"
    if path.exists():
        return "cached"
    if event["competitions"][0]["status"]["type"]["name"] != "STATUS_FINAL":
        return "unfinished"
    try:
        game = _extract(get_json(f"summary?event={event['id']}"), event)
    except (requests.HTTPError, KeyError, IndexError):
        return "missing"
    path.write_text(json.dumps(game))
    return "ok"


def update_cache(seasons: list[int], workers: int = 6, until: date | None = None) -> None:
    CACHE.mkdir(exist_ok=True)
    until = until or date.today() - timedelta(days=1)
    days = []
    for s in seasons:
        first, last = SEASONS[s]
        d = first
        while d <= min(last, until):
            days.append(d)
            d += timedelta(days=1)
    with ThreadPoolExecutor(workers) as pool:
        events = [e for evs in pool.map(_events, days) for e in evs]
        results = list(pool.map(_fetch_game, events))
    print(f"cache update: {pd.Series(results).value_counts().to_dict()}")


def load_games() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (games, players) sorted by date."""
    games, players = [], []
    for f in CACHE.glob("*.json"):
        g = json.loads(f.read_text())
        games.append({k: g[k] for k in ("game_id", "date", "home", "away")})
        players += [{"game_id": g["game_id"], "date": g["date"], **p} for p in g["players"]]
    out = []
    for rows in (games, players):
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        out.append(df.sort_values(["date", "game_id"]).reset_index(drop=True))
    return tuple(out)


if __name__ == "__main__":
    import sys
    update_cache([int(s) for s in sys.argv[1:]] or list(SEASONS))
