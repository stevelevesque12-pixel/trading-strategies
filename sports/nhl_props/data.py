"""Download and cache NHL regular-season boxscores from the public NHL API.

Each finished game is reduced to one small JSON file in ``cache/`` so later
runs only fetch new games. ``load_games`` turns the cache into two tables:
one row per skater-game and one row per goalie-game.
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd
import requests

API = "https://api-web.nhle.com/v1"
CACHE = Path(__file__).resolve().parent / "cache"
MAX_GAMES_PER_SEASON = 1312
FINISHED = ("OFF", "FINAL")

_session = requests.Session()


def get_json(path: str, retries: int = 5) -> dict:
    for attempt in range(retries):
        r = _session.get(f"{API}/{path}", timeout=20)
        if r.status_code != 429 and r.status_code < 500:
            break
        time.sleep(2 ** attempt)  # rate-limited or server error: back off
    r.raise_for_status()
    return r.json()


def toi_minutes(toi: str) -> float:
    m, s = toi.split(":")
    return int(m) + int(s) / 60


def _extract(box: dict) -> dict:
    """Keep only the fields the model needs."""
    game = {
        "game_id": box["id"],
        "date": box["gameDate"],
        "home": box["homeTeam"]["abbrev"],
        "away": box["awayTeam"]["abbrev"],
        "home_sog": box["homeTeam"].get("sog"),
        "away_sog": box["awayTeam"].get("sog"),
        "skaters": [],
        "goalies": [],
    }
    for side in ("home", "away"):
        team = game[side]
        opp = game["away" if side == "home" else "home"]
        stats = box["playerByGameStats"][f"{side}Team"]
        for p in stats["forwards"] + stats["defense"]:
            game["skaters"].append({
                "player_id": p["playerId"], "name": p["name"]["default"],
                "pos": "D" if p["position"] == "D" else "F",
                "team": team, "opp": opp, "is_home": side == "home",
                "sog": p["sog"], "toi": toi_minutes(p["toi"]),
            })
        for g in stats["goalies"]:
            game["goalies"].append({
                "player_id": g["playerId"], "name": g["name"]["default"],
                "team": team, "opp": opp, "is_home": side == "home",
                "starter": bool(g.get("starter")), "toi": toi_minutes(g["toi"]),
                "shots_against": g.get("shotsAgainst", 0), "saves": g.get("saves", 0),
            })
    return game


def _fetch_game(game_id: int) -> str:
    """Return 'ok', 'cached', 'unfinished' or 'missing'."""
    path = CACHE / f"{game_id}.json"
    if path.exists():
        return "cached"
    try:
        box = get_json(f"gamecenter/{game_id}/boxscore")
    except requests.HTTPError:
        return "missing"
    if box.get("gameState") not in FINISHED:
        return "unfinished"
    path.write_text(json.dumps(_extract(box)))
    return "ok"


def update_cache(seasons: list[int], workers: int = 4) -> None:
    """Fetch every finished regular-season game for the given start years (2025 = 2025-26)."""
    CACHE.mkdir(exist_ok=True)
    ids = [int(f"{s}02{n:04d}") for s in seasons for n in range(1, MAX_GAMES_PER_SEASON + 1)]
    with ThreadPoolExecutor(workers) as pool:
        results = list(pool.map(_fetch_game, ids))
    counts = pd.Series(results).value_counts().to_dict()
    print(f"cache update: {counts}")


def load_games() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Return (games, skaters, goalies) tables sorted by date."""
    games, skaters, goalies = [], [], []
    for f in CACHE.glob("*.json"):
        g = json.loads(f.read_text())
        base = {"game_id": g["game_id"], "date": g["date"]}
        games.append({k: g[k] for k in ("game_id", "date", "home", "away", "home_sog", "away_sog")})
        skaters += [{**base, **s} for s in g["skaters"]]
        goalies += [{**base, **s} for s in g["goalies"]]
    out = []
    for rows in (games, skaters, goalies):
        df = pd.DataFrame(rows)
        df["date"] = pd.to_datetime(df["date"])
        out.append(df.sort_values(["date", "game_id"]).reset_index(drop=True))
    return tuple(out)


if __name__ == "__main__":
    import sys
    update_cache([int(s) for s in sys.argv[1:]] or [2024, 2025, 2026])
