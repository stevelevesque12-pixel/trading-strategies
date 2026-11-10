"""Paper-trade the NBA prop model: log every bet the rule recommends, grade it next morning.

    python -m sports.nba_props.paper pick ODDS.csv   # price the slate, log every recommended bet
    python -m sports.nba_props.paper grade           # grade finished picks, print the daily summary

ODDS.csv has one row per player and market:
    date, game ("Away Team @ Home Team"), start_time_ET, team, player,
    market (Rebounds / Assists / Threes / Points), line, over_odds, under_odds

The rule comes from the 2025-27 out-of-sample backtest (python -m sports.nba_props.backtest):

- Unders only. Model overs hit 2-20 points below their stated probability.
- Rebounds, assists and threes only. Points unders were 4-7 points overconfident.
- Assists and threes get a 1.5-point haircut (they ran slightly hot); rebounds none.
  After that the rule hit 69.4% against 69.7% predicted on ~21,800 past bets.
- Adjusted probability of at least 62%; expected value +3% to +15% at the actual odds.
- Player has 15+ games of history and played in his team's last game.
- One bet per player (the best EV across markets), at most 2 per game.

Stakes are tracked two ways from a $230 start: a flat $10, and the NHL system's
edge-sized stake (half Kelly, 5% a bet, 30% a night, drawdown brake).
A player who doesn't play is void (stake returned).
"""
from __future__ import annotations

import argparse
import re
import unicodedata
from pathlib import Path

import pandas as pd

from ..nhl_props.paper import payout, restake
from .data import CACHE, SEASONS, load_games, update_cache
from .model import Engine, Params, prob_over, season_of, walk_forward

LOG = Path(__file__).resolve().parent / "paper_bets.csv"
STAKE = 10.0
MARKETS = {"reb": "reb", "rebound": "reb", "ast": "ast", "assist": "ast", "three": "threes", "3": "threes",
           "point": "pts", "pts": "pts"}
HAIRCUT = {"reb": 0.0, "ast": 0.015, "threes": 0.015}
MIN_GP, MIN_P, MIN_EV, MAX_EV, MAX_PER_GAME = 15, 0.62, 0.03, 0.15, 2
TEAMS = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN", "Charlotte Hornets": "CHA",
    "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE", "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN",
    "Detroit Pistons": "DET", "Golden State Warriors": "GS", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL", "Memphis Grizzlies": "MEM",
    "Miami Heat": "MIA", "Milwaukee Bucks": "MIL", "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NO",
    "New York Knicks": "NY", "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC", "San Antonio Spurs": "SA",
    "Toronto Raptors": "TOR", "Utah Jazz": "UTAH", "Washington Wizards": "WSH",
}


def name_key(name: str) -> str:
    plain = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode().lower()
    words = [w for w in re.sub(r"[^a-z ]", "", plain.replace("-", " ")).split() if w not in {"jr", "sr", "ii", "iii", "iv"}]
    return " ".join(words)


def market_of(text: str) -> str | None:
    t = text.lower()
    for k, v in MARKETS.items():
        if k in t:
            return v
    return None


def read_odds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    away_home = df["game"].str.split(r" @ | at ", regex=True)
    df["away"] = [TEAMS[a] for a, _ in away_home]
    df["home"] = [TEAMS[h] for _, h in away_home]
    df["stat"] = df["market"].map(market_of)
    df["start_utc"] = (pd.to_datetime(df["date"] + " " + df["start_time_ET"].astype(str), format="mixed")
                       .dt.tz_localize("America/New_York").dt.tz_convert("UTC"))
    return df.dropna(subset=["stat"])



def refresh_cache() -> None:
    """Fetch the current season; on a fresh checkout (the cache is git-ignored) fetch every season,
    since projections need the earlier seasons' history and the 15-game minimum would block every bet."""
    cached = len(list(CACHE.glob("*.json"))) if CACHE.exists() else 0
    update_cache([2026] if cached >= 2000 else list(SEASONS))


def price(odds_csv: str, update: bool = True) -> tuple[pd.DataFrame, set[int]]:
    if update:
        refresh_cache()
    games, players = load_games()
    eng, _ = walk_forward(games, players, Params())
    odds = read_odds(odds_csv)
    season = season_of(odds["start_utc"].min().tz_convert("America/New_York"))
    by_key: dict[tuple[str, str], int] = {(name_key(i["name"]), i["team"]): pid for pid, i in eng.info.items()}
    last = players.sort_values("date").groupby("team")["game_id"].last()
    active = set(players.loc[players["game_id"].isin(last.values) & (players["min"] > 0), "player_id"])
    rows, missing = [], set()
    for r in odds.itertuples():
        pid = team = None
        for t in (r.home, r.away):
            if (name_key(r.player), t) in by_key:
                pid, team = by_key[(name_key(r.player), t)], t
                break
        if pid is None:
            missing.add(r.player)
            continue
        info = eng.info[pid]
        opp = r.away if team == r.home else r.home
        pr = Engine.project(eng, pid, info["pos"], opp, season)
        mean = pr[f"mean_{r.stat}"]
        p_over = prob_over(float(r.line), mean, eng.p.size[r.stat])
        for side, odds_, p in (("Over", r.over_odds, p_over), ("Under", r.under_odds, 1 - p_over)):
            rows.append({"date": r.date, "game": f"{r.away}@{r.home}", "player": r.player, "player_id": pid,
                         "team": team, "stat": r.stat, "bet": f"{side} {r.line}", "odds": float(odds_),
                         "proj": round(mean, 2), "proj_min": round(pr["proj_min"], 1), "gp": pr["gp"],
                         "model_p": p, "ev": p * float(odds_) - 1})
    if missing:
        print(f"No history found for: {', '.join(sorted(missing))}")
    return pd.DataFrame(rows), active


def select(res: pd.DataFrame, active: set[int]) -> pd.DataFrame:
    r = res[res["bet"].str.startswith("Under") & res["stat"].isin(list(HAIRCUT))].copy()
    r["adj_p"] = r["model_p"] - r["stat"].map(HAIRCUT)
    r["adj_ev"] = r["adj_p"] * r["odds"] - 1
    ok = (r["adj_p"] >= MIN_P) & r["adj_ev"].between(MIN_EV, MAX_EV) & (r["gp"] >= MIN_GP) & r["player_id"].isin(active)
    cand = r[ok].sort_values("adj_ev", ascending=False).drop_duplicates("player_id")
    return cand[cand.groupby("game").cumcount() < MAX_PER_GAME]


def pick(odds_csv: str) -> pd.DataFrame:
    date = str(read_odds(odds_csv)["date"].min())
    log = pd.read_csv(LOG, dtype={"result": "object"}) if LOG.exists() else pd.DataFrame()
    if len(log) and (log["date"] == date).any():
        print(f"Picks for {date} are already logged:")
        return log[log["date"] == date]
    res, active = price(odds_csv)
    picks = select(res, active).assign(stake=STAKE, result=None, actual=None, profit=None)
    cols = ["date", "game", "player", "player_id", "stat", "bet", "odds", "proj", "proj_min", "model_p", "adj_p",
            "adj_ev", "stake", "result", "actual", "profit"]
    log = restake(pd.concat([log, picks[cols]], ignore_index=True))
    log.to_csv(LOG, index=False)
    return log[log["date"] == date]


def grade() -> pd.DataFrame:
    log = pd.read_csv(LOG, dtype={"result": "object"})
    if log["result"].isna().any():
        refresh_cache()
    games, players = load_games()
    players["d"] = players["date"].dt.strftime("%Y-%m-%d")
    played = set(zip(games["date"].dt.strftime("%Y-%m-%d"), games["away"] + "@" + games["home"]))
    for i in log.index[log["result"].isna()]:
        r = log.loc[i]
        if (r["date"], r["game"]) not in played:
            continue
        row = players[(players["player_id"] == r["player_id"]) & (players["d"] == r["date"])]
        if row.empty or row["min"].iloc[0] <= 0:
            log.loc[i, ["result", "actual", "profit"]] = ["void", None, 0.0]
            continue
        actual = int(row[r["stat"]].iloc[0])
        side, line = r["bet"].split()
        won = actual > float(line) if side == "Over" else actual < float(line)
        result = "win" if won else "loss"
        log.loc[i, ["result", "actual", "profit"]] = [result, actual, payout(result, r["stake"], r["odds"])]
    log = restake(log)
    log.to_csv(LOG, index=False)
    return log


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pick").add_argument("odds_csv")
    sub.add_parser("grade")
    args = ap.parse_args()
    pd.set_option("display.width", 220)
    if args.cmd == "pick":
        print(pick(args.odds_csv).to_string(index=False))
    else:
        log = grade()
        g = log[log["result"].isin(["win", "loss"])]
        print(log.tail(15).to_string(index=False))
        print(f"\nRecord {(g['result'] == 'win').sum()}-{(g['result'] == 'loss').sum()}, "
              f"flat ${g['profit'].sum():+.2f}, edge-sized ${g['kelly_profit'].sum():+.2f}")


if __name__ == "__main__":
    main()
