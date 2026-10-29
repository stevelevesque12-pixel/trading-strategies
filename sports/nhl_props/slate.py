"""Price a Sports Interaction NHL odds export against the shots-on-goal model.

    python -m sports.nhl_props.slate ODDS.csv [--update] [--min-edge 0.03] [--bankroll 500]

The CSV needs the columns game, market, selection, decimal_odds and status
(the export format: "Away Team at Home Team", "Connor McDavid Shots on Goal",
"Over 3.5"). Two-way "Shots on Goal" markets and one-way "N+ Shots on Goal
Milestone" markets are priced. Projections use every game finished before the
slate (run with --update first to pull last night's games).
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

from .data import load_games, update_cache
from .model import Params, prob_over, walk_forward

TEAMS = {
    "Anaheim Ducks": "ANA", "Boston Bruins": "BOS", "Buffalo Sabres": "BUF", "Calgary Flames": "CGY",
    "Carolina Hurricanes": "CAR", "Chicago Blackhawks": "CHI", "Colorado Avalanche": "COL",
    "Columbus Blue Jackets": "CBJ", "Dallas Stars": "DAL", "Detroit Red Wings": "DET",
    "Edmonton Oilers": "EDM", "Florida Panthers": "FLA", "Los Angeles Kings": "LAK",
    "Minnesota Wild": "MIN", "Montreal Canadiens": "MTL", "Montréal Canadiens": "MTL",
    "Nashville Predators": "NSH", "New Jersey Devils": "NJD", "New York Islanders": "NYI",
    "New York Rangers": "NYR", "Ottawa Senators": "OTT", "Philadelphia Flyers": "PHI",
    "Pittsburgh Penguins": "PIT", "San Jose Sharks": "SJS", "Seattle Kraken": "SEA",
    "St. Louis Blues": "STL", "St Louis Blues": "STL", "Tampa Bay Lightning": "TBL",
    "Toronto Maple Leafs": "TOR", "Utah Mammoth": "UTA", "Utah Hockey Club": "UTA",
    "Vancouver Canucks": "VAN", "Vegas Golden Knights": "VGK", "Washington Capitals": "WSH",
    "Winnipeg Jets": "WPG",
}
TWO_WAY = re.compile(r"^(?P<player>.+) Shots on Goal$")
MILESTONE = re.compile(r"^(?P<player>.+) (?P<k>\d+)\+ Shots on Goal( Milestone)?$")


def parse_odds(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8-sig")
    df = df[df["status"].eq("Open")]
    rows = []
    for r in df.itertuples():
        away, home = r.game.split(" at ")
        base = {"game": r.game, "away": TEAMS[away], "home": TEAMS[home], "odds": float(r.decimal_odds)}
        if m := MILESTONE.match(r.market):
            if r.selection == "Yes":
                rows.append({**base, "type": "milestone", "player": m["player"],
                             "side": "Over", "line": int(m["k"]) - 0.5})
        elif m := TWO_WAY.match(r.market):
            side, line = r.selection.split()
            rows.append({**base, "type": "two_way", "player": m["player"], "side": side, "line": float(line)})
    return pd.DataFrame(rows)


def _key(name: str) -> str:
    """'Connor McDavid' and 'C. McDavid' both become 'c mcdavid'."""
    parts = name.replace(".", "").lower().split()
    return f"{parts[0][0]} {' '.join(parts[1:])}"


def match_players(odds: pd.DataFrame, eng) -> pd.DataFrame:
    """Attach the model's player id, using the player's most recent team to break ties."""
    lookup: dict[tuple[str, str], int] = {}
    for pid, info in eng.player_info.items():
        if info["pos"] != "G":
            lookup[(_key(info["name"]), info["team"])] = pid
    ids, teams = [], []
    for r in odds.itertuples():
        pid = team = None
        for t in (r.home, r.away):
            if (_key(r.player), t) in lookup:
                pid, team = lookup[(_key(r.player), t)], t
                break
        ids.append(pid)
        teams.append(team)
    return odds.assign(player_id=ids, team=teams)


def kelly(p: float, odds: float) -> float:
    return max(0.0, (p * odds - 1) / (odds - 1))


def price(odds_path: str, params: Params | None = None, update: bool = False) -> pd.DataFrame:
    if update:
        update_cache([2024, 2025, 2026])
    params = params or Params()
    games, skaters, goalies = load_games()
    eng, sk, _ = walk_forward(games, skaters, goalies, params)
    size = params.skater_size
    odds = match_players(parse_odds(odds_path), eng)
    out = []
    for r in odds.itertuples():
        if r.player_id is None or pd.isna(r.player_id):
            continue
        info = eng.player_info[r.player_id]
        is_home = r.team == r.home
        pr = eng.project_skater(r.player_id, info["pos"], r.away if is_home else r.home, is_home)
        p_over = prob_over(r.line, pr["mean"], size)
        p = p_over if r.side == "Over" else 1 - p_over
        out.append({"game": f"{r.away}@{r.home}", "player": r.player, "player_id": int(r.player_id),
                    "team": r.team, "pos": info["pos"],
                    "type": r.type, "bet": f"{r.side} {r.line}", "odds": r.odds,
                    "proj_sog": pr["mean"], "proj_toi": pr["proj_toi"], "gp": pr["gp"],
                    "model_p": p, "book_p": 1 / r.odds, "ev": p * r.odds - 1, "kelly": kelly(p, r.odds)})
    res = pd.DataFrame(out)
    # no-vig book probability for two-way markets
    tw = res["type"].eq("two_way")
    overround = res[tw].groupby(["player", res[tw]["bet"].str.split().str[1]])["book_p"].transform("sum")
    res.loc[tw, "fair_book_p"] = res.loc[tw, "book_p"] / overround
    missing = sorted(set(odds.loc[odds["player_id"].isna(), "player"]))
    if missing:
        print(f"No history found for: {', '.join(missing)}")
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("odds_csv")
    ap.add_argument("--update", action="store_true", help="fetch newly finished games first")
    ap.add_argument("--min-edge", type=float, default=0.03, help="minimum expected value per $1")
    ap.add_argument("--bankroll", type=float, default=0.0, help="print quarter-Kelly stakes for this bankroll")
    ap.add_argument("--out", help="write every priced line to this CSV")
    args = ap.parse_args()
    res = price(args.odds_csv, update=args.update)
    if args.out:
        res.to_csv(args.out, index=False)
    pd.set_option("display.width", 200)
    fmt = {"odds": "{:.2f}", "proj_sog": "{:.2f}", "proj_toi": "{:.1f}", "model_p": "{:.1%}",
           "fair_book_p": "{:.1%}", "book_p": "{:.1%}", "ev": "{:+.1%}"}
    for kind, cols in (("two_way", ["game", "player", "pos", "bet", "odds", "proj_sog", "proj_toi", "gp",
                                    "model_p", "fair_book_p", "ev"]),
                       ("milestone", ["game", "player", "pos", "bet", "odds", "proj_sog", "gp",
                                      "model_p", "book_p", "ev"])):
        picks = res[(res["type"] == kind) & (res["ev"] >= args.min_edge) & (res["gp"] >= 10)]
        picks = picks.sort_values("model_p", ascending=False)
        if args.bankroll:
            picks = picks.assign(stake=(picks["kelly"] / 4 * args.bankroll).round(2))
            cols = cols + ["stake"]
        print(f"\n=== {kind.replace('_', '-')} bets with EV >= {args.min_edge:.0%} "
              f"({len(picks)} of {(res['type'] == kind).sum()} priced) ===")
        if len(picks):
            print(picks[cols].to_string(index=False, formatters={k: v.format for k, v in fmt.items() if k in cols}))


if __name__ == "__main__":
    main()
