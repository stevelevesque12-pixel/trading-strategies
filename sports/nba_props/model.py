"""Walk-forward projections for NBA player rebounds, assists, threes and points.

    expected stat = per-minute rate x projected minutes x opponent factor

The per-minute rate is recency-weighted and shrunk toward the league rate for
the player's position group (G, F or C). Minutes are a short recency-weighted
average that ignores abnormally short games (injury exits, blowout benchings).
The opponent factor is how much of that stat the opponent allows per game
relative to the league, shrunk toward 1. Counts are negative binomial, as in
the NHL model. Every projection uses only games played before that date.
"""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import pandas as pd

from ..nhl_props.model import Decayed, prob_over  # noqa: F401  (prob_over re-exported)

from .data import STATS

def season_of(day) -> int:
    """NBA season start year: games before August belong to the season that began the previous fall."""
    day = pd.Timestamp(day)
    return day.year - (day.month < 8)


GROUP = {"G": "G", "PG": "G", "SG": "G", "F": "F", "SF": "F", "PF": "F", "C": "C"}


@dataclass
class Params:
    rate_halflife: float = 40.0     # player games
    min_halflife: float = 5.0       # player games
    rate_prior_min: float = 150.0   # minutes of league-average production added to each player
    team_halflife: float = 25.0     # team games
    team_prior_games: float = 10.0
    short_game_frac: float = 0.0    # 0 keeps every game: NBA blowout minutes recur, so dropping them biased minutes up
    season_carry_rate: float = 0.5  # weight kept on last season's per-minute rates at a player's first game
    season_carry_min: float = 0.2   # weight kept on last season's minutes
    league_daily_decay: float = 0.995  # league averages fade (half-life ~ 140 days)
    # negative-binomial spread per stat, fitted on 2024-25 (python -m sports.nba_props.backtest)
    size: dict = field(default_factory=lambda: {"reb": 9.0, "ast": 8.9, "threes": 6.7, "pts": 5.3})


class Engine:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.rate = defaultdict(lambda: Decayed(self.p.rate_halflife))
        self.mins = defaultdict(lambda: Decayed(self.p.min_halflife))
        self.allowed = defaultdict(lambda: Decayed(self.p.team_halflife))
        self.games = defaultdict(int)
        self.info: dict[int, dict] = {}
        self.season: dict[int, int] = {}
        self.lg = defaultdict(float)

    def new_day(self) -> None:
        for k in self.lg:
            self.lg[k] *= self.p.league_daily_decay

    def lg_rate(self, stat: str, group: str) -> float:
        m = self.lg[f"min_{group}"]
        return self.lg[f"{stat}_{group}"] / m if m else {"reb": 0.18, "ast": 0.1, "threes": 0.05, "pts": 0.45}[stat]

    def opp_factor(self, opp: str, stat: str) -> float:
        g = self.lg["team_games"]
        lg = self.lg[f"team_{stat}"] / g if g else 1.0
        if not lg:
            return 1.0
        t = self.allowed[opp]
        k = self.p.team_prior_games
        return (t.sums[stat] + k * lg) / (t.w + k) / lg

    def project(self, pid: int, pos: str, opp: str, season: int | None = None) -> dict:
        if season is not None:
            self._carry_over(pid, season)
        group = GROUP.get(pos, "F")
        m = self.mins[pid]
        proj_min = m.sums["min"] / m.w if m.w else 20.0
        r = self.rate[pid]
        k = self.p.rate_prior_min
        out = {"proj_min": proj_min, "gp": self.games[pid]}
        for stat in STATS:
            rate = (r.sums[stat] + k * self.lg_rate(stat, group)) / (r.sums["min"] + k)
            out[f"mean_{stat}"] = rate * proj_min * self.opp_factor(opp, stat)
        return out

    def _carry_over(self, pid: int, season: int) -> None:
        """Down-weight a player's history at his first game of a new season."""
        if self.season.get(pid, season) != season:
            for state, keep in ((self.rate[pid], self.p.season_carry_rate), (self.mins[pid], self.p.season_carry_min)):
                state.w *= keep
                for k in state.sums:
                    state.sums[k] *= keep
        self.season[pid] = season

    def update(self, game_players: pd.DataFrame, season: int) -> None:
        for p in game_players.itertuples():
            if p.min <= 0:
                continue
            self._carry_over(p.player_id, season)
            group = GROUP.get(p.pos, "F")
            self.rate[p.player_id].add(min=p.min, **{s: getattr(p, s) for s in STATS})
            m = self.mins[p.player_id]
            if not (m.w >= 3 and p.min < self.p.short_game_frac * m.sums["min"] / m.w):
                m.add(min=p.min)
            self.games[p.player_id] += 1
            self.info[p.player_id] = {"name": p.name, "pos": p.pos, "team": p.team}
            self.lg[f"min_{group}"] += p.min
            for s in STATS:
                self.lg[f"{s}_{group}"] += getattr(p, s)
        for team, rows in game_players.groupby("team"):
            opp = rows["opp"].iloc[0]
            totals = {s: float(rows[s].sum()) for s in STATS}
            self.allowed[opp].add(**totals)  # what the opponent allowed
            self.lg["team_games"] += 1
            for s in STATS:
                self.lg[f"team_{s}"] += totals[s]


def walk_forward(games: pd.DataFrame, players: pd.DataFrame, params: Params | None = None):
    """Project every player-game from prior dates only, then update."""
    eng = Engine(params)
    by_game = dict(tuple(players.groupby("game_id")))
    rows = []
    for d, day in games.groupby("date", sort=True):
        eng.new_day()
        season = season_of(d)
        for g in day.itertuples():
            for p in by_game.get(g.game_id, pd.DataFrame()).itertuples():
                if p.min > 0:
                    rows.append({"game_id": g.game_id, "date": g.date, "season": season, "player_id": p.player_id,
                                 "name": p.name, "pos": p.pos, "team": p.team, "opp": p.opp, "min": p.min,
                                 **{s: getattr(p, s) for s in STATS},
                                 **eng.project(p.player_id, p.pos, p.opp, season)})
        for g in day.itertuples():
            if g.game_id in by_game:
                eng.update(by_game[g.game_id], season)
    return eng, pd.DataFrame(rows)
