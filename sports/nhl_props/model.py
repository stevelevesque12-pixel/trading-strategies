"""Walk-forward projections for NHL player shots on goal and goalie saves.

Shots on goal
    expected SOG = shot rate per 60 min  x  projected TOI / 60
                   x  opponent shots-allowed factor  x  home/away factor
    The shot rate is a recency-weighted average shrunk toward the league
    rate for the player's position (F or D), so a hot 5-game streak doesn't
    move it much.

Goalie saves
    expected shots against = league avg  x  opponent shots-for factor
                             x  own team shots-against factor  x  home/away
    expected saves = expected shots against  x  shrunk save %

Both counts are modelled as negative binomial (a Poisson with extra spread),
which gives P(over / under) for any line. Every projection uses only games
played before that game's date, so the same code runs the backtest and today's slate.
"""
from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass

import pandas as pd


@dataclass
class Params:
    rate_halflife: float = 60.0   # skater games, for shot rate
    toi_halflife: float = 3.0     # skater games, for ice time
    short_game_frac: float = 0.5  # games under half the usual ice time are left out of the TOI average
    rate_prior_min: float = 150.0  # minutes of league-average shooting added to each player
    team_halflife: float = 20.0   # team games, for shots for/against
    team_prior_games: float = 8.0
    sv_halflife: float = 40.0     # goalie games, for save %
    sv_prior_shots: float = 1500.0
    home_factor: float = 1.04     # home teams take ~4% more shots (2024-26 data)
    starter_share: float = 0.96   # starters face ~96% of team shots (pulls, relief)
    skater_size: float = 17.0     # negative-binomial size (higher = closer to Poisson)
    saves_size: float = 29.0 


class Decayed:
    """Exponentially weighted sums: every update multiplies old weight by 0.5**(1/halflife)."""

    def __init__(self, halflife: float):
        self.decay = 0.5 ** (1 / halflife)
        self.w = 0.0
        self.sums: dict[str, float] = defaultdict(float)

    def add(self, **vals: float) -> None:
        self.w = self.w * self.decay + 1
        for k in set(self.sums) | set(vals):
            self.sums[k] = self.sums[k] * self.decay + vals.get(k, 0.0)


# ---------------------------------------------------------------- distributions

def nb_pmf(k: int, mean: float, size: float) -> float:
    if mean <= 0:
        return 1.0 if k == 0 else 0.0
    p = size / (size + mean)
    return math.exp(math.lgamma(k + size) - math.lgamma(size) - math.lgamma(k + 1)
                    + size * math.log(p) + k * math.log1p(-p))


def prob_over(line: float, mean: float, size: float) -> float:
    """P(X > line) for a half-point line such as 2.5."""
    return 1.0 - sum(nb_pmf(k, mean, size) for k in range(int(math.floor(line)) + 1))


def fair_odds(p: float) -> float:
    return 1 / p if p > 0 else float("inf")


# ---------------------------------------------------------------- walk-forward engine

class Engine:
    def __init__(self, params: Params | None = None):
        self.p = params or Params()
        self.rate = defaultdict(lambda: Decayed(self.p.rate_halflife))
        self.toi = defaultdict(lambda: Decayed(self.p.toi_halflife))
        self.games = defaultdict(int)
        self.team = defaultdict(lambda: Decayed(self.p.team_halflife))
        self.sv = defaultdict(lambda: Decayed(self.p.sv_halflife))
        self.player_info: dict[int, dict] = {}
        # league running totals
        self.lg = defaultdict(float)

    # league averages (with sane defaults before any data)
    def lg_rate(self, pos: str) -> float:
        toi = self.lg[f"toi_{pos}"]
        return self.lg[f"sog_{pos}"] / toi * 60 if toi else (8.0 if pos == "F" else 5.0)

    def lg_team_sog(self) -> float:
        return self.lg["team_sog"] / self.lg["team_games"] if self.lg["team_games"] else 29.0

    def lg_sv(self) -> float:
        return self.lg["saves"] / self.lg["sa"] if self.lg["sa"] else 0.900

    def team_factor(self, team: str, key: str) -> float:
        """Shrunk ratio of a team's shots for/against per game to the league average."""
        lg = self.lg_team_sog()
        t = self.team[team]
        k = self.p.team_prior_games
        return (t.sums[key] + k * lg) / (t.w + k) / lg

    def venue(self, is_home: bool, for_shooter: bool) -> float:
        h = math.sqrt(self.p.home_factor)
        return h if is_home == for_shooter else 1 / h

    # ---- projections
    def project_skater(self, pid: int, pos: str, opp: str, is_home: bool, toi: float | None = None) -> dict:
        r = self.rate[pid]
        prior = self.p.rate_prior_min
        rate = (r.sums["sog"] + self.lg_rate(pos) * prior / 60) / (r.sums["toi"] + prior) * 60
        t = self.toi[pid]
        proj_toi = toi if toi is not None else (t.sums["toi"] / t.w if t.w else (15.0 if pos == "F" else 19.0))
        mean = rate * proj_toi / 60 * self.team_factor(opp, "against") * self.venue(is_home, True)
        return {"mean": mean, "rate60": rate, "proj_toi": proj_toi, "gp": self.games[pid]}

    def project_goalie(self, pid: int, team: str, opp: str, is_home: bool) -> dict:
        exp_sa = (self.lg_team_sog() * self.team_factor(opp, "for")
                  * self.team_factor(team, "against") * self.venue(is_home, False)
                  * self.p.starter_share)
        g = self.sv[pid]
        k = self.p.sv_prior_shots
        sv = (g.sums["saves"] + k * self.lg_sv()) / (g.sums["sa"] + k)
        return {"mean": exp_sa * sv, "exp_sa": exp_sa, "sv_pct": sv, "gp": self.games[pid]}

    # ---- updates
    def update(self, game: pd.Series, skaters: pd.DataFrame, goalies: pd.DataFrame) -> None:
        for s in skaters.itertuples():
            if s.toi <= 0:
                continue
            self.rate[s.player_id].add(sog=s.sog, toi=s.toi)
            t = self.toi[s.player_id]
            # an injury exit or ejection says nothing about next game's ice time
            if not (t.w >= 3 and s.toi < self.p.short_game_frac * t.sums["toi"] / t.w):
                t.add(toi=s.toi)
            self.games[s.player_id] += 1
            self.player_info[s.player_id] = {"name": s.name, "pos": s.pos, "team": s.team}
            self.lg[f"sog_{s.pos}"] += s.sog
            self.lg[f"toi_{s.pos}"] += s.toi
        for g in goalies.itertuples():
            if g.shots_against > 0:
                self.sv[g.player_id].add(saves=g.saves, sa=g.shots_against)
                self.lg["saves"] += g.saves
                self.lg["sa"] += g.shots_against
            if g.toi > 0:
                self.games[g.player_id] += 1
                self.player_info[g.player_id] = {"name": g.name, "pos": "G", "team": g.team}
        for team, sog_for, sog_against in ((game.home, game.home_sog, game.away_sog),
                                           (game.away, game.away_sog, game.home_sog)):
            self.team[team].add(**{"for": sog_for, "against": sog_against})
            self.lg["team_sog"] += sog_for
            self.lg["team_games"] += 1


def walk_forward(games: pd.DataFrame, skaters: pd.DataFrame, goalies: pd.DataFrame,
                 params: Params | None = None) -> tuple[Engine, pd.DataFrame, pd.DataFrame]:
    """Project every skater and starting goalie from prior games only, then update.

    Games on the same date are all projected before any of them is added,
    so nothing from that day leaks into another game's projection.
    """
    eng = Engine(params)
    sk_by_game = dict(tuple(skaters.groupby("game_id")))
    gl_by_game = dict(tuple(goalies.groupby("game_id")))
    sk_rows, gl_rows = [], []
    for _, day in games.groupby("date", sort=True):
        for game in day.itertuples():
            for s in sk_by_game.get(game.game_id, pd.DataFrame()).itertuples():
                if s.toi > 0:
                    pr = eng.project_skater(s.player_id, s.pos, s.opp, s.is_home)
                    sk_rows.append({"game_id": game.game_id, "date": game.date, "player_id": s.player_id,
                                    "name": s.name, "pos": s.pos, "team": s.team, "opp": s.opp,
                                    "sog": s.sog, "toi": s.toi, **pr})
            for g in gl_by_game.get(game.game_id, pd.DataFrame()).itertuples():
                if g.starter:
                    pr = eng.project_goalie(g.player_id, g.team, g.opp, g.is_home)
                    gl_rows.append({"game_id": game.game_id, "date": game.date, "player_id": g.player_id,
                                    "name": g.name, "team": g.team, "opp": g.opp, "saves": g.saves,
                                    "shots_against": g.shots_against, "toi": g.toi, **pr})
        for game in day.itertuples():
            eng.update(game, sk_by_game.get(game.game_id, pd.DataFrame(columns=skaters.columns)),
                       gl_by_game.get(game.game_id, pd.DataFrame(columns=goalies.columns)))
    return eng, pd.DataFrame(sk_rows), pd.DataFrame(gl_rows)


def fit_size(y: pd.Series, mean: pd.Series) -> float:
    """Method-of-moments negative-binomial size: Var = mean + mean^2 / size."""
    excess = ((y - mean) ** 2 - mean).sum()
    return float((mean ** 2).sum() / excess) if excess > 0 else 1000.0
