"""Walk-forward calibration test of the NBA prop model.

Out of sample = 2025-26 onward (2024-25 only fits the negative-binomial spread).
Proxy line per bet = floor(player's last-10 average) + 0.5, as in the NHL test.

Usage: python -m sports.nba_props.backtest
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..nhl_props.model import fit_size, prob_over
from .data import STATS, load_games
from .model import walk_forward

MIN_GP = 15


def frame(stat: str, df: pd.DataFrame, size: float) -> pd.DataFrame:
    d = df.sort_values(["date", "game_id"]).copy()
    d["naive"] = d.groupby("player_id")[stat].transform(lambda s: s.shift(1).rolling(10, min_periods=5).mean())
    d = d.dropna(subset=["naive"])
    d = d[(d["gp"] >= MIN_GP) & (d["naive"] >= 0.5)]
    d["line"] = np.floor(d["naive"]) + 0.5
    d["p_over"] = [prob_over(l, m, size) for l, m in zip(d["line"], d[f"mean_{stat}"])]
    d["side"] = np.where(d["p_over"] >= 0.5, "Over", "Under")
    d["p"] = np.maximum(d["p_over"], 1 - d["p_over"])
    d["hit"] = np.where(d["side"] == "Over", d[stat] > d["line"], d[stat] < d["line"])
    d["stat"] = stat
    return d


def run() -> pd.DataFrame:
    games, players = load_games()
    _, df = walk_forward(games, players)
    out = []
    for stat in STATS:
        train = df[df["season"] == 2024]
        size = fit_size(train[stat], train[f"mean_{stat}"])
        out.append(frame(stat, df[df["season"] > 2024], size).assign(size=size))
    return pd.concat(out)


if __name__ == "__main__":
    d = run()
    bands = pd.cut(d["p"], [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0])
    t = d.groupby(["stat", "side", bands], observed=True).agg(n=("hit", "size"), hit=("hit", "mean"), pred=("p", "mean"))
    t["hit-pred"] = t["hit"] - t["pred"]
    pd.set_option("display.max_rows", 200)
    print(d.groupby("stat")["size"].first().round(1).to_dict())
    print(t[t["n"] >= 100].round(3).to_string())
