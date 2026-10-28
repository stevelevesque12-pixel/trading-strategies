"""Walk-forward backtest of the NHL shots/saves model.

There is no free archive of historical prop odds, so this checks the two
things that can be checked honestly:

1. Accuracy: log loss on over/under outcomes versus a naive baseline
   (Poisson around the player's last-10-game average, roughly what a
   casual bettor eyeballs).
2. Calibration: when the model says 60%, does it hit about 60%? Tested on
   a proxy line set at the player's recent average (floor + 0.5).

Whether it beats Sports Interaction's actual prices can only be learned by
logging real bets (see ``slate.py price --log``).

Usage: python -m sports.nhl_props.backtest [--tune]
"""
from __future__ import annotations

import argparse
import itertools
import math

import numpy as np
import pandas as pd

from .data import load_games
from .model import Params, fit_size, prob_over, walk_forward

TRAIN_SEASON = 2024  # 2024-25 is used to pick parameters, later seasons are out of sample
MIN_GP = 10


def add_naive(df: pd.DataFrame, y: str, window: int = 10) -> pd.DataFrame:
    df = df.sort_values(["date", "game_id"]).copy()
    df["naive"] = df.groupby("player_id")[y].transform(lambda s: s.shift(1).rolling(window, min_periods=3).mean())
    df["line"] = np.floor(df["naive"]) + 0.5
    return df.dropna(subset=["naive"])


def evaluate(df: pd.DataFrame, y: str, size: float) -> dict:
    p_model = np.array([prob_over(l, m, size) for l, m in zip(df["line"], df["mean"])])
    p_naive = np.array([prob_over(l, m, 1e6) for l, m in zip(df["line"], df["naive"])])
    hit = (df[y] > df["line"]).to_numpy()

    def logloss(p):
        p = np.clip(p, 1e-4, 1 - 1e-4)
        return float(-np.mean(hit * np.log(p) + (~hit) * np.log(1 - p)))

    return {"n": len(df), "over_rate": hit.mean(), "logloss_model": logloss(p_model),
            "logloss_naive": logloss(p_naive), "p_model": p_model, "hit": hit}


def calibration(p: np.ndarray, hit: np.ndarray) -> pd.DataFrame:
    """For each confidence band, take the model's favoured side and check its hit rate."""
    side_p = np.maximum(p, 1 - p)
    side_hit = np.where(p >= 0.5, hit, ~hit)
    is_under = p < 0.5
    bands = [0.5, 0.55, 0.6, 0.65, 0.7, 0.8, 1.0]
    rows = []
    for lo, hi in zip(bands[:-1], bands[1:]):
        m = (side_p >= lo) & (side_p < hi)
        if m.sum():
            rows.append({"model_conf": f"{lo:.0%}-{hi:.0%}", "bets": int(m.sum()),
                         "avg_pred": side_p[m].mean(), "hit_rate": side_hit[m].mean(),
                         "share_unders": is_under[m].mean(),
                         "breakeven_odds": 1 / side_hit[m].mean()})
    return pd.DataFrame(rows)


def run(params: Params, frames=None, verbose: bool = True) -> dict:
    games, skaters, goalies = frames or load_games()
    _, sk, gl = walk_forward(games, skaters, goalies, params)
    out = {}
    for name, df, y in (("shots", sk, "sog"), ("saves", gl, "saves")):
        df = add_naive(df[df["gp"] >= MIN_GP], y)
        season = df["game_id"] // 1_000_000
        train, test = df[season == TRAIN_SEASON], df[season > TRAIN_SEASON]
        size = fit_size(train[y], train["mean"])
        res_train, res_test = evaluate(train, y, size), evaluate(test, y, size)
        out[name] = {"size": size, "train": res_train, "test": res_test}
        if verbose:
            print(f"\n=== {name.upper()}  (NB size fitted on 2024-25 = {size:.1f}) ===")
            for label, r in (("2024-25 (tuning)", res_train), ("2025-26 + 2026-27 (out of sample)", res_test)):
                print(f"{label:36s} n={r['n']:6d}  over-rate={r['over_rate']:.3f}  "
                      f"logloss model={r['logloss_model']:.4f}  naive={r['logloss_naive']:.4f}")
            print("\nOut-of-sample calibration on proxy lines (model's favoured side):")
            print(calibration(res_test["p_model"], res_test["hit"]).to_string(index=False, float_format="%.3f"))
    return out


def tune() -> None:
    frames = load_games()
    grid = {"rate_halflife": [15, 30, 60], "toi_halflife": [3, 6, 12], "rate_prior_min": [75, 150, 300]}
    results = []
    for vals in itertools.product(*grid.values()):
        p = Params(**dict(zip(grid, vals)))
        r = run(p, frames, verbose=False)
        results.append({**dict(zip(grid, vals)), "shots_train_ll": r["shots"]["train"]["logloss_model"]})
    print(pd.DataFrame(results).sort_values("shots_train_ll").head(8).to_string(index=False))
    grid_g = {"team_halflife": [10, 20, 40], "sv_prior_shots": [750, 1500, 3000]}
    results = []
    for vals in itertools.product(*grid_g.values()):
        p = Params(**dict(zip(grid_g, vals)))
        r = run(p, frames, verbose=False)
        results.append({**dict(zip(grid_g, vals)), "saves_train_ll": r["saves"]["train"]["logloss_model"]})
    print(pd.DataFrame(results).sort_values("saves_train_ll").head(5).to_string(index=False))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--tune", action="store_true", help="grid-search parameters on 2024-25 only")
    args = ap.parse_args()
    tune() if args.tune else run(Params())
