"""Real-money stakes for one night, across the NHL and NBA paper picks.

    python -m sports.real_stakes 2026-11-12 219.31

Every paper pick stays in its paper log; this only decides which of them get real money and how much.
Stakes use the NHL sizing (half Kelly, 5% a bet, 30% of the balance a night), shared across both sports
because they come out of one Sports Interaction balance.

Real-money exclusions (paper logging continues so they can earn their way back):
- NBA rebound unders: 3-6 on paper over the first two nights (5.8 wins expected), all losses on big men.
  Paused 2026-11-12; revisit after ~30 paper rebound bets.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .nhl_props.paper import day_stakes

ROOT = Path(__file__).resolve().parent
NBA_REAL_EXCLUDE_STATS = {"reb"}


def tonight(date: str) -> pd.DataFrame:
    nhl = pd.read_csv(ROOT / "nhl_props" / "paper_bets.csv")
    nhl = nhl[nhl["date"] == date].assign(sport="NHL", market="shots")
    nba_path = ROOT / "nba_props" / "paper_bets.csv"
    nba = pd.read_csv(nba_path) if nba_path.exists() else pd.DataFrame(columns=["date"])
    nba = nba[nba["date"] == date].assign(sport="NBA")
    nba = nba.rename(columns={"stat": "market"})
    cols = ["sport", "game", "player", "market", "bet", "odds", "adj_p", "adj_ev"]
    picks = pd.concat([nhl[cols], nba[cols]], ignore_index=True)
    picks["real"] = ~((picks["sport"] == "NBA") & picks["market"].isin(NBA_REAL_EXCLUDE_STATS))
    return picks


def size(date: str, bankroll: float) -> pd.DataFrame:
    picks = tonight(date)
    real = picks[picks["real"]]
    picks["stake"] = 0.0
    if len(real):
        picks.loc[real.index, "stake"] = day_stakes(real["adj_p"], real["odds"], bankroll)
    picks["min_odds"] = (1.03 / picks["adj_p"]).round(2)   # below this the edge is under 3%
    return picks


if __name__ == "__main__":
    out = size(sys.argv[1], float(sys.argv[2]))
    print(out.to_string(index=False))
    print(f"\nReal total: ${out['stake'].sum():.2f} on {int((out['stake'] > 0).sum())} bets; "
          f"paper only: {int((~out['real']).sum())}")
