"""Paper-trade the model: log every bet the rule recommends and grade it two ways,
at a flat $10 and at a quarter-Kelly stake on a notional $1,000 bankroll.

    python -m sports.nhl_props.paper pick ODDS.csv   # price the slate, log every recommended bet
    python -m sports.nhl_props.paper grade           # grade finished picks, print the daily summary

The picks follow a fixed "win-rate" rule so the record measures the system,
not anyone's judgment. Each part comes from the 2025-27 out-of-sample backtest,
where it raised the hit rate of the model's side from 68% to 71.5%:

- Unders by default. Model unders hit slightly above their stated probability;
  overs hit ~2.5 points below it, so overs get a 2.5-point haircut and are only
  allowed against a leaky defence (opponent allows 5%+ more shots than average).
- No unders against a leaky defence (they hit below their stated probability).
- Lines 1.5 to 3.5 only (0.5 lines and extreme lines were miscalibrated).
- Adjusted model probability of at least 62%.
- Expected value +3% to +15% at the actual odds. Bigger "edges" usually mean
  missing news (injury, line change).
- Player has 20+ games of history and played in his team's last game (a
  player who missed it is hurt, scratched or just back, so his data is stale
  and the bet may be void). One bet per player.
- At most 2 bets per game, keeping the highest expected value. Bets in one
  game win or lose together, so this caps the damage from one odd game.
- Every other bet that passes is logged, however many there are that day.

Two stakes are tracked: a flat $10, and quarter Kelly on a notional $1,000
bankroll (capped at $30), which bets more where the edge is bigger.

A pick on a player who doesn't dress is void (stake returned), as at the sportsbook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import load_games, update_cache
from .model import Params, walk_forward
from .slate import price

LOG = Path(__file__).resolve().parent / "paper_bets.csv"
STAKE = 10.0
MIN_GP, MIN_EV, MAX_EV = 20, 0.03, 0.15
MIN_P, OVER_HAIRCUT, LEAKY = 0.62, 0.025, 1.05
MAX_PER_GAME = 2
KELLY_BANKROLL, KELLY_FRACTION, KELLY_CAP = 1000.0, 0.25, 30.0


def kelly_stake(p: float, odds: float) -> float:
    full = max(0.0, (p * odds - 1) / (odds - 1))
    return float(min(KELLY_CAP, round(KELLY_BANKROLL * KELLY_FRACTION * full)))


def played_last_game(skaters: pd.DataFrame) -> set[int]:
    """Players who dressed in their team's most recent game."""
    last = skaters.sort_values("date").groupby("team")["game_id"].last()
    in_last = skaters[skaters["game_id"].isin(last.values) & (skaters["toi"] > 0)]
    return set(in_last.loc[in_last["game_id"] == in_last["team"].map(last), "player_id"])


def select(res: pd.DataFrame, opp_factor: dict[str, float], active: set[int] | None = None) -> pd.DataFrame:
    """Apply the rule. ``opp_factor`` maps team -> shots allowed vs league average;
    ``active`` is the set of players who played their team's last game."""
    res = res[res["type"] == "two_way"].copy()
    side = res["bet"].str.split().str[0]
    line = res["bet"].str.split().str[1].astype(float)
    opp = [a if t == h else h for a, h, t in zip(res["game"].str[:3], res["game"].str[4:], res["team"])]
    leaky = pd.Series([opp_factor.get(o, 1.0) >= LEAKY for o in opp], index=res.index)
    res["adj_p"] = res["model_p"] - OVER_HAIRCUT * (side == "Over")
    res["adj_ev"] = res["adj_p"] * res["odds"] - 1
    ok = (((side == "Under") & ~leaky) | ((side == "Over") & leaky)) & line.between(1.5, 3.5)
    ok &= (res["adj_p"] >= MIN_P) & res["adj_ev"].between(MIN_EV, MAX_EV) & (res["gp"] >= MIN_GP)
    if active is not None:
        ok &= res["player_id"].isin(active)
    cand = res[ok].sort_values("adj_ev", ascending=False).drop_duplicates("player_id")
    cand = cand[cand.groupby("game").cumcount() < MAX_PER_GAME]
    return cand.assign(kelly_stake=[kelly_stake(p, o) for p, o in zip(cand["adj_p"], cand["odds"])])


def slate_date(odds_csv: str) -> str:
    start = pd.read_csv(odds_csv, encoding="utf-8-sig")["start_utc"]
    return str(pd.to_datetime(start, utc=True).dt.tz_convert("America/New_York").dt.date.min())


def pick(odds_csv: str) -> pd.DataFrame:
    date = slate_date(odds_csv)
    log = pd.read_csv(LOG) if LOG.exists() else pd.DataFrame()
    if len(log) and (log["date"] == date).any():
        print(f"Picks for {date} are already logged:")
        return log[log["date"] == date]
    res = price(odds_csv, update=True)
    games, skaters, goalies = load_games()
    eng, _, _ = walk_forward(games, skaters, goalies, Params())
    factors = {t: eng.team_factor(t, "against") for t in eng.team}
    picks = select(res, factors, played_last_game(skaters)).assign(date=date, stake=STAKE)
    picks = picks[["date", "game", "player", "player_id", "pos", "bet", "odds", "proj_sog",
                   "model_p", "adj_p", "fair_book_p", "adj_ev", "stake", "kelly_stake"]]
    picks = picks.assign(result=None, sog=None, profit=None, kelly_profit=None)
    pd.concat([log, picks]).to_csv(LOG, index=False)
    return picks


def grade() -> pd.DataFrame:
    log = pd.read_csv(LOG, dtype={"result": "object"})
    open_rows = log["result"].isna() if "result" in log else pd.Series(True, index=log.index)
    if open_rows.any():
        update_cache([2026])
    games, skaters, _ = load_games()
    skaters["d"] = skaters["date"].dt.strftime("%Y-%m-%d")
    finished_dates = set(games["date"].dt.strftime("%Y-%m-%d"))
    for i in log.index[open_rows]:
        r = log.loc[i]
        team_games = games[games["date"].dt.strftime("%Y-%m-%d").eq(r["date"])]
        away, home = r["game"].split("@")
        played = team_games[(team_games["home"] == home) & (team_games["away"] == away)]
        if r["date"] not in finished_dates or played.empty:
            continue  # game not finished yet
        row = skaters[(skaters["player_id"] == r["player_id"]) & (skaters["d"] == r["date"])]
        if row.empty or row["toi"].iloc[0] <= 0:
            log.loc[i, ["result", "sog", "profit", "kelly_profit"]] = ["void", None, 0.0, 0.0]
            continue
        sog = int(row["sog"].iloc[0])
        side, line = r["bet"].split()
        won = sog > float(line) if side == "Over" else sog < float(line)
        payout = (r["odds"] - 1) if won else -1.0
        log.loc[i, ["result", "sog", "profit", "kelly_profit"]] = [
            "win" if won else "loss", sog, round(r["stake"] * payout, 2), round(r["kelly_stake"] * payout, 2)]
    log.to_csv(LOG, index=False)
    return log


def daily_summary(log: pd.DataFrame) -> pd.DataFrame:
    """One row per graded day: bets, wins, expected wins, staked, net and running total."""
    rows = []
    for date, d in log[log["result"].isin(["win", "loss"])].groupby("date"):
        rows.append({"date": date, "bets": len(d), "wins": int((d["result"] == "win").sum()),
                     "expected_wins": round(d["adj_p"].sum(), 2), "staked": d["stake"].sum(),
                     "net": round(d["profit"].sum(), 2), "kelly_net": round(d["kelly_profit"].sum(), 2)})
    out = pd.DataFrame(rows, columns=["date", "bets", "wins", "expected_wins", "staked", "net", "kelly_net"])
    out["cumulative"] = out["net"].cumsum().round(2)
    out["kelly_cumulative"] = out["kelly_net"].cumsum().round(2)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("pick").add_argument("odds_csv")
    sub.add_parser("grade")
    args = ap.parse_args()
    pd.set_option("display.width", 200)
    if args.cmd == "pick":
        print(pick(args.odds_csv).to_string(index=False))
    else:
        log = grade()
        print(log.tail(15).to_string(index=False))
        print("\nDaily results ($10 flat):")
        print(daily_summary(log).to_string(index=False))


if __name__ == "__main__":
    main()
