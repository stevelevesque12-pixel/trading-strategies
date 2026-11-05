"""Paper-trade the model: log every bet the rule recommends and grade it two ways,
at a flat $10 and at a quarter-Kelly stake sized from the running balance.

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

Two stakes are tracked from a $230 starting balance: a flat $10, and quarter
Kelly (capped at 3% of the balance), which bets more where the edge is bigger.
Edge-sized stakes use the balance after the last graded day, so they grow
after wins and shrink after losses. ``restake`` recomputes them for the whole
log, so changing the starting balance re-sizes every past bet consistently.

A pick on a player who doesn't dress is void (stake returned), as at the sportsbook.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from .data import load_games, update_cache
from .model import Params, walk_forward
from .slate import price, read_odds

LOG = Path(__file__).resolve().parent / "paper_bets.csv"
STAKE = 10.0
MIN_GP, MIN_EV, MAX_EV = 20, 0.03, 0.15
MIN_P, OVER_HAIRCUT, LEAKY = 0.62, 0.025, 1.05
MAX_PER_GAME = 2
STARTING_BANKROLL, KELLY_FRACTION, KELLY_CAP = 230.0, 0.25, 0.03


def kelly_stake(p: float, odds: float, bankroll: float) -> float:
    full = max(0.0, (p * odds - 1) / (odds - 1))
    return round(bankroll * min(KELLY_CAP, KELLY_FRACTION * full), 2)


def payout(result, stake: float, odds: float):
    if result == "win":
        return round(stake * (odds - 1), 2)
    if result == "loss":
        return -stake
    return 0.0 if result == "void" else float("nan")


def restake(log: pd.DataFrame) -> pd.DataFrame:
    """Size every edge-sized bet from the balance at the start of its day."""
    log = log.copy()
    log[["balance_before", "kelly_stake", "kelly_profit"]] = float("nan")
    balance = STARTING_BANKROLL
    for date in sorted(log["date"].unique()):
        day = log["date"] == date
        log.loc[day, "balance_before"] = round(balance, 2)
        log.loc[day, "kelly_stake"] = [kelly_stake(p, o, balance) for p, o in zip(log.loc[day, "adj_p"], log.loc[day, "odds"])]
        log.loc[day, "kelly_profit"] = [payout(r, k, o) for r, k, o in
                                        zip(log.loc[day, "result"], log.loc[day, "kelly_stake"], log.loc[day, "odds"])]
        balance += log.loc[day, "kelly_profit"].fillna(0).sum()
    return log


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
    return cand


def slate_date(odds_csv: str) -> str:
    start = read_odds(odds_csv)["start_utc"]
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
                   "model_p", "adj_p", "fair_book_p", "adj_ev", "stake"]]
    picks = picks.assign(result=None, sog=None, profit=None)
    log = restake(pd.concat([log, picks]))
    log.to_csv(LOG, index=False)
    return log[log["date"] == date]


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
            log.loc[i, ["result", "sog", "profit"]] = ["void", None, 0.0]
            continue
        sog = int(row["sog"].iloc[0])
        side, line = r["bet"].split()
        won = sog > float(line) if side == "Over" else sog < float(line)
        result = "win" if won else "loss"
        log.loc[i, ["result", "sog", "profit"]] = [result, sog, payout(result, r["stake"], r["odds"])]
    log = restake(log)
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
