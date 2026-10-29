import pandas as pd

from sports.nhl_props.model import Decayed, nb_pmf, prob_over
from sports.nhl_props.slate import _key, parse_odds


def test_nb_pmf_sums_to_one_and_matches_mean():
    pmf = [nb_pmf(k, 2.4, 17.0) for k in range(60)]
    assert abs(sum(pmf) - 1) < 1e-9
    assert abs(sum(k * p for k, p in enumerate(pmf)) - 2.4) < 1e-6


def test_prob_over_half_point_lines():
    assert abs(prob_over(0.5, 2.0, 1e9) - (1 - pow(2.718281828, -2.0))) < 1e-6
    assert prob_over(1.5, 2.0, 17) > prob_over(2.5, 2.0, 17)


def test_decayed_weights_recent_games_more():
    d = Decayed(halflife=1)
    d.add(x=0.0)
    d.add(x=10.0)
    assert d.sums["x"] / d.w > 5


def test_name_key_matches_api_and_book_formats():
    assert _key("Connor McDavid") == _key("C. McDavid")
    assert _key("Pierre-Luc Dubois") == _key("P. Dubois")


def test_parse_odds(tmp_path):
    p = tmp_path / "odds.csv"
    pd.DataFrame([
        ["Calgary Flames at Edmonton Oilers", "Connor McDavid Shots on Goal", "Over 3.5", 1.72, "Open"],
        ["Calgary Flames at Edmonton Oilers", "Connor McDavid 4+ Shots on Goal Milestone", "Yes", 1.72, "Open"],
        ["Calgary Flames at Edmonton Oilers", "Connor McDavid 2+ Shots on Goal", "Yes", 1.2, "Open"],
        ["Calgary Flames at Edmonton Oilers", "Moneyline", "Oilers", 1.5, "Open"],
    ], columns=["game", "market", "selection", "decimal_odds", "status"]).to_csv(p, index=False)
    df = parse_odds(str(p))
    assert list(df["type"]) == ["two_way", "milestone", "milestone"]
    assert list(df["line"]) == [3.5, 3.5, 1.5]
    assert (df["home"] == "EDM").all()


def test_select_caps_two_bets_per_game_by_ev_and_skips_inactive():
    from sports.nhl_props.paper import kelly_stake, select
    rows = [
        # game, player_id, bet, odds, model_p
        ("MTL@DAL", 1, "Under 2.5", 1.60, 0.71),
        ("MTL@DAL", 2, "Under 1.5", 1.78, 0.64),
        ("MTL@DAL", 3, "Under 1.5", 1.62, 0.65),
        ("BOS@STL", 4, "Under 3.5", 1.75, 0.66),
    ]
    res = pd.DataFrame([{"type": "two_way", "game": g, "player_id": pid, "team": g[:3], "bet": b,
                         "odds": o, "model_p": p, "gp": 50} for g, pid, b, o, p in rows])
    out = select(res, {}, active={1, 2, 3})
    assert list(out["player_id"]) == [2, 1]  # highest EV first; game capped at 2; player 4 inactive
    assert kelly_stake(0.64, 1.80) == 30.0 and kelly_stake(0.55, 1.80) == 0.0
