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
        ["Calgary Flames at Edmonton Oilers", "Moneyline", "Oilers", 1.5, "Open"],
    ], columns=["game", "market", "selection", "decimal_odds", "status"]).to_csv(p, index=False)
    df = parse_odds(str(p))
    assert list(df["type"]) == ["two_way", "milestone"]
    assert list(df["line"]) == [3.5, 3.5]
    assert (df["home"] == "EDM").all()
