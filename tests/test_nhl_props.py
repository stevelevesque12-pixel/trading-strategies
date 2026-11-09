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
    assert _key("Tim Stutzle") == _key("T. Stützle")


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
    assert kelly_stake(0.64, 1.80, 230) == 11.5  # capped at 5% of the balance
    assert kelly_stake(0.55, 1.80, 230) == 0.0
    assert kelly_stake(0.62, 1.70, 230) == 8.87  # half Kelly


def test_day_stakes_capped_at_30_percent():
    from sports.nhl_props.paper import day_stakes
    stakes = day_stakes([0.70] * 10, [1.60] * 10, 200)
    assert abs(sum(stakes) - 60) < 0.05 and len(set(stakes)) == 1


def test_restake_compounds_and_brakes():
    from sports.nhl_props.paper import restake
    log = pd.DataFrame([
        {"date": "2026-10-29", "adj_p": 0.70, "odds": 1.60, "result": "win"},
        {"date": "2026-10-29", "adj_p": 0.65, "odds": 1.80, "result": "loss"},
        {"date": "2026-10-30", "adj_p": 0.70, "odds": 1.60, "result": None},
    ])
    out = restake(log)
    assert list(out.iloc[:2]["kelly_stake"]) == [11.5, 11.5] and list(out.iloc[:2]["kelly_profit"]) == [6.9, -11.5]
    assert out.iloc[2]["balance_before"] == 225.4 and pd.isna(out.iloc[2]["kelly_profit"])
    # 10% edge at 1.60: half Kelly bets 8.3% (capped to 5%), quarter Kelly 4.2%
    crash = pd.DataFrame([{"date": f"2026-11-{d:02d}", "adj_p": 0.6875, "odds": 1.60, "result": "loss"}
                          for d in range(1, 9)])
    out = restake(crash)
    braked = out[out["balance_before"] < 0.75 * 230]
    assert len(braked) and (braked["kelly_stake"] < 0.045 * braked["balance_before"]).all()


def test_read_odds_one_row_per_player(tmp_path):
    from sports.nhl_props.slate import read_odds
    p = tmp_path / "wide.csv"
    p.write_text("date,game,start_time_ET,team,player,line,over_odds,under_odds\n"
                 "2026-11-05,Anaheim Ducks @ Edmonton Oilers,8:08 PM,Oilers,Connor McDavid,3.5,1.909,1.75\n"
                 "2026-11-03,Utah Mammoth at Toronto Maple Leafs,19:08,Leafs,Auston Matthews,3.5,2.2,1.57\n")
    df = read_odds(str(p))
    assert list(df["selection"]) == ["Over 3.5", "Under 3.5", "Over 3.5", "Under 3.5"]
    assert df["game"].iloc[0] == "Anaheim Ducks at Edmonton Oilers"
    assert list(df["start_utc"].unique()) == ["2026-11-06T01:08:00Z", "2026-11-04T00:08:00Z"]
