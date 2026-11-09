import pandas as pd

from sports.nba_props.paper import market_of, name_key, read_odds, select


def test_name_key_ignores_accents_suffixes_and_punctuation():
    assert name_key("Luka Dončić") == name_key("Luka Doncic")
    assert name_key("Jaren Jackson Jr.") == name_key("Jaren Jackson")
    assert name_key("Shai Gilgeous-Alexander") == "shai gilgeous alexander"


def test_market_names():
    assert [market_of(m) for m in ("Rebounds", "Assists", "Threes made", "3-pointers", "Points")] == \
        ["reb", "ast", "threes", "threes", "pts"]


def test_read_odds(tmp_path):
    p = tmp_path / "odds.csv"
    p.write_text("date,game,start_time_ET,team,player,market,line,over_odds,under_odds\n"
                 "2026-11-10,Golden State Warriors @ Utah Jazz,9:00 PM,Jazz,Lauri Markkanen,Rebounds,6.5,1.9,1.9\n")
    df = read_odds(str(p))
    assert (df["away"].iloc[0], df["home"].iloc[0], df["stat"].iloc[0]) == ("GS", "UTAH", "reb")


def test_select_unders_only_no_points_haircut_and_caps():
    def row(pid, game, stat, bet, odds, p, gp=50):
        return {"player_id": pid, "game": game, "stat": stat, "bet": bet, "odds": odds, "model_p": p, "gp": gp}
    res = pd.DataFrame([
        row(1, "A@B", "reb", "Under 6.5", 1.70, 0.66),     # passes
        row(2, "A@B", "pts", "Under 20.5", 1.70, 0.70),    # points excluded
        row(3, "A@B", "reb", "Over 6.5", 1.90, 0.66),      # overs excluded
        row(4, "A@B", "ast", "Under 4.5", 1.70, 0.63),     # 63% - 1.5 haircut < 62%
        row(5, "A@B", "threes", "Under 1.5", 1.70, 0.68),  # passes
        row(6, "A@B", "reb", "Under 9.5", 1.70, 0.65),     # third bet in the game: capped
        row(7, "C@D", "reb", "Under 4.5", 1.70, 0.66),     # did not play last game
    ])
    out = select(res, active={1, 2, 3, 4, 5, 6})
    assert sorted(out["player_id"]) == [1, 5]
