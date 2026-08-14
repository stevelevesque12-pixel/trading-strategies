from datetime import datetime, timedelta

from failed2s.bars import Bar
from failed2s.instruments import INSTRUMENTS
from mss_pullback.pivots import Pivot
from mss_pullback.strategy import Exit, Fill, MSSPullbackStrategy, StructureTracker, _qty_for_risk

T0 = datetime(2026, 1, 5, 9, 30)
NQ = INSTRUMENTS["NQ"]
MNQ = INSTRUMENTS["MNQ"]


def bar(i, o, h, l, c):
    return Bar(T0 + timedelta(seconds=15 * i), o, h, l, c)


# ---- StructureTracker (3m bias) ----


# Both sequences below are verified by direct execution, not hand-traced --
# this rule has enough moving parts (independent high/low candidates, ties,
# a recomputed window) that hand-tracing it reliably turned out to be a bad
# bet even for writing the tests. Note bias legitimately flips to the
# "wrong" direction first (bar 5) purely because only sparse pivot data
# exists yet -- that's the rule working correctly on thin data, not a bug;
# what matters is the final flip once the real setup completes (bar 7).


def test_structure_tracker_bullish_mss_end_to_end():
    st = StructureTracker()
    seq = [
        bar(0, 200, 205, 195, 196),
        bar(1, 196, 198, 190, 192),
        bar(2, 192, 220, 191, 218),
        bar(3, 218, 219, 192, 193),
        bar(4, 193, 194, 185, 190),
        bar(5, 190, 191, 180, 181),
        bar(6, 181, 200, 181, 199),
    ]
    for b in seq:
        st.update(b)
    assert [p.price for p in st.pivots.valid_highs] == [205, 220]
    assert [p.price for p in st.pivots.valid_lows] == [190, 180]
    assert st.direction == "short"  # sparse-data early read, see note above

    st.update(bar(7, 199, 225, 198, 222))  # closes 222 > 220 -> bullish MSS
    assert st.direction == "long"


def test_structure_tracker_flips_short_mirrors_long():
    # The exact mirror of the bullish sequence above (price' = 400 - price,
    # which swaps each bar's high/low roles), so the bearish path exercises
    # the same code with the sign flipped.
    st = StructureTracker()
    seq = [
        bar(0, 200, 205, 195, 204),
        bar(1, 204, 210, 202, 208),
        bar(2, 208, 209, 180, 182),
        bar(3, 182, 208, 181, 207),
        bar(4, 207, 215, 206, 210),
        bar(5, 210, 220, 209, 219),
        bar(6, 219, 219, 200, 201),
    ]
    for b in seq:
        st.update(b)
    assert [p.price for p in st.pivots.valid_highs] == [210, 220]
    assert [p.price for p in st.pivots.valid_lows] == [195, 180]
    assert st.direction == "long"  # sparse-data early read, mirroring the bullish test

    st.update(bar(7, 201, 202, 175, 178))  # closes 178 < 180 -> bearish MSS
    assert st.direction == "short"


# ---- _qty_for_risk ----


def test_qty_for_risk_basic_and_zero_distance():
    assert _qty_for_risk(500, 6000, 5990, point_value=5.0, max_contracts=10) == 10
    assert _qty_for_risk(500, 6000, 6000, point_value=5.0, max_contracts=10) == 0
    assert _qty_for_risk(10, 6000, 5000, point_value=5.0, max_contracts=10) == 1  # floors to min 1


# ---- Full 15s walkthrough (verified by execution) ----
#
# Long scenario: two valid lows confirm (190 @ bar1, 180 @ bar5) with exactly
# one confirmed valid high between them (220 @ bar2) -> Entry 1 rests at 220.
# Price breaks through it at bar 7, filling Entry 1 (stop = 180, the leg low)
# and confirming the 15s MSS in the same instant. The peak trails to 225
# (bar 7's own high), 50% of the leg (202.50) fills Entry 2 at bar 9 without
# the peak ever confirming first, and price reaches the floored 1:1 target
# (260) at bar 10.
WALKTHROUGH_BARS = [
    bar(0, 200, 205, 195, 196),
    bar(1, 196, 198, 190, 192),
    bar(2, 192, 220, 191, 218),
    bar(3, 218, 219, 192, 193),
    bar(4, 193, 194, 185, 186),
    bar(5, 186, 188, 180, 181),
    bar(6, 181, 200, 181, 199),
    bar(7, 199, 225, 198, 224),  # Entry 1 fills at 220.00, 15s MSS confirmed
    bar(8, 224, 224, 210, 212),
    bar(9, 212, 213, 200, 205),  # low touches 202.50 -> Entry 2 fills
    bar(10, 205, 265, 204, 262),  # reaches the floored target (260)
]


def _new_strategy(risk_usd=1000.0, entry1_risk_share=0.5, max_contracts=50) -> MSSPullbackStrategy:
    strat = MSSPullbackStrategy(MNQ, risk_usd=risk_usd, entry1_risk_share=entry1_risk_share, max_contracts=max_contracts)
    strat.struct_3m.direction = "long"  # 3m bias assumed already bullish
    return strat


def test_entry1_fires_at_the_shared_resistance_with_frozen_stop():
    strat = _new_strategy()
    events = []
    for b in WALKTHROUGH_BARS[:8]:  # through bar 7, the fill bar
        events += strat.on_15s_bar(b)

    fills = [e for e in events if isinstance(e, Fill)]
    assert len(fills) == 1
    fill = fills[0]
    assert fill.order == "entry1"
    assert fill.direction == "long"
    assert fill.price == 220
    assert fill.qty == 6  # floor(1000*0.5 / ((220-180)*2)) = floor(500/80) = 6
    assert fill.stop_price == 180  # the leg low: the valid low confirmed at bar 5
    assert strat.direction == "long"
    assert strat.stop_price == 180


def test_target_floors_at_1r_when_no_3m_level_exists():
    strat = _new_strategy()
    events = []
    for b in WALKTHROUGH_BARS[:8]:
        events += strat.on_15s_bar(b)
    fill = next(e for e in events if isinstance(e, Fill))
    # No 3m valid highs recorded -> target floors to entry + (entry - stop) = 220 + (220-180) = 260.
    assert fill.target_price == 260
    assert strat.target_price == 260


def test_target_uses_real_3m_level_when_it_clears_the_1r_floor():
    strat = _new_strategy()
    strat.struct_3m.pivots.valid_highs.append(Pivot(T0, 300))  # a 3m valid high well past the 1R floor (260)
    events = []
    for b in WALKTHROUGH_BARS[:8]:
        events += strat.on_15s_bar(b)
    fill = next(e for e in events if isinstance(e, Fill))
    assert fill.target_price == 300


def test_target_floors_up_when_nearest_3m_level_is_too_close():
    strat = _new_strategy()
    strat.struct_3m.pivots.valid_highs.append(Pivot(T0, 230))  # closer than 1:1 (260) above entry (220)
    events = []
    for b in WALKTHROUGH_BARS[:8]:
        events += strat.on_15s_bar(b)
    fill = next(e for e in events if isinstance(e, Fill))
    assert fill.target_price == 260  # floored up to at least 1:1, not the closer 230


def test_entry2_fires_at_fifty_percent_and_adds_to_the_position():
    strat = _new_strategy()
    events = []
    for b in WALKTHROUGH_BARS[:10]:  # through bar 9, the entry-2 fill bar
        events += strat.on_15s_bar(b)

    fills = [e for e in events if isinstance(e, Fill)]
    assert [f.order for f in fills] == ["entry1", "entry2"]
    entry2 = fills[1]
    assert entry2.direction == "long"
    assert entry2.price == 202.5  # 180 + 0.5*(225-180)
    assert entry2.qty == 11  # floor(1000*0.5 / ((202.5-180)*2)) = floor(500/45) = 11
    assert entry2.stop_price == 180
    assert entry2.target_price == fills[0].target_price  # shared target


def test_position_reaches_target_and_resets_for_a_fresh_setup():
    strat = _new_strategy()
    events = []
    for b in WALKTHROUGH_BARS:
        events += strat.on_15s_bar(b)

    exits = [e for e in events if isinstance(e, Exit)]
    assert len(exits) == 1
    exit_event = exits[0]
    assert exit_event.reason == "target"
    assert exit_event.direction == "long"
    assert exit_event.price == 260  # the floored 1R target, reached at bar 10 (high=265 >= 260)
    assert exit_event.qty == 17  # 6 (entry 1) + 11 (entry 2)
    assert strat.direction is None  # flat again, ready for the next 15s MSS


# ---- Entry 2 cancellation vs. same-bar fill priority ----


def test_entry2_cancelled_when_candidate_high_confirms_even_if_it_also_touches_fifty_percent():
    strat = MSSPullbackStrategy(NQ, risk_usd=200.0, entry1_risk_share=0.5, max_contracts=10)
    strat.direction = "long"
    strat.stop_price = 80.0
    strat.target_price = 200.0

    events_a = strat.on_15s_bar(bar(0, 110, 115, 108, 112))  # candidate high: 115, low=108 -> entry2 @ 97.5
    assert events_a == []
    assert strat.entry2_price == 97.5

    # This bar's low (95) would touch 97.5, but its close (97) also confirms
    # the candidate high (115) by clearing its opposite wick (108) -- the
    # confirmation must cancel Entry 2 instead of letting it fill.
    events_b = strat.on_15s_bar(bar(1, 112, 112, 95, 97))
    assert events_b == []
    assert strat._entry2_cancelled is True
    assert strat._entry2_filled is False
    assert strat.direction == "long"  # trade stays open, managed by SL/TP only from here


def test_stop_exit_wins_over_target_in_the_same_bar():
    strat = MSSPullbackStrategy(NQ, risk_usd=200.0, entry1_risk_share=0.5, max_contracts=10)
    strat.direction = "long"
    strat.stop_price = 100.0
    strat.target_price = 110.0
    strat._entry1_qty = 3

    events = strat.on_15s_bar(bar(0, 105, 112, 95, 108))  # range covers both stop(100) and target(110)
    exits = [e for e in events if isinstance(e, Exit)]
    assert len(exits) == 1
    assert exits[0].reason == "stop"
    assert exits[0].price == 100.0
    assert strat.direction is None
