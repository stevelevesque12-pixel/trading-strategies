import pytest

from polymarket_mm.book import OrderBook, ceil_tick, floor_tick
from polymarket_mm.engine import MarketInfo, MarketMaker, RiskParams
from polymarket_mm.paper import PaperBroker
from polymarket_mm.quoting import QuoteParams, compute_quotes
from polymarket_mm.report import markouts, summary
from polymarket_mm.rewards import order_score, our_share, q_min

import pandas as pd


def lv(*pairs):
    return [{"price": str(p), "size": str(s)} for p, s in pairs]


# ---- book ----------------------------------------------------------------

def test_book_snapshot_and_updates():
    b = OrderBook()
    b.load_snapshot(lv((0.48, 100), (0.47, 50)), lv((0.52, 80), (0.53, 10)))
    assert (b.best_bid, b.best_ask, b.mid) == (0.48, 0.52, 0.5)
    b.set_level("BUY", "0.49", "20")
    b.set_level("SELL", "0.52", "0")  # size 0 removes the level
    assert (b.best_bid, b.best_ask) == (0.49, 0.53)
    assert b.size_at("BUY", 0.47) == 50


def test_tick_rounding_is_float_safe():
    assert floor_tick(0.29, 0.01) == 0.29
    assert ceil_tick(0.57, 0.01) == 0.57
    assert floor_tick(0.4999, 0.001) == 0.499


# ---- quoting -------------------------------------------------------------

def test_quotes_sit_inside_reward_band_and_join_book():
    # mid 0.50, max_spread 4c, band 0.5 -> target 0.48 / 0.52; book best is 0.49/0.51
    q = compute_quotes(0.49, 0.51, 0, 4.0, 20, 0.01, QuoteParams())
    assert (q.bid, q.ask) == (0.48, 0.52)
    assert q.bid_size == q.ask_size == 20


def test_join_only_never_steps_ahead_of_best():
    # wide book: target 0.48/0.52 would improve on 0.46/0.54
    q = compute_quotes(0.46, 0.54, 0, 4.0, 20, 0.01, QuoteParams(max_book_spread=0.10))
    assert (q.bid, q.ask) == (0.46, 0.54)
    q = compute_quotes(0.46, 0.54, 0, 4.0, 20, 0.01, QuoteParams(max_book_spread=0.10, join_only=False))
    assert (q.bid, q.ask) == (0.48, 0.52)


def test_inventory_skew_and_limit():
    long_q = compute_quotes(0.49, 0.51, 400, 4.0, 20, 0.01, QuoteParams())
    assert long_q.bid < 0.48  # long -> bid backs off
    capped = compute_quotes(0.49, 0.51, 1000, 4.0, 20, 0.01, QuoteParams())
    assert capped.bid is None and capped.ask is not None
    short_capped = compute_quotes(0.49, 0.51, -1000, 4.0, 20, 0.01, QuoteParams())
    assert short_capped.ask is None and short_capped.bid is not None


def test_stand_aside_on_extreme_mid_or_wide_book():
    assert compute_quotes(0.03, 0.04, 0, 4.0, 20, 0.01, QuoteParams()).bid is None
    assert compute_quotes(0.40, 0.60, 0, 4.0, 20, 0.01, QuoteParams()).bid is None


def test_quotes_never_cross():
    # 1-tick book, tiny band: bid must stay below ask
    q = compute_quotes(0.50, 0.51, 0, 0.5, 5, 0.01, QuoteParams(join_only=False))
    assert q.bid <= 0.50 and q.ask >= 0.51


# ---- rewards -------------------------------------------------------------

def test_order_score_shape():
    assert order_score(0.0, 4.0, 100) == 100
    assert order_score(0.02, 4.0, 100) == pytest.approx(25)  # half the band -> 1/4
    assert order_score(0.05, 4.0, 100) == 0


def test_q_min_single_sided_penalty():
    assert q_min(90, 0, 0.5) == 30  # mid range: one side counts at 1/3
    assert q_min(90, 0, 0.95) == 0  # extremes: must be two-sided
    assert q_min(90, 60, 0.5) == 60


def test_our_share_grows_with_tighter_quotes():
    b = OrderBook()
    b.load_snapshot(lv((0.49, 100)), lv((0.51, 100)))
    wide = our_share(b, [(0.47, 100)], [(0.53, 100)], 4.0, 20)
    tight = our_share(b, [(0.49, 100)], [(0.51, 100)], 4.0, 20)
    assert 0 < wide < tight < 1
    assert our_share(b, [(0.49, 10)], [(0.51, 10)], 4.0, 20) == 0  # below min_size


# ---- paper fills ---------------------------------------------------------

def test_queue_position_trade_at_our_price():
    br = PaperBroker(latency=0)
    br.place("BUY", 0.48, 50, queue_ahead=100, now=0)
    assert br.on_trade("SELL", 0.48, 80, now=1) == []      # eats queue ahead
    fills = br.on_trade("SELL", 0.48, 40, now=2)           # 20 ahead left, 20 for us
    assert [f.size for f in fills] == [20]
    assert br.position == 20 and br.cash == pytest.approx(-9.6)


def test_trade_through_fills_in_full():
    br = PaperBroker(latency=0)
    br.place("SELL", 0.52, 50, queue_ahead=1000, now=0)
    fills = br.on_trade("BUY", 0.53, 5, now=1)
    assert [f.size for f in fills] == [50] and br.position == -50


def test_book_cross_fills():
    br = PaperBroker(latency=0)
    br.place("BUY", 0.48, 30, queue_ahead=500, now=0)
    assert br.on_book(0.47, 0.49, now=1) == []
    assert [f.size for f in br.on_book(0.46, 0.48, now=2)] == [30]


def test_latency_order_not_live_and_cancel_in_flight_still_fills():
    br = PaperBroker(latency=0.5)
    o = br.place("BUY", 0.48, 30, queue_ahead=0, now=0)
    assert br.on_trade("SELL", 0.40, 10, now=0.2) == []   # not live yet
    br.cancel(o, now=1.0)                                   # cancel lands at 1.5
    assert [f.size for f in br.on_trade("SELL", 0.40, 10, now=1.2)] == [30]


def test_wrong_side_trade_ignored():
    br = PaperBroker(latency=0)
    br.place("BUY", 0.48, 30, queue_ahead=0, now=0)
    assert br.on_trade("BUY", 0.47, 100, now=1) == []


# ---- engine --------------------------------------------------------------

class _Log:
    def __init__(self):
        self.recs = []

    def write(self, **r):
        self.recs.append(r)


def _mm(**risk):
    info = MarketInfo("0xc", "Q?", "q", "YES", "NO", tick=0.01, min_size=20, max_spread=4.0, daily_rate=100)
    return MarketMaker(info, QuoteParams(), RiskParams(min_requote_interval=0, **risk), _Log(),
                       PaperBroker(latency=0))


def test_engine_quotes_and_fills_from_no_token_trade():
    mm = _mm()
    mm.on_book("YES", lv((0.49, 100)), lv((0.51, 100)), now=0)
    bids = [o for o in mm.broker.working(0) if o.side == "BUY"]
    assert [o.price for o in bids] == [0.48]
    # someone buys NO at 0.53 == sells YES at 0.47 -> through our 0.48 bid
    mm.on_trade({"asset_id": "NO", "price": "0.53", "size": "10", "side": "BUY", "transaction_hash": "t1"}, now=1)
    assert mm.broker.position == 20
    # the same match reported on the YES token is not double counted
    mm.on_trade({"asset_id": "YES", "price": "0.47", "size": "10", "side": "SELL", "transaction_hash": "t1"}, now=1)
    assert mm.broker.position == 20


def test_engine_pulls_quotes_on_fast_move():
    mm = _mm(pull_move=0.02, cooldown=30)
    mm.on_book("YES", lv((0.49, 100)), lv((0.51, 100)), now=0)
    mm.on_price_changes([{"asset_id": "YES", "side": "BUY", "price": "0.49", "size": "0"},
                         {"asset_id": "YES", "side": "BUY", "price": "0.52", "size": "50"},
                         {"asset_id": "YES", "side": "SELL", "price": "0.51", "size": "0"},
                         {"asset_id": "YES", "side": "SELL", "price": "0.54", "size": "50"}], now=1)
    assert any(r["type"] == "pull" for r in mm.log.recs)
    assert mm.broker.working(1) == []
    mm.on_timer(40)  # cooldown over -> quoting again
    assert len(mm.broker.working(40)) == 2


def test_engine_halts_at_loss_limit():
    mm = _mm(max_loss=1.0, pull_move=1.0)
    mm.on_book("YES", lv((0.49, 100)), lv((0.51, 100)), now=0)
    mm.on_trade({"asset_id": "YES", "price": "0.40", "size": "5", "side": "SELL", "transaction_hash": "x"}, now=1)
    mm.on_book("YES", lv((0.39, 100)), lv((0.41, 100)), now=2)  # 20 sh long, mid -8c -> -1.6
    assert mm.halted and mm.broker.working(3) == []


def test_engine_accrues_rewards_while_quoting():
    mm = _mm()
    mm.on_book("YES", lv((0.49, 100)), lv((0.51, 100)), now=0)
    mm.on_timer(1)
    mm.on_timer(3601)
    assert 0 < mm.rewards < 100 / 24
    mm.flush(3601)
    logged = sum(r["amount"] for r in mm.log.recs if r["type"] == "reward")
    assert logged == pytest.approx(mm.rewards)


# ---- report --------------------------------------------------------------

def test_markouts_sign_and_horizon():
    fills = pd.DataFrame([{"market": "m", "ts": 0.0, "side": "BUY", "price": 0.48, "size": 10, "fee": 0},
                          {"market": "m", "ts": 0.0, "side": "SELL", "price": 0.52, "size": 10, "fee": 0}])
    mids = pd.DataFrame([{"market": "m", "ts": 0.0, "mid": 0.50}, {"market": "m", "ts": 5.0, "mid": 0.55},
                         {"market": "m", "ts": 100.0, "mid": 0.55}])
    mo = markouts(fills, mids, horizons=(0, 10, 1000))
    assert mo["mo_0s"].round(6).tolist() == [2.0, 2.0]
    assert mo["mo_10s"].round(6).tolist() == [7.0, -3.0]
    assert mo["mo_1000s"].isna().all()  # log ended before t+h


def test_summary_pnl():
    logs = {
        "fill": pd.DataFrame([{"market": "m", "ts": 0.0, "side": "BUY", "price": 0.48, "size": 10, "fee": 0}]),
        "mid": pd.DataFrame([{"market": "m", "ts": 0.0, "mid": 0.50}, {"market": "m", "ts": 9.0, "mid": 0.45}]),
        "reward": pd.DataFrame([{"market": "m", "amount": 0.25}]),
    }
    s = summary(logs, horizons=(0,))
    row = s[s["market"] == "m"].iloc[0]
    assert row["trading_pnl"] == pytest.approx(-0.30)  # bought 10 @ .48, marked .45
    assert row["total"] == pytest.approx(-0.05)
