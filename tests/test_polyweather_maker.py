"""Offline tests for the market maker and reward scoring."""

from datetime import datetime, timezone

import pytest

from polyweather.maker import Inventory, MakerConfig, fills_from_prints, in_blackout, make_quotes, usual_obs_minutes
from polyweather.orderbook import OrderBook
from polyweather.rewards import (RewardConfig, adjusted_mid, book_q, expected_daily_reward, order_score,
                                 our_q, parse_reward_config, q_min)
from tests.test_polyweather import make_event

RC = RewardConfig("c", daily_rate=100.0, max_spread=0.03, min_size=20.0)


def test_order_score_matches_docs_example():
    # docs: max spread 3c, orders 1c and 2c from mid
    assert order_score(0.01, 0.03) == pytest.approx((2 / 3) ** 2)
    assert order_score(0.02, 0.03) == pytest.approx((1 / 3) ** 2)
    assert order_score(0.031, 0.03) == 0.0


def test_q_min_single_sided_penalty_and_tails():
    assert q_min(90, 0, 0.5) == pytest.approx(30)  # one-sided at a third
    assert q_min(90, 60, 0.5) == 60
    assert q_min(90, 0, 0.05) == 0  # below 0.10 mid: must be two-sided


def test_our_q_two_sided():
    q = our_q((0.49, 100), (0.49, 100), 0.50, RC)  # YES bid 0.49, NO bid 0.49 == YES ask 0.51
    assert q == pytest.approx(100 * (2 / 3) ** 2)
    assert our_q((0.49, 10), None, 0.5, RC) == 0  # below min size


def test_reward_share_and_parse():
    assert expected_daily_reward(10, 30, RC) == pytest.approx(25.0)
    assert expected_daily_reward(0, 30, RC) == 0
    cfg = parse_reward_config({"conditionId": "x", "clobRewards": [{"rewardsDailyRate": 56}],
                               "rewardsMaxSpread": 4.5, "rewardsMinSize": 20})
    assert cfg.daily_rate == 56 and cfg.max_spread == pytest.approx(0.045)
    assert parse_reward_config({"clobRewards": None}) is None


def test_adjusted_mid_skips_dust_and_book_q():
    yes = OrderBook("y", bids=[(0.48, 5), (0.47, 100)], asks=[(0.50, 100)])
    assert adjusted_mid(yes, 20) == pytest.approx(0.485)
    no = OrderBook("n", bids=[(0.50, 100)], asks=[])  # NO bid 0.50 == YES ask 0.50
    assert book_q(yes, no, 0.485, RC) > book_q(yes, None, 0.485, RC)


def _event_with_rewards():
    ev = make_event(labels=["59°F or below", "60-61°F", "62-63°F", "64°F or higher"])
    for i, b in enumerate(ev.buckets):
        b.condition_id = f"c{i}"
    rewards = {"c1": RewardConfig("c1", 50, 0.045, 20), "c2": RewardConfig("c2", 50, 0.045, 20)}
    return ev, rewards


def test_make_quotes_brackets_fair_inside_band():
    ev, rewards = _event_with_rewards()
    qs = make_quotes(ev, [0.1, 0.4, 0.45, 0.05], [0.1, 0.4, 0.45, 0.05], rewards, Inventory(), MakerConfig(max_side_inventory_usd=40))
    assert {(q.bucket, q.side) for q in qs} == {(1, "YES"), (1, "NO"), (2, "YES"), (2, "NO")}  # rewarded only
    y = next(q for q in qs if q.bucket == 1 and q.side == "YES")
    n = next(q for q in qs if q.bucket == 1 and q.side == "NO")
    assert y.price == pytest.approx(0.39) and n.price == pytest.approx(0.59)
    assert y.price + n.price < 1  # both filling locks in a profit


def test_make_quotes_respects_inventory_and_never_crosses():
    ev, rewards = _event_with_rewards()
    inv = Inventory()
    inv.add("YES", 1, 100, 39.0)
    qs = make_quotes(ev, [None, 0.4, None, None], [None, 0.4, None, None], rewards, inv, MakerConfig())
    assert [q.side for q in qs] == ["NO"]  # YES side at its cap
    # long 100 YES: fair skews down, so the NO bid gets keener (up to the mid) to flatten
    assert qs[0].price == pytest.approx(0.60)
    qs = make_quotes(ev, [None, 0.4, None, None], [None, 0.4, None, None], rewards, Inventory(), MakerConfig(),
                     best_asks=[(None, None), (0.385, 0.70), (None, None), (None, None)])
    y = next(q for q in qs if q.side == "YES")
    assert y.price < 0.385


def test_blackout_window():
    cfg = MakerConfig(blackout_before_min=5, blackout_after_min=8)
    t = lambda m: datetime(2026, 10, 19, 14, m, tzinfo=timezone.utc)
    assert in_blackout(t(48), "UTC", [51], cfg) and in_blackout(t(58), "UTC", [51], cfg)
    assert not in_blackout(t(30), "UTC", [51], cfg) and not in_blackout(t(0), "UTC", [51], cfg)
    assert usual_obs_minutes([t(51)] * 3 + [t(20)]) == [51]


def test_fills_from_prints():
    ev, _ = _event_with_rewards()
    b = ev.buckets[1]
    from polyweather.maker import Quote
    q = Quote(1, "YES", b.yes_token, 0.40, 50, 0.41)
    trades = [
        {"ts": 5, "asset": b.yes_token, "side": "SELL", "price": 0.35, "size": 20},  # before window
        {"ts": 15, "asset": b.yes_token, "side": "SELL", "price": 0.42, "size": 100},  # above our bid
        {"ts": 16, "asset": b.yes_token, "side": "SELL", "price": 0.38, "size": 20},  # through us: full
        {"ts": 17, "asset": b.yes_token, "side": "SELL", "price": 0.40, "size": 20},  # at our price: half
        {"ts": 18, "asset": b.no_token, "side": "BUY", "price": 0.61, "size": 100},  # complement buy >= 0.60
    ]
    assert fills_from_prints(q, ev, trades, 10, 20) == pytest.approx(50)  # 20 + 10 + capped at 50
    assert fills_from_prints(q, ev, trades[:3], 10, 20) == pytest.approx(20)


def test_inventory_payout_pairs_redeem():
    inv = Inventory()
    inv.add("YES", 1, 10, 4.0)
    inv.add("NO", 1, 10, 5.5)
    assert inv.payout(1) == 10 and inv.payout(2) == 10  # YES+NO on one bucket always pays $1/pair


def test_stop_hour_by_kind():
    from polyweather.maker import quoting_allowed
    ev, _ = _event_with_rewards()  # NYC high on 2026-10-19
    from datetime import timedelta
    et = lambda d, h: datetime(2026, 10, d, tzinfo=timezone.utc) + timedelta(hours=h + 4)  # EDT = UTC-4
    cfg = MakerConfig()
    assert quoting_allowed(ev, et(18, 15), cfg)  # day before
    assert quoting_allowed(ev, et(19, 11), cfg)
    assert not quoting_allowed(ev, et(19, 12), cfg)  # target-day afternoon
    ev.kind = "low"
    assert quoting_allowed(ev, et(18, 23), cfg) and not quoting_allowed(ev, et(19, 1), cfg)
    assert quoting_allowed(ev, et(19, 15), MakerConfig(stop_hour_low=None))
