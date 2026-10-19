"""Pilot, exchange broker, treasury and alerts against fakes of the unified SDK (no network, no keys)."""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from types import SimpleNamespace as NS

import pytest

from polyweather import pilot as P
from polyweather.alerts import Alerter
from polyweather.exchange import ExchangeBroker
from polyweather.ledger import Ledger
from polyweather.maker import Inventory, MakerConfig, Quote, make_quotes
from polyweather.rewards import RewardConfig
from polyweather.treasury import Treasury, TreasuryConfig
from tests.test_polyweather import make_event

WALLET = "0xabc"


def pages(*items):
    return [NS(items=tuple(items))]


@dataclass
class FakeSDK:
    wallet: str = "0xABC"
    wallet_type: str = "POLY_PROXY"
    orders: list = field(default_factory=list)
    trades: list = field(default_factory=list)
    positions: list = field(default_factory=list)
    placed: list = field(default_factory=list)
    cancelled: list = field(default_factory=list)
    merged: list = field(default_factory=list)
    redeemed: list = field(default_factory=list)
    fail_merge: bool = False

    def list_open_orders(self, **kw):
        return pages(*self.orders)

    def cancel_orders(self, *, order_ids):
        self.cancelled += list(order_ids)

    def cancel_all(self):
        self.orders = []

    def place_limit_order(self, **kw):
        self.placed.append(kw)
        if kw["price"] == "0.999":
            return NS(ok=False, code="post_only_would_cross")
        return NS(ok=True, order_id=f"o{len(self.placed)}")

    def list_account_trades(self, **kw):
        return pages(*self.trades)

    def get_orders_scoring(self, *, order_ids):
        return {i: i != order_ids[0] for i in order_ids}

    def list_positions(self, **kw):
        return pages(*self.positions)

    def merge_positions(self, **kw):
        if self.fail_merge:
            raise RuntimeError("relayer down")
        self.merged.append(kw["condition_id"])
        return NS(wait=lambda: NS(transaction_hash="0xmerge"))

    def redeem_positions(self, **kw):
        self.redeemed.append(kw["condition_id"])
        return NS(wait=lambda: NS(transaction_hash="0xredeem"))


def broker(**kw):
    return ExchangeBroker(client=FakeSDK(**kw))


def quote_event():
    ev = make_event(labels=["59°F or below", "60-61°F", "62°F or higher"])
    for i, b in enumerate(ev.buckets):
        b.condition_id = f"c{i}"
    return ev


# ---------------------------------------------------------------- exchange broker


def test_sync_posts_post_only_bids_and_cancels_stale():
    b = broker()
    ev = quote_event()
    q = Quote(1, "YES", ev.buckets[1].yes_token, 0.39, 20, 0.40)
    b.sync({ev.slug: [q]}, {ev.slug: ev})
    kw = b.client.placed[0]
    assert kw["side"] == "BUY" and kw["post_only"] and kw["price"] == "0.390" and kw["size"] == "20.00"
    assert kw["expiration"] is not None  # heartbeat never started -> short-lived GTD orders
    b.sync({ev.slug: [q]}, {ev.slug: ev})
    assert len(b.client.placed) == 1  # unchanged quote is left alone
    b.sync({ev.slug: [Quote(1, "YES", q.token_id, 0.38, 20, 0.40)]}, {ev.slug: ev})
    assert b.client.cancelled == ["o1"] and len(b.client.placed) == 2


def test_rejections_are_counted_not_tracked():
    b = broker()
    ev = quote_event()
    b.sync({ev.slug: [Quote(1, "YES", ev.buckets[1].yes_token, 0.999, 20, 1.0)]}, {ev.slug: ev})
    assert b.open == {} and b.rejections == {"post_only_would_cross": 1}


def test_reconcile_and_fills_and_scoring():
    b = broker()
    b.open = {("s", 1, "YES"): {"id": "mine", "price": 0.4, "shares": 20, "expires": None},
              ("s", 2, "NO"): {"id": "gone", "price": 0.5, "shares": 20, "expires": None}}
    b.client.orders = [NS(id="mine"), NS(id="stranger")]
    assert b.reconcile() == (1, 1) and b.client.cancelled == ["stranger"]
    b.client.trades = [NS(id="t1", matched_at=datetime.fromtimestamp(100, timezone.utc), maker_orders=(
        NS(maker_address="0xABC", order_id="o1", asset_id="y1", price="0.41", matched_amount="12"),
        NS(maker_address="0xother", order_id="o2", asset_id="y1", price="0.41", matched_amount="50")))]
    assert b.fills_since(0) == [{"id": "t1:o1", "ts": 100.0, "token": "y1", "price": 0.41, "shares": 12.0}]
    b.open[("s", 3, "YES")] = {"id": "x", "price": 0.3, "shares": 20, "expires": None}
    assert b.scoring_fraction() == pytest.approx(0.5)


def test_heartbeat_falls_back_to_gtd_when_unavailable():
    b = broker()
    assert b.start_heartbeat() is False  # fake has no _ctx -> unavailable
    assert b.heartbeat_ok() and not b.heartbeat_supported


# ---------------------------------------------------------------- treasury


def _ledger(tmp_path):
    led = Ledger(str(tmp_path / "t.db"))
    led.db.executescript(P.PILOT_SCHEMA)
    return led


def _pos(cid, token, size, mergeable=False, redeemable=False, value=0.0):
    return NS(condition_id=cid, asset_id=token, current_size=size, mergeable=mergeable, redeemable=redeemable,
              archived=False, current_value=value, event_slug="ev")


def test_treasury_merges_pairs_and_books_cash(tmp_path):
    led = _ledger(tmp_path)
    led.add_fill("ev", 1, "YES", "y1", "nyc", "high", "2026-10-19", "60-61", 30, 12.0)
    led.add_fill("ev", 1, "NO", "n1", "nyc", "high", "2026-10-19", "60-61", 20, 11.0)
    inv = {"ev": Inventory()}
    inv["ev"].add("YES", 1, 30, 12.0)
    inv["ev"].add("NO", 1, 20, 11.0)
    b = broker(positions=[_pos("c1", "y1", 30, mergeable=True), _pos("c1", "n1", 20, mergeable=True)])
    sent = []
    t = Treasury(b, led, Alerter(_post=lambda url, **kw: sent.append(url)), inventories=inv)
    out = t.run(now=1e9)
    assert b.client.merged == ["c1"] and out["merged_pairs"] == 20
    rows = {r.side: r for r in led.positions()}
    assert rows["YES"].shares == pytest.approx(10) and rows["NO"].shares == pytest.approx(0)
    # 20 pairs cost 20*12/30 + 11 = 19 -> $20 cash, +$1 realized
    realized = led.db.execute("SELECT realized FROM pilot_treasury WHERE action='merge'").fetchone()[0]
    assert realized == pytest.approx(1.0)
    assert inv["ev"].cost("NO", 1) == pytest.approx(0) and inv["ev"].yes[1][0] == pytest.approx(10)


def test_treasury_redeems_and_throttles_and_alerts_on_repeated_failure(tmp_path):
    led = _ledger(tmp_path)
    b = broker(positions=[_pos("c9", "y9", 50, redeemable=True, value=50.0)])
    msgs = []
    al = Alerter(telegram_token="t", telegram_chat="c", _post=lambda url, **kw: msgs.append(kw["json"]["text"]))
    t = Treasury(b, led, al, cfg=TreasuryConfig(give_up_after=2))
    assert t.run(now=1e9)["redeemed"] == 1 and b.client.redeemed == ["c9"]
    assert t.run(now=1e9 + 60) == {}  # throttled
    b.client.positions = [_pos("c2", "y2", 30, mergeable=True), _pos("c2", "n2", 30, mergeable=True)]
    b.client.fail_merge = True
    t.run(now=1e9 + 1000)
    t.run(now=1e9 + 2000)
    assert any("merge keeps failing" in m for m in msgs)
    t.run(now=1e9 + 3000)
    assert b.client.merged == []  # gave up on that condition


# ---------------------------------------------------------------- alerts


def test_alert_channels_and_throttle():
    calls = []
    a = Alerter(ntfy_topic="topic", webhook_url="https://discord.com/api/webhooks/x",
                _post=lambda url, **kw: calls.append((url, kw)))
    assert a.channels == ["ntfy", "webhook"]
    assert a.send("x", "y", level="warn", key="k")
    assert not a.send("x", "y", level="warn", key="k")  # throttled
    assert a.send("HALTED", "boom", level="urgent")
    ntfy = [kw for url, kw in calls if "ntfy" in url]
    assert ntfy[-1]["headers"]["Priority"] == "urgent"
    assert "content" in [kw for url, kw in calls if "discord" in url][0]["json"]
    # titles with non-latin-1 characters must not break HTTP headers
    a.send("62°F → done", level="info")


def test_alert_delivery_failure_is_reported():
    def boom(url, **kw):
        raise RuntimeError("429")
    assert not Alerter(ntfy_topic="t", _post=boom).send("x")


# ---------------------------------------------------------------- pilot guards


def _runner(tmp_path, b, **cfg):
    pc = P.PilotConfig(db=str(tmp_path / "p.db"), kill_file=str(tmp_path / "STOP"), **cfg)
    return P.PilotRunner(pc, Ledger(pc.db), b, Alerter(_post=lambda *a, **k: None))


def test_kill_file_heartbeat_loss_limit_and_geoblock(tmp_path, monkeypatch):
    b = broker()
    r = _runner(tmp_path, b, daily_loss_limit_usd=40)
    monkeypatch.setattr(r, "day_pnl", lambda: 0.0)
    r.n = 1
    r.check_guards()
    (tmp_path / "STOP").write_text("")
    with pytest.raises(P.Halt, match="kill file"):
        r.check_guards()
    (tmp_path / "STOP").unlink()
    monkeypatch.setattr(b, "heartbeat_ok", lambda: False)
    with pytest.raises(P.Halt, match="heartbeat"):
        r.check_guards()
    monkeypatch.setattr(b, "heartbeat_ok", lambda: True)
    monkeypatch.setattr(r, "day_pnl", lambda: -41.0)
    with pytest.raises(P.Halt, match="loss limit"):
        r.check_guards()
    monkeypatch.setattr(r, "day_pnl", lambda: 0.0)
    monkeypatch.setattr(P, "geoblock_status", lambda: {"blocked": True, "country": "US"})
    r.n = 20
    with pytest.raises(P.Halt, match="geoblocked"):
        r.check_guards()


def test_record_fills_dedupes(tmp_path):
    b = broker()
    r = _runner(tmp_path, b)
    ev = quote_event()
    r.events = [ev]
    b.token_map = {ev.buckets[1].yes_token: (ev.slug, 1, "YES")}
    b.client.trades = [NS(id="t1", matched_at=datetime.fromtimestamp(100, timezone.utc), maker_orders=(
        NS(maker_address=WALLET, order_id="o1", asset_id=ev.buckets[1].yes_token, price="0.4", matched_amount="20"),))]
    r.record_fills()
    r.record_fills()
    assert r.inv[ev.slug].yes[1] == (20.0, pytest.approx(8.0))
    assert r.ledger.positions()[0].shares == 20.0


def test_preflight_refuses_when_geoblocked(monkeypatch):
    monkeypatch.setattr(P, "geoblock_status", lambda: {"blocked": True, "country": "US", "region": "OH"})
    for k in P.REQUIRED_ENV + ("POLYWEATHER_LIVE",):
        monkeypatch.setenv(k, "1")
    ok, lines = P.preflight(P.PilotConfig(), Alerter())
    assert not ok and "BLOCKED" in lines[0]


def test_daily_summary(tmp_path):
    b = broker()
    r = _runner(tmp_path, b)
    db = r.ledger.db
    db.execute("INSERT INTO maker_rewards VALUES (?,?,?,?,?,?,?,?)", (86400 * 20000 + 5, "live", "s", 1, 50, 1, 1, 12.5))
    db.execute("INSERT INTO pilot_rewards_actual VALUES (?,?,?,?)", ("2024-10-04", 10.0, 0, "[]"))
    text = P.daily_summary(r.ledger, datetime(2024, 10, 4).date())
    assert "actual $10.00 (80% of estimate)" in text and "estimate $12.50" in text


def test_global_cap_switches_to_reduce_only():
    ev = quote_event()
    rewards = {"c1": RewardConfig("c1", 50, 0.045, 20)}
    cfg = MakerConfig(quote_shares=20, max_side_inventory_usd=40, max_total_inventory_usd=100)
    inv = Inventory()
    inv.add("YES", 1, 30, 12.0)
    fair = [None, 0.4, None]
    assert {q.side for q in make_quotes(ev, fair, fair, rewards, inv, cfg, total_inventory=50)} == {"YES", "NO"}
    qs = make_quotes(ev, fair, fair, rewards, inv, cfg, total_inventory=100)
    assert [(q.side, q.shares) for q in qs] == [("NO", 20.0)]  # only the side that pairs off held YES
