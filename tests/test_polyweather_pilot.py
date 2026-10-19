"""Pilot safety rails, against a fake exchange client (no network, no keys)."""

import pytest

pytest.importorskip("py_clob_client")

from polyweather import pilot as P  # noqa: E402
from polyweather.ledger import Ledger  # noqa: E402
from polyweather.maker import Inventory, MakerConfig, make_quotes  # noqa: E402
from polyweather.rewards import RewardConfig  # noqa: E402
from tests.test_polyweather import make_event  # noqa: E402

FUNDER = "0xabc"


class FakeClient:
    def __init__(self):
        self.orders = []
        self.cancelled = []
        self.trades = []

    def get_orders(self, *a, **k):
        return self.orders

    def cancel_orders(self, ids):
        self.cancelled += ids

    def cancel_all(self):
        self.orders = []

    def get_trades(self, params=None):
        return self.trades


def fake_broker():
    b = P.PilotBroker.__new__(P.PilotBroker)
    b.client = FakeClient()
    b.funder = FUNDER
    b.sig_type = 1
    b.open = {}
    b.on_fill = None
    b.token_map = {}
    b._hb_thread = None
    b._hb_fail = 0
    return b


def test_reconcile_cancels_orphans_and_forgets_gone():
    b = fake_broker()
    b.open = {("s", 1, "YES"): {"id": "mine", "price": 0.4, "shares": 20, "matched": 0},
              ("s", 2, "NO"): {"id": "gone", "price": 0.5, "shares": 20, "matched": 0}}
    b.client.orders = [{"id": "mine"}, {"id": "stranger"}]
    assert b.reconcile() == (1, 1)
    assert b.client.cancelled == ["stranger"] and list(b.open) == [("s", 1, "YES")]


def test_fills_from_trade_history_only_ours():
    b = fake_broker()
    b.client.trades = [{"id": "t1", "match_time": "100", "maker_orders": [
        {"maker_address": "0xABC", "order_id": "o1", "asset_id": "y1", "price": "0.41", "matched_amount": "12"},
        {"maker_address": "0xother", "order_id": "o2", "asset_id": "y1", "price": "0.41", "matched_amount": "50"}]}]
    fills = b.fills_since(0)
    assert fills == [{"id": "t1:o1", "ts": 100.0, "token": "y1", "price": 0.41, "shares": 12.0}]


def _runner(tmp_path, broker, **cfg):
    pc = P.PilotConfig(db=str(tmp_path / "p.db"), kill_file=str(tmp_path / "STOP"), **cfg)
    r = P.PilotRunner.__new__(P.PilotRunner)
    r.pcfg, r.pbroker, r.broker = pc, broker, broker
    r.ledger = Ledger(pc.db)
    r.ledger.db.executescript(P.PILOT_SCHEMA)
    from polyweather.maker_runner import MAKER_SCHEMA
    r.ledger.db.executescript(MAKER_SCHEMA)
    r.events, r.inv, r.n, r.last_fill_ts, r.mode = [], {}, 1, 0, "live"
    return r


def test_kill_file_and_heartbeat_halt(tmp_path, monkeypatch):
    b = fake_broker()
    r = _runner(tmp_path, b)
    monkeypatch.setattr(b, "heartbeat_ok", lambda: True)
    monkeypatch.setattr(r, "day_pnl", lambda: 0.0)
    r.check_guards()  # fine
    (tmp_path / "STOP").write_text("")
    with pytest.raises(P.Halt, match="kill file"):
        r.check_guards()
    (tmp_path / "STOP").unlink()
    monkeypatch.setattr(b, "heartbeat_ok", lambda: False)
    with pytest.raises(P.Halt, match="heartbeat"):
        r.check_guards()


def test_loss_limit_and_geoblock_halt(tmp_path, monkeypatch):
    b = fake_broker()
    r = _runner(tmp_path, b, daily_loss_limit_usd=40)
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
    b = fake_broker()
    r = _runner(tmp_path, b)
    ev = make_event(labels=["59°F or below", "60-61°F", "62°F or higher"])
    r.events = [ev]
    b.token_map = {ev.buckets[1].yes_token: (ev.slug, 1, "YES")}
    b.client.trades = [{"id": "t1", "match_time": "100", "maker_orders": [
        {"maker_address": FUNDER, "order_id": "o1", "asset_id": ev.buckets[1].yes_token, "price": "0.4",
         "matched_amount": "20"}]}]
    r.record_fills()
    r.record_fills()  # same trade again
    assert r.inv[ev.slug].yes[1] == (20.0, pytest.approx(8.0))
    assert r.ledger.positions()[0].shares == 20.0


def test_preflight_refuses_when_geoblocked(monkeypatch):
    monkeypatch.setattr(P, "geoblock_status", lambda: {"blocked": True, "country": "US", "region": "OH"})
    for k in ("POLY_PRIVATE_KEY", "POLY_FUNDER", "POLY_SIGNATURE_TYPE", "POLYWEATHER_LIVE"):
        monkeypatch.setenv(k, "1")
    ok, lines = P.preflight(P.PilotConfig())
    assert not ok and "BLOCKED" in lines[0]


def test_global_cap_switches_to_reduce_only():
    ev = make_event(labels=["59°F or below", "60-61°F", "62°F or higher"])
    for i, b in enumerate(ev.buckets):
        b.condition_id = f"c{i}"
    rewards = {"c1": RewardConfig("c1", 50, 0.045, 20)}
    cfg = MakerConfig(quote_shares=20, max_total_inventory_usd=100)
    inv = Inventory()
    inv.add("YES", 1, 30, 12.0)
    fair = [None, 0.4, None]
    qs = make_quotes(ev, fair, fair, rewards, inv, cfg, total_inventory=50)
    assert {q.side for q in qs} == {"YES", "NO"}
    qs = make_quotes(ev, fair, fair, rewards, inv, cfg, total_inventory=100)
    assert [(q.side, q.shares) for q in qs] == [("NO", 20.0)]  # only the side that pairs off held YES
