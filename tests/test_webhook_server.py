import importlib

import pytest
from fastapi.testclient import TestClient


def _make_client(monkeypatch, **extra_env):
    monkeypatch.setenv("WEBHOOK_SECRET", "test-secret")
    monkeypatch.setenv("TRADOVATE_ACCOUNT_SPEC", "test-account")
    monkeypatch.setenv("DRY_RUN", "true")
    monkeypatch.setenv("RISK_PER_TRADE_USD", "200")
    monkeypatch.setenv("MAX_CONTRACTS", "5")
    monkeypatch.setenv("MAX_DAILY_TRADES", "10")
    for k, v in extra_env.items():
        monkeypatch.setenv(k, v)

    import webhook.server as server
    importlib.reload(server)  # re-read the env vars set above
    return TestClient(server.app)


BUY_PAYLOAD = {
    "secret": "test-secret", "root": "MES", "symbol": "MESZ2026", "action": "buy",
    "entry_price": 6000, "stop_price": 5990, "target_price": 6020,
}


def test_rejects_bad_secret(monkeypatch):
    client = _make_client(monkeypatch)
    resp = client.post("/tradingview-webhook", json={**BUY_PAYLOAD, "secret": "wrong"})
    assert resp.status_code == 401


def test_dry_run_buy_computes_contracts(monkeypatch):
    client = _make_client(monkeypatch)
    resp = client.post("/tradingview-webhook", json=BUY_PAYLOAD)
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "dry_run"
    assert body["contracts"] == 4  # $200 risk / (10 pts * $5/pt) = 4


def test_dry_run_flatten(monkeypatch):
    client = _make_client(monkeypatch)
    resp = client.post("/tradingview-webhook", json={"secret": "test-secret", "action": "flatten", "symbol": "MESZ2026"})
    assert resp.status_code == 200
    assert resp.json()["status"] == "flattened"


def test_unknown_action_rejected(monkeypatch):
    client = _make_client(monkeypatch)
    resp = client.post("/tradingview-webhook", json={"secret": "test-secret", "action": "wat"})
    assert resp.status_code == 400


def test_missing_price_field_rejected(monkeypatch):
    client = _make_client(monkeypatch)
    bad = {k: v for k, v in BUY_PAYLOAD.items() if k != "stop_price"}
    resp = client.post("/tradingview-webhook", json=bad)
    assert resp.status_code == 400


def test_risk_manager_blocks_after_daily_trade_cap(monkeypatch):
    client = _make_client(monkeypatch, MAX_DAILY_TRADES="1")

    first = client.post("/tradingview-webhook", json=BUY_PAYLOAD)
    assert first.json()["status"] == "dry_run"

    second = client.post("/tradingview-webhook", json=BUY_PAYLOAD)
    assert second.json()["status"] == "blocked_by_risk_manager"


def test_larger_risk_budget_scales_contracts(monkeypatch):
    client = _make_client(monkeypatch, RISK_PER_TRADE_USD="1000")
    resp = client.post("/tradingview-webhook", json=BUY_PAYLOAD)
    assert resp.json()["contracts"] == 5  # would be 20, clamped to MAX_CONTRACTS=5


def _with_positions(monkeypatch, client, positions):
    import webhook.server as server
    monkeypatch.setattr(server, "_open_positions", lambda _client: positions)
    return client


def test_hedging_guard_blocks_opposing_position_on_another_account(monkeypatch):
    from failed2s.hedging import Position
    client = _with_positions(monkeypatch, _make_client(monkeypatch), [Position(999, "NQZ6", -1)])
    resp = client.post("/tradingview-webhook", json=BUY_PAYLOAD)
    body = resp.json()
    assert body["status"] == "blocked_by_hedging_guard"
    assert "NQ" in body["reason"]


def test_hedging_guard_allows_same_direction(monkeypatch):
    from failed2s.hedging import Position
    client = _with_positions(monkeypatch, _make_client(monkeypatch), [Position(999, "NQZ6", 1)])
    assert client.post("/tradingview-webhook", json=BUY_PAYLOAD).json()["status"] == "dry_run"


def test_blocked_entry_does_not_use_up_daily_trade_cap(monkeypatch):
    from failed2s.hedging import Position
    import webhook.server as server
    client = _make_client(monkeypatch, MAX_DAILY_TRADES="1")
    monkeypatch.setattr(server, "_open_positions", lambda _c: [Position(999, "NQZ6", -1)])
    assert client.post("/tradingview-webhook", json=BUY_PAYLOAD).json()["status"] == "blocked_by_hedging_guard"
    monkeypatch.setattr(server, "_open_positions", lambda _c: [])
    assert client.post("/tradingview-webhook", json=BUY_PAYLOAD).json()["status"] == "dry_run"


def test_entry_blocked_when_positions_unavailable(monkeypatch):
    import webhook.server as server
    client = _make_client(monkeypatch)

    def boom(_client):
        raise RuntimeError("network down")

    monkeypatch.setattr(server, "_open_positions", boom)
    assert client.post("/tradingview-webhook", json=BUY_PAYLOAD).json()["status"] == "blocked_positions_unavailable"


def test_contract_limit_env(monkeypatch):
    client = _make_client(monkeypatch, ACCOUNT_CONTRACT_LIMIT_MINIS="0.3")
    body = client.post("/tradingview-webhook", json=BUY_PAYLOAD).json()  # 4 MES = 0.4 minis
    assert body["status"] == "blocked_by_hedging_guard"
    assert body["reason"].startswith("contract limit:")
