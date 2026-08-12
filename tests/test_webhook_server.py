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
