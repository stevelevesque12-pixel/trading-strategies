"""
TradingView webhook -> risk-sized Tradovate bracket order, for MES.

Deploy this with a public HTTPS endpoint (TradingView's alert servers call
out to it) -- it does not run usefully from a sandboxed environment with no
public ingress and no outbound access to tradovateapi.com. See
../TRADINGVIEW_WEBHOOK.md for the full setup: installing the Pine script,
configuring the TradingView alert, and deployment options.

Run:
    uvicorn webhook.server:app --host 0.0.0.0 --port 8000

Required environment variables:
    WEBHOOK_SECRET            shared secret checked against the alert payload
    TRADOVATE_ACCOUNT_SPEC    Tradovate account username/spec
    TRADOVATE_USERNAME, TRADOVATE_PASSWORD, TRADOVATE_APP_ID, TRADOVATE_CID,
    TRADOVATE_SEC             Tradovate API credentials (see live/tradovate_client.py)
Optional:
    RISK_PER_TRADE_USD        dollars risked per trade; contract count is derived
                               from this and the signal's stop distance (default 200)
    MAX_CONTRACTS             hard cap regardless of risk math (default 5)
    DAILY_LOSS_LIMIT_USD      (default 1000)
    MAX_DAILY_TRADES          (default 10)
    ACCOUNT_CONTRACT_LIMIT_MINIS  your Tradeify account's contract limit, in
                               mini-equivalents (10 micros = 1 mini); unset = no check
    TRADOVATE_ENV             "demo" (default) or "live"
    DRY_RUN                   "true" (default) or "false" -- log-only vs. real orders

Known gap: this process has no fill/close feedback from Tradovate. Every
accepted signal is recorded against MAX_DAILY_TRADES immediately, so that
cap works correctly as an entry-rate limiter. DAILY_LOSS_LIMIT_USD does
NOT work the same way -- with no real P&L to record, it never actually
trips. Treat it as a placeholder until a fills listener is wired in, and
don't run this unattended past your real daily loss limit without also
checking the account manually.

Hedging guard: before every entry, open positions on EVERY account this
Tradovate login can see are checked against Tradeify's hedging rule (no
opposing positions in the same Product Group, across accounts, any contract
size -- see failed2s/hedging.py). If positions can't be fetched, the entry is
blocked rather than sent blind. It cannot see accounts behind other logins,
TradersPost, or manual trades on other platforms -- see TRADINGVIEW_WEBHOOK.md.
"""

import logging
import os
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException, Request

from failed2s.hedging import check_entry
from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from live.tradovate_client import TradovateClient

from .sizing import contracts_for_risk

logging.basicConfig(level=logging.INFO)
log = logging.getLogger("webhook")

WEBHOOK_SECRET = os.environ["WEBHOOK_SECRET"]
ACCOUNT_SPEC = os.environ.get("TRADOVATE_ACCOUNT_SPEC", "")
RISK_PER_TRADE_USD = float(os.environ.get("RISK_PER_TRADE_USD", "200"))
MAX_CONTRACTS = int(os.environ.get("MAX_CONTRACTS", "5"))
DAILY_LOSS_LIMIT_USD = float(os.environ.get("DAILY_LOSS_LIMIT_USD", "1000"))
MAX_DAILY_TRADES = int(os.environ.get("MAX_DAILY_TRADES", "10"))
_limit = os.environ.get("ACCOUNT_CONTRACT_LIMIT_MINIS")
ACCOUNT_CONTRACT_LIMIT_MINIS = float(_limit) if _limit else None
TRADOVATE_ENV = os.environ.get("TRADOVATE_ENV", "demo")
DRY_RUN = os.environ.get("DRY_RUN", "true").lower() != "false"

INSTRUMENT = INSTRUMENTS["MES"]

app = FastAPI()
risk_manager = RiskManager(daily_loss_limit=DAILY_LOSS_LIMIT_USD, max_daily_trades=MAX_DAILY_TRADES)

_client = None
_account_id = None


def _get_client():
    global _client, _account_id
    if _client is None:
        _client = TradovateClient(env=TRADOVATE_ENV)
        if not DRY_RUN:
            _client.authenticate()
            accounts = _client.list_accounts()
            if not accounts:
                raise RuntimeError("No Tradovate accounts returned for this login")
            _account_id = accounts[0]["id"]
    return _client


def _open_positions(client):
    """All open positions across every account on this login (none known in dry run)."""
    if DRY_RUN:
        return []
    return client.open_positions()


@app.post("/tradingview-webhook")
async def tradingview_webhook(request: Request):
    payload = await request.json()
    log.info("Webhook received: %s", payload)

    if payload.get("secret") != WEBHOOK_SECRET:
        raise HTTPException(status_code=401, detail="bad secret")

    action = payload.get("action")
    symbol = payload.get("symbol", "MES")
    client = _get_client()
    today = datetime.now(timezone.utc).date()

    if action == "flatten":
        if not DRY_RUN:
            client.liquidate_position(_account_id, symbol)
        log.info("Flatten%s: %s", " (dry run)" if DRY_RUN else "", symbol)
        return {"status": "flattened", "dry_run": DRY_RUN}

    if action not in ("buy", "sell"):
        raise HTTPException(status_code=400, detail=f"unknown action: {action}")

    try:
        entry_price = float(payload["entry_price"])
        stop_price = float(payload["stop_price"])
        target_price = float(payload["target_price"])
    except (KeyError, TypeError, ValueError) as e:
        raise HTTPException(status_code=400, detail=f"bad/missing price field: {e}")

    if not risk_manager.can_enter(today):
        log.warning("Blocked by risk manager (daily loss limit or trade cap already hit)")
        return {"status": "blocked_by_risk_manager"}

    if not DRY_RUN and client.get_position_size(_account_id) != 0:
        log.warning("Blocked: Tradovate account already has an open position")
        return {"status": "blocked_position_already_open"}

    try:
        contracts = contracts_for_risk(INSTRUMENT, entry_price, stop_price, RISK_PER_TRADE_USD, MAX_CONTRACTS)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    log.info(
        "Sizing: risk=$%.2f, stop distance=%.2f pts -> %s contract(s)",
        RISK_PER_TRADE_USD, abs(entry_price - stop_price), contracts,
    )

    try:
        positions = _open_positions(client)
    except Exception:
        log.exception("Blocked: couldn't fetch open positions for the hedging check")
        return {"status": "blocked_positions_unavailable"}

    signed_qty = contracts if action == "buy" else -contracts
    reason = check_entry(symbol, signed_qty, positions, _account_id, ACCOUNT_CONTRACT_LIMIT_MINIS)
    if reason:
        log.warning("Blocked by hedging guard: %s", reason)
        return {"status": "blocked_by_hedging_guard", "reason": reason}

    # Counts this entry against MAX_DAILY_TRADES immediately (in both dry-run
    # and live mode, so the cap is testable before going live). pnl=0.0 here
    # is a placeholder, not a real fill result -- see the module docstring's
    # "Known gap": DAILY_LOSS_LIMIT_USD only reflects orders placed through
    # this process, not actual fills, so it's not a real stop-loss backstop.
    risk_manager.record_trade(today, 0.0)

    if DRY_RUN:
        log.info(
            "DRY RUN: would place %s %s x%s stop=%.2f target=%.2f",
            action, symbol, contracts, stop_price, target_price,
        )
        return {"status": "dry_run", "contracts": contracts}

    order = client.place_bracket_order(
        account_id=_account_id,
        account_spec=ACCOUNT_SPEC,
        symbol=symbol,
        action="Buy" if action == "buy" else "Sell",
        qty=contracts,
        stop_price=stop_price,
        target_price=target_price,
    )
    return {"status": "order_placed", "contracts": contracts, "order": order}
