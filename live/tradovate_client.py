"""
Minimal Tradovate REST + WebSocket client.

IMPORTANT: this could not be exercised against a live Tradovate account from
this environment (no credentials, and outbound network access to
tradovateapi.com is not available here). The endpoint names, payload shapes,
and WebSocket frame protocol follow Tradovate's publicly documented API
(https://api.tradovate.com, https://partner.tradovate.com) as of this
writing, but have NOT been end-to-end tested by the assistant.

Before trading real money:
  1. Validate every call against the current Tradovate API docs.
  2. Run against the demo/paper environment first (TRADOVATE_ENV=demo,
     the default).
  3. Confirm order fills and bracket behavior manually in the Tradovate UI
     before relying on this unattended.

Required environment variables:
  TRADOVATE_USERNAME, TRADOVATE_PASSWORD, TRADOVATE_APP_ID, TRADOVATE_CID,
  TRADOVATE_SEC
Optional:
  TRADOVATE_APP_VERSION (default "1.0"), TRADOVATE_DEVICE_ID,
  TRADOVATE_ENV ("demo" default, or "live")
"""

import json
import os
import threading
import time as _time
from typing import Optional

import requests

DEMO_BASE = "https://demo.tradovateapi.com/v1"
LIVE_BASE = "https://live.tradovateapi.com/v1"
DEMO_MD_WS = "wss://md-demo.tradovateapi.com/v1/websocket"
LIVE_MD_WS = "wss://md.tradovateapi.com/v1/websocket"


class TradovateAuthError(RuntimeError):
    pass


class TradovateClient:
    def __init__(self, env: str = "demo"):
        if env not in ("demo", "live"):
            raise ValueError("env must be 'demo' or 'live'")
        self.env = env
        self.base_url = DEMO_BASE if env == "demo" else LIVE_BASE
        self.md_ws_url = DEMO_MD_WS if env == "demo" else LIVE_MD_WS
        self.access_token: Optional[str] = None
        self.md_access_token: Optional[str] = None
        self.expiration_time: Optional[str] = None
        self._session = requests.Session()

    @classmethod
    def from_env(cls, env: Optional[str] = None) -> "TradovateClient":
        return cls(env=env or os.environ.get("TRADOVATE_ENV", "demo"))

    # -- auth -----------------------------------------------------------
    def authenticate(self) -> None:
        payload = {
            "name": os.environ["TRADOVATE_USERNAME"],
            "password": os.environ["TRADOVATE_PASSWORD"],
            "appId": os.environ["TRADOVATE_APP_ID"],
            "appVersion": os.environ.get("TRADOVATE_APP_VERSION", "1.0"),
            "deviceId": os.environ.get("TRADOVATE_DEVICE_ID", "failed2s-bot"),
            "cid": os.environ["TRADOVATE_CID"],
            "sec": os.environ["TRADOVATE_SEC"],
        }
        resp = self._session.post(f"{self.base_url}/auth/accessTokenRequest", json=payload)
        resp.raise_for_status()
        data = resp.json()
        if "accessToken" not in data:
            raise TradovateAuthError(f"Tradovate auth failed: {data}")
        self.access_token = data["accessToken"]
        self.md_access_token = data.get("mdAccessToken")
        self.expiration_time = data.get("expirationTime")
        self._session.headers.update({"Authorization": f"Bearer {self.access_token}"})

    # -- REST helpers -----------------------------------------------------
    def _get(self, path: str, **kwargs):
        resp = self._session.get(f"{self.base_url}{path}", **kwargs)
        resp.raise_for_status()
        return resp.json()

    def _post(self, path: str, payload: dict):
        resp = self._session.post(f"{self.base_url}{path}", json=payload)
        resp.raise_for_status()
        return resp.json()

    def list_accounts(self):
        return self._get("/account/list")

    def find_contract(self, symbol: str):
        return self._get("/contract/find", params={"name": symbol})

    def place_bracket_order(
        self,
        account_id: int,
        account_spec: str,
        symbol: str,
        action: str,  # "Buy" or "Sell"
        qty: int,
        stop_price: float,
        target_price: float,
    ) -> dict:
        """Market entry with attached stop-loss + take-profit legs (OSO bracket)."""
        exit_action = "Sell" if action == "Buy" else "Buy"
        payload = {
            "accountSpec": account_spec,
            "accountId": account_id,
            "action": action,
            "symbol": symbol,
            "orderQty": qty,
            "orderType": "Market",
            "isAutomated": True,
            "bracket1": {"action": exit_action, "orderType": "Stop", "stopPrice": stop_price},
            "bracket2": {"action": exit_action, "orderType": "Limit", "price": target_price},
        }
        return self._post("/order/placeOSO", payload)

    def liquidate_position(self, account_id: int, symbol: str) -> dict:
        return self._post("/order/liquidateposition", {"accountId": account_id, "symbol": symbol})

    def get_position_size(self, account_id: int) -> int:
        """
        Net open position size across ALL contracts held in `account_id`
        (positive=long, negative=short, 0=flat).

        Caveat: Tradovate's /position/list entries key off contractId, not
        a human symbol, and this sums every position on the account without
        filtering by symbol. That's correct for an account that only ever
        trades one instrument (the intended use here); if the same account
        also trades other symbols, this will conflate them -- add proper
        contractId filtering via find_contract() before relying on this in
        that case.
        """
        positions = self._get("/position/list")
        return sum(p.get("netPos", 0) for p in positions if p.get("accountId") == account_id)


class TradovateChartFeed:
    """
    Best-effort realtime bar feed over Tradovate's market-data WebSocket
    (md/getChart). Requires `websocket-client` (pip install websocket-client).

    Frame protocol (per Tradovate docs): text frames of the form
    "{endpoint}\\n{request_id}\\n\\n{json_body}", server responses are
    JSON arrays prefixed with 'a', heartbeats are bare 'h'.

    UNVERIFIED end-to-end from this environment -- treat as a starting
    point and confirm against a demo account before depending on it live.
    """

    def __init__(self, client: TradovateClient, on_bar):
        self._client = client
        self._on_bar = on_bar
        self._ws = None
        self._req_id = 0
        self._stop = threading.Event()

    def _next_id(self) -> int:
        self._req_id += 1
        return self._req_id

    def subscribe_chart(self, symbol: str, element_size_minutes: int, history_bars: int = 50) -> None:
        import websocket  # websocket-client

        def on_open(ws):
            ws.send(f"authorize\n{self._next_id()}\n\n{self._client.access_token}")
            body = {
                "symbol": symbol,
                "chartDescription": {
                    "underlyingType": "MinuteBar",
                    "elementSize": element_size_minutes,
                    "elementSizeUnit": "UnderlyingUnits",
                    "withHistogram": False,
                },
                "timeRange": {"asMuchAsElements": history_bars},
            }
            ws.send(f"md/getChart\n{self._next_id()}\n\n{json.dumps(body)}")
            threading.Thread(target=self._heartbeat_loop, args=(ws,), daemon=True).start()

        def on_message(ws, message):
            if not message or message == "h":
                return
            if message[0] != "a":
                return
            try:
                frames = json.loads(message[1:])
            except json.JSONDecodeError:
                return
            for frame in frames:
                data = frame.get("d") if isinstance(frame, dict) else None
                if not data:
                    continue
                for chart in data.get("charts", []):
                    for bar in chart.get("bars", []):
                        self._on_bar(bar)

        self._ws = websocket.WebSocketApp(
            self._client.md_ws_url, on_open=on_open, on_message=on_message
        )
        self._ws.run_forever()

    def _heartbeat_loop(self, ws) -> None:
        while not self._stop.is_set():
            _time.sleep(2.5)
            try:
                ws.send("[]")
            except Exception:
                return

    def stop(self) -> None:
        self._stop.set()
        if self._ws is not None:
            self._ws.close()
