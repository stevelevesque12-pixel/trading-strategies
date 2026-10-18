"""
Polymarket public market websocket, with keepalive and auto-reconnect.

On every (re)connect the venue sends fresh `book` snapshots for each
subscribed token, so a reconnect also resyncs the books.
"""

import json
import threading
import time

import websocket

WS_URL = "wss://ws-subscriptions-clob.polymarket.com/ws/market"


class MarketFeed:
    def __init__(self, token_ids: list[str], on_event, url: str = WS_URL):
        """on_event(event_dict) is called from the feed thread for every event."""
        self.token_ids = token_ids
        self.on_event = on_event
        self.url = url
        self._stop = threading.Event()
        self._ws = None
        self._thread = threading.Thread(target=self._run, daemon=True)
        self.reconnects = 0

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._ws:
            self._ws.close()
        self._thread.join(timeout=5)

    def _run(self) -> None:
        backoff = 1.0
        while not self._stop.is_set():
            try:
                self._ws = websocket.create_connection(self.url, timeout=10)
                self._ws.send(json.dumps({"assets_ids": self.token_ids, "type": "market"}))
                backoff, last_ping = 1.0, time.time()
                while not self._stop.is_set():
                    if time.time() - last_ping > 10:
                        self._ws.send("PING")
                        last_ping = time.time()
                    try:
                        msg = self._ws.recv()
                    except websocket.WebSocketTimeoutException:
                        continue
                    if not msg or msg[0] not in "[{":
                        continue  # PONG
                    data = json.loads(msg)
                    for ev in data if isinstance(data, list) else [data]:
                        self.on_event(ev)
            except Exception as e:  # network blips: reconnect and resync
                if self._stop.is_set():
                    break
                self.reconnects += 1
                print(f"[feed] disconnected ({e!r}); reconnecting in {backoff:.0f}s")
                time.sleep(backoff)
                backoff = min(backoff * 2, 30)
