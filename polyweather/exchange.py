"""Exchange access for live market making, on Polymarket's unified SDK (`polymarket-client`).

The legacy `py-clob-client` path in execution.py predates pUSD collateral and
Deposit Wallets. This broker uses `polymarket.SecureClient`, which resolves
the account's wallet type itself (legacy email/Magic proxy wallet, Safe, or
Deposit Wallet) and relays wallet transactions (merge, redeem) gaslessly
through Polymarket's relayer.

Environment:
    POLY_PRIVATE_KEY               signer key (email/Magic login: export it from polymarket.com)
    POLY_WALLET                    the account wallet address shown in your Polymarket profile menu
    POLY_RELAYER_API_KEY           polymarket.com -> Settings -> API Keys -> Relayer API Keys
    POLY_RELAYER_API_KEY_ADDRESS   the address that key was created for
"""

import os
import threading
import time
from decimal import Decimal
from typing import Iterable, Optional

ORDER_TTL_S = 900  # GTD expiry used only if the exchange heartbeat can't be started
REFRESH_BEFORE_S = 300


def _items(paginator, max_pages: int = 50) -> Iterable:
    for i, page in enumerate(paginator):
        yield from page.items
        if i + 1 >= max_pages:
            break


def _fmt(x: float, places: int) -> str:
    return f"{Decimal(str(x)).quantize(Decimal(1).scaleb(-places))}"


class ExchangeBroker:
    mode = "live"

    def __init__(self, client=None):
        if client is None:
            from polymarket import RelayerApiKey, SecureClient
            client = SecureClient.create(
                private_key=os.environ["POLY_PRIVATE_KEY"],
                wallet=os.environ["POLY_WALLET"],
                api_key=RelayerApiKey(key=os.environ["POLY_RELAYER_API_KEY"],
                                      address=os.environ["POLY_RELAYER_API_KEY_ADDRESS"]),
            )
        self.client = client
        self.wallet = str(client.wallet).lower()
        self.wallet_type = str(getattr(client, "wallet_type", "?"))
        self.open: dict[tuple, dict] = {}  # (slug, bucket, side) -> {"id", "price", "shares", "expires"}
        self.token_map: dict[str, tuple[str, int, str]] = {}  # token -> (slug, bucket, side)
        self.on_fill = None
        self.rejections: dict[str, int] = {}
        self._hb_id: Optional[str] = ""
        self._hb_fail = 0
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self.heartbeat_supported = False

    # heartbeat (dead-man switch) ---------------------------------------------
    def _post_heartbeat(self) -> None:
        # Same authenticated transport the SDK uses for its own L2 POSTs.
        resp = self.client._ctx.secure_clob.post_json("/v1/heartbeats", json={"heartbeat_id": self._hb_id}) or {}
        self._hb_id = resp.get("heartbeat_id", self._hb_id)

    def start_heartbeat(self, every_s: float = 5.0) -> bool:
        try:
            self._post_heartbeat()
        except Exception:
            # Fall back to short-lived GTD orders: a dead process's quotes expire on their own.
            self.heartbeat_supported = False
            return False
        self.heartbeat_supported = True

        def run():
            while not self._hb_stop.wait(every_s):
                try:
                    self._post_heartbeat()
                    self._hb_fail = 0
                except Exception:
                    self._hb_fail += 1
        self._hb_thread = threading.Thread(target=run, daemon=True, name="clob-heartbeat")
        self._hb_thread.start()
        return True

    def heartbeat_ok(self) -> bool:
        if not self.heartbeat_supported:
            return True  # GTD-expiry mode: nothing to keep alive
        return self._hb_thread is not None and self._hb_thread.is_alive() and self._hb_fail < 2

    def stop_heartbeat(self) -> None:
        self._hb_stop.set()

    # account -----------------------------------------------------------------
    def collateral(self) -> tuple[float, float]:
        """(pUSD balance, max exchange allowance), in dollars."""
        ba = self.client.get_balance_allowance(asset_type="COLLATERAL")
        allowances = list((ba.allowances or {}).values())
        return ba.balance / 1e6, (max(allowances) / 1e6) if allowances else 0.0

    def approvals_state(self):
        return self.client.get_trading_approvals_state()

    # orders ------------------------------------------------------------------
    def sync(self, desired: dict, events: dict) -> None:
        """Make resting orders match `desired` ({slug: [Quote]}): cancel stale, post missing."""
        now = time.time()
        want = {}
        for slug, qs in desired.items():
            for q in qs:
                want[(slug, q.bucket, q.side)] = q
                self.token_map[q.token_id] = (slug, q.bucket, q.side)
        stale = []
        for key, o in list(self.open.items()):
            q = want.get(key)
            expiring = o["expires"] is not None and o["expires"] - now < REFRESH_BEFORE_S
            if q is None or abs(q.price - o["price"]) > 1e-9 or abs(q.shares - o["shares"]) > 1e-6 or expiring:
                stale.append(o["id"])
                del self.open[key]
        if stale:
            self.client.cancel_orders(order_ids=stale)
        for key, q in want.items():
            if key in self.open:
                continue
            ev = events[key[0]]
            expires = None if self.heartbeat_supported else int(now) + ORDER_TTL_S
            places = max(0, len(f"{ev.tick_size:.6f}".rstrip("0").split(".")[1]))
            resp = self.client.place_limit_order(
                token_id=q.token_id, price=_fmt(q.price, places), size=_fmt(q.shares, 2),
                side="BUY", post_only=True, expiration=expires)
            if getattr(resp, "ok", False):
                self.open[key] = {"id": resp.order_id, "price": q.price, "shares": q.shares, "expires": expires}
            else:
                code = str(getattr(resp, "code", "unknown"))
                self.rejections[code] = self.rejections.get(code, 0) + 1

    def cancel_all(self) -> None:
        try:
            self.client.cancel_all()
        finally:
            self.open.clear()

    def reconcile(self) -> tuple[int, int]:
        """Cancel orders on the exchange we don't know about; forget ours that are gone."""
        live = {o.id for o in _items(self.client.list_open_orders())}
        known = {o["id"] for o in self.open.values()}
        orphans = [oid for oid in live if oid not in known]
        if orphans:
            self.client.cancel_orders(order_ids=orphans)
        gone = [k for k, o in self.open.items() if o["id"] not in live]
        for k in gone:
            del self.open[k]
        return len(orphans), len(gone)

    def fills_since(self, after_ts: int) -> list[dict]:
        """Our maker fills from the CLOB trade history."""
        out = []
        for t in _items(self.client.list_account_trades(maker_address=self.wallet, after=str(int(after_ts)))):
            for mo in t.maker_orders:
                if str(mo.maker_address).lower() != self.wallet:
                    continue
                out.append({"id": f"{t.id}:{mo.order_id}", "ts": t.matched_at.timestamp(), "token": str(mo.asset_id),
                            "price": float(mo.price), "shares": float(mo.matched_amount)})
        return out

    def scoring_fraction(self) -> Optional[float]:
        ids = [o["id"] for o in self.open.values()]
        if not ids:
            return None
        r = self.client.get_orders_scoring(order_ids=ids) or {}
        return sum(1 for v in r.values() if v) / len(ids)

    # rewards -----------------------------------------------------------------
    def earnings(self, day) -> tuple[float, list]:
        rows = list(self.client.get_total_earnings_for_user_for_day(date=day.isoformat()))
        total = sum(float(r.earnings) * float(r.asset_rate or 1) for r in rows)
        return total, [{"asset": r.asset_address, "earnings": float(r.earnings), "rate": float(r.asset_rate)}
                       for r in rows]

    def reward_percentages(self) -> dict:
        r = self.client.get_reward_percentages()
        return dict(getattr(r, "root", r) or {})

    # positions (treasury) ----------------------------------------------------
    def positions(self) -> list:
        return list(_items(self.client.list_positions(user=self.wallet)))

    def merge(self, condition_id: str):
        return self.client.merge_positions(condition_id=condition_id, amount="max",
                                           metadata="polyweather: merge YES+NO pairs")

    def redeem(self, condition_id: str):
        return self.client.redeem_positions(condition_id=condition_id, metadata="polyweather: redeem resolved")
