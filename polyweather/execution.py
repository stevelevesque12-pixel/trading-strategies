"""Brokers: paper (default) and live via Polymarket's py-clob-client.

Live trading needs, on top of `--live`:
  POLYWEATHER_LIVE=1          explicit second switch
  POLY_PRIVATE_KEY            key of the wallet that signs orders
  POLY_FUNDER                 address holding the funds (your Polymarket proxy wallet)
  POLY_SIGNATURE_TYPE         0 = plain EOA, 1 = email/Magic login, 2 = browser-wallet proxy
and `pip install py-clob-client`. Orders go out as FAK (fill-and-kill)
limit orders at the worst book level the planner priced, so the bot never
pays more than it computed edge on and never leaves resting orders behind.
"""

import math
import os
from dataclasses import dataclass
from typing import Optional

from .fees import taker_fee
from .markets import WeatherEvent
from .strategy import OrderIntent

CLOB_HOST = "https://clob.polymarket.com"
POLYGON = 137


@dataclass
class Fill:
    shares: float
    cost: float  # USDC paid incl. fee (BUY) or received net of fee (SELL)
    fee: float
    status: str
    exchange_id: Optional[str] = None


class PaperBroker:
    """Fills at the levels the planner walked (it already priced depth + fees)."""

    mode = "paper"

    def __init__(self, slippage: float = 0.0):
        self.slippage = slippage

    def execute(self, o: OrderIntent, ev: WeatherEvent) -> Fill:
        slip = self.slippage * o.shares
        if o.action == "BUY":
            return Fill(o.shares, o.est_cost + slip, o.est_fee, "filled")
        return Fill(o.shares, o.est_cost - slip, o.est_fee, "filled")


def _round_to_tick(price: float, tick: float, up: bool) -> float:
    n = price / tick
    n = math.ceil(n - 1e-9) if up else math.floor(n + 1e-9)
    return round(min(max(n * tick, tick), 1 - tick), 6)


class LiveBroker:
    mode = "live"

    def __init__(self):
        print("WARNING: no version of this strategy showed a reliable edge in an honest backtest "
              "(see polyweather/README.md). Live orders risk real money.")
        if os.environ.get("POLYWEATHER_LIVE") != "1":
            raise RuntimeError("refusing to trade live: set POLYWEATHER_LIVE=1 as well as --live")
        try:
            from py_clob_client.client import ClobClient
        except ImportError as exc:
            raise RuntimeError("pip install py-clob-client to trade live") from exc
        key = os.environ["POLY_PRIVATE_KEY"]
        funder = os.environ.get("POLY_FUNDER")
        sig = int(os.environ.get("POLY_SIGNATURE_TYPE", "0"))
        self.client = ClobClient(CLOB_HOST, chain_id=POLYGON, key=key, signature_type=sig, funder=funder)
        self.client.set_api_creds(self.client.create_or_derive_api_creds())

    def execute(self, o: OrderIntent, ev: WeatherEvent) -> Fill:
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
        from py_clob_client.order_builder.constants import BUY, SELL

        buy = o.action == "BUY"
        price = _round_to_tick(o.limit_price, ev.tick_size, up=buy)
        args = OrderArgs(token_id=o.token_id, price=price, size=round(o.shares, 2), side=BUY if buy else SELL)
        opts = PartialCreateOrderOptions(tick_size=_tick_str(ev.tick_size), neg_risk=ev.neg_risk)
        signed = self.client.create_order(args, opts)
        resp = self.client.post_order(signed, OrderType.FAK)
        oid = (resp or {}).get("orderID")
        if not resp or not resp.get("success", True) or not oid:
            return Fill(0.0, 0.0, 0.0, f"rejected: {resp}", oid)
        matched, avg = self._fill_info(oid, price)
        fee = taker_fee(matched, avg, ev.fee_rate, ev.fee_exponent)
        cost = matched * avg + fee if buy else matched * avg - fee
        return Fill(matched, cost, fee, "filled" if matched > 0 else "unfilled", oid)

    def _fill_info(self, oid: str, limit: float) -> tuple[float, float]:
        try:
            od = self.client.get_order(oid) or {}
            matched = float(od.get("size_matched") or 0.0)
            # The CLOB reports the order's limit price; actual fills can only be better.
            return matched, float(od.get("price") or limit)
        except Exception:
            return 0.0, limit


def _tick_str(tick: float) -> str:
    for s in ("0.1", "0.01", "0.001", "0.0001"):
        if abs(float(s) - tick) < 1e-12:
            return s
    return "0.001"


class LiveMakerBroker(LiveBroker):
    """Keeps resting post-only GTC bids in line with the desired quotes.

    Each sync: record fills on our open orders (size_matched growth), cancel
    orders whose price/size is no longer wanted, post the missing ones.
    """

    def __init__(self, ledger=None, on_fill=None):
        super().__init__()
        self.open: dict[tuple, dict] = {}  # (slug, bucket, side) -> {"id", "price", "shares", "matched"}
        self.on_fill = on_fill

    def sync(self, desired: dict, events: dict) -> None:
        from py_clob_client.clob_types import OrderArgs, OrderType, PartialCreateOrderOptions
        from py_clob_client.order_builder.constants import BUY

        want = {(slug, q.bucket, q.side): q for slug, qs in desired.items() for q in qs}
        for key, o in list(self.open.items()):
            matched, _ = self._fill_info(o["id"], o["price"])
            if matched > o["matched"] + 1e-9 and self.on_fill:
                self.on_fill(key, matched - o["matched"], o["price"])
            o["matched"] = matched
            q = want.get(key)
            stale = q is None or abs(q.price - o["price"]) > 1e-9 or matched >= o["shares"] - 1e-9
            if stale:
                try:
                    self.client.cancel(o["id"])
                except Exception:
                    pass
                del self.open[key]
        for key, q in want.items():
            if key in self.open:
                continue
            ev = events[key[0]]
            args = OrderArgs(token_id=q.token_id, price=q.price, size=round(q.shares, 2), side=BUY)
            opts = PartialCreateOrderOptions(tick_size=_tick_str(ev.tick_size), neg_risk=ev.neg_risk)
            try:
                resp = self.client.post_order(self.client.create_order(args, opts), OrderType.GTC, post_only=True)
            except Exception as exc:
                print(f"  post failed {key}: {exc}")
                continue
            if resp and resp.get("orderID"):
                self.open[key] = {"id": resp["orderID"], "price": q.price, "shares": q.shares, "matched": 0.0}

    def cancel_all(self) -> None:
        try:
            self.client.cancel_all()
        finally:
            self.open.clear()
