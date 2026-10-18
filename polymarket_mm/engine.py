"""
Per-market paper market maker: routes feed events into the book and the
paper broker, re-quotes, enforces risk limits, accrues estimated rewards and
writes everything to a JSONL event log (read back by polymarket_mm.report).
"""

import json
from collections import deque
from dataclasses import dataclass

from .book import OrderBook
from .paper import PaperBroker
from .quoting import QuoteParams, compute_quotes
from .rewards import our_share


@dataclass
class MarketInfo:
    condition_id: str
    question: str
    slug: str
    yes_token: str
    no_token: str
    tick: float
    min_size: float           # reward min order size (shares)
    max_spread: float         # reward max spread (cents)
    daily_rate: float         # reward pool, USDC/day
    end_date: str = ""


@dataclass
class RiskParams:
    pull_move: float = 0.02       # mid moves this much within pull_window -> pull quotes
    pull_window: float = 10.0     # seconds
    cooldown: float = 30.0        # seconds to stay out after a pull
    max_loss: float = 50.0        # per-market P&L (incl. est. rewards) floor -> stop quoting it
    min_requote_interval: float = 1.0


class EventLog:
    def __init__(self, path):
        self._f = open(path, "a", buffering=1)

    def write(self, **rec) -> None:
        self._f.write(json.dumps(rec) + "\n")

    def close(self) -> None:
        self._f.close()


class MarketMaker:
    def __init__(self, info: MarketInfo, quote_params: QuoteParams, risk: RiskParams,
                 log: EventLog, broker: PaperBroker | None = None):
        self.info = info
        self.qp = quote_params
        self.risk = risk
        self.log = log
        self.broker = broker or PaperBroker()
        self.book = OrderBook()
        self.rewards = 0.0
        self.halted = False
        self.cooldown_until = 0.0
        self._mids: deque[tuple[float, float]] = deque()
        self._last_mid = None
        self._last_requote = -1e9
        self._last_reward_ts = None
        self._reward_buf = (0.0, 0.0, 0.0)  # (amount, share*seconds, seconds) not yet logged
        self._last_reward_log = None
        self._seen_trades: set = set()

    # ---- feed events -------------------------------------------------
    def on_book(self, asset_id: str, bids, asks, now: float) -> None:
        if asset_id != self.info.yes_token:
            return  # the NO book is the mirror image
        self.book.load_snapshot(bids, asks)
        self._after_book_change(now)

    def on_price_changes(self, changes, now: float) -> None:
        touched = False
        for c in changes:
            if c.get("asset_id") == self.info.yes_token:
                self.book.set_level(c["side"], c["price"], c["size"])
                touched = True
        if touched:
            self._after_book_change(now)

    def on_trade(self, ev: dict, now: float) -> None:
        price, size, side = float(ev["price"]), float(ev["size"]), ev["side"]
        if ev["asset_id"] == self.info.no_token:
            # buying NO at q == selling YES at 1-q
            price, side = round(1 - price, 6), ("SELL" if side == "BUY" else "BUY")
        elif ev["asset_id"] != self.info.yes_token:
            return
        key = (ev.get("transaction_hash"), side, price, size)
        if key in self._seen_trades:  # same match reported on both tokens
            return
        self._seen_trades.add(key)
        if len(self._seen_trades) > 5000:
            self._seen_trades.clear()
        self.log.write(type="trade", ts=now, market=self.info.condition_id, side=side, price=price, size=size)
        self._record_fills(self.broker.on_trade(side, price, size, now), now)
        self._requote(now)

    def on_tick_size(self, new_tick, now: float) -> None:
        self.info.tick = float(new_tick)
        self._requote(now, force=True)

    # ---- periodic ----------------------------------------------------
    def on_timer(self, now: float) -> None:
        """Call every few seconds: accrues rewards for the time just passed
        and lets in-flight orders/cooldowns resolve into a requote."""
        if self._last_reward_ts is not None and not self.halted:
            dt = now - self._last_reward_ts
            live = self.broker.live(now)
            share = our_share(self.book,
                              [(o.price, o.remaining) for o in live if o.side == "BUY"],
                              [(o.price, o.remaining) for o in live if o.side == "SELL"],
                              self.info.max_spread, self.info.min_size)
            earned = self.info.daily_rate * share * dt / 86_400
            self.rewards += earned
            amt, share_s, secs = self._reward_buf
            self._reward_buf = (amt + earned, share_s + share * dt, secs + dt)
        self._last_reward_ts = now
        if self._last_reward_log is None:
            self._last_reward_log = now
        elif now - self._last_reward_log >= 60:
            self.flush(now)
        self._check_risk(now)
        self._requote(now)

    def flush(self, now: float) -> None:
        """Write accrued-but-unlogged rewards (once a minute, and at shutdown)."""
        amt, share_s, secs = self._reward_buf
        if secs > 0:
            self.log.write(type="reward", ts=now, market=self.info.condition_id,
                           amount=amt, share=share_s / secs, seconds=secs)
        self._reward_buf = (0.0, 0.0, 0.0)
        self._last_reward_log = now

    def pnl(self) -> float:
        return self.broker.pnl(self.book.mid)

    # ---- internals ---------------------------------------------------
    def _after_book_change(self, now: float) -> None:
        mid = self.book.mid
        if mid is not None and mid != self._last_mid:
            self._last_mid = mid
            self.log.write(type="mid", ts=now, market=self.info.condition_id, mid=mid,
                           bid=self.book.best_bid, ask=self.book.best_ask)
            self._mids.append((now, mid))
            while self._mids and self._mids[0][0] < now - self.risk.pull_window:
                self._mids.popleft()
            lo, hi = min(m for _, m in self._mids), max(m for _, m in self._mids)
            if hi - lo >= self.risk.pull_move - 1e-9 and now >= self.cooldown_until:
                self.cooldown_until = now + self.risk.cooldown
                self.broker.cancel_all(now)
                self.log.write(type="pull", ts=now, market=self.info.condition_id, move=round(hi - lo, 6))
        self._record_fills(self.broker.on_book(self.book.best_bid, self.book.best_ask, now), now)
        self._requote(now)

    def _record_fills(self, fills, now: float) -> None:
        for f in fills:
            self.log.write(type="fill", ts=f.ts, market=self.info.condition_id, order_id=f.order_id,
                           side=f.side, price=f.price, size=f.size, fee=f.fee, mid=self.book.mid,
                           position=self.broker.position)
        if fills:
            self._check_risk(now)

    def _check_risk(self, now: float) -> None:
        if not self.halted and self.pnl() + self.rewards < -self.risk.max_loss:
            self.halted = True
            self.broker.cancel_all(now)
            self.log.write(type="halt", ts=now, market=self.info.condition_id,
                           pnl=self.pnl(), rewards=self.rewards)

    def _requote(self, now: float, force: bool = False) -> None:
        working = self.broker.working(now)
        if self.halted or now < self.cooldown_until:
            for o in working:
                self.broker.cancel(o, now)
            return
        if not force and now - self._last_requote < self.risk.min_requote_interval:
            return
        q = compute_quotes(self.book.best_bid, self.book.best_ask, self.broker.position,
                           self.info.max_spread, self.info.min_size, self.info.tick, self.qp)
        changed = False
        for side, price, size in (("BUY", q.bid, q.bid_size), ("SELL", q.ask, q.ask_size)):
            mine = [o for o in working if o.side == side]
            keep = [o for o in mine if price is not None and abs(o.price - price) < 1e-9]
            for o in mine:
                if o not in keep[:1]:
                    self.broker.cancel(o, now)
                    changed = True
            if price is not None and not keep:
                self.broker.place(side, price, size, self.book.size_at(side, price), now)
                changed = True
        if changed:
            self._last_requote = now
            self.log.write(type="quote", ts=now, market=self.info.condition_id,
                           bid=q.bid, ask=q.ask, size=max(q.bid_size, q.ask_size),
                           position=self.broker.position)
