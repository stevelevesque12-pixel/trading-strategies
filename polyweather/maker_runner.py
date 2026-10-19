"""Live market-making loop (paper by default).

Paper mode is meant as the *measurement* of the strategy, not a toy:
* quotes are computed from the real order books every cycle;
* fills come only from taker trades that actually printed through our
  price since the last cycle (Polymarket data API) -- the same rule as the
  backtest;
* reward income is accrued each cycle from the real book: pool x our score
  / (our score + everyone else's score), per the liquidity-rewards formula.
Everything is written to the ledger (tables maker_fills, maker_rewards,
plus positions), so `python -m polyweather maker-report` can tell after a
few days whether rewards outrun adverse selection.
"""

import math
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Optional
from zoneinfo import ZoneInfo

from .ledger import Ledger
from .maker import (Inventory, MakerConfig, Quote, fills_from_prints, in_blackout, make_quotes, quoting_allowed,
                    usual_obs_minutes)
from .markets import WeatherEvent, fetch_event, fetch_open_events
from .observations import day_window_utc, fetch_awc_many, obs_from_awc, resolved_extreme
from .orderbook import fetch_books, fetch_trades
from .rewards import (RewardConfig, adjusted_mid, book_q, expected_daily_reward, fetch_reward_configs, our_q,
                      q_sides)

MAKER_SCHEMA = """
CREATE TABLE IF NOT EXISTS maker_fills (
    ts REAL, mode TEXT, slug TEXT, city TEXT, kind TEXT, target_date TEXT, bucket INTEGER, label TEXT,
    side TEXT, price REAL, shares REAL, mid REAL
);
CREATE TABLE IF NOT EXISTS maker_rewards (
    ts REAL, mode TEXT, slug TEXT, bucket INTEGER, pool REAL, our_q REAL, comp_q REAL, usd REAL
);
"""


@dataclass
class MakerRunConfig:
    maker: MakerConfig = field(default_factory=MakerConfig)
    cycle_s: int = 60
    refresh_events_s: int = 600
    cities: Optional[list[str]] = None


class MakerRunner:
    def __init__(self, cfg: MakerRunConfig, ledger: Ledger, live_broker=None):
        self.cfg = cfg
        self.ledger = ledger
        self.ledger.db.executescript(MAKER_SCHEMA)
        self.broker = live_broker  # None = paper
        self.mode = "live" if live_broker else "paper"
        if live_broker is not None:
            live_broker.on_fill = self._live_fill
        self.events: list[WeatherEvent] = []
        self.rewards: dict[str, RewardConfig] = {}
        self.resting: dict[str, list[Quote]] = {}  # slug -> quotes resting since last_ts
        self.last_ts: Optional[float] = None
        self._refreshed = 0.0
        self.inv: dict[str, Inventory] = {}
        for p in ledger.positions():
            self.inv.setdefault(p.slug, Inventory()).add(p.side, p.bucket, p.shares, p.cost)

    # ------------------------------------------------------------------ helpers
    def _refresh(self):
        evs = [e for e in fetch_open_events() if e.tradeable]
        if self.cfg.cities:
            evs = [e for e in evs if e.city in self.cfg.cities]
        self.rewards = fetch_reward_configs()
        self.events = [e for e in evs if any(b.condition_id in self.rewards for b in e.buckets)]
        self._refreshed = time.time()

    def _fair_and_guards(self, ev: WeatherEvent, books, awc, now: datetime):
        """Fair YES per bucket (market mid) with hard observation guards; None = don't quote."""
        st = ev.station
        start, end = day_window_utc(ev.target_date, st.tz)
        if now >= end:
            return None  # local day over, awaiting resolution: nothing left to make a market in
        if not quoting_allowed(ev, now, self.cfg.maker):
            return None
        rows = awc.get(st.icao, [])
        obs = obs_from_awc(rows, ev.unit, include_specials=not ev.hourly_only).window(start, end)
        ext = resolved_extreme(obs, ev.kind)
        fair, mids, asks = [], [], []
        for b in ev.buckets:
            yb, nb = books.get(b.yes_token), books.get(b.no_token)
            rc = self.rewards.get(b.condition_id)
            mid = adjusted_mid(yb, rc.min_size if rc else 20.0) if yb else None
            dead = ext is not None and ((ev.kind == "high" and b.hi is not None and b.hi < ext) or
                                        (ev.kind == "low" and b.lo is not None and b.lo > ext))
            fair.append(None if dead else mid)
            mids.append(mid)
            asks.append((yb.best_ask if yb else None, nb.best_ask if nb else None))
        minutes = usual_obs_minutes([t for t, _, _ in rows], min_count=8)
        blackout = (now >= start - timedelta(hours=1)) and in_blackout(now, st.tz, minutes, self.cfg.maker)
        return fair, mids, asks, blackout

    # ------------------------------------------------------------------ cycle
    def cycle(self) -> None:
        now = datetime.now(timezone.utc)
        if time.time() - self._refreshed > self.cfg.refresh_events_s or not self.events:
            self._refresh()
        by_slug = {e.slug: e for e in self.events}
        toks = [t for e in self.events for b in e.buckets if b.condition_id in self.rewards
                for t in (b.yes_token, b.no_token)]
        books = fetch_books(toks)
        awc = fetch_awc_many(sorted({e.station.icao for e in self.events}))
        t_now = now.timestamp()

        # 1) fills and reward accrual for the quotes that rested since the last cycle
        if self.last_ts is not None and self.broker is None:
            dt = t_now - self.last_ts
            live = [(slug, qs) for slug, qs in self.resting.items() if qs and slug in by_slug]

            def recent(slug):  # newest page is enough for a one-cycle window
                try:
                    return slug, fetch_trades([b.condition_id for b in by_slug[slug].buckets], max_pages=1)
                except Exception:
                    return slug, []
            with ThreadPoolExecutor(8) as ex:
                prints = dict(ex.map(recent, [slug for slug, _ in live]))
            for slug, quotes in live:
                ev = by_slug[slug]
                trades = prints.get(slug, [])
                inv = self.inv.setdefault(slug, Inventory())
                for q in quotes:
                    got = fills_from_prints(q, ev, trades, self.last_ts, t_now)
                    if got > 0:
                        self._record_fill(ev, q, got, books)
                        inv.add(q.side, q.bucket, got, got * q.price)
                self._accrue(ev, quotes, books, dt)

        # live: our orders are in the book now, so estimate rewards with them taken out of it
        if self.last_ts is not None and self.broker is not None:
            dt = t_now - self.last_ts
            for slug, quotes in self.resting.items():
                if quotes and slug in by_slug:
                    self._accrue(by_slug[slug], quotes, books, dt, own_in_book=True)

        # 2) new quotes
        self.resting = {}
        n_q = 0
        for ev in self.events:
            g = self._fair_and_guards(ev, books, awc, now)
            if g is None:
                continue
            fair, mids, asks, blackout = g
            if blackout:
                continue
            inv = self.inv.setdefault(ev.slug, Inventory())
            total = sum(i.total_cost() for i in self.inv.values())
            quotes = make_quotes(ev, fair, mids, self.rewards, inv, self.cfg.maker, best_asks=asks,
                                 total_inventory=total)
            self.resting[ev.slug] = quotes
            n_q += len(quotes)
        if self.broker is not None:
            self.broker.sync(self.resting, by_slug)
        self.last_ts = t_now
        self.ledger.commit()
        held = sum(i.total_cost() for i in self.inv.values())
        rew = self.ledger.db.execute("SELECT COALESCE(SUM(usd),0) FROM maker_rewards WHERE mode=?",
                                     (self.mode,)).fetchone()[0]
        nf = self.ledger.db.execute("SELECT COUNT(*) FROM maker_fills WHERE mode=?", (self.mode,)).fetchone()[0]
        print(f"[{now:%H:%M:%S}Z] {len(self.events)} events, {n_q} quotes, inventory ${held:,.0f}, "
              f"fills {nf}, est rewards ${rew:,.2f}", flush=True)

    def _live_fill(self, key, shares: float, price: float) -> None:
        slug, bucket, side = key
        ev = next((e for e in self.events if e.slug == slug), None)
        if ev is None:
            return
        b = ev.buckets[bucket]
        q = Quote(bucket, side, b.yes_token if side == "YES" else b.no_token, price, shares, price)
        self._record_fill(ev, q, shares, {})
        self.inv.setdefault(slug, Inventory()).add(side, bucket, shares, shares * price)

    def _record_fill(self, ev, q: Quote, shares: float, books) -> None:
        b = ev.buckets[q.bucket]
        yb = books.get(b.yes_token)
        mid = adjusted_mid(yb, 20.0) if yb else None
        self.ledger.db.execute("INSERT INTO maker_fills VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                               (time.time(), self.mode, ev.slug, ev.city, ev.kind, str(ev.target_date),
                                q.bucket, b.label, q.side, q.price, shares, mid))
        self.ledger.add_fill(ev.slug, q.bucket, q.side, q.token_id, ev.city, ev.kind, ev.target_date,
                             b.label, shares, shares * q.price)

    def _accrue(self, ev, quotes: list[Quote], books, dt: float, own_in_book: bool = False) -> None:
        by_bucket: dict[int, dict] = {}
        for q in quotes:
            by_bucket.setdefault(q.bucket, {})[q.side] = q
        for i, sides in by_bucket.items():
            b = ev.buckets[i]
            rc = self.rewards.get(b.condition_id)
            yb, nb = books.get(b.yes_token), books.get(b.no_token)
            if rc is None or yb is None:
                continue
            mid = adjusted_mid(yb, rc.min_size)
            if mid is None:
                continue
            y, n = sides.get("YES"), sides.get("NO")
            ours = our_q((y.price, y.shares) if y else None, (n.price, n.shares) if n else None, mid, rc)
            comp = book_q(yb, nb, mid, rc)
            if own_in_book:
                q1, q2 = q_sides([(y.price, y.shares)] if y else [], [(1 - n.price, n.shares)] if n else [], mid, rc)
                comp = max(0.0, comp - q1 - q2)
            usd = expected_daily_reward(ours, comp, rc) * dt / 86400.0
            self.ledger.db.execute("INSERT INTO maker_rewards VALUES (?,?,?,?,?,?,?,?)",
                                   (time.time(), self.mode, ev.slug, i, rc.daily_rate, ours, comp, usd))

    def settle(self) -> float:
        pnl = 0.0
        for slug in {p.slug for p in self.ledger.positions()}:
            ev = fetch_event(slug)
            if ev is None or not ev.closed or ev.winner() is None:
                continue
            p = self.ledger.settle(slug, ev.winner())
            self.inv.pop(slug, None)
            print(f"  settled {slug}: winner {ev.buckets[ev.winner()].label}  pnl {p:+.2f}", flush=True)
            pnl += p
        return pnl

    def loop(self) -> None:
        last_settle = 0.0
        while True:
            t0 = time.time()
            if t0 - last_settle > 1800:
                try:
                    self.settle()
                except Exception as exc:
                    print(f"  settle failed: {exc}", flush=True)
                last_settle = t0
            try:
                self.cycle()
            except Exception as exc:
                print(f"  cycle failed: {exc}", flush=True)
            time.sleep(max(5.0, self.cfg.cycle_s - (time.time() - t0)))


def maker_report(ledger: Ledger, mode: str = "paper") -> None:
    db = ledger.db
    db.executescript(MAKER_SCHEMA)
    rew = db.execute("SELECT COALESCE(SUM(usd),0), MIN(ts), MAX(ts) FROM maker_rewards WHERE mode=?", (mode,)).fetchone()
    hours = ((rew[2] or 0) - (rew[1] or 0)) / 3600.0
    fills = db.execute("SELECT COUNT(*), COALESCE(SUM(price*shares),0) FROM maker_fills WHERE mode=?", (mode,)).fetchone()
    settled = db.execute("SELECT COALESCE(SUM(payout-cost),0), COUNT(*) FROM positions WHERE settled=1").fetchone()
    open_cost = db.execute("SELECT COALESCE(SUM(cost),0) FROM positions WHERE settled=0").fetchone()[0]
    print(f"{mode}: {hours:.1f}h of quoting")
    print(f"  est rewards ${rew[0]:,.2f}" + (f"  (${rew[0] / hours * 24:,.0f}/day)" if hours > 0.5 else ""))
    print(f"  fills {fills[0]}  bought ${fills[1]:,.0f}  open inventory ${open_cost:,.0f}")
    print(f"  settled trading pnl ${settled[0]:+,.2f} over {settled[1]} positions")
