"""Small real-money pilot of the market maker, with the safety rails it needs.

    python -m polyweather pilot --check     # preflight only: eligibility, keys, balance, allowance
    python -m polyweather pilot             # run (needs POLYWEATHER_LIVE=1)
    python -m polyweather pilot-report      # estimated vs ACTUAL rewards, fills, settled PnL by day
    touch STOP                              # kill switch: cancels every order and exits

Any automatic halt (loss limit, heartbeat, geoblock) writes the kill file too,
so a supervisor restarting the process can't resume trading on its own.

Rails, in the order they act:
* Eligibility: refuses to start (and halts) if polymarket.com/api/geoblock says
  this connection is blocked. There is deliberately no override.
* Exchange heartbeat: once started, the CLOB cancels all our orders if a
  heartbeat is missed for ~10s. A crashed process or dead server can't leave
  quotes resting.
* Hard capital cap across all events; above it only risk-reducing quotes.
* Daily loss limit on settled PnL plus open inventory marked to mid; on
  breach: cancel everything and exit.
* Reconciliation every cycle: orders on the exchange we don't know about are
  cancelled; orders we think are live but aren't are forgotten. Fills come
  from our own trade history (not from guessing), de-duplicated by trade id.
* Reward check: which of our orders the exchange says are scoring, our live
  reward percentages, and each day's actual earnings, next to the estimate.
"""

import json
import os
import threading
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from . import http
from .execution import LiveMakerBroker
from .ledger import Ledger
from .maker import Inventory, MakerConfig, Quote
from .maker_runner import MakerRunConfig, MakerRunner
from .orderbook import fetch_books
from .rewards import adjusted_mid

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
PILOT_SCHEMA = """
CREATE TABLE IF NOT EXISTS pilot_fills (
    fill_id TEXT PRIMARY KEY, ts REAL, slug TEXT, bucket INTEGER, side TEXT, token_id TEXT,
    price REAL, shares REAL
);
CREATE TABLE IF NOT EXISTS pilot_rewards_actual (
    day TEXT PRIMARY KEY, earnings REAL, fetched_ts REAL, detail TEXT
);
CREATE TABLE IF NOT EXISTS pilot_events (ts REAL, kind TEXT, detail TEXT);
"""


@dataclass
class PilotConfig:
    capital_usd: float = 400.0  # hard cap on open inventory cost, all events together
    max_event_usd: float = 60.0
    max_side_usd: float = 15.0  # per bucket side
    quote_shares: float = 20.0  # the usual reward minimum; never quote below a market's min size
    edge: float = 0.01
    daily_loss_limit_usd: float = 40.0
    cities: Optional[list[str]] = None
    cycle_s: int = 30
    kill_file: str = "STOP"
    db: str = "pilot.db"

    @classmethod
    def from_env(cls, **overrides) -> "PilotConfig":
        env = os.environ
        cfg = cls(
            capital_usd=float(env.get("PILOT_CAPITAL_USD", cls.capital_usd)),
            max_event_usd=float(env.get("PILOT_MAX_EVENT_USD", cls.max_event_usd)),
            max_side_usd=float(env.get("PILOT_MAX_SIDE_USD", cls.max_side_usd)),
            quote_shares=float(env.get("PILOT_QUOTE_SHARES", cls.quote_shares)),
            edge=float(env.get("PILOT_EDGE", cls.edge)),
            daily_loss_limit_usd=float(env.get("PILOT_DAILY_LOSS_LIMIT_USD", cls.daily_loss_limit_usd)),
            cities=[c for c in env.get("PILOT_CITIES", "").split(",") if c] or None,
            cycle_s=int(env.get("PILOT_CYCLE_S", cls.cycle_s)),
            kill_file=env.get("PILOT_KILL_FILE", cls.kill_file),
            db=env.get("PILOT_DB", cls.db),
        )
        for k, v in overrides.items():
            if v is not None:
                setattr(cfg, k, v)
        return cfg

    def maker_config(self) -> MakerConfig:
        return MakerConfig(quote_shares=self.quote_shares, min_edge=self.edge,
                           max_side_inventory_usd=self.max_side_usd, max_event_inventory_usd=self.max_event_usd,
                           max_total_inventory_usd=self.capital_usd)


class Halt(RuntimeError):
    pass


def geoblock_status() -> dict:
    return http.get(GEOBLOCK_URL, retries=2, timeout=15)


# ---------------------------------------------------------------- exchange plumbing


class PilotBroker(LiveMakerBroker):
    """LiveMakerBroker plus heartbeat, reconciliation, trade-history fills and reward queries."""

    def __init__(self):
        super().__init__()
        self.funder = (os.environ.get("POLY_FUNDER") or self.client.get_address() or "").lower()
        self.sig_type = int(os.environ.get("POLY_SIGNATURE_TYPE", "0"))
        self._hb_id: Optional[str] = ""
        self._hb_fail = 0
        self._hb_stop = threading.Event()
        self._hb_thread: Optional[threading.Thread] = None
        self.token_map: dict[str, tuple[str, int, str]] = {}  # token -> (slug, bucket, side)

    # heartbeat -------------------------------------------------------------
    def start_heartbeat(self, every_s: float = 5.0) -> None:
        def run():
            while not self._hb_stop.wait(every_s):
                try:
                    resp = self.client.post_heartbeat(self._hb_id) or {}
                    self._hb_id = resp.get("heartbeat_id", self._hb_id)
                    self._hb_fail = 0
                except Exception:
                    self._hb_fail += 1
        self._hb_thread = threading.Thread(target=run, daemon=True, name="clob-heartbeat")
        self._hb_thread.start()

    def heartbeat_ok(self) -> bool:
        return self._hb_thread is not None and self._hb_thread.is_alive() and self._hb_fail < 2

    def stop_heartbeat(self) -> None:
        self._hb_stop.set()

    # account ---------------------------------------------------------------
    def collateral(self) -> tuple[float, float]:
        """(balance, allowance) in USDC."""
        from py_clob_client.clob_types import AssetType, BalanceAllowanceParams
        r = self.client.get_balance_allowance(BalanceAllowanceParams(asset_type=AssetType.COLLATERAL,
                                                                     signature_type=self.sig_type)) or {}
        bal = float(r.get("balance") or 0) / 1e6
        allowances = r.get("allowances") or {}
        allow = max([float(v) for v in allowances.values()] or [float(r.get("allowance") or 0)]) / 1e6
        return bal, allow

    # orders ----------------------------------------------------------------
    def sync(self, desired: dict, events: dict) -> None:
        for slug, qs in desired.items():
            for q in qs:
                self.token_map[q.token_id] = (slug, q.bucket, q.side)
        on_fill, self.on_fill = self.on_fill, None  # fills come from trade history, not size_matched
        try:
            super().sync(desired, events)
        finally:
            self.on_fill = on_fill

    def reconcile(self) -> tuple[int, int]:
        """Cancel orphan exchange orders; forget local orders the exchange no longer has."""
        live = {o["id"]: o for o in (self.client.get_orders() or [])}
        known = {o["id"] for o in self.open.values()}
        orphans = [oid for oid in live if oid not in known]
        if orphans:
            self.client.cancel_orders(orphans)
        gone = [k for k, o in self.open.items() if o["id"] not in live]
        for k in gone:
            del self.open[k]
        return len(orphans), len(gone)

    def fills_since(self, after_ts: int) -> list[dict]:
        """Our maker fills from the CLOB trade history: [{"id","ts","token","price","shares"}]."""
        from py_clob_client.clob_types import TradeParams
        out = []
        for t in self.client.get_trades(TradeParams(maker_address=self.funder, after=int(after_ts))) or []:
            for mo in t.get("maker_orders") or []:
                if str(mo.get("maker_address", "")).lower() != self.funder:
                    continue
                out.append({"id": f"{t.get('id')}:{mo.get('order_id')}", "ts": float(t.get("match_time") or 0),
                            "token": str(mo.get("asset_id")), "price": float(mo.get("price")),
                            "shares": float(mo.get("matched_amount"))})
        return out

    def scoring_fraction(self) -> Optional[float]:
        from py_clob_client.clob_types import OrdersScoringParams
        ids = [o["id"] for o in self.open.values()]
        if not ids:
            return None
        r = self.client.are_orders_scoring(OrdersScoringParams(orderIds=ids)) or {}
        return sum(1 for v in r.values() if v) / len(ids)

    # rewards (L2-authenticated GETs not wrapped by py-clob-client) -----------
    def _l2_get(self, path: str, params: dict):
        from py_clob_client.clob_types import RequestArgs
        from py_clob_client.headers.headers import create_level_2_headers
        headers = create_level_2_headers(self.client.signer, self.client.creds, RequestArgs(method="GET", request_path=path))
        r = http.session().get(self.client.host + path, params=params, headers=headers, timeout=30)
        r.raise_for_status()
        return r.json()

    def earnings(self, day: date) -> tuple[float, list]:
        rows, cursor = [], ""
        while True:
            params = {"date": day.isoformat(), "signature_type": self.sig_type, "maker_address": self.funder}
            if cursor:
                params["next_cursor"] = cursor
            d = self._l2_get("/rewards/user", params)
            rows.extend(d.get("data", []))
            cursor = d.get("next_cursor", "LTE=")
            if not cursor or cursor == "LTE=":
                break
        return sum(float(r.get("earnings") or 0) * float(r.get("asset_rate") or 1) for r in rows), rows

    def reward_percentages(self) -> dict:
        return self._l2_get("/rewards/user/percentages", {"signature_type": self.sig_type,
                                                          "maker_address": self.funder}) or {}


# ---------------------------------------------------------------- runner


class PilotRunner(MakerRunner):
    def __init__(self, cfg: PilotConfig, ledger: Ledger, broker: PilotBroker):
        super().__init__(MakerRunConfig(maker=cfg.maker_config(), cycle_s=cfg.cycle_s, cities=cfg.cities),
                         ledger, broker)
        self.pcfg = cfg
        self.pbroker = broker
        ledger.db.executescript(PILOT_SCHEMA)
        self.last_fill_ts = int(time.time()) - 3600
        self.n = 0

    def log(self, kind: str, detail) -> None:
        self.ledger.db.execute("INSERT INTO pilot_events VALUES (?,?,?)", (time.time(), kind, json.dumps(detail)))
        self.ledger.commit()
        print(f"  [{kind}] {detail}", flush=True)

    # guards ----------------------------------------------------------------
    def check_guards(self) -> None:
        if Path(self.pcfg.kill_file).exists():
            raise Halt(f"kill file {self.pcfg.kill_file} present")
        if not self.pbroker.heartbeat_ok():
            raise Halt("exchange heartbeat failing (orders will be auto-cancelled)")
        if self.n % 20 == 0:
            g = geoblock_status()
            if g.get("blocked"):
                raise Halt(f"geoblocked: {g.get('country')}/{g.get('region')}")
        pnl = self.day_pnl()
        if pnl <= -abs(self.pcfg.daily_loss_limit_usd):
            raise Halt(f"daily loss limit: {pnl:+.2f} <= -{self.pcfg.daily_loss_limit_usd}")

    def day_pnl(self) -> float:
        """Settled PnL today (UTC) + open inventory marked to mid."""
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        db = self.ledger.db
        # settled positions don't carry a timestamp; approximate "today" by resolutions recorded today
        settled = db.execute("""SELECT COALESCE(SUM(p.payout - p.cost), 0) FROM positions p
                                JOIN pilot_events e ON e.kind='settled' AND json_extract(e.detail, '$.slug') = p.slug
                                WHERE p.settled=1 AND e.ts >= ?""", (start,)).fetchone()[0]
        open_pos = self.ledger.positions()
        if not open_pos:
            return float(settled)
        books = fetch_books([p.token_id for p in open_pos])
        mark = 0.0
        for p in open_pos:
            ob = books.get(p.token_id)
            mid = adjusted_mid(ob, 1.0) if ob else None
            mark += (p.shares * mid - p.cost) if mid is not None else -p.cost
        return float(settled) + mark

    # cycle -----------------------------------------------------------------
    def cycle(self) -> None:
        self.n += 1
        self.check_guards()
        self.record_fills()
        orphans, gone = self.pbroker.reconcile()
        if orphans:
            self.log("orphans_cancelled", {"n": orphans})
        super().cycle()
        if self.n % 10 == 0:
            frac = self.pbroker.scoring_fraction()
            if frac is not None:
                print(f"  scoring: {100 * frac:.0f}% of {len(self.pbroker.open)} open orders", flush=True)

    def record_fills(self) -> None:
        for f in self.pbroker.fills_since(self.last_fill_ts - 120):
            ref = self.pbroker.token_map.get(f["token"])
            if ref is None:
                continue
            cur = self.ledger.db.execute("INSERT OR IGNORE INTO pilot_fills VALUES (?,?,?,?,?,?,?,?)",
                                         (f["id"], f["ts"], ref[0], ref[1], ref[2], f["token"], f["price"], f["shares"]))
            if cur.rowcount == 0:
                continue  # already recorded
            slug, bucket, side = ref
            ev = next((e for e in self.events if e.slug == slug), None)
            if ev is not None:
                q = Quote(bucket, side, f["token"], f["price"], f["shares"], f["price"])
                self._record_fill(ev, q, f["shares"], {})
                self.inv.setdefault(slug, Inventory()).add(side, bucket, f["shares"], f["shares"] * f["price"])
            self.last_fill_ts = max(self.last_fill_ts, int(f["ts"]))
        self.ledger.commit()

    def settle(self) -> float:
        before = {p.slug for p in self.ledger.positions()}
        pnl = super().settle()
        after = {p.slug for p in self.ledger.positions()}
        for slug in before - after:
            self.log("settled", {"slug": slug})
        return pnl

    def fetch_actual_rewards(self, days_back: int = 3) -> None:
        today = datetime.now(timezone.utc).date()
        for k in range(1, days_back + 1):
            d = today - timedelta(days=k)
            try:
                total, rows = self.pbroker.earnings(d)
            except Exception as exc:
                print(f"  earnings {d}: {exc}", flush=True)
                continue
            self.ledger.db.execute("INSERT OR REPLACE INTO pilot_rewards_actual VALUES (?,?,?,?)",
                                   (d.isoformat(), total, time.time(), json.dumps(rows)[:100000]))
        self.ledger.commit()

    def run(self) -> None:
        self.pbroker.cancel_all()  # clean slate
        self.pbroker.start_heartbeat()
        self.log("start", {"capital": self.pcfg.capital_usd, "loss_limit": self.pcfg.daily_loss_limit_usd,
                           "cities": self.pcfg.cities})
        last_settle = last_rewards = 0.0
        try:
            while True:
                t0 = time.time()
                if t0 - last_settle > 1800:
                    self.settle()
                    last_settle = t0
                if t0 - last_rewards > 6 * 3600:
                    self.fetch_actual_rewards()
                    last_rewards = t0
                try:
                    self.cycle()
                except Halt:
                    raise
                except Exception as exc:
                    self.log("cycle_error", {"error": str(exc)[:300]})
                time.sleep(max(2.0, self.pcfg.cycle_s - (time.time() - t0)))
        except Halt as h:
            self.log("halt", {"reason": str(h)})
            # Latch: a restart must not quietly resume. A person deletes the kill file to go again.
            kf = Path(self.pcfg.kill_file)
            if not kf.exists():
                kf.write_text(f"{datetime.now(timezone.utc).isoformat()} halted: {h}\n")
        finally:
            try:
                self.pbroker.cancel_all()
            finally:
                self.pbroker.stop_heartbeat()


# ---------------------------------------------------------------- preflight & report


def preflight(cfg: PilotConfig) -> tuple[bool, list[str]]:
    """Checks that need no keys first, then the account. Returns (ok, lines)."""
    lines, ok = [], True
    try:
        g = geoblock_status()
        blocked = bool(g.get("blocked"))
        lines.append(f"{'FAIL' if blocked else 'ok  '} eligibility: geoblock says "
                     f"{'BLOCKED' if blocked else 'allowed'} ({g.get('country')}/{g.get('region')})")
        ok &= not blocked
    except Exception as exc:
        lines.append(f"FAIL eligibility check unavailable: {exc}")
        ok = False
    for var in ("POLY_PRIVATE_KEY", "POLY_FUNDER", "POLY_SIGNATURE_TYPE"):
        present = bool(os.environ.get(var))
        lines.append(f"{'ok  ' if present else 'FAIL'} {var} {'set' if present else 'missing'}")
        ok &= present
    live = os.environ.get("POLYWEATHER_LIVE") == "1"
    lines.append(f"{'ok  ' if live else 'FAIL'} POLYWEATHER_LIVE=1 {'set' if live else 'missing'}")
    ok &= live
    try:
        import py_clob_client  # noqa: F401
        lines.append("ok   py-clob-client installed")
    except ImportError:
        lines.append("FAIL py-clob-client not installed (pip install py-clob-client)")
        return False, lines
    if not ok:
        return ok, lines
    try:
        b = PilotBroker()
        bal, allow = b.collateral()
        enough = bal >= cfg.capital_usd
        lines.append(f"{'ok  ' if enough else 'FAIL'} USDC balance ${bal:,.2f} (pilot capital cap ${cfg.capital_usd:,.0f})")
        lines.append(f"{'ok  ' if allow >= cfg.capital_usd else 'FAIL'} exchange allowance ${allow:,.2f}")
        ok &= enough and allow >= cfg.capital_usd
        n_open = len(b.client.get_orders() or [])
        lines.append(f"info {n_open} open orders on the account (the pilot cancels them on start)")
    except Exception as exc:
        lines.append(f"FAIL account check: {exc}")
        ok = False
    lines.append(f"info caps: capital ${cfg.capital_usd:,.0f}, per event ${cfg.max_event_usd:,.0f}, "
                 f"per bucket side ${cfg.max_side_usd:,.0f}, {cfg.quote_shares:.0f} sh quotes, "
                 f"daily loss limit ${cfg.daily_loss_limit_usd:,.0f}, kill file '{cfg.kill_file}'")
    return ok, lines


def pilot_report(ledger: Ledger) -> None:
    db = ledger.db
    db.executescript(PILOT_SCHEMA)
    from .maker_runner import MAKER_SCHEMA
    db.executescript(MAKER_SCHEMA)
    est = dict(db.execute("""SELECT date(ts, 'unixepoch'), SUM(usd) FROM maker_rewards WHERE mode='live'
                             GROUP BY 1""").fetchall())
    act = dict(db.execute("SELECT day, earnings FROM pilot_rewards_actual").fetchall())
    fills = {d: (n, v) for d, n, v in db.execute("""SELECT date(ts, 'unixepoch'), COUNT(*), SUM(price*shares)
                                                    FROM pilot_fills GROUP BY 1""")}
    days = sorted(set(est) | set(act) | set(fills))
    print(f"{'day':10} {'est rewards':>12} {'ACTUAL':>10} {'actual/est':>10} {'fills':>6} {'bought':>9}")
    for d in days:
        e, a = est.get(d), act.get(d)
        ratio = f"{a / e:.2f}" if e and a is not None else "-"
        n, v = fills.get(d, (0, 0.0))
        print(f"{d:10} {('$%.2f' % e) if e else '-':>12} {('$%.2f' % a) if a is not None else '-':>10} "
              f"{ratio:>10} {n:6d} {('$%.0f' % (v or 0)):>9}")
    settled = db.execute("SELECT COALESCE(SUM(payout-cost),0), COUNT(*) FROM positions WHERE settled=1").fetchone()
    open_cost = db.execute("SELECT COALESCE(SUM(cost),0) FROM positions WHERE settled=0").fetchone()[0]
    tot_act = sum(v for v in act.values() if v)
    print(f"\nsettled trading pnl ${settled[0]:+,.2f} ({settled[1]} positions); open inventory ${open_cost:,.0f}")
    print(f"actual rewards to date ${tot_act:,.2f}  ->  net ${tot_act + settled[0]:+,.2f} before open inventory")
    for ts, kind, detail in db.execute("SELECT ts, kind, detail FROM pilot_events WHERE kind IN "
                                       "('halt','cycle_error','orphans_cancelled') ORDER BY ts DESC LIMIT 5"):
        print(f"  {datetime.fromtimestamp(ts, timezone.utc):%m-%d %H:%M} {kind}: {detail}")
