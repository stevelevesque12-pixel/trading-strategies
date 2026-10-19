"""Small real-money pilot of the market maker, fully automated, with the safety rails it needs.

    python -m polyweather pilot --check         # preflight only: eligibility, keys, balance, approvals
    python -m polyweather pilot --test-alert    # send a test alert to every configured channel
    python -m polyweather pilot                 # run (needs POLYWEATHER_LIVE=1)
    python -m polyweather pilot-report          # estimated vs ACTUAL rewards, fills, cash-outs, PnL by day
    touch STOP                                  # kill switch: cancels every order and exits

Any automatic halt (loss limit, heartbeat, geoblock) writes the kill file too,
so a supervisor restarting the process can't resume trading on its own.

What runs without anyone watching:
* quoting, fills and reconciliation every cycle;
* settlement of resolved markets in the ledger;
* treasury: merge matched YES+NO pairs back to cash, redeem resolved winnings
  (gasless, via Polymarket's relayer);
* alerts: start/halt, money-path failures, cash-outs, settlements, and a daily
  report with yesterday's estimated vs actual rewards.

Rails, in the order they act:
* Eligibility: refuses to start (and halts) if polymarket.com/api/geoblock says
  this connection is blocked. There is deliberately no override.
* Exchange heartbeat: once started, the CLOB cancels all our orders if a
  heartbeat is missed for ~10s. If heartbeats can't be started, orders are
  placed as short-lived GTD orders instead, so they still expire on their own.
* Hard capital cap across all events; above it only risk-reducing quotes.
* Daily loss limit on settled PnL plus open inventory marked to mid.
* Reconciliation, fills from our own trade history (de-duplicated), and
  the exchange's own word on which orders score for rewards.
"""

import json
import os
import time
from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from . import http
from .alerts import Alerter
from .exchange import ExchangeBroker
from .ledger import Ledger
from .maker import Inventory, MakerConfig, Quote
from .maker_runner import MAKER_SCHEMA, MakerRunConfig, MakerRunner
from .orderbook import fetch_books
from .rewards import adjusted_mid
from .treasury import TREASURY_SCHEMA, Treasury

GEOBLOCK_URL = "https://polymarket.com/api/geoblock"
REQUIRED_ENV = ("POLY_PRIVATE_KEY", "POLY_WALLET", "POLY_RELAYER_API_KEY", "POLY_RELAYER_API_KEY_ADDRESS")
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
    quote_shares: float = 20.0  # the usual reward minimum
    edge: float = 0.01
    daily_loss_limit_usd: float = 40.0
    cities: Optional[list[str]] = None
    cycle_s: int = 30
    kill_file: str = "STOP"
    db: str = "pilot.db"
    report_hour_utc: int = 1  # daily report after Polymarket's midnight-UTC reward payout

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


class PilotRunner(MakerRunner):
    def __init__(self, cfg: PilotConfig, ledger: Ledger, broker: ExchangeBroker, alerter: Optional[Alerter] = None):
        super().__init__(MakerRunConfig(maker=cfg.maker_config(), cycle_s=cfg.cycle_s, cities=cfg.cities),
                         ledger, broker)
        self.pcfg = cfg
        self.pbroker = broker
        self.alerter = alerter or Alerter.from_env()
        ledger.db.executescript(PILOT_SCHEMA)
        self.treasury = Treasury(broker, ledger, self.alerter, inventories=self.inv)
        self.last_fill_ts = int(time.time()) - 3600
        self.n = 0
        self._errors: list[float] = []
        self._last_report_day: Optional[date] = None

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
        """Today's (UTC) settled PnL and merge gains, plus open inventory marked to mid."""
        start = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        db = self.ledger.db
        settled = db.execute("""SELECT COALESCE(SUM(p.payout - p.cost), 0) FROM positions p
                                JOIN pilot_events e ON e.kind='settled' AND json_extract(e.detail, '$.slug') = p.slug
                                WHERE p.settled=1 AND e.ts >= ?""", (start,)).fetchone()[0]
        merged = db.execute("SELECT COALESCE(SUM(realized), 0) FROM pilot_treasury WHERE action='merge' "
                            "AND status='ok' AND ts >= ?", (start,)).fetchone()[0]
        open_pos = [p for p in self.ledger.positions() if p.shares > 1e-9]
        mark = 0.0
        if open_pos:
            books = fetch_books([p.token_id for p in open_pos])
            for p in open_pos:
                ob = books.get(p.token_id)
                mid = adjusted_mid(ob, 1.0) if ob else None
                mark += (p.shares * mid - p.cost) if mid is not None else -p.cost
        return float(settled) + float(merged) + mark

    # cycle -----------------------------------------------------------------
    def cycle(self) -> None:
        self.n += 1
        self.check_guards()
        self.record_fills()
        orphans, _ = self.pbroker.reconcile()
        if orphans:
            self.log("orphans_cancelled", {"n": orphans})
        super().cycle()
        try:
            self.treasury.run()
        except Exception as exc:
            self.alerter.send("treasury error", str(exc)[:300], level="warn", key="treasury")
        if self.n % 10 == 0:
            frac = self.pbroker.scoring_fraction()
            if frac is not None:
                print(f"  scoring: {100 * frac:.0f}% of {len(self.pbroker.open)} open orders", flush=True)
                if frac < 0.5 and len(self.pbroker.open) >= 4:
                    self.alerter.send("most orders not scoring for rewards",
                                      f"{100 * frac:.0f}% of {len(self.pbroker.open)} open orders score",
                                      level="warn", key="scoring")
        self.maybe_daily_report()

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
        newly = before - {p.slug for p in self.ledger.positions()}
        for slug in newly:
            self.log("settled", {"slug": slug})
        if newly:
            self.alerter.send("markets settled", f"{len(newly)} events, trading pnl {pnl:+.2f}", level="info")
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

    def maybe_daily_report(self) -> None:
        now = datetime.now(timezone.utc)
        if now.hour < self.pcfg.report_hour_utc or self._last_report_day == now.date():
            return
        self._last_report_day = now.date()
        self.fetch_actual_rewards()
        self.alerter.send("daily report", daily_summary(self.ledger, now.date() - timedelta(days=1)), level="info")

    def run(self) -> None:
        self.pbroker.cancel_all()  # clean slate
        hb = self.pbroker.start_heartbeat()
        self.log("start", {"capital": self.pcfg.capital_usd, "loss_limit": self.pcfg.daily_loss_limit_usd,
                           "cities": self.pcfg.cities, "wallet_type": self.pbroker.wallet_type, "heartbeat": hb})
        self.alerter.send("pilot started", f"capital cap ${self.pcfg.capital_usd:,.0f}, loss limit "
                          f"${self.pcfg.daily_loss_limit_usd:,.0f}, heartbeat {'on' if hb else 'off (GTD expiry)'}")
        last_settle = 0.0
        try:
            while True:
                t0 = time.time()
                if t0 - last_settle > 1800:
                    self.settle()
                    last_settle = t0
                try:
                    self.cycle()
                except Halt:
                    raise
                except Exception as exc:
                    self.log("cycle_error", {"error": str(exc)[:300]})
                    self._errors = [t for t in self._errors if t > t0 - 600] + [t0]
                    if len(self._errors) >= 3:
                        self.alerter.send("repeated cycle errors", str(exc)[:300], level="warn", key="cycle")
                time.sleep(max(2.0, self.pcfg.cycle_s - (time.time() - t0)))
        except Halt as h:
            self.log("halt", {"reason": str(h)})
            # Latch: a restart must not quietly resume. A person deletes the kill file to go again.
            kf = Path(self.pcfg.kill_file)
            if not kf.exists():
                kf.write_text(f"{datetime.now(timezone.utc).isoformat()} halted: {h}\n")
            self.alerter.send("HALTED, all orders cancelled", f"{h}\nDelete {self.pcfg.kill_file} to restart.",
                              level="urgent")
        except BaseException as exc:
            self.alerter.send("pilot crashed", f"{type(exc).__name__}: {str(exc)[:300]}", level="urgent")
            raise
        finally:
            try:
                self.pbroker.cancel_all()
            finally:
                self.pbroker.stop_heartbeat()


# ---------------------------------------------------------------- preflight & reports


def preflight(cfg: PilotConfig, alerter: Optional[Alerter] = None) -> tuple[bool, list[str]]:
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
    for var in REQUIRED_ENV:
        present = bool(os.environ.get(var))
        lines.append(f"{'ok  ' if present else 'FAIL'} {var} {'set' if present else 'missing'}")
        ok &= present
    live = os.environ.get("POLYWEATHER_LIVE") == "1"
    lines.append(f"{'ok  ' if live else 'FAIL'} POLYWEATHER_LIVE=1 {'set' if live else 'missing'}")
    ok &= live
    channels = (alerter or Alerter.from_env()).channels
    lines.append(f"{'ok  ' if channels else 'warn'} alerts: {', '.join(channels) or 'none configured (print only)'}")
    try:
        import polymarket  # noqa: F401
        lines.append("ok   polymarket-client installed")
    except ImportError:
        lines.append("FAIL polymarket-client not installed (pip install polymarket-client)")
        return False, lines
    if not ok:
        return ok, lines
    try:
        b = ExchangeBroker()
        lines.append(f"ok   connected: wallet {b.wallet} ({b.wallet_type})")
        bal, allow = b.collateral()
        enough = bal >= cfg.capital_usd
        lines.append(f"{'ok  ' if enough else 'FAIL'} pUSD balance ${bal:,.2f} (capital cap ${cfg.capital_usd:,.0f})")
        ok &= enough
        try:
            lines.append(f"info trading approvals: {b.approvals_state()}")
        except Exception as exc:
            lines.append(f"warn couldn't read trading approvals ({exc}); order placement sets them up if missing")
        n_open = sum(1 for _ in b.client.list_open_orders().first_page().items)
        lines.append(f"info {n_open} open orders on the account (the pilot cancels them on start)")
    except Exception as exc:
        lines.append(f"FAIL account check: {exc}")
        ok = False
    lines.append(f"info caps: capital ${cfg.capital_usd:,.0f}, per event ${cfg.max_event_usd:,.0f}, "
                 f"per bucket side ${cfg.max_side_usd:,.0f}, {cfg.quote_shares:.0f} sh quotes, "
                 f"daily loss limit ${cfg.daily_loss_limit_usd:,.0f}, kill file '{cfg.kill_file}'")
    return ok, lines


def _ensure_schema(db) -> None:
    db.executescript(PILOT_SCHEMA)
    db.executescript(MAKER_SCHEMA)
    db.executescript(TREASURY_SCHEMA)


def daily_summary(ledger: Ledger, day: date) -> str:
    db = ledger.db
    _ensure_schema(db)
    d = day.isoformat()
    est = db.execute("SELECT COALESCE(SUM(usd),0) FROM maker_rewards WHERE mode='live' AND date(ts,'unixepoch')=?",
                     (d,)).fetchone()[0]
    act = db.execute("SELECT earnings FROM pilot_rewards_actual WHERE day=?", (d,)).fetchone()
    n, bought = db.execute("SELECT COUNT(*), COALESCE(SUM(price*shares),0) FROM pilot_fills "
                           "WHERE date(ts,'unixepoch')=?", (d,)).fetchone()
    merged, mreal = db.execute("SELECT COALESCE(SUM(pairs),0), COALESCE(SUM(realized),0) FROM pilot_treasury "
                               "WHERE action='merge' AND status='ok' AND date(ts,'unixepoch')=?", (d,)).fetchone()
    redeemed = db.execute("SELECT COUNT(*) FROM pilot_treasury WHERE action='redeem' AND status='ok' "
                          "AND date(ts,'unixepoch')=?", (d,)).fetchone()[0]
    settled = db.execute("SELECT COALESCE(SUM(payout-cost),0) FROM positions WHERE settled=1").fetchone()[0]
    open_cost = db.execute("SELECT COALESCE(SUM(cost),0) FROM positions WHERE settled=0").fetchone()[0]
    actual = f"${act[0]:.2f}" if act and act[0] is not None else "not yet reported"
    ratio = f" ({act[0] / est:.0%} of estimate)" if act and act[0] is not None and est else ""
    return (f"{d}: rewards actual {actual}{ratio}, estimate ${est:.2f}\n"
            f"fills {n} (${bought:,.0f}), merged {merged:.0f} pairs (+${mreal:.2f}), redeemed {redeemed}\n"
            f"settled trading pnl to date ${settled:+,.2f}; open inventory ${open_cost:,.0f}")


def pilot_report(ledger: Ledger) -> None:
    db = ledger.db
    _ensure_schema(db)
    est = dict(db.execute("""SELECT date(ts, 'unixepoch'), SUM(usd) FROM maker_rewards WHERE mode='live'
                             GROUP BY 1""").fetchall())
    act = dict(db.execute("SELECT day, earnings FROM pilot_rewards_actual").fetchall())
    fills = {d: (n, v) for d, n, v in db.execute("""SELECT date(ts, 'unixepoch'), COUNT(*), SUM(price*shares)
                                                    FROM pilot_fills GROUP BY 1""")}
    merges = {d: (p, r) for d, p, r in db.execute("""SELECT date(ts,'unixepoch'), SUM(pairs), SUM(realized)
                                                     FROM pilot_treasury WHERE action='merge' AND status='ok'
                                                     GROUP BY 1""")}
    days = sorted(set(est) | set(act) | set(fills) | set(merges))
    print(f"{'day':10} {'est rewards':>12} {'ACTUAL':>10} {'actual/est':>10} {'fills':>6} {'bought':>9} {'merged':>8}")
    for d in days:
        e, a = est.get(d), act.get(d)
        ratio = f"{a / e:.2f}" if e and a is not None else "-"
        n, v = fills.get(d, (0, 0.0))
        mp, _ = merges.get(d, (0.0, 0.0))
        print(f"{d:10} {('$%.2f' % e) if e else '-':>12} {('$%.2f' % a) if a is not None else '-':>10} "
              f"{ratio:>10} {n:6d} {('$%.0f' % (v or 0)):>9} {mp or 0:8.0f}")
    settled = db.execute("SELECT COALESCE(SUM(payout-cost),0), COUNT(*) FROM positions WHERE settled=1").fetchone()
    open_cost = db.execute("SELECT COALESCE(SUM(cost),0) FROM positions WHERE settled=0").fetchone()[0]
    merge_gain = db.execute("SELECT COALESCE(SUM(realized),0) FROM pilot_treasury WHERE action='merge' "
                            "AND status='ok'").fetchone()[0]
    tot_act = sum(v for v in act.values() if v)
    print(f"\nsettled trading pnl ${settled[0]:+,.2f} ({settled[1]} positions), merge gains ${merge_gain:+,.2f}; "
          f"open inventory ${open_cost:,.0f}")
    print(f"actual rewards to date ${tot_act:,.2f}  ->  net ${tot_act + settled[0] + merge_gain:+,.2f} "
          "before open inventory")
    for ts, kind, detail in db.execute("SELECT ts, kind, detail FROM pilot_events WHERE kind IN "
                                       "('halt','cycle_error','orphans_cancelled') ORDER BY ts DESC LIMIT 5"):
        print(f"  {datetime.fromtimestamp(ts, timezone.utc):%m-%d %H:%M} {kind}: {detail}")
