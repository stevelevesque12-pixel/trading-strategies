"""Live/paper trading loop: discover -> forecast -> price -> size -> execute -> log -> settle."""

import time
from dataclasses import dataclass, field
from datetime import timedelta
from typing import Optional

from .calibration import Calibrator
from .engine import Evaluation, ForecastCache, evaluate_event, now_utc
from .ledger import Ledger
from .markets import WeatherEvent, fetch_event, fetch_open_events
from .observations import day_obs, resolved_extreme
from .orderbook import fetch_books
from .risk import RiskConfig, RiskManager
from .strategy import OrderIntent, StrategyConfig, find_arbitrage, plan_event


@dataclass
class RunConfig:
    bankroll: float = 1000.0
    cities: Optional[list[str]] = None  # None = every city with a METAR settlement station
    max_lead_days: int = 2
    min_liquidity: float = 500.0
    use_nbm: bool = True
    use_ensembles: bool = True
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    risk: RiskConfig = field(default_factory=RiskConfig)


def select_events(events: list[WeatherEvent], cfg: RunConfig, asof) -> list[WeatherEvent]:
    from .engine import local_lead
    out = []
    for ev in events:
        if not ev.tradeable or ev.liquidity < cfg.min_liquidity:
            continue
        if cfg.cities and ev.city not in cfg.cities:
            continue
        lead = local_lead(ev, asof)
        if lead > cfg.max_lead_days or lead < -1:
            continue
        out.append(ev)
    return sorted(out, key=lambda e: (e.target_date, e.city, e.kind))


class Trader:
    def __init__(self, cfg: RunConfig, ledger: Ledger, broker, calibrator: Calibrator):
        self.cfg = cfg
        self.ledger = ledger
        self.broker = broker
        self.cal = calibrator
        self.cache = ForecastCache()
        self.risk = RiskManager(cfg.risk, cfg.bankroll)

    def evaluate_all(self, asof=None) -> list[tuple[Evaluation, dict]]:
        asof = asof or now_utc()
        events = select_events(fetch_open_events(), self.cfg, asof)
        tokens = [t for ev in events for b in ev.buckets for t in (b.yes_token, b.no_token)]
        books = fetch_books(tokens)
        out = []
        for ev in events:
            try:
                e = evaluate_event(ev, asof, self.cal, self.cache, self.cfg.use_nbm, self.cfg.use_ensembles)
            except Exception as exc:
                print(f"  {ev.slug}: evaluation failed: {exc}")
                continue
            out.append((e, books))
        return out

    def cycle(self, execute: bool = True) -> list[OrderIntent]:
        placed = []
        for e, books in self.evaluate_all():
            ev, out = e.event, e.output
            if out is None:
                continue
            self._log(e, books)
            positions = self.ledger.positions()
            budget = self.risk.budget_for(ev, positions)
            plan = plan_event(ev, out.probs, books, self.cfg.bankroll, positions, self.cfg.strategy,
                              observed_extreme=out.observed_extreme, final=out.final, risk_budget=budget,
                              calibrated_frac=out.calibrated_frac)
            orders = find_arbitrage(ev, books, self.cfg.strategy) + plan.orders
            for o in orders:
                placed.append(o)
                print(f"  {o.action} {o.shares:.2f} {o.side} {ev.city} {ev.kind} {ev.target_date} "
                      f"[{o.label}] <= {o.limit_price:.3f}  p={o.prob:.3f} edge={o.edge:+.3f} ({o.reason})")
                if execute:
                    self._execute(o, ev)
        self.ledger.commit()
        return placed

    def _execute(self, o: OrderIntent, ev: WeatherEvent) -> None:
        try:
            fill = self.broker.execute(o, ev)
        except Exception as exc:
            fill = None
            status = f"error: {exc}"
        else:
            status = fill.status
        self.ledger.record_order(
            mode=self.broker.mode, slug=ev.slug, city=ev.city, kind=ev.kind, target_date=str(ev.target_date),
            bucket=o.bucket, label=o.label, side=f"{o.action}_{o.side}", token_id=o.token_id,
            limit_price=o.limit_price, shares=o.shares, filled_shares=fill.shares if fill else 0.0,
            cost=fill.cost if fill else 0.0, fee=fill.fee if fill else 0.0, model_p=o.model_prob,
            edge=o.edge, status=status, exchange_id=fill.exchange_id if fill else None, note=o.reason)
        if fill and fill.shares > 0:
            sign = 1.0 if o.action == "BUY" else -1.0
            self.ledger.add_fill(ev.slug, o.bucket, o.side, o.token_id, ev.city, ev.kind, ev.target_date,
                                 o.label, sign * fill.shares, sign * fill.cost)

    def _log(self, e: Evaluation, books) -> None:
        ev, out = e.event, e.output
        n_obs = e.obs.n if e.obs else 0
        for v in out.sources:
            self.ledger.log_forecast(ev.slug, ev.city, ev.kind, ev.target_date, max(e.lead, 0), v.source,
                                     v.center, v.spread, v.kernel_sd, out.observed_extreme, n_obs)
        rows = []
        for i, b in enumerate(ev.buckets):
            yb, nb = books.get(b.yes_token), books.get(b.no_token)
            rows.append((i, b.label, float(out.probs[i]), None,
                         yb.best_bid if yb else None, yb.best_ask if yb else None, nb.best_ask if nb else None))
        self.ledger.log_probs(ev.slug, rows)

    def settle(self) -> float:
        """Settle resolved positions and record resolutions for calibration."""
        pnl = 0.0
        q = """SELECT DISTINCT slug, city, kind, target_date FROM forecasts
               WHERE slug NOT IN (SELECT slug FROM resolutions)
               UNION SELECT DISTINCT slug, city, kind, target_date FROM positions WHERE settled=0"""
        today = now_utc().date()
        for slug, city, kind, tdate in list(self.ledger.db.execute(q)):
            if str(tdate) >= str(today + timedelta(days=1)):
                continue
            ev = fetch_event(slug)
            if ev is None or not ev.closed or ev.winner() is None:
                continue
            value = None
            if ev.station is not None and ev.station.source == "metar":
                try:
                    obs = day_obs(ev.station.icao, ev.target_date, ev.station.tz, ev.unit, not ev.hourly_only)
                    value = resolved_extreme(obs, ev.kind)
                except Exception:
                    pass
            self.ledger.record_resolution(slug, city, kind, tdate, ev.winner(), value)
            p = self.ledger.settle(slug, ev.winner())
            if p:
                print(f"  settled {slug}: winner {ev.buckets[ev.winner()].label}  pnl {p:+.2f}")
            pnl += p
        self.risk.record_pnl(pnl)
        return pnl

    def loop(self, interval_s: int = 300, execute: bool = True) -> None:
        last_settle = 0.0
        while True:
            t0 = time.time()
            print(f"[{now_utc():%Y-%m-%d %H:%M:%S}Z] cycle")
            if t0 - last_settle > 1800:
                self.settle()
                last_settle = t0
            try:
                self.cycle(execute=execute)
            except Exception as exc:
                print(f"  cycle failed: {exc}")
            time.sleep(max(5.0, interval_s - (time.time() - t0)))
