"""Walk-forward backtest on resolved Polymarket events, US cities (NBM as issued).

What is real:
* events, buckets, fee schedule and the winning bucket: Polymarket Gamma API
* market prices at each decision time: CLOB price history for every bucket
* forecasts: the NBM bulletin that was public at the decision time (IEM archive)
* observations up to the decision time: IEM METAR archive
* calibration: refit before every decision day using only events resolved earlier

What is assumed:
* the order book. There's no public history of it, so each decision trades
  against a synthetic one-level book around the historical mid, with the
  spread and depth of a *measured* live book of the same kind and price
  band (`CostModel`, from a snapshot of all 247 open events on 2026-10-18).
  That matters enormously: "lowest temperature" books are nearly empty
  (median spread 30-80c near the money), so a mid-price backtest of them
  is fiction. `--costs flat` reproduces the naive mid +/- 1c assumption
  for comparison.
* fills. With `fills="trades"` (default) an order only fills against size
  that actually printed: taker trades in the following `fill_window_min`
  that bought our token at or below our limit (or sold its complement at or
  above 1 - limit), at those prints' prices, and we only get
  `print_capture` of that volume (someone else took it in reality). This
  is the closest a book-less backtest gets to "was this executable".
  `fills="book"` trades against the synthetic book alone.
* global ensembles are excluded: Open-Meteo doesn't serve archived
  ensemble members, so including them would mean using forecasts we
  couldn't have had.
"""

import csv
import math
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from . import forecast as fc
from .calibration import Calibrator, fit, fit_intraday, intraday_prior
from .markets import WeatherEvent, event_slug, fetch_event
from .model import SourceCalibration, bucket_probabilities, default_calibration, remaining_extreme
from .observations import Obs, ObsState, day_window_utc, fetch_hf_iem, fetch_obs_iem, resolved_extreme
from .fees import fee_per_share
from .orderbook import OrderBook, fetch_price_history, fetch_trades
from .ledger import Position
from .stations import STATIONS
from .strategy import StrategyConfig, market_probs, plan_event

US_CITIES = [c for c, s in STATIONS.items() if s.nbm]
CALIB_POINTS = ((1, 10), (2, 10))  # (lead, local hour) where calibration samples are taken


@dataclass
class BacktestConfig:
    days: int = 90
    end: Optional[date] = None  # last target date (default: 2 days ago)
    cities: Optional[list[str]] = None
    kinds: tuple = ("high", "low")
    # (days before target, local hour): when the bot looks at each event
    decisions: tuple = ((1, 10), (1, 18), (0, 8), (0, 12), (0, 15))
    bankroll: float = 1000.0
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    costs: str = "measured"  # "measured" | "conservative" | "flat"
    half_spread: float = 0.01  # only for costs="flat"
    depth_usd: float = 150.0  # only for costs="flat"
    max_price_age_h: float = 0.25
    # The model may only use METARs at least this much older than the price snapshot it trades
    # against (~publication latency), so it never knows more than the market could have.
    obs_lag_min: float = 5.0
    use_hf_obs: bool = True  # 5-minute ASOS feed as the nowcast anchor
    intraday_refit: bool = True  # walk-forward refit of the intraday (bias, sd) table
    fills: str = "trades"  # "trades" | "book"
    fill_window_min: float = 30.0
    print_capture: float = 0.5
    calib_window_days: int = 60
    calib_min_samples: int = 10
    workers: int = 8
    out_dir: str = "backtest_out"


# ------------------------------------------------------------------ data


def _asof(target: date, lead: int, hour: int, tz: str) -> datetime:
    return datetime.combine(target - timedelta(days=lead), time(hour), tzinfo=ZoneInfo(tz)).astimezone(timezone.utc)


def _load_events(cities, kinds, dates, workers) -> list[WeatherEvent]:
    jobs = [(k, c, d) for c in cities for k in kinds for d in dates]

    def get(job):
        k, c, d = job
        try:
            ev = fetch_event(event_slug(k, c, d), cache=True)
        except Exception:
            return None
        return ev if ev is not None and ev.closed and ev.winner() is not None and ev.station else None

    with ThreadPoolExecutor(workers) as ex:
        return [e for e in ex.map(get, jobs) if e is not None]


def _load_obs(events, start: date, end: date) -> dict:
    """{(icao, include_specials): Obs} spanning the whole test range."""
    need = {(e.station.icao, not e.hourly_only, e.unit, e.station.tz) for e in events}
    out = {}
    for icao, specials, unit, tz in need:
        s, _ = day_window_utc(start - timedelta(days=1), tz)
        _, e = day_window_utc(end, tz)
        out[(icao, specials)] = fetch_obs_iem(icao, s, e, unit, include_specials=specials, cache=True)
    return out


def _load_hf(events, start: date, end: date, workers) -> dict:
    """{icao: Obs} of 5-minute readings, fetched in 10-day chunks."""
    st_by = {e.station.icao: e for e in events}
    jobs = []
    d = start - timedelta(days=1)
    while d <= end:
        for icao in st_by:
            jobs.append((icao, d, min(d + timedelta(days=10), end + timedelta(days=1))))
        d += timedelta(days=10)

    def get(job):
        icao, a, b = job
        ev = st_by[icao]
        s, _ = day_window_utc(a, ev.station.tz)
        _, e = day_window_utc(b, ev.station.tz)
        try:
            return icao, fetch_hf_iem(icao, s, e, ev.unit, cache=True)
        except Exception:
            return icao, Obs()

    out: dict[str, Obs] = {}
    with ThreadPoolExecutor(min(workers, 4)) as ex:
        for icao, o in ex.map(get, jobs):
            acc = out.setdefault(icao, Obs())
            acc.times.extend(o.times)
            acc.values.extend(o.values)
    for o in out.values():
        order = sorted(range(len(o.times)), key=o.times.__getitem__)
        o.times = [o.times[i] for i in order]
        o.values = [o.values[i] for i in order]
    return out


def _load_prices(events, workers) -> dict[str, list]:
    toks = [(b.yes_token, e) for e in events for b in e.buckets]

    def get(item):
        tok, ev = item
        s, end = day_window_utc(ev.target_date, ev.station.tz)
        try:
            return tok, fetch_price_history(tok, int(s.timestamp()) - 4 * 86400, int(end.timestamp()),
                                            fidelity_min=10, cache=True)
        except Exception:
            return tok, []

    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(get, toks))


def _load_trades(events, workers) -> dict[str, list]:
    def get(ev):
        try:
            return ev.slug, fetch_trades([b.condition_id for b in ev.buckets if b.condition_id], cache=True)
        except Exception:
            return ev.slug, []
    with ThreadPoolExecutor(workers) as ex:
        return dict(ex.map(get, events))


def _trade_fill(ev, o, trades, asof: datetime, cfg) -> tuple[float, float, float]:
    """(shares, cost incl. fees, fees) we could have bought from real prints after `asof`."""
    b = ev.buckets[o.bucket]
    comp = b.no_token if o.token_id == b.yes_token else b.yes_token
    t0, t1 = asof.timestamp(), asof.timestamp() + cfg.fill_window_min * 60
    want, got, cost, fees = o.shares, 0.0, 0.0, 0.0
    for t in trades:
        if t["ts"] <= t0:
            continue
        if t["ts"] > t1 or got >= want - 1e-9:
            break
        if t["asset"] == o.token_id and t["side"] == "BUY" and t["price"] <= o.limit_price + 1e-9:
            px = t["price"]
        elif t["asset"] == comp and t["side"] == "SELL" and t["price"] >= 1 - o.limit_price - 1e-9:
            px = 1 - t["price"]
        else:
            continue
        take = min(t["size"] * cfg.print_capture, want - got)
        f = fee_per_share(px, ev.fee_rate, ev.fee_exponent)
        got += take
        cost += take * (px + f)
        fees += take * f
    return got, cost, fees


class _NbmStore:
    def __init__(self):
        self.mem = {}

    def _get(self, icao, cycle):
        key = (icao, cycle)
        if key not in self.mem:
            self.mem[key] = fc.fetch_nbm(icao, cycle, cache=True)
        return self.mem[key]

    def at(self, icao: str, asof: datetime):
        return fc.latest_nbm(icao, asof, fetch=lambda cyc: self._get(icao, cyc))

    def prefetch(self, jobs, workers):
        def get(job):
            icao, asof = job
            return self.at(icao, asof)
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(get, jobs))


def _price_at(hist, ts: float, max_age_s: float) -> Optional[tuple[float, float]]:
    """(timestamp, price) of the latest point at or before ts."""
    best = None
    for t, p in hist:
        if t <= ts:
            best = (t, p)
        else:
            break
    if best is None or ts - best[0] > max_age_s:
        return None
    return best


# Live CLOB snapshot, 2026-10-18 (all open daily-temperature events): YES spread
# (median, 75th pct) and $ within 2c of the best ask (median), by kind/region/mid band.
_BANDS = (0.1, 0.3, 0.7, 0.9, 1.01)
_MEASURED = {
    ("high", True): [(0.008, 0.017, 4), (0.030, 0.050, 30), (0.030, 0.047, 147), (0.070, 0.085, 54), (0.03, 0.05, 100)],
    ("high", False): [(0.009, 0.016, 4), (0.021, 0.030, 44), (0.020, 0.030, 221), (0.040, 0.060, 714), (0.017, 0.032, 432)],
    ("low", True): [(0.029, 0.049, 4), (0.310, 0.350, 13), (0.400, 0.608, 25), (0.227, 0.227, 15), (0.077, 0.095, 35)],
    ("low", False): [(0.019, 0.050, 3), (0.270, 0.380, 10), (0.810, 0.920, 27), (0.227, 0.227, 15), (0.049, 0.080, 15)],
}


def _cost_params(ev, p, cfg):
    if cfg.costs == "flat":
        return cfg.half_spread, cfg.depth_usd
    band = next(i for i, hi in enumerate(_BANDS) if p < hi)
    med, p75, depth = _MEASURED[(ev.kind, bool(ev.station.nbm))][band]
    spread = p75 if cfg.costs == "conservative" else med
    return max(spread / 2, 0.0005), max(depth, 1.0)


def _synthetic_books(ev, prices, cfg) -> dict[str, OrderBook]:
    books = {}
    for b, p in zip(ev.buckets, prices):
        if p is None:
            continue
        half, depth_usd = _cost_params(ev, p, cfg)
        ya = min(0.999, p + half)
        yb = max(0.001, p - half)
        na = min(0.999, 1 - p + half)
        nb = max(0.001, 1 - p - half)
        books[b.yes_token] = OrderBook(b.yes_token, [(yb, depth_usd / yb)], [(ya, depth_usd / ya)])
        books[b.no_token] = OrderBook(b.no_token, [(nb, depth_usd / nb)], [(na, depth_usd / na)])
    return books


# ------------------------------------------------------------------ scoring


def _brier(p, k):
    y = np.zeros(len(p))
    y[k] = 1
    return float(np.sum((p - y) ** 2))


def _logloss(p, k):
    return -math.log(max(float(p[k]), 1e-4))


# ------------------------------------------------------------------ main


def run_backtest(cfg: BacktestConfig) -> dict:
    cities = cfg.cities or US_CITIES
    cities = [c for c in cities if STATIONS.get(c) and STATIONS[c].nbm]
    end = cfg.end or (datetime.now(timezone.utc).date() - timedelta(days=2))
    start = end - timedelta(days=cfg.days - 1)
    dates = [start + timedelta(days=i) for i in range(cfg.days)]
    print(f"backtest {start} .. {end}, cities {cities}")

    events = _load_events(cities, cfg.kinds, dates, cfg.workers)
    print(f"  {len(events)} resolved events")
    obs_all = _load_obs(events, start, end)
    hf_all = _load_hf(events, start, end, cfg.workers) if cfg.use_hf_obs else {}
    print("  observations loaded" + (f" (+5-min feed for {len(hf_all)} stations)" if hf_all else ""))
    nbm = _NbmStore()
    nbm.prefetch([(e.station.icao, _asof(e.target_date, lead, hr, e.station.tz))
                  for e in events for lead, hr in tuple(cfg.decisions) + CALIB_POINTS], cfg.workers)
    print("  NBM loaded")
    prices = _load_prices(events, cfg.workers)
    trades_by = _load_trades(events, cfg.workers) if cfg.fills == "trades" else {}
    print("  prices loaded" + (f", trades for {sum(1 for v in trades_by.values() if v)} events" if trades_by else ""))

    events.sort(key=lambda e: e.target_date)
    # calibration samples: (city, kind, lead) -> [(target_date, center, spread, observed)]
    samples = defaultdict(list)
    intraday_samples = defaultdict(list)  # (kind, local hour) -> [(date, residual)], pooled over cities
    rows, trades = [], []
    positions: dict[str, list[Position]] = defaultdict(list)
    recon_ok = recon_n = 0

    for ev in events:
        st = ev.station
        day_obs = obs_all[(st.icao, not ev.hourly_only)].window(*day_window_utc(ev.target_date, st.tz))
        day_hf = hf_all[st.icao].window(*day_window_utc(ev.target_date, st.tz)) if st.icao in hf_all else None
        value = resolved_extreme(day_obs, ev.kind)
        winner = ev.winner()
        if value is not None:
            recon_n += 1
            recon_ok += int(ev.bucket_index_for(value) == winner)

        # Raw forecast vs outcome at the calibration points, used by later events' decisions
        # (whatever decision points this run trades).
        for lead, hour in CALIB_POINTS:
            if value is None:
                break
            asof = _asof(ev.target_date, lead, hour, st.tz)
            runtime, nrows = nbm.at(st.icao, asof)
            f = fc.nbm_day_forecast(nrows, runtime, st, ev.target_date, ev.kind) if nrows else None
            R = remaining_extreme(f, SourceCalibration(), ev.kind, None, asof.timestamp())[0] if f else None
            if R is not None:
                samples[(ev.city, ev.kind, lead)].append((ev.target_date, float(R.mean()), float(R.std()), value))

        for lead, hour in cfg.decisions:
            asof = _asof(ev.target_date, lead, hour, st.tz)
            runtime, nrows = nbm.at(st.icao, asof)
            if not nrows:
                continue
            f = fc.nbm_day_forecast(nrows, runtime, st, ev.target_date, ev.kind)
            if f is None:
                continue
            # walk-forward calibration from events that resolved before this decision
            cutoff = asof.astimezone(ZoneInfo(st.tz)).date() - timedelta(days=1)
            lead_c = max(lead, 1)
            hist = [(c, s, o) for d, c, s, o in samples[(ev.city, ev.kind, lead_c)]
                    if d < cutoff and d >= cutoff - timedelta(days=cfg.calib_window_days)]
            prior = default_calibration("nbm", lead, ev.unit, ev.kind, ev.hourly_only)
            cal = prior
            if len(hist) >= cfg.calib_min_samples:
                fitted = fit(hist, default_calibration("nbm", lead_c, ev.unit, ev.kind, ev.hourly_only))
                cal = fitted if lead >= 1 else type(prior)(bias=fitted.bias, spread_scale=fitted.spread_scale,
                                                           kernel_sd=prior.kernel_sd, weight=prior.weight,
                                                           n=fitted.n)
            snaps = [_price_at(prices.get(b.yes_token, []), asof.timestamp(), cfg.max_price_age_h * 3600)
                     for b in ev.buckets]
            px = [sn[1] if sn else None for sn in snaps]
            snap_ts = min((sn[0] for sn in snaps if sn), default=asof.timestamp())
            obs_cut = datetime.fromtimestamp(min(snap_ts, asof.timestamp()) - cfg.obs_lag_min * 60, timezone.utc)
            obs_state = ObsState.from_obs(day_obs, ev.kind, obs_cut, day_hf) if lead == 0 else None
            if obs_state is not None and obs_state.last_value is not None:
                prior_i = intraday_prior(ev.kind, hour, ev.unit, "nbm")
                hist_i = [r for d, r in intraday_samples[(ev.kind, hour)]
                          if cutoff - timedelta(days=cfg.calib_window_days) <= d < cutoff]
                cal = fit_intraday(hist_i, prior_i) if cfg.intraday_refit else prior_i
                # residual for later events: realised remaining extreme vs forecast
                fut = [v for t, v in zip(day_obs.times, day_obs.values) if t > obs_cut]
                R0, _ = remaining_extreme(f, SourceCalibration(intraday=True), ev.kind, obs_state, asof.timestamp())
                if fut and R0 is not None:
                    tgt = max(fut) if ev.kind == "high" else min(fut)
                    intraday_samples[(ev.kind, hour)].append((ev.target_date, tgt - float(R0.mean())))
            out = bucket_probabilities(ev, [(f, cal)], obs_state, asof.timestamp(), st.tz)
            if out is None:
                continue

            if sum(p is not None for p in px) < len(px) - 1:
                continue
            books = _synthetic_books(ev, px, cfg)
            mkt = market_probs(ev, _synthetic_books(ev, px, BacktestConfig(costs="flat")), max_spread=1.0)
            mkt_full = np.where(np.isnan(mkt), 0.0, mkt)
            mkt_full = np.clip(mkt_full, 1e-3, None)
            mkt_full = mkt_full / mkt_full.sum()

            plan = plan_event(ev, out.probs, books, cfg.bankroll, positions[ev.slug], cfg.strategy,
                              observed_extreme=out.observed_extreme, final=out.final,
                              calibrated_frac=1.0 if cal.n else 0.0)
            rows.append({
                "slug": ev.slug, "city": ev.city, "kind": ev.kind, "date": ev.target_date.isoformat(),
                "lead": lead, "hour": hour, "winner": winner, "calibrated": int(cal.n > 0),
                "brier_model": _brier(out.probs, winner), "brier_blend": _brier(plan.q, winner),
                "brier_market": _brier(mkt_full, winner),
                "ll_model": _logloss(out.probs, winner), "ll_blend": _logloss(plan.q, winner),
                "ll_market": _logloss(mkt_full, winner),
                "p_model_win": float(out.probs[winner]), "p_market_win": float(mkt_full[winner]),
            })
            for o in plan.orders:
                if o.action != "BUY":
                    continue
                shares, cost, fee = o.shares, o.est_cost, o.est_fee
                if cfg.fills == "trades":
                    shares, cost, fee = _trade_fill(ev, o, trades_by.get(ev.slug, []), asof, cfg)
                    if shares < 0.01:
                        continue
                won = (o.bucket == winner) if o.side == "YES" else (o.bucket != winner)
                pnl = (shares if won else 0.0) - cost
                positions[ev.slug].append(Position(ev.slug, o.bucket, o.side, o.token_id, ev.city, ev.kind,
                                                   ev.target_date.isoformat(), o.label, shares, cost,
                                                   False, 0.0))
                trades.append({
                    "date": ev.target_date.isoformat(), "asof": asof.isoformat(), "city": ev.city,
                    "kind": ev.kind, "lead": lead, "hour": hour, "bucket": o.label, "side": o.side,
                    "shares": round(shares, 2), "price": round(cost / shares, 4),
                    "cost": round(cost, 2), "fee": round(fee, 3), "q": round(o.prob, 4),
                    "p_model": round(o.model_prob, 4), "won": int(won), "pnl": round(pnl, 2),
                })

    return _report(cfg, rows, trades, recon_ok, recon_n)


def _report(cfg, rows, trades, recon_ok, recon_n) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, data in (("decisions.csv", rows), ("trades.csv", trades)):
        if data:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)

    print(f"\nresolution reconstruction from METARs: {recon_ok}/{recon_n} "
          f"({100 * recon_ok / max(recon_n, 1):.1f}%) match Polymarket's winner")
    print("\nforecast skill (lower is better), by decision point:")
    print(f"  {'lead/hour':>10} {'n':>5} {'brier mdl':>9} {'blend':>7} {'market':>7}   "
          f"{'logloss mdl':>11} {'blend':>7} {'market':>7}")
    groups = defaultdict(list)
    for r in rows:
        groups[(r["lead"], r["hour"])].append(r)
    summary = {"decision_skill": {}}
    for key in sorted(groups, key=lambda k: (-k[0], k[1])):
        g = groups[key]
        m = {k: float(np.mean([r[k] for r in g])) for k in
             ("brier_model", "brier_blend", "brier_market", "ll_model", "ll_blend", "ll_market")}
        summary["decision_skill"][f"{key[0]}d@{key[1]:02d}h"] = {"n": len(g), **m}
        print(f"  {key[0]:>4}d@{key[1]:02d}h {len(g):>5} {m['brier_model']:9.4f} {m['brier_blend']:7.4f} "
              f"{m['brier_market']:7.4f}   {m['ll_model']:11.4f} {m['ll_blend']:7.4f} {m['ll_market']:7.4f}")

    if not trades:
        print("\nno trades")
        return summary
    pnl = np.array([t["pnl"] for t in trades])
    cost = np.array([t["cost"] for t in trades])
    fees = sum(t["fee"] for t in trades)
    print(f"\ntrades {len(trades)}  staked ${cost.sum():,.0f}  fees ${fees:,.0f}  pnl ${pnl.sum():+,.0f}  "
          f"ROI {100 * pnl.sum() / cost.sum():+.1f}%  hit rate {100 * np.mean([t['won'] for t in trades]):.0f}%")
    by_day = defaultdict(float)
    for t in trades:
        by_day[t["date"]] += t["pnl"]
    daily = np.array([by_day[d] for d in sorted(by_day)])
    if len(daily) > 1 and daily.std() > 0:
        print(f"  daily pnl mean ${daily.mean():+.1f} sd ${daily.std():.1f}  "
              f"sharpe(ann.) {daily.mean() / daily.std() * math.sqrt(365):.2f}  "
              f"worst day ${daily.min():+.0f}  max drawdown ${_max_dd(np.cumsum(daily)):,.0f}")
    for label, keyf in (("decision", lambda t: f"{t['lead']}d@{t['hour']:02d}h"), ("side", lambda t: t["side"]),
                        ("kind", lambda t: t["kind"]), ("city", lambda t: t["city"]),
                        ("price", lambda t: _price_band(t["price"]))):
        g = defaultdict(list)
        for t in trades:
            g[keyf(t)].append(t)
        print(f"  by {label}:")
        for k in sorted(g):
            ts = g[k]
            c = sum(t["cost"] for t in ts)
            p = sum(t["pnl"] for t in ts)
            print(f"    {k:>14}: n={len(ts):4d} staked ${c:8,.0f} pnl ${p:+8,.0f} ROI {100 * p / max(c, 1e-9):+6.1f}%")
    summary.update(trades=len(trades), staked=float(cost.sum()), pnl=float(pnl.sum()), fees=float(fees))
    with open(out / "daily_pnl.csv", "w") as f:
        f.write("date,pnl,cum\n")
        cum = 0.0
        for d in sorted(by_day):
            cum += by_day[d]
            f.write(f"{d},{by_day[d]:.2f},{cum:.2f}\n")
    return summary


def _price_band(p):
    for hi, name in ((0.1, "<0.10"), (0.3, "0.10-0.30"), (0.5, "0.30-0.50"), (0.7, "0.50-0.70"), (0.9, "0.70-0.90")):
        if p < hi:
            return name
    return ">=0.90"


def _max_dd(cum):
    peak = np.maximum.accumulate(np.concatenate([[0.0], cum]))
    return float(np.max(peak[1:] - cum))


# ------------------------------------------------------------------ calibration / validation CLIs


def calibrate_nbm_archive(cal: Calibrator, cities, start: date, end: date, hour: int = 10, workers: int = 8):
    """Fit NBM bias/spread per US station, kind and lead from the IEM archive.

    Settlement flavour (routine-only vs all reports) follows the city's most
    recent resolved event, since that's what the next events will use.
    """
    cities = [c for c in (cities or US_CITIES) if STATIONS.get(c) and STATIONS[c].nbm]
    store = _NbmStore()
    days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
    for city in cities:
        st = STATIONS[city]
        recent = next((e for d in (end, end - timedelta(days=1), end - timedelta(days=2))
                       for e in [fetch_event(event_slug("high", city, d), cache=True)] if e is not None), None)
        hourly_only = bool(recent and recent.hourly_only)
        s, _ = day_window_utc(start, st.tz)
        _, e = day_window_utc(end, st.tz)
        obs = fetch_obs_iem(st.icao, s, e, st.unit, include_specials=not hourly_only, cache=True)
        for lead in (1, 2):
            store.prefetch([(st.icao, _asof(d, lead, hour, st.tz)) for d in days], workers)
            for kind in ("high", "low"):
                smp = []
                for d in days:
                    asof = _asof(d, lead, hour, st.tz)
                    rt, rows = store.at(st.icao, asof)
                    f = fc.nbm_day_forecast(rows, rt, st, d, kind) if rows else None
                    v = resolved_extreme(obs.window(*day_window_utc(d, st.tz)), kind)
                    if f is None or v is None:
                        continue
                    R, _ = remaining_extreme(f, SourceCalibration(), kind, None, asof.timestamp())
                    if R is not None:
                        smp.append((float(R.mean()), float(R.std()), v))
                c = cal.fit_group(city, kind, "nbm", lead, smp, st.unit, hourly_only=hourly_only)
                print(f"  {city:>14} {kind:4} nbm lead {lead}: n={c.n:3d} bias {c.bias:+.2f} "
                      f"kernel {c.kernel_sd:.2f} scale {c.spread_scale:.2f} w {c.weight:.2f}")
    return cal


def validate_resolutions(cities, days: int, workers: int = 8):
    """How often does max/min of IEM METARs land in Polymarket's winning bucket?"""
    cities = cities or [c for c, s in STATIONS.items() if s.source == "metar"]
    end = datetime.now(timezone.utc).date() - timedelta(days=2)
    dates = [end - timedelta(days=i) for i in range(days)]
    events = _load_events(cities, ("high", "low"), dates, workers)
    obs_all = _load_obs(events, min(dates), end)
    per = defaultdict(lambda: [0, 0, []])
    for ev in events:
        st = ev.station
        o = obs_all[(st.icao, not ev.hourly_only)].window(*day_window_utc(ev.target_date, st.tz))
        v = resolved_extreme(o, ev.kind)
        if v is None:
            continue
        k = ev.bucket_index_for(v)
        rec = per[(ev.city, ev.kind)]
        rec[1] += 1
        if k == ev.winner():
            rec[0] += 1
        else:
            rec[2].append(f"{ev.target_date} obs {v:g} -> {ev.buckets[k].label if k is not None else '?'}, "
                          f"won {ev.buckets[ev.winner()].label}")
    tot_ok = sum(r[0] for r in per.values())
    tot = sum(r[1] for r in per.values())
    for (city, kind), (ok, n, misses) in sorted(per.items()):
        print(f"  {city:>14} {kind:4}: {ok:3d}/{n:3d}" + (f"   misses: {'; '.join(misses[:3])}" if misses else ""))
    print(f"overall {tot_ok}/{tot} = {100 * tot_ok / max(tot, 1):.1f}%")
