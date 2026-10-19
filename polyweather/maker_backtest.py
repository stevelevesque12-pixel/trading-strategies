"""Backtest the market maker on resolved events: fills from real prints, settle at resolution.

Each event is quoted every `cycle_min` from `start_hour` local the day before
until the end of the target day. Fair value is the walk-forward NBM model
blended with the mid (US) or the mid alone (international, no forecast
archive). A resting bid fills only against taker trades that really
printed through it during the cycle (see maker.fills_from_prints).

Reward income is *not* in this PnL: the pools and the competing quotes at
the time aren't archived. The report gives the quoting uptime and the
reward score we'd have carried, and `python -m polyweather rewards`
estimates $/day from live books -- combine the two.
"""

import bisect
import csv
import math
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

import numpy as np

from . import forecast as fc
from .backtest import (CALIB_POINTS, _asof, _load_events, _load_hf, _load_obs, _load_prices, _load_trades,
                       _NbmStore)
from .calibration import fit, intraday_prior
from .maker import Inventory, MakerConfig, fills_from_prints, in_blackout, make_quotes, usual_obs_minutes
from .model import SourceCalibration, bucket_probabilities, default_calibration, remaining_extreme
from .observations import ObsState, day_window_utc, resolved_extreme
from .rewards import RewardConfig, our_q
from .stations import STATIONS


@dataclass
class MakerBacktestConfig:
    days: int = 60
    end: Optional[date] = None
    cities: Optional[list[str]] = None
    kinds: tuple = ("high",)
    cycle_min: int = 15
    start_hour: int = 8  # local hour on the day before the target day
    maker: MakerConfig = field(default_factory=MakerConfig)
    use_model: bool = True  # US: blend NBM model with mid; False: mid only everywhere
    max_price_age_h: float = 0.25
    obs_lag_min: float = 5.0
    reward_buckets: int = 3  # assume pools sit on the N buckets nearest the money (as observed live)
    workers: int = 8
    out_dir: str = "maker_backtest_out"


class _Prices:
    def __init__(self, hist):
        self.t = [h[0] for h in hist]
        self.p = [h[1] for h in hist]

    def at(self, ts: float, max_age: float) -> Optional[float]:
        i = bisect.bisect_right(self.t, ts) - 1
        if i < 0 or ts - self.t[i] > max_age:
            return None
        return self.p[i]


def _model_fair(ev, st, asof, nbm, samples, day_obs, day_hf, obs_cut, cache):
    """Walk-forward NBM model probabilities at `asof` (None if unavailable)."""
    runtime, rows = nbm.at(st.icao, asof)
    if not rows:
        return None
    key = (ev.slug, runtime)
    f = cache.get(key)
    if f is None:
        f = cache[key] = fc.nbm_day_forecast(rows, runtime, st, ev.target_date, ev.kind)
    if f is None:
        return None
    lead = (ev.target_date - asof.astimezone(ZoneInfo(st.tz)).date()).days
    cutoff = asof.astimezone(ZoneInfo(st.tz)).date() - timedelta(days=1)
    hist = [(c, s, o) for d, c, s, o in samples[(ev.city, ev.kind, 1)] if cutoff - timedelta(days=60) <= d < cutoff]
    cal = default_calibration("nbm", max(lead, 0), ev.unit, ev.kind, ev.hourly_only)
    if len(hist) >= 10:
        cal = fit(hist, default_calibration("nbm", 1, ev.unit, ev.kind, ev.hourly_only))
    obs_state = None
    if lead <= 0:
        obs_state = ObsState.from_obs(day_obs, ev.kind, obs_cut, day_hf)
        if obs_state.last_value is not None:
            hour = asof.astimezone(ZoneInfo(st.tz)).hour
            cal = intraday_prior(ev.kind, hour, ev.unit, "nbm")
    out = bucket_probabilities(ev, [(f, cal)], obs_state, asof.timestamp(), st.tz)
    return out


def run_maker_backtest(cfg: MakerBacktestConfig) -> dict:
    cities = cfg.cities or [c for c, s in STATIONS.items() if s.source == "metar"]
    end = cfg.end or (datetime.now(timezone.utc).date() - timedelta(days=2))
    start = end - timedelta(days=cfg.days - 1)
    dates = [start + timedelta(days=i) for i in range(cfg.days)]
    print(f"maker backtest {start} .. {end}, {len(cities)} cities, kinds {cfg.kinds}")
    events = _load_events(cities, cfg.kinds, dates, cfg.workers)
    print(f"  {len(events)} resolved events")
    obs_all = _load_obs(events, start - timedelta(days=3), end)
    us_events = [e for e in events if e.station.nbm]
    hf_all = _load_hf(us_events, start, end, cfg.workers) if (cfg.use_model and us_events) else {}
    nbm = _NbmStore()
    if cfg.use_model and us_events:
        jobs = []
        for e in us_events:
            t0 = _asof(e.target_date, 1, cfg.start_hour, e.station.tz)
            for h in range(0, 42, 6):
                jobs.append((e.station.icao, t0 + timedelta(hours=h)))
            jobs += [(e.station.icao, _asof(e.target_date, l, h, e.station.tz)) for l, h in CALIB_POINTS]
        nbm.prefetch(jobs, cfg.workers)
    prices = _load_prices(events, cfg.workers)
    trades_by = _load_trades(events, cfg.workers)
    print("  data loaded")

    events.sort(key=lambda e: e.target_date)
    samples = defaultdict(list)
    fc_cache: dict = {}
    fills, per_event = [], []
    for ev in events:
        st = ev.station
        win = day_window_utc(ev.target_date, st.tz)
        obs_series = obs_all[(st.icao, not ev.hourly_only)]
        day_obs = obs_series.window(*win)
        day_hf = hf_all[st.icao].window(*win) if st.icao in hf_all else None
        prior_obs = obs_series.window(win[0] - timedelta(days=3), win[0])
        obs_minutes = usual_obs_minutes(prior_obs.times, min_count=20)
        value = resolved_extreme(day_obs, ev.kind)
        winner = ev.winner()
        use_model = cfg.use_model and st.nbm

        if use_model and value is not None:
            for lead, hour in CALIB_POINTS:
                a = _asof(ev.target_date, lead, hour, st.tz)
                rt, rows = nbm.at(st.icao, a)
                f = fc.nbm_day_forecast(rows, rt, st, ev.target_date, ev.kind) if rows else None
                R = remaining_extreme(f, SourceCalibration(), ev.kind, None, a.timestamp())[0] if f else None
                if R is not None:
                    samples[(ev.city, ev.kind, lead)].append((ev.target_date, float(R.mean()), float(R.std()), value))

        px = {b.yes_token: _Prices(prices.get(b.yes_token, [])) for b in ev.buckets}
        trades = trades_by.get(ev.slug, [])
        inv = Inventory()
        t = _asof(ev.target_date, 1, cfg.start_hour, st.tz)
        stop = win[1]
        n_cycles = n_quoted = 0
        q_score = 0.0
        while t < stop:
            t1 = t + timedelta(minutes=cfg.cycle_min)
            n_cycles += 1
            mids = [px[b.yes_token].at(t.timestamp(), cfg.max_price_age_h * 3600) for b in ev.buckets]
            lead = (ev.target_date - t.astimezone(ZoneInfo(st.tz)).date()).days
            # Pull quotes around each expected reading on the target day (and the one before
            # midnight on the eve, which can set a "lowest" market).
            if (lead <= 0 or t1 > win[0] - timedelta(hours=1)) and in_blackout(t, st.tz, obs_minutes, cfg.maker):
                t = t1
                continue
            if sum(m is not None for m in mids) < len(mids) - 2:
                t = t1
                continue
            fair = list(mids)
            if use_model:
                snap = min((px[b.yes_token].t[bisect.bisect_right(px[b.yes_token].t, t.timestamp()) - 1]
                            for b in ev.buckets if px[b.yes_token].t and px[b.yes_token].t[0] <= t.timestamp()),
                           default=t.timestamp())
                obs_cut = datetime.fromtimestamp(min(snap, t.timestamp()) - cfg.obs_lag_min * 60, timezone.utc)
                out = _model_fair(ev, st, t, nbm, samples, day_obs, day_hf, obs_cut, fc_cache)
                if out is not None:
                    w = cfg.maker.model_weight
                    fair = [None if m is None else w * float(p) + (1 - w) * m for p, m in zip(out.probs, mids)]
            # Reward pools: assume the N buckets nearest the money, as seen on live markets.
            ranked = sorted((i for i, m in enumerate(mids) if m is not None), key=lambda i: -mids[i])
            rewards = {ev.buckets[i].condition_id: RewardConfig(ev.buckets[i].condition_id, 0.0, 0.045, 20.0)
                       for i in ranked[:cfg.reward_buckets]}
            quotes = make_quotes(ev, fair, mids, rewards, inv, cfg.maker)
            if quotes:
                n_quoted += 1
            by_bucket = defaultdict(dict)
            for q in quotes:
                by_bucket[q.bucket][q.side] = q
                got = fills_from_prints(q, ev, trades, t.timestamp(), t1.timestamp())
                if got <= 0:
                    continue
                inv.add(q.side, q.bucket, got, got * q.price)
                won = (q.bucket == winner) if q.side == "YES" else (q.bucket != winner)
                later = px[ev.buckets[q.bucket].yes_token].at(t.timestamp() + 3600, 6 * 3600)
                if later is not None and q.side == "NO":
                    later = 1 - later
                fills.append({
                    "slug": ev.slug, "city": ev.city, "kind": ev.kind, "us": int(st.nbm), "lead": lead,
                    "hour": t.astimezone(ZoneInfo(st.tz)).hour, "time": t.isoformat(), "bucket": ev.buckets[q.bucket].label,
                    "side": q.side, "price": q.price, "shares": round(got, 2), "fair": round(q.fair, 4),
                    "mark_1h": None if later is None else round(later, 4), "won": int(won),
                    "pnl": round((1.0 if won else 0.0) * got - got * q.price, 4),
                })
            for i, sides in by_bucket.items():
                m = mids[i]
                yb = sides.get("YES")
                nb = sides.get("NO")
                rc = rewards.get(ev.buckets[i].condition_id)
                if rc and m is not None:
                    q_score += our_q((yb.price, yb.shares) if yb else None, (nb.price, nb.shares) if nb else None, m, rc)
            t = t1
        payout = inv.payout(winner) if winner is not None else 0.0
        per_event.append({"slug": ev.slug, "city": ev.city, "kind": ev.kind, "us": int(st.nbm),
                          "date": ev.target_date.isoformat(), "cycles": n_cycles, "quoted": n_quoted,
                          "cost": round(inv.total_cost(), 2), "pnl": round(payout - inv.total_cost(), 2),
                          "q_score_cycles": round(q_score, 1)})
    return _report(cfg, fills, per_event)


def _report(cfg, fills, per_event) -> dict:
    out = Path(cfg.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("fills.csv", fills), ("events.csv", per_event)):
        if rows:
            with open(out / name, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0]))
                w.writeheader()
                w.writerows(rows)
    n_ev = len(per_event)
    days = len({e["date"] for e in per_event}) or 1
    pnl = sum(e["pnl"] for e in per_event)
    vol = sum(f["price"] * f["shares"] for f in fills)
    uptime = sum(e["quoted"] for e in per_event) / max(sum(e["cycles"] for e in per_event), 1)
    print(f"\nevents {n_ev} over {days} days; quoting uptime {100 * uptime:.0f}% of cycles")
    print(f"fills {len(fills)}  bought ${vol:,.0f}  trading pnl ${pnl:+,.0f}  "
          f"({100 * pnl / max(vol, 1e-9):+.1f}% of volume, ${pnl / days:+.1f}/day)")
    if fills:
        mk = [f for f in fills if f["mark_1h"] is not None]
        edge_fair = np.mean([f["fair"] - f["price"] for f in fills])
        edge_mark = np.mean([f["mark_1h"] - f["price"] for f in mk]) if mk else float("nan")
        print(f"  avg edge at fill vs our fair {edge_fair:+.3f}/share; vs mid 1h later {edge_mark:+.3f}/share "
              f"(adverse selection = the drop between the two)")
    for label, keyf in (("region", lambda r: "US" if r["us"] else "intl"), ("kind", lambda r: r["kind"]),
                        ("lead", lambda r: r["lead"]), ("side", lambda r: r["side"]),
                        ("hour", lambda r: r["hour"] if r["lead"] <= 0 else -1)):
        g = defaultdict(list)
        for f in fills:
            g[keyf(f)].append(f)
        print(f"  fills by {label}:")
        for k in sorted(g):
            fs = g[k]
            v = sum(f["price"] * f["shares"] for f in fs)
            p = sum(f["pnl"] for f in fs)
            mk = [f["mark_1h"] - f["price"] for f in fs if f["mark_1h"] is not None]
            print(f"    {str(k):>5}: n={len(fs):5d} bought ${v:8,.0f} pnl ${p:+8,.0f} ({100 * p / max(v, 1e-9):+6.1f}%)"
                  f"  1h markout {np.mean(mk) if mk else float('nan'):+.3f}")
    g = defaultdict(lambda: [0.0, 0.0])
    for e in per_event:
        g[e["city"]][0] += e["pnl"]
        g[e["city"]][1] += e["cost"]
    print("  pnl by city:", ", ".join(f"{c} {p:+.0f}" for c, (p, _) in sorted(g.items(), key=lambda x: x[1][0])))
    return {"events": n_ev, "fills": len(fills), "volume": vol, "pnl": pnl, "uptime": uptime, "days": days}
