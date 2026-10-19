"""CLI.

    python -m polyweather scan                 # model vs market for every open event, no orders
    python -m polyweather trade --loop 300     # paper trade every 5 min (default broker = paper)
    python -m polyweather trade --live         # real orders (also needs POLYWEATHER_LIVE=1, see execution.py)
    python -m polyweather settle               # settle resolved paper/live positions, record outcomes
    python -m polyweather calibrate --nbm-days 120 --from-ledger
    python -m polyweather validate --days 30   # do we reconstruct Polymarket's resolutions from METARs?
    python -m polyweather backtest --days 90   # US cities, NBM-as-issued vs Polymarket price history
"""

import argparse
from datetime import date, timedelta

from .calibration import Calibrator
from .engine import now_utc
from .ledger import Ledger
from .runner import RunConfig, Trader
from .strategy import StrategyConfig, plan_event


def _common(p):
    p.add_argument("--bankroll", type=float, default=1000.0)
    p.add_argument("--cities", nargs="*", default=None)
    p.add_argument("--max-lead", type=int, default=2)
    p.add_argument("--db", default="polyweather.db")
    p.add_argument("--calibration", default="polyweather/calibration.json")
    p.add_argument("--no-nbm", action="store_true")
    p.add_argument("--no-ensembles", action="store_true")
    p.add_argument("--model-weight", type=float, default=StrategyConfig.model_weight)
    p.add_argument("--kelly", type=float, default=StrategyConfig.kelly_fraction)
    p.add_argument("--min-edge", type=float, default=StrategyConfig.min_edge)
    p.add_argument("--min-price-yes", type=float, default=StrategyConfig.min_price_yes)


def _run_config(a) -> RunConfig:
    s = StrategyConfig(model_weight=a.model_weight, kelly_fraction=a.kelly, min_edge=a.min_edge,
                       min_price_yes=a.min_price_yes)
    return RunConfig(bankroll=a.bankroll, cities=a.cities, max_lead_days=a.max_lead,
                     use_nbm=not a.no_nbm, use_ensembles=not a.no_ensembles, strategy=s)


def cmd_scan(a):
    from .execution import PaperBroker
    cfg = _run_config(a)
    trader = Trader(cfg, Ledger(a.db), PaperBroker(), Calibrator(a.calibration))
    for e, books in trader.evaluate_all():
        ev, out = e.event, e.output
        if out is None:
            print(f"\n{ev.slug}: no forecast")
            continue
        srcs = ", ".join(f"{v.source}={v.center:.1f}±{v.spread:.1f}" for v in out.sources)
        obs = f" obs-{ev.kind}={out.observed_extreme:g}" if out.observed_extreme is not None else ""
        print(f"\n{ev.title} [{ev.station.icao}, lead {e.lead}]{obs}  E={out.expected:.1f}{ev.unit}  ({srcs})")
        plan = plan_event(ev, out.probs, books, cfg.bankroll, [], cfg.strategy,
                          observed_extreme=out.observed_extreme, final=out.final,
                          calibrated_frac=out.calibrated_frac)
        for i, b in enumerate(ev.buckets):
            yb, nb = books.get(b.yes_token), books.get(b.no_token)
            ya = yb.best_ask if yb else None
            ybid = yb.best_bid if yb else None
            na = nb.best_ask if nb else None
            flag = "".join(f" <- {o.action} {o.side} {o.shares:.0f}sh edge {o.edge:+.3f}"
                           for o in plan.orders if o.bucket == i)
            print(f"  {b.label:>16}  model {out.probs[i]:6.3f}  q {plan.q[i]:6.3f}  "
                  f"yes {_f(ybid)}/{_f(ya)}  no-ask {_f(na)}{flag}")


def _f(x):
    return f"{x:.3f}" if x is not None else "  -  "


def cmd_trade(a):
    from .execution import LiveBroker, PaperBroker
    broker = LiveBroker() if a.live else PaperBroker()
    trader = Trader(_run_config(a), Ledger(a.db), broker, Calibrator(a.calibration))
    if a.loop:
        trader.loop(a.loop)
    else:
        trader.settle()
        trader.cycle()


def cmd_settle(a):
    from .execution import PaperBroker
    trader = Trader(_run_config(a), Ledger(a.db), PaperBroker(), Calibrator(a.calibration))
    print(f"settled pnl {trader.settle():+.2f}; lifetime realized {trader.ledger.realized_pnl():+.2f}")


def cmd_calibrate(a):
    from .backtest import calibrate_nbm_archive
    cal = Calibrator(a.calibration)
    if a.nbm_days:
        end = now_utc().date() - timedelta(days=1)
        calibrate_nbm_archive(cal, a.cities, end - timedelta(days=a.nbm_days), end)
    if a.from_ledger:
        from .calibration import fit
        from .model import default_calibration
        from .stations import STATIONS
        groups = {}
        for city, kind, source, lead, center, spread, value in Ledger(a.db).forecast_samples():
            groups.setdefault((city, kind, source, lead), []).append((center, spread, value))
        for (city, kind, source, lead), samples in groups.items():
            st = STATIONS.get(city)
            if st is None:
                continue
            c = cal.fit_group(city, kind, source, lead, samples, st.unit, st.unit == "F")
            print(f"  {city:>14} {kind:4} {source:10} lead {lead}: n={c.n:3d} bias {c.bias:+.2f} "
                  f"kernel {c.kernel_sd:.2f} scale {c.spread_scale:.2f} w {c.weight:.2f}")
    cal.save(a.calibration)
    print(f"wrote {a.calibration} ({len(cal.table)} entries)")


def cmd_validate(a):
    from .backtest import validate_resolutions
    validate_resolutions(a.cities, a.days)


def cmd_backtest(a):
    from .backtest import BacktestConfig, run_backtest
    cfg = BacktestConfig(days=a.days, cities=a.cities, bankroll=a.bankroll,
                         strategy=StrategyConfig(model_weight=a.model_weight, kelly_fraction=a.kelly,
                                                 min_edge=a.min_edge, min_price_yes=a.min_price_yes),
                         end=date.fromisoformat(a.end) if a.end else None,
                         costs=a.costs, fills=a.fills, half_spread=a.half_spread, out_dir=a.out,
                         kinds=tuple(a.kinds), decisions=tuple(_decisions(a.decisions)))
    run_backtest(cfg)


def cmd_make(a):
    from .execution import LiveMakerBroker
    from .maker import MakerConfig
    from .maker_runner import MakerRunConfig, MakerRunner
    mcfg = MakerConfig(quote_shares=a.shares, min_edge=a.edge, max_side_inventory_usd=a.max_side,
                       max_event_inventory_usd=a.max_event)
    ledger = Ledger(a.db)
    broker = LiveMakerBroker() if a.live else None
    runner = MakerRunner(MakerRunConfig(maker=mcfg, cycle_s=a.cycle, cities=a.cities), ledger, broker)
    try:
        if a.once:
            runner.cycle()
        else:
            runner.loop()
    finally:
        if broker is not None:
            broker.cancel_all()


def cmd_maker_report(a):
    from .maker_runner import maker_report
    maker_report(Ledger(a.db), "live" if a.live else "paper")


def cmd_rewards(a):
    from collections import defaultdict
    from .markets import fetch_open_events
    from .orderbook import fetch_books
    from .rewards import fetch_reward_configs, snapshot_estimate
    evs = [e for e in fetch_open_events() if e.station is not None]
    cfgs = fetch_reward_configs()
    books = fetch_books([t for e in evs for b in e.buckets if b.condition_id in cfgs for t in (b.yes_token, b.no_token)])
    rows = snapshot_estimate(evs, books, cfgs, a.shares, a.edge)
    tot = defaultdict(lambda: [0.0, 0.0])
    for ev, lab, rate, mid, ours, comp, est in rows:
        k = f"{'US' if ev.station.nbm else 'intl'} {ev.kind}"
        tot[k][0] += est
        tot[k][1] += rate
    print(f"reward pools now: ${sum(v[1] for v in tot.values()):,.0f}/day on {len(rows)} buckets")
    for k, (est, pool) in sorted(tot.items()):
        print(f"  {k:10} pool ${pool:8,.0f}/day   est. our share quoting {a.shares:.0f} sh at mid±{a.edge}: ${est:7,.0f}/day")
    print("(snapshot: assumes the book stays as it is and we quote 100% of the time)")
    for ev, lab, rate, mid, ours, comp, est in sorted(rows, key=lambda r: -r[6])[:a.top]:
        print(f"  {ev.slug[:52]:52} {lab:>9} pool ${rate:4.0f} mid {mid:.3f} competing Q {comp:7.1f} -> ${est:5.1f}/day")


def cmd_pilot(a):
    from .pilot import PilotBroker, PilotConfig, PilotRunner, preflight
    cfg = PilotConfig.from_env(capital_usd=a.capital, cities=a.cities or None, db=a.pilot_db)
    ok, lines = preflight(cfg)
    print("preflight:")
    for line in lines:
        print("  " + line)
    if a.check:
        raise SystemExit(0 if ok else 1)
    if not ok:
        raise SystemExit("preflight failed: not starting")
    from pathlib import Path
    if Path(cfg.kill_file).exists():
        raise SystemExit(f"kill file {cfg.kill_file} exists ({Path(cfg.kill_file).read_text().strip()}); "
                         "delete it to start")
    PilotRunner(cfg, Ledger(cfg.db), PilotBroker()).run()


def cmd_pilot_report(a):
    from .pilot import PilotConfig, pilot_report
    pilot_report(Ledger(a.pilot_db or PilotConfig.from_env().db))


def cmd_maker_backtest(a):
    from .maker import MakerConfig
    from .maker_backtest import MakerBacktestConfig, run_maker_backtest
    cfg = MakerBacktestConfig(days=a.days, end=date.fromisoformat(a.end) if a.end else None, cities=a.cities,
                              kinds=tuple(a.kinds), use_model=not a.mid_only, out_dir=a.out,
                              maker=MakerConfig(quote_shares=a.shares, min_edge=a.edge,
                                                max_side_inventory_usd=a.max_side, max_event_inventory_usd=a.max_event))
    run_maker_backtest(cfg)


def _maker_args(p):
    p.add_argument("--shares", type=float, default=50.0, help="quote size per side per bucket")
    p.add_argument("--edge", type=float, default=0.01, help="bid this far below fair value")
    p.add_argument("--max-side", type=float, default=40.0, help="max $ inventory per bucket side")
    p.add_argument("--max-event", type=float, default=150.0, help="max $ inventory per event")


def _decisions(items):
    out = []
    for it in items:
        lead, hour = it.split("@")
        out.append((int(lead), int(hour)))
    return out


def main():
    ap = argparse.ArgumentParser(prog="polyweather")
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("scan"); _common(p); p.set_defaults(fn=cmd_scan)
    p = sub.add_parser("trade"); _common(p)
    p.add_argument("--live", action="store_true")
    p.add_argument("--loop", type=int, default=0, help="seconds between cycles (0 = run once)")
    p.set_defaults(fn=cmd_trade)
    p = sub.add_parser("settle"); _common(p); p.set_defaults(fn=cmd_settle)
    p = sub.add_parser("calibrate"); _common(p)
    p.add_argument("--nbm-days", type=int, default=0)
    p.add_argument("--from-ledger", action="store_true")
    p.set_defaults(fn=cmd_calibrate)
    p = sub.add_parser("validate"); _common(p)
    p.add_argument("--days", type=int, default=30)
    p.set_defaults(fn=cmd_validate)
    p = sub.add_parser("backtest"); _common(p)
    p.add_argument("--days", type=int, default=90)
    p.add_argument("--costs", choices=["measured", "conservative", "flat"], default="measured")
    p.add_argument("--fills", choices=["trades", "book"], default="trades",
                   help="trades: fill only against real taker prints after the decision")
    p.add_argument("--half-spread", type=float, default=0.01, help="with --costs flat")
    p.add_argument("--kinds", nargs="*", default=["high", "low"])
    p.add_argument("--decisions", nargs="*", default=["1@10", "1@18", "0@8", "0@12", "0@15"],
                   help="LEAD@LOCALHOUR decision points")
    p.add_argument("--out", default="backtest_out")
    p.add_argument("--end", default=None, help="last target date YYYY-MM-DD (default: 2 days ago)")
    p.set_defaults(fn=cmd_backtest)
    p = sub.add_parser("make", help="market-make rewarded buckets (paper unless --live)"); _common(p); _maker_args(p)
    p.add_argument("--live", action="store_true")
    p.add_argument("--cycle", type=int, default=60)
    p.add_argument("--once", action="store_true")
    p.set_defaults(fn=cmd_make)
    p = sub.add_parser("maker-report"); _common(p)
    p.add_argument("--live", action="store_true")
    p.set_defaults(fn=cmd_maker_report)
    p = sub.add_parser("rewards", help="live reward pools and our estimated share"); _common(p); _maker_args(p)
    p.add_argument("--top", type=int, default=15)
    p.set_defaults(fn=cmd_rewards)
    p = sub.add_parser("pilot", help="small real-money maker pilot with safety rails")
    p.add_argument("--check", action="store_true", help="preflight only")
    p.add_argument("--capital", type=float, default=None, help="override PILOT_CAPITAL_USD")
    p.add_argument("--cities", nargs="*", default=None)
    p.add_argument("--pilot-db", default=None)
    p.set_defaults(fn=cmd_pilot)
    p = sub.add_parser("pilot-report")
    p.add_argument("--pilot-db", default=None)
    p.set_defaults(fn=cmd_pilot_report)
    p = sub.add_parser("maker-backtest"); _common(p); _maker_args(p)
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--end", default=None)
    p.add_argument("--kinds", nargs="*", default=["high"])
    p.add_argument("--mid-only", action="store_true", help="fair value = market mid (no model)")
    p.add_argument("--out", default="maker_backtest_out")
    p.set_defaults(fn=cmd_maker_backtest)
    a = ap.parse_args()
    a.fn(a)


if __name__ == "__main__":
    main()
