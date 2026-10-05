"""
Walk-forward portfolio of several families traded on one account.

Each member is walk-forward re-optimized (anchored, see validate.py). Then, for
each $-risk level, every window's pick is re-simulated at that risk (integer
contracts, real per-contract costs) and only its test-window trades are kept.
Members' trades are merged; a shared account-level daily loss stop is applied
in time order (a member's trade is skipped once the day's realized loss hits
the stop). Lucid 50K eval simulated on the merged daily P&L.

  python -m trend_lab.portfolio --families trend_dip_atr,trend_dip_rsi --n 300
"""

import argparse
from datetime import datetime, timezone

from .data import load_bars
from .metrics import compute, daily_pnl, equity_points
from .optimize import load_registry, save_registry
from .sim import SimConfig, simulate
from .strategies import ALL_FAMILIES
from .validate import walk_forward

RISKS = (200, 300, 400, 500, 600)


def member_trades(fam, windows, risk, df):
    out, cache = [], {}
    for w in windows:
        k = repr(sorted(w["params"].items()))
        if k not in cache:
            cache[k] = simulate(df, fam.generate(df, w["params"]),
                                SimConfig(session=w["params"]["session"], risk_usd=risk,
                                          daily_loss_stop=2.25 * risk))
        te0, te1 = w["test"].split("..")
        out += [t for t in cache[k] if te0 <= str(t.trade_day) <= te1]
    return out


def merge(trade_lists, daily_loss_stop, exclusive=False):
    """exclusive=True: one position at a time (first signal wins), as a single Pine strategy trades."""
    allt = sorted((t for ts in trade_lists for t in ts), key=lambda t: t.entry_time)
    kept, day, day_pnl, busy_until = [], None, 0.0, None
    for t in allt:
        if t.trade_day != day:
            day, day_pnl = t.trade_day, 0.0
        if -day_pnl >= daily_loss_stop:
            continue
        if exclusive and busy_until is not None and t.entry_time < busy_until:
            continue
        busy_until = t.exit_time
        kept.append(t)
        day_pnl += t.pnl
    return sorted(kept, key=lambda t: t.exit_time)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", required=True)
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--select", default="best")
    ap.add_argument("--exclusive", action="store_true", help="one position at a time across members")
    args = ap.parse_args()
    names = args.families.split(",")
    df = load_bars("15min")

    wins = {}
    test_days = None
    for name in names:
        _, _, test_days, windows = walk_forward(ALL_FAMILIES[name], args.n, anchored=True, select=args.select)
        wins[name] = windows
        print(f"walk-forward done: {name}", flush=True)

    sweep = []
    for risk in RISKS:
        members = {n: member_trades(ALL_FAMILIES[n], wins[n], risk, df) for n in names}
        merged = merge(members.values(), daily_loss_stop=2.25 * risk, exclusive=args.exclusive)
        m = compute(merged, test_days)
        row = {"risk": risk, **{k: m.get(k) for k in ("trades", "win_rate", "profit_factor", "net", "max_dd",
                                                       "eod_dd", "trades_per_week", "lucid_attempts",
                                                       "lucid_pass_pct", "lucid_fail_pct", "lucid_median_days",
                                                       "best_day", "worst_day")}}
        row["members"] = {n: compute(ts, test_days).get("profit_factor") for n, ts in members.items()}
        row["daily"] = [round(float(v), 2) for v in daily_pnl(merged, test_days).to_numpy()]
        sweep.append((row, merged))
        print(f"risk ${risk}: { {k: v for k, v in row.items() if k != 'daily'} }", flush=True)

    tag = "portfolio:" + "+".join(names) + ("|exclusive" if args.exclusive else "")
    best_row, best_trades = max(sweep, key=lambda x: ((x[0]["lucid_pass_pct"] or 0) - 2 * (x[0]["lucid_fail_pct"] or 0)))
    reg = load_registry()
    reg.setdefault("validations", {})[tag] = {
        "tag": tag, "family": "+".join(names), "anchored": True, "select": args.select,
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "walk_forward": compute([t for t in sweep[0][1]], test_days),
        "equity": equity_points(sweep[0][1]), "start": str(test_days[0]), "end": str(test_days[-1]),
        "windows": {n: w for n, w in wins.items()}, "neighborhood": None,
        "risk_sweep": [r for r, _ in sweep], "chosen_risk": best_row["risk"],
    }
    save_registry(reg)


if __name__ == "__main__":
    main()
