"""
Prop-firm variant of the Step Range Breakout: long-only, intraday-only,
risk-sized micros, daily loss stop -- plus a rolling evaluation simulator.

Trading rules:
- Indicator logic unchanged (`step_range.strategy`); only bullish breakouts
  are taken, and only when the breakout bar CLOSES inside the entry window.
- Entry at the next bar's open; exit on a resting stop at the trail, or at
  the next open after a close through the trail, or flat at `flatten_at`
  (whichever first). Never held overnight.
- Size = floor(risk_per_trade / (stop distance * point value)), capped at
  `max_contracts`; a setup whose stop is too wide for 1 contract is skipped.
- Personal daily stop (`daily_loss`) and max trades per day, via the
  shared `failed2s.risk.RiskManager`.
- `--rth-bars` computes the indicator on session bars only (overnight bars
  dropped), instead of on the full 23h series.

Evaluation simulator (defaults ~ a Topstep-style 50K combine): starting on
every trading day in the data, trade forward until the balance reaches
+target (with at least `min_days` trading days) = PASS, or touches the
trailing max-loss line = FAIL. The line trails the highest end-of-day
balance by `max_drawdown` and stops trailing once it reaches the starting
balance. The FAIL check uses each trade's worst open drawdown (MAE), not
just closed P&L. Starts too close to the end of the data that neither
pass nor fail are reported as `unresolved`.

Usage:
    python -m step_range.prop --symbols MNQ,MGC --tf 15min,1h
"""

import argparse
from dataclasses import dataclass
from datetime import time
from typing import List, Optional

import numpy as np
import pandas as pd

from failed2s.instruments import INSTRUMENTS, Instrument
from failed2s.risk import RiskManager

from .backtest import DATA_SYMBOL, load_bars
from .strategy import StepRangeParams, find_breakouts

# Micros trade at the full-size contract's price; reuse the 10-year files.
PRICE_SOURCE = {"MNQ": "NQ", "MES": "ES", "MGC": "GC", "MCL": "CL"}


def _t(s: str) -> time:
    return time.fromisoformat(s)


@dataclass(frozen=True)
class PropConfig:
    entry_start: time = time(9, 30)   # breakout bar must close within [start, end]
    entry_end: time = time(15, 0)
    flatten_at: time = time(15, 45)   # exit at the open of the first bar starting >= this
    rth_start: time = time(9, 30)     # session used by --rth-bars
    rth_end: time = time(16, 0)
    rth_bars: bool = False
    risk_per_trade: float = 400.0
    max_contracts: int = 20
    daily_loss: float = 500.0
    max_trades_per_day: int = 3
    slip_ticks: float = 1.0
    commission: float = 1.0           # round trip per micro contract


@dataclass
class PropTrade:
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    contracts: int
    entry_price: float
    exit_price: float
    stop_price: float
    exit_reason: str
    pnl_dollars: float
    mae_dollars: float   # worst open P&L during the trade (<= 0), after costs
    r_multiple: float


def run_prop(df: pd.DataFrame, inst: Instrument, params: StepRangeParams, cfg: PropConfig,
             stats: Optional[dict] = None) -> List[PropTrade]:
    """`stats`, if given, gets counts of signals taken vs skipped for sizing."""
    stats = {} if stats is None else stats
    stats.update(signals=0, too_wide=0)
    if cfg.rth_bars:
        t = df.index.time
        df = df[(t >= cfg.rth_start) & (t < cfg.rth_end)]
    bar_len = df.index.to_series().diff().mode().iloc[0]
    o, h, lo = (df[c].to_numpy(float) for c in ("open", "high", "low"))
    idx = df.index
    n = len(df)
    slip = cfg.slip_ticks * inst.tick_size
    risk = RiskManager(daily_loss_limit=cfg.daily_loss, max_daily_trades=cfg.max_trades_per_day)

    trades: List[PropTrade] = []
    for b in find_breakouts(df, params):
        if b.direction != "long":
            continue
        s = b.signal_idx
        if s + 1 >= n:
            break
        closes_at = idx[s] + bar_len
        e = s + 1
        day = idx[e].date()
        if not (cfg.entry_start <= closes_at.time() <= cfg.entry_end):
            continue
        if idx[e].date() != closes_at.date() or idx[e].time() >= cfg.flatten_at:
            continue
        if not risk.can_enter(day):
            continue
        stats["signals"] += 1

        entry = o[e] + slip
        stop_dist = entry - b.trail[0]
        if stop_dist <= 0:
            continue
        qty = min(cfg.max_contracts, int(cfg.risk_per_trade // (stop_dist * inst.point_value)))
        if qty < 1:
            stats["too_wide"] += 1
            continue

        exit_px: Optional[float] = None
        worst = entry
        last = b.exit_signal_idx if b.exit_signal_idx >= 0 else n - 1
        j = e
        while j < n:
            if idx[j].date() != day or (j > e and idx[j].time() >= cfg.flatten_at):
                exit_px, reason = o[j] - slip, "flatten"
                break
            k = j - 1 - s
            stop = b.trail[k] if k < len(b.trail) else b.trail[-1]
            if lo[j] <= stop:
                exit_px, reason = min(o[j], stop) - slip, "stop"
                break
            worst = min(worst, lo[j])
            if j > last:  # the previous bar closed through the trail
                exit_px, reason = o[j] - slip, "trail_close"
                break
            j += 1
        if exit_px is None:
            break  # ran off the end of the data mid-trade
        worst = min(worst, exit_px)

        cost = cfg.commission * qty
        pnl = (exit_px - entry) * inst.point_value * qty - cost
        trades.append(PropTrade(
            entry_time=idx[e], exit_time=idx[j], contracts=qty, entry_price=entry, exit_price=exit_px,
            stop_price=b.trail[0], exit_reason=reason, pnl_dollars=pnl,
            mae_dollars=min(0.0, (worst - entry) * inst.point_value * qty) - cost,
            r_multiple=(exit_px - entry) / stop_dist,
        ))
        risk.record_trade(day, pnl)
    return trades


@dataclass(frozen=True)
class EvalRules:
    start_balance: float = 50_000.0
    profit_target: float = 3_000.0
    max_drawdown: float = 2_000.0
    min_days: int = 2


def simulate_evals(trades: List[PropTrade], trading_days: pd.Index, rules: EvalRules) -> pd.DataFrame:
    """One evaluation per starting trading day; returns outcome + days used."""
    by_day: dict = {}
    for t in trades:
        by_day.setdefault(t.entry_time.date(), []).append(t)
    days = list(trading_days)
    out = []
    for start_i in range(len(days)):
        bal = peak_eod = rules.start_balance
        mll = bal - rules.max_drawdown
        outcome, used = "unresolved", 0
        for d in days[start_i:]:
            used += 1
            for t in by_day.get(d, []):
                if bal + t.mae_dollars <= mll:
                    outcome = "fail"
                    break
                bal += t.pnl_dollars
            if outcome == "fail":
                break
            if bal - rules.start_balance >= rules.profit_target and used >= rules.min_days:
                outcome = "pass"
                break
            peak_eod = max(peak_eod, bal)
            mll = max(mll, min(peak_eod - rules.max_drawdown, rules.start_balance))
        out.append((days[start_i], outcome, used))
    return pd.DataFrame(out, columns=["start", "outcome", "days"])


def summarize(trades: List[PropTrade], evals: pd.DataFrame, stats: dict) -> dict:
    skipped = round(stats["too_wide"] / stats["signals"] * 100) if stats.get("signals") else 0
    if not trades:
        return {"trades": 0, "skipped_pct": skipped}
    pnl = np.array([t.pnl_dollars for t in trades])
    t = pd.DataFrame([vars(x) for x in trades])
    daily = t.groupby(t.entry_time.dt.date)["pnl_dollars"].sum()
    eq = np.cumsum(pnl)
    yearly = t.groupby(t.exit_time.dt.year)["pnl_dollars"].sum()
    passes = evals[evals.outcome == "pass"]
    share = evals.outcome.value_counts(normalize=True).mul(100).round(1)
    return {
        "trades": len(trades),
        "skipped_pct": skipped,
        "trades_per_wk": round(len(trades) / max(1, (t.entry_time.iloc[-1] - t.entry_time.iloc[0]).days / 7), 2),
        "win_pct": round((pnl > 0).mean() * 100, 1),
        "pf": round(pnl[pnl > 0].sum() / -pnl[pnl <= 0].sum(), 2),
        "net": round(pnl.sum()),
        "max_dd": round((np.maximum.accumulate(eq) - eq).max()),
        "avg_qty": round(t.contracts.mean(), 1),
        "worst_day": round(daily.min()),
        "flatten_pct": round((t.exit_reason == "flatten").mean() * 100),
        "years_pos": f"{int((yearly > 0).sum())}/{len(yearly)}",
        "pass_pct": share.get("pass", 0.0),
        "fail_pct": share.get("fail", 0.0),
        "open_pct": share.get("unresolved", 0.0),
        "median_days_to_pass": int(passes.days.median()) if len(passes) else None,
        "yearly": yearly.round(0).to_dict(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--symbols", default="MNQ,MGC")
    ap.add_argument("--tf", default="15min")
    ap.add_argument("--entry-window", default="09:30-15:00", help="HH:MM-HH:MM ET; comma-separate to compare")
    ap.add_argument("--flatten-at", default="15:45")
    ap.add_argument("--rth-bars", default="no", help="Compute indicator on session bars only: no, yes, or both")
    ap.add_argument("--risk", type=float, default=400.0, help="$ risk per trade")
    ap.add_argument("--max-contracts", type=int, default=20)
    ap.add_argument("--daily-loss", type=float, default=500.0)
    ap.add_argument("--max-trades-per-day", type=int, default=3)
    ap.add_argument("--slip-ticks", type=float, default=1.0)
    ap.add_argument("--commission", type=float, default=1.0, help="Round trip per micro")
    ap.add_argument("--length", type=int, default=20)
    ap.add_argument("--consolidation-bars", type=int, default=5)
    ap.add_argument("--atr-length", type=int, default=14)
    ap.add_argument("--atr-mult", type=float, default=2.0)
    ap.add_argument("--account", type=float, default=50_000.0)
    ap.add_argument("--target", type=float, default=3_000.0)
    ap.add_argument("--max-dd", type=float, default=2_000.0, help="Trailing (end-of-day) max loss")
    ap.add_argument("--min-days", type=int, default=2)
    ap.add_argument("--start", default=None, help="Only count trades/evals from this date")
    ap.add_argument("--yearly", action="store_true")
    ap.add_argument("--trades-out", default=None)
    args = ap.parse_args()

    params = StepRangeParams(args.length, args.consolidation_bars, args.atr_length, args.atr_mult)
    rules = EvalRules(args.account, args.target, args.max_dd, args.min_days)
    rows, all_trades = [], []
    for sym in [s.strip().upper() for s in args.symbols.split(",")]:
        inst = INSTRUMENTS[sym]
        src = PRICE_SOURCE.get(sym, sym)
        if src not in DATA_SYMBOL:
            raise SystemExit(f"No price data mapped for {sym}")
        for tf in args.tf.split(","):
            df = load_bars(src, tf)
            for window in args.entry_window.split(","):
                ws, we = window.split("-")
                for rth in args.rth_bars.split(","):
                    cfg = PropConfig(
                        entry_start=_t(ws), entry_end=_t(we), flatten_at=_t(args.flatten_at),
                        rth_bars=rth == "yes", risk_per_trade=args.risk, max_contracts=args.max_contracts,
                        daily_loss=args.daily_loss, max_trades_per_day=args.max_trades_per_day,
                        slip_ticks=args.slip_ticks, commission=args.commission,
                    )
                    stats: dict = {}
                    trades = run_prop(df, inst, params, cfg, stats)
                    days = pd.Index(sorted({d for d in df.index.date if d.weekday() < 5}))
                    if args.start:
                        cutoff = pd.Timestamp(args.start).date()
                        trades = [t for t in trades if t.entry_time.date() >= cutoff]
                        days = days[days >= cutoff]
                    evals = simulate_evals(trades, days, rules)
                    m = summarize(trades, evals, stats)
                    rows.append({"symbol": sym, "tf": tf, "window": window, "rth_bars": rth, **m})
                    all_trades += [{"symbol": sym, "tf": tf, "window": window, "rth_bars": rth, **vars(t)} for t in trades]

    table = pd.DataFrame(rows).drop(columns="yearly", errors="ignore")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(table.to_string(index=False))
    if args.yearly:
        print()
        for r in rows:
            print(f"{r['symbol']:>4} {r['tf']:>5} {r['window']} rth={r['rth_bars']:<3} " +
                  "  ".join(f"{y}:{v:+,.0f}" for y, v in r.get("yearly", {}).items()))
    if args.trades_out:
        pd.DataFrame(all_trades).to_csv(args.trades_out, index=False)
        print(f"\nTrades: {args.trades_out}")


if __name__ == "__main__":
    main()
