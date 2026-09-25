"""
Prop-firm evaluation simulator for the TR1 breakout.

Replays the backtest's trade stream from every possible start day and checks
whether a typical futures-prop evaluation would pass or fail. Defaults model
a generic 50K account (the common shape across Topstep/Apex/TPT/etc. -- the
exact numbers differ by firm and change often, so set them to yours):

    +$3,000 target, $2,000 end-of-day trailing max loss that stops trailing
    once it reaches the starting balance, optional daily loss limit, and a
    max position size in MNQ.

    python -m tr_breakout.prop_sim --data <parquet> --risk 200 250 300 400 --from 2022-01-01

Limitations: breaches are checked on closed-trade balance, not open P&L, so
an intraday trailing drawdown (Apex-style) is looser here than in reality.
"""

import argparse
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from backtest.data import load_1m_csv
from tr_breakout.strategy import Config, run_backtest, trades_frame


@dataclass
class PropRules:
    account: float = 50_000
    target: float = 3_000
    max_loss: float = 2_000          # trailing, measured from the highest EOD balance
    lock_at_start: bool = True       # trailing floor stops at the starting balance
    daily_loss: Optional[float] = None
    max_contracts: int = 50          # e.g. 5 NQ minis = 50 MNQ
    max_days: Optional[int] = None   # None = no time limit


def daily_pnl_paths(trades: pd.DataFrame):
    """Per day: list of cumulative closed-trade P&L after each trade."""
    return {d: g["pnl"].cumsum().tolist() for d, g in trades.groupby("day")}


def run_eval(days, paths, start_idx, rules: PropRules):
    """Returns (result, trading_days_used). result in {'pass','fail','open'}."""
    bal = rules.account
    peak_eod = bal
    used = 0
    for d in days[start_idx:]:
        if rules.max_days is not None and used >= rules.max_days:
            return "open", used
        floor_ = peak_eod - rules.max_loss
        if rules.lock_at_start:
            floor_ = min(floor_, rules.account)
        day_pnl = 0.0
        if d in paths:
            used += 1
            for cum in paths[d]:
                day_pnl = cum
                if bal + day_pnl <= floor_:
                    return "fail", used
                if rules.daily_loss is not None and day_pnl <= -rules.daily_loss:
                    break  # day halted; later trades that day never happen
        bal += day_pnl
        peak_eod = max(peak_eod, bal)
        if bal >= rules.account + rules.target:
            return "pass", used
    return "open", used


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--risk", type=float, nargs="+", default=[200, 250, 300, 400, 500])
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--target", type=float, default=3000)
    ap.add_argument("--max-loss", type=float, default=2000)
    ap.add_argument("--daily-loss", type=float, default=None)
    ap.add_argument("--max-contracts", type=int, default=50)
    args = ap.parse_args()

    df = load_1m_csv(args.data)
    rules = PropRules(target=args.target, max_loss=args.max_loss,
                      daily_loss=args.daily_loss, max_contracts=args.max_contracts)
    rows = []
    for risk in args.risk:
        cfg = Config(start_equity=rules.account, risk_pct=risk / rules.account,
                     max_contracts=rules.max_contracts)
        tf = trades_frame(run_backtest(df, cfg))
        if args.start:
            tf = tf[pd.to_datetime(tf["day"]) >= args.start]
        all_days = sorted(set(pd.to_datetime(df.index.date)))
        all_days = [d.date() for d in all_days if not args.start or d >= pd.Timestamp(args.start)]
        paths = daily_pnl_paths(tf)
        res = [run_eval(all_days, paths, i, rules) for i in range(len(all_days))]
        done = [r for r in res if r[0] != "open"]
        passes = [u for r, u in done if r == "pass"]
        fails = [u for r, u in done if r == "fail"]
        traded_days = tf["day"].nunique()
        rows.append({
            "risk_$": risk,
            "days_skipped_%": round(100 * (1 - traded_days / len(all_days)), 1),
            "avg_contracts": round(tf["contracts"].mean(), 1),
            "pass_%": round(100 * len(passes) / len(done), 1) if done else None,
            "median_days_to_pass": int(pd.Series(passes).median()) if passes else None,
            "median_days_to_fail": int(pd.Series(fails).median()) if fails else None,
            "evals_simulated": len(done),
        })
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
