"""
CLI: simulate whether the Fair Value Theory strategy would have passed a
prop-firm evaluation within N trading days, across every historical window
in the data -- not just a single point-in-time backtest.

Usage:
    python -m backtest.run_prop_compliance --data sample_data/real_nq_1min_2022_2025.csv \
        --symbol NQ --windows AM --target-risk 250 --eval-days 10

See backtest/prop_compliance.py's module docstring: the built-in account
profiles are illustrative/generic, not verified terms for any specific real
firm. Pass --target-risk to test different position sizings against them.
"""

import argparse

from failed2s.instruments import INSTRUMENTS
from fair_value.strategy import DEFAULT_WINDOWS, FairValueStrategy

from .fair_value_engine import FairValueBacktestEngine
from .prop_compliance import GENERIC_PROFILES, rolling_eval_pass_rate

WINDOW_CHOICES = [w.name for w in DEFAULT_WINDOWS]


def main() -> None:
    parser = argparse.ArgumentParser(description="Simulate prop-firm eval pass rate over rolling historical windows")
    parser.add_argument("--data", required=True)
    parser.add_argument("--symbol", default="NQ", choices=list(INSTRUMENTS.keys()))
    parser.add_argument("--target-risk", type=float, default=1000.0, help="Target $ risk per trade for contract sizing")
    parser.add_argument("--min-contracts", type=int, default=1)
    parser.add_argument("--max-contracts", type=int, default=3)
    parser.add_argument("--fixed-contracts", type=int, default=None)
    parser.add_argument("--eval-days", type=int, default=10, help="Evaluation window length in trading days (10 ~= 2 calendar weeks)")
    parser.add_argument("--windows", default=",".join(WINDOW_CHOICES), help=f"Choices: {','.join(WINDOW_CHOICES)}")
    parser.add_argument("--restrict-to-first-hour", action="store_true")
    parser.add_argument("--commission-per-contract-rt", type=float, default=9.0, help="Round-turn commission per contract ($)")
    parser.add_argument("--slippage-ticks-rt", type=float, default=2.0, help="Round-turn slippage in ticks per contract")
    args = parser.parse_args()

    window_names = [w.strip() for w in args.windows.split(",")]
    windows = [w for w in DEFAULT_WINDOWS if w.name in window_names]

    instrument = INSTRUMENTS[args.symbol]
    strategy = FairValueStrategy(tick_size=instrument.tick_size, windows=windows, restrict_to_first_hour=args.restrict_to_first_hour)
    engine = FairValueBacktestEngine(
        instrument=instrument,
        strategy=strategy,
        target_risk=args.target_risk,
        min_contracts=args.min_contracts,
        max_contracts=args.max_contracts,
        fixed_contracts=args.fixed_contracts,
        commission_per_contract_rt=args.commission_per_contract_rt,
        slippage_ticks_rt=args.slippage_ticks_rt,
        track_equity=True,
    )
    engine.run(args.data)

    if not engine.equity_curve:
        raise SystemExit("No bars processed -- check --data path")

    print(f"Symbol: {args.symbol}  Windows: {window_names}  Target risk/trade: ${args.target_risk:,.0f}")
    print(f"Costs modeled: ${args.commission_per_contract_rt:.2f}/contract commission + {args.slippage_ticks_rt:.1f} ticks slippage (round-turn)")
    print(f"Eval window: {args.eval_days} trading days\n")

    header = f"{'Profile':<48} {'windows':>8} {'pass%':>7} {'breach%':>8} {'neither%':>9} {'avg days to pass':>17}"
    print(header)
    print("-" * len(header))
    for profile in GENERIC_PROFILES:
        report = rolling_eval_pass_rate(engine.equity_curve, profile, eval_trading_days=args.eval_days)
        avg_days = f"{report['avg_days_to_pass']}" if report["avg_days_to_pass"] is not None else "n/a"
        print(
            f"{profile.name:<48} {report['num_windows']:>8} {report['pass_rate_pct']:>6.1f}% "
            f"{report['breach_rate_pct']:>7.1f}% {report['neither_pct']:>8.1f}% {avg_days:>17}"
        )


if __name__ == "__main__":
    main()
