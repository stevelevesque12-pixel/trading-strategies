"""
Backtest CrudePut30 on WTI futures bars, with an implied-vol sensitivity sweep.

Usage:
    python -m crude_options.run --data sample_data/real_multi_instrument/real_mcl_15m_2025-09-30_2026-08-25.parquet
"""

import argparse
from dataclasses import asdict, replace

import pandas as pd

from backtest.data import load_1m_csv
from backtest.metrics import compute_metrics

from .put30 import Put30Config, run_put30


def main() -> None:
    p = argparse.ArgumentParser(description="Backtest the CrudePut30 short-put strategy")
    p.add_argument("--data", required=True)
    p.add_argument("--multiplier", type=float, default=1000.0, help="bbl per option: LO 1000, Micro WTI 100")
    p.add_argument("--iv-mults", default="0.9,1.0,1.2,1.4")
    p.add_argument("--stop-mult", type=float, default=None)
    p.add_argument("--out", default="put30_trades.csv")
    args = p.parse_args()

    bars = load_1m_csv(args.data)
    base = Put30Config(multiplier=args.multiplier, stop_mult=args.stop_mult)

    rows = []
    for m in [float(x) for x in args.iv_mults.split(",")]:
        trades = run_put30(bars, replace(base, iv_mult=m))
        rows.append({"iv_mult": m, **compute_metrics(trades)})
        if m == 1.0:
            pd.DataFrame([asdict(t) for t in trades]).to_csv(args.out, index=False)
            print(pd.DataFrame([asdict(t) for t in trades])[
                ["entry_time", "expiry", "strike", "entry_fut", "exit_fut", "credit", "exit_price", "reason", "pnl_dollars"]
            ].to_string(index=False))
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
