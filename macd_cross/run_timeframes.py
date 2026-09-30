"""
CLI: test MACD strategies (1: signal-line cross, 2: zero-line cross) across timeframes.

Each timeframe is built from the finest real MES file that can produce it:
  1m, 2m, 3m, 4m   <- 1m file  (2026-08-02 .. 2026-08-25)
  5m, 10m          <- 5m file  (2026-05-10 .. 2026-08-25)
  15m              <- 15m file (2025-09-30 .. 2026-08-25)
Sub-minute timeframes (15s/30s/45s) need sub-minute source data; if no
`--seconds-data` file is given they are reported as untestable instead of
being faked from 1m bars.

`--common-window` instead runs every minute timeframe on the 1m file only,
so all rows cover the same ~3.5 weeks (apples-to-apples).

Usage:
    python -m macd_cross.run_timeframes
    python -m macd_cross.run_timeframes --common-window --mode rth
"""

import argparse
from pathlib import Path

import pandas as pd

from backtest.data import load_1m_csv, resample_ohlc
from failed2s.instruments import INSTRUMENTS

from .backtest import CostModel, SessionRules, metrics, run_backtest
from .strategy import SIGNAL_RULES, MACDParams

DATA_DIR = Path(__file__).resolve().parents[1] / "sample_data" / "real_multi_instrument"
SOURCES = {
    "1min": "real_{sym}_1m_2026-08-02_2026-08-25.parquet",
    "5min": "real_{sym}_5m_2026-05-10_2026-08-25.parquet",
    "15min": "real_{sym}_15m_2025-09-30_2026-08-25.parquet",
}
TIMEFRAMES = ["15s", "30s", "45s", "1min", "2min", "3min", "4min", "5min", "10min", "15min"]


def pick_source(tf: pd.Timedelta, common: bool) -> str:
    if common:
        return "1min"
    for src in ("15min", "5min", "1min"):
        td = pd.Timedelta(src)
        if tf >= td and tf % td == pd.Timedelta(0):
            return src
    raise ValueError(tf)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--symbol", default="MES")
    p.add_argument("--timeframes", default=",".join(TIMEFRAMES))
    p.add_argument("--mode", choices=["24h", "rth"], default="24h")
    p.add_argument("--common-window", action="store_true")
    p.add_argument("--seconds-data", help="Optional sub-minute OHLCV file (CSV/parquet) for 15s/30s/45s")
    p.add_argument("--rule", choices=list(SIGNAL_RULES), default="signal_cross",
                   help="signal_cross = Strategy 1; zero_cross / zero_cross_hist = Strategy 2")
    p.add_argument("--fast", type=int, default=12)
    p.add_argument("--slow", type=int, default=26)
    p.add_argument("--signal", type=int, default=9)
    p.add_argument("--trend-ema", type=int, default=0,
                   help="Confluence: only enter with the trend of this EMA (e.g. 200); 0 = base strategy")
    p.add_argument("--vwap", action="store_true",
                   help="Confluence: only enter on the matching side of session VWAP (use with --mode rth)")
    p.add_argument("--commission", type=float, default=0.62, help="$ per contract per side")
    p.add_argument("--slippage-ticks", type=float, default=1.0, help="ticks per side")
    p.add_argument("--out", default="macd_timeframes.csv")
    p.add_argument("--trades-dir", default=None, help="If set, write per-timeframe trade logs here")
    args = p.parse_args()

    inst = INSTRUMENTS[args.symbol]
    params = MACDParams(args.fast, args.slow, args.signal)
    costs = CostModel(args.commission, args.slippage_ticks)
    rules = SessionRules(mode=args.mode)
    cache = {}

    rows = []
    for tf_name in [t.strip() for t in args.timeframes.split(",")]:
        tf = pd.Timedelta(tf_name)
        row = {"timeframe": tf_name}
        if tf < pd.Timedelta("1min"):
            if not args.seconds_data:
                row["note"] = "no sub-minute data available - not tested"
                rows.append(row)
                continue
            key, path = "seconds", args.seconds_data
        else:
            key = pick_source(tf, args.common_window)
            path = str(DATA_DIR / SOURCES[key].format(sym=args.symbol.lower()))
        if key not in cache:
            cache[key] = load_1m_csv(path)
        bars = resample_ohlc(cache[key], tf_name)
        trades = run_backtest(bars, tf, inst, params, costs, rules, trend_ema=args.trend_ema, vwap=args.vwap, rule=args.rule)
        days = bars.index.normalize().nunique()
        row.update({"source": key, "start": bars.index[0].date(), "end": bars.index[-1].date(), "bars": len(bars)})
        row.update(metrics(trades, days))
        rows.append(row)
        if args.trades_dir:
            Path(args.trades_dir).mkdir(parents=True, exist_ok=True)
            pd.DataFrame([t.__dict__ for t in trades]).to_csv(
                Path(args.trades_dir) / f"trades_{args.rule}_{args.mode}_{tf_name}.csv", index=False)

    df = pd.DataFrame(rows)
    df.to_csv(args.out, index=False)
    print(f"{args.symbol}  rule={args.rule}  MACD({params.fast},{params.slow},{params.signal})  mode={args.mode}  "
          f"trend_ema={args.trend_ema or 'off'}  vwap={args.vwap}  costs/RT=${costs.round_turn(inst):.2f}  common_window={args.common_window}\n")
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(df.to_string(index=False))


if __name__ == "__main__":
    main()
