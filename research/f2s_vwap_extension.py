"""
Does VWAP extension at entry improve Failed-2s signals? (NQ 1m, 2022-12 .. 2025-12)

1. Run the unchanged Failed-2s backtest engine for each timeframe pair with
   no risk caps, so every signal it would take (one position at a time) is kept.
2. Tag each trade with what price looked like when the entry bar closed (no lookahead):
     ext_atr  = distance from session VWAP in units of a 30-minute ATR, signed
                so that positive = the trade fades the stretch (long below VWAP,
                short above VWAP), negative = trading in the direction of the stretch.
     adr_used = today's range so far / 14-day average daily range.
3. Re-simulate each signal on 1-minute bars with two exits:
     1R  : the strategy's own stop and 1R target
     VWAP: same stop, target = session VWAP at entry (only when it's beyond entry)
   Stop is checked before target inside a bar (conservative). Flat at 15:55 ET.
   Costs: 0.5 NQ points per round trip (~1 tick slippage each side + commission).

Usage:  python -m research.f2s_vwap_extension
"""

import numpy as np
import pandas as pd

from backtest.data import load_1m_csv
from backtest.engine import BacktestEngine
from failed2s.instruments import INSTRUMENTS
from failed2s.risk import RiskManager
from failed2s.strategy import PAIRS, Failed2sStrategy

DATA = "sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet"
COST_PTS = 0.5
FLAT = pd.Timestamp("15:55").time()


def session_features(base: pd.DataFrame) -> pd.DataFrame:
    rth = base.between_time("09:30", "15:59").copy()
    day = rth.index.date
    tp = (rth.high + rth.low + rth.close) / 3
    vol = rth.volume.replace(0, np.nan).fillna(1.0)
    rth["vwap"] = (tp * vol).groupby(day).cumsum() / vol.groupby(day).cumsum()
    tr = pd.concat(
        [rth.high - rth.low, (rth.high - rth.close.shift()).abs(), (rth.low - rth.close.shift()).abs()], axis=1
    ).max(axis=1)
    rth["atr30"] = tr.rolling(390).mean() * np.sqrt(30)
    hi, lo = rth.high.groupby(day).cummax(), rth.low.groupby(day).cummin()
    daily = rth.groupby(day).agg(h=("high", "max"), l=("low", "min"))
    adr = (daily.h - daily.l).rolling(14).mean().shift()
    rth["adr_used"] = (hi - lo) / pd.Series(day, index=rth.index).map(adr)
    return rth


def simulate(rth: pd.DataFrame, entry_ts, direction, entry, stop, target):
    sign = 1 if direction == "long" else -1
    day = rth.loc[entry_ts.normalize() : entry_ts.normalize() + pd.Timedelta(days=1)]
    fwd = day[day.index > entry_ts]
    for ts, b in fwd.iterrows():
        if ts.time() >= FLAT:
            return (b.close - entry) * sign
        if (sign == 1 and b.low <= stop) or (sign == -1 and b.high >= stop):
            return (stop - entry) * sign
        if (sign == 1 and b.high >= target) or (sign == -1 and b.low <= target):
            return (target - entry) * sign
    return (fwd.close.iloc[-1] - entry) * sign if len(fwd) else 0.0


def run_pair(pair_name: str, rth: pd.DataFrame) -> pd.DataFrame:
    inst = INSTRUMENTS["NQ"]
    eng = BacktestEngine(
        pair=PAIRS[pair_name],
        instrument=inst,
        strategy=Failed2sStrategy(tick_size=inst.tick_size),
        risk=RiskManager(daily_loss_limit=float("inf"), max_daily_trades=10_000),
    )
    trades = eng.run(DATA)
    entry_len = pd.Timedelta(PAIRS[pair_name].entry_tf)
    rows = []
    for t in trades:
        # Signal timestamp is the entry bar's open; its last 1m bar opens 1 minute before it closes.
        decision_ts = t.entry_time + entry_len - pd.Timedelta("1min")
        if decision_ts not in rth.index:
            continue
        f = rth.loc[decision_ts]
        if not np.isfinite(f.atr30) or f.atr30 <= 0:
            continue
        sign = 1 if t.direction == "long" else -1
        risk = abs(t.entry_price - t.stop_price)
        ext = (f.vwap - f.close) * sign / f.atr30  # + = fading the stretch
        r1 = simulate(rth, decision_ts, t.direction, t.entry_price, t.stop_price, t.target_price)
        vw_ok = (f.vwap - t.entry_price) * sign > 0
        rv = simulate(rth, decision_ts, t.direction, t.entry_price, t.stop_price, f.vwap) if vw_ok else np.nan
        rows.append(
            dict(
                pair=pair_name,
                entry_time=t.entry_time,
                direction=t.direction,
                risk_pts=risk,
                ext_atr=ext,
                adr_used=f.adr_used,
                engine_r=t.r_multiple,
                r_1r=(r1 - COST_PTS) / risk,
                r_vwap=(rv - COST_PTS) / risk if vw_ok else np.nan,
            )
        )
    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, col: str) -> str:
    s = df[col].dropna()
    if len(s) == 0:
        return "n=0"
    se = s.std(ddof=1) / np.sqrt(len(s)) if len(s) > 1 else np.nan
    return f"n={len(s):4d}  avgR={s.mean():+.3f} (±{se:.3f})  win={(s > 0).mean():.0%}  totR={s.sum():+7.1f}"


def main() -> None:
    base = load_1m_csv(DATA)
    rth = session_features(base)
    all_trades = pd.concat([run_pair(p, rth) for p in PAIRS], ignore_index=True)
    all_trades.to_csv("research/f2s_vwap_extension_trades.csv", index=False)

    buckets = [(-99, -1.0, "with stretch  < -1"), (-1.0, 0.0, "with stretch -1..0"),
               (0.0, 1.0, "fade  0..1"), (1.0, 2.0, "fade  1..2"), (2.0, 99, "fade  >2")]
    for pair, d in all_trades.groupby("pair", sort=False):
        print(f"\n=== {pair} ===")
        print(f"  ALL (1R exit)          {summarize(d, 'r_1r')}")
        for lo, hi, name in buckets:
            m = (d.ext_atr >= lo) & (d.ext_atr < hi)
            print(f"  ext {name:20s} 1R:   {summarize(d[m], 'r_1r')}")
            print(f"  {'':24s} VWAP: {summarize(d[m], 'r_vwap')}")
        for lo, hi in [(0, 0.8), (0.8, 1.2), (1.2, 99)]:
            m = (d.adr_used >= lo) & (d.adr_used < hi)
            print(f"  ADR used {lo}-{hi:<4}         1R:   {summarize(d[m], 'r_1r')}")
        d = d.assign(year=pd.to_datetime(d.entry_time, utc=True).dt.year)
        for y, dy in d.groupby("year"):
            print(f"  {y}: all {summarize(dy, 'r_1r')} | fade>=1 {summarize(dy[dy.ext_atr >= 1], 'r_1r')}")


if __name__ == "__main__":
    main()
