"""IB breach-rate statistics (descriptive, not a backtest).

Per RTH session: IB = first `ib_minutes` from 09:30 ET. Classifies each day by
  - IB close vs IB midpoint (close of the last IB bar)
  - HL order: which IB extreme was printed last (days where one bar set both
    are left unclassified)
and reports how often IB high / IB low get breached (a trade through the
level, i.e. bar high > IB high / bar low < IB low) by noon and by 16:00.

It also reports *follow-through*: after the first breach of the predicted side,
how often price extends `ext` x IB range beyond the level before coming back
to the IB midpoint -- that's what decides whether a breach is tradable.

Usage:
    python -m ib_break.stats --data sample_data/real_multi_instrument/real_nq_15m_2016-05-29_2026-08-25.parquet
"""

import argparse

import numpy as np
import pandas as pd

from backtest.data import load_1m_csv


def session_table(df: pd.DataFrame, ib_minutes: int = 60, ext: float = 0.5) -> pd.DataFrame:
    rth = df.between_time("09:30", "15:59")
    ib_end = (pd.Timestamp("09:30") + pd.Timedelta(minutes=ib_minutes)).time()
    rows = []
    for day, g in rth.groupby(rth.index.date):
        t = g.index.time
        ib = g[t < ib_end]
        post = g[t >= ib_end]
        if ib.empty or post.empty or ib.index[0].time() != pd.Timestamp("09:30").time():
            continue
        hi, lo = ib.high.max(), ib.low.min()
        mid = (hi + lo) / 2
        hi_bar = ib.high.values.argmax()  # first bar printing the extreme
        lo_bar = ib.low.values.argmin()
        order = "low_first" if lo_bar < hi_bar else "high_first" if hi_bar < lo_bar else "same_bar"
        noon = post[post.index.time < pd.Timestamp("12:00").time()]

        row = {
            "date": day, "ib_high": hi, "ib_low": lo, "ib_range": hi - lo,
            "close_above_mid": ib.close.iloc[-1] > mid, "order": order,
            "high_noon": bool((noon.high > hi).any()), "low_noon": bool((noon.low < lo).any()),
            "high_close": bool((post.high > hi).any()), "low_close": bool((post.low < lo).any()),
        }
        # Follow-through: after first breach of each side, does price reach the
        # extension before returning to IB mid? (same-bar ambiguity -> counted as fail)
        for side in ("high", "low"):
            hit = (post.high > hi) if side == "high" else (post.low < lo)
            res = np.nan
            if hit.any():
                after = post.loc[hit.idxmax():]
                if side == "high":
                    tgt, fail = hi + ext * (hi - lo), after.low <= mid
                    win = after.high >= tgt
                else:
                    tgt, fail = lo - ext * (hi - lo), after.high >= mid
                    win = after.low <= tgt
                first_win = win.values.argmax() if win.any() else len(after)
                first_fail = fail.values.argmax() if fail.any() else len(after)
                res = float(first_win < first_fail) if win.any() else 0.0
            row[f"{side}_follow"] = res
        rows.append(row)
    return pd.DataFrame(rows)


def _pct(s: pd.Series) -> str:
    return f"{s.mean() * 100:5.1f}%" if len(s) else "  n/a"


def report(t: pd.DataFrame, ext: float) -> None:
    print(f"Qualifying days: {len(t)}  ({t.date.min()} .. {t.date.max()})")
    print(f"{'subset':38s} {'n':>5s}  {'hi@noon':>8s} {'hi@close':>8s} {'lo@noon':>8s} {'lo@close':>8s}"
          f"  {'hi follow':>9s} {'lo follow':>9s}")
    subsets = {
        "all": t,
        "close above mid": t[t.close_above_mid],
        "close below mid": t[~t.close_above_mid],
        "low first / high last": t[t.order == "low_first"],
        "high first / low last": t[t.order == "high_first"],
        "above mid + high last (bull confl.)": t[t.close_above_mid & (t.order == "low_first")],
        "below mid + low last (bear confl.)": t[~t.close_above_mid & (t.order == "high_first")],
        "conflicting signals": t[(t.close_above_mid & (t.order == "high_first"))
                                 | (~t.close_above_mid & (t.order == "low_first"))],
    }
    for name, s in subsets.items():
        print(f"{name:38s} {len(s):5d}  {_pct(s.high_noon):>8s} {_pct(s.high_close):>8s} "
              f"{_pct(s.low_noon):>8s} {_pct(s.low_close):>8s}  "
              f"{_pct(s.high_follow.dropna()):>9s} {_pct(s.low_follow.dropna()):>9s}")
    either_noon = (t.high_noon | t.low_noon).mean() * 100
    either_close = (t.high_close | t.low_close).mean() * 100
    print(f"either side breached: noon {either_noon:.1f}%  close {either_close:.1f}%")
    print(f"follow = after first breach, reached {ext}x IB range beyond the level before IB mid")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--data", required=True)
    p.add_argument("--ib-minutes", type=int, default=60)
    p.add_argument("--ext", type=float, default=0.5)
    args = p.parse_args()
    t = session_table(load_1m_csv(args.data), args.ib_minutes, args.ext)
    report(t, args.ext)


if __name__ == "__main__":
    main()
