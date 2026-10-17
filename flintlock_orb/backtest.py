"""Replay flintlock_orb.signals() into trades using the same exit rules.

Fill at the next candle's open; stop first when a candle touches both; close at the open
of the 16:00 candle or the session's last candle close. Results in points and R.

Run: python -m flintlock_orb.backtest bars.(csv|parquet) [--point-value 20] [--cost-pts 0.5]
"""
import argparse

import numpy as np
import pandas as pd

from flintlock_orb import flintlock_orb as fo


def load(path):
    if str(path).endswith(".parquet"):
        df = pd.read_parquet(path)
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df[["open", "high", "low", "close", "volume"]].sort_index()
    return fo.load_candles(path)


def trades(df, s, tick=fo.TICK_SIZE):
    o, h, l, c = (df[k].to_numpy(np.float64) for k in ("open", "high", "low", "close"))
    _, _, session = fo._clock(df.index)
    flat = s["long_exit"]
    n, out = len(o), []
    for i in np.flatnonzero(s["long_entry"] | s["short_entry"]):
        j = i + 1
        if j >= n or session[j] != session[i]:
            continue
        side = 1 if s["long_entry"][i] else -1
        entry, stop, tgt = o[j], s["stop_price"][i], s["target_price"][i]
        if side * (entry - stop) < tick:
            stop = entry - side * tick
        if side * (tgt - entry) < tick:
            tgt = entry + side * tick
        exit_px, why, t = np.nan, "", j
        while t < n:
            if t > j and flat[t - 1]:
                exit_px, why = o[t], "16:00"
                break
            hit_stop = l[t] <= stop if side == 1 else h[t] >= stop
            hit_tgt = h[t] >= tgt if side == 1 else l[t] <= tgt
            if hit_stop:
                exit_px, why = stop, "stop"
                break
            if hit_tgt:
                exit_px, why = tgt, "target"
                break
            if t == n - 1 or session[t + 1] != session[t]:
                exit_px, why = c[t], "session"
                break
            t += 1
        risk = abs(entry - stop)
        pts = side * (exit_px - entry)
        out.append({"signal_time": df.index[i], "entry_time": df.index[j],
                    "exit_time": df.index[t], "side": "long" if side == 1 else "short",
                    "entry": entry, "stop": stop, "target": tgt, "exit": exit_px,
                    "reason": why, "risk_pts": risk, "pnl_pts": pts, "r": pts / risk})
    return pd.DataFrame(out)


def summary(tr, cost_pts=0.0, point_value=20.0):
    if tr.empty:
        return "no trades"
    net = tr["pnl_pts"] - cost_pts
    r = net / tr["risk_pts"]
    eq = net.cumsum() * point_value
    dd = (eq - eq.cummax()).min()
    gw, gl = net[net > 0].sum(), -net[net < 0].sum()
    lines = [
        f"trades {len(tr)}  (long {sum(tr.side == 'long')}, short {sum(tr.side == 'short')})",
        f"win rate {np.mean(net > 0):.1%}   avg R {r.mean():+.3f}   total R {r.sum():+.1f}",
        f"net points {net.sum():+.1f}   ${net.sum() * point_value:+,.0f} per contract"
        f"   profit factor {gw / gl if gl else np.inf:.2f}   max DD ${dd:,.0f}",
        f"avg risk {tr.risk_pts.mean():.1f} pts   exits: "
        + ", ".join(f"{k} {v}" for k, v in tr.reason.value_counts().items()),
    ]
    by = pd.DataFrame({"y": tr.entry_time.dt.tz_convert("America/New_York").dt.year,
                       "net": net, "r": r, "win": net > 0}).groupby("y")
    lines.append("by year:")
    for y, g in by:
        lines.append(f"  {y}: {len(g):4d} trades  win {g.win.mean():.1%}  "
                     f"R {g.r.sum():+6.1f}  pts {g.net.sum():+8.1f}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--point-value", type=float, default=20.0)
    ap.add_argument("--cost-pts", type=float, default=0.0, help="round-trip cost per trade, points")
    ap.add_argument("--out")
    a = ap.parse_args()
    df = load(a.path)
    tr = trades(df, fo.signals(df))
    print(summary(tr, a.cost_pts, a.point_value))
    if a.out:
        tr.to_csv(a.out, index=False)


if __name__ == "__main__":
    main()
