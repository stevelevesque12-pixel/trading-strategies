"""
Single-position simulator that mirrors tradingview/mcl_trend_dip.pine:
one strategy, one position at a time, engine A has priority on the same bar,
each trade carries its own engine's exits (target, breakeven, trail, RSI exit,
session window + flatten), shared daily counters (trades, loss stop, profit
lock), optional equity-curve kill switch (shadow trades) and cushion sizing.

  python -m trend_lab.combined
"""

import numpy as np
import pandas as pd

from .cross_market import ENGINE_A, ENGINE_B
from .data import load_bars
from .metrics import compute, daily_pnl
from .sim import POINT_VALUE, SESSIONS, TICK, Trade
from .strategies import ALL_FAMILIES
from . import indicators as ind


def run(df, risk_fixed=300.0, cushion=None, ecf_len=None, slip_ticks=1.0, comm_rt=1.24,
        daily_loss_mult=2.25, profit_lock=1200.0, max_trades=6, max_stop=1.5, max_contracts=20,
        eia_filter=True, a_max_bars=None, a_scale_r=None):
    """
    a_scale_r: Engine A scale-out -- take half the contracts off at +a_scale_r R (rest keeps the normal exits).
    cushion: None for fixed risk, else (frac, lo, hi) of (balance - Lucid max-loss line) of LIVE trades.
    ecf_len: None = always live; N = live only while closed equity >= mean of its last N values.
    Returns (all_trades, live_trades).
    """
    pa, pb = ENGINE_A[1], ENGINE_B[1]
    sa = ALL_FAMILIES[ENGINE_A[0]].generate(df, pa)
    sb = ALL_FAMILIES[ENGINE_B[0]].generate(df, pb)
    rsi = ind.rsi(df["close"], pb["rsi_n"]).fillna(50).to_numpy()
    atr = ind.atr(df, 14).to_numpy()
    n = len(df)
    o, h, lo, c = (df[k].to_numpy() for k in ("open", "high", "low", "close"))
    idx = df.index
    tf = int(pd.Series(idx).diff().median().total_seconds() / 60)
    mod = (idx.hour * 60 + idx.minute).to_numpy()
    cmod = (mod + tf) % (24 * 60)
    wed = idx.weekday.to_numpy() == 2
    tday = pd.factorize(df["trade_day"])[0]
    tvals = df["trade_day"].to_numpy()

    def window(sess):
        s = SESSIONS[sess]
        w = (cmod >= s["start"]) & (cmod <= s["end"]) & ~((mod >= 16 * 60) & (mod < 18 * 60))
        if eia_filter:
            w &= ~(wed & (cmod >= 10 * 60 + 15) & (cmod <= 10 * 60 + 45))
        return w, (mod >= s["flatten"]) & (mod < 18 * 60)

    winA, flatA = window(pa["session"])
    winB, flatB = window(pb["session"])
    eng_cfg = {
        1: dict(target=pa["target_r"], be=pa["be_r"], trail=pa["trail_k"], flat=flatA, rsi_exit=None,
                max_bars=a_max_bars),
        2: dict(target=pb["target_r"], be=pb["be_r"], trail=pb["trail_k"], flat=flatB, rsi_exit=pb["exit_rsi"],
                max_bars=None),
    }
    slip = slip_ticks * TICK
    cost_pc = comm_rt + 2 * slip * POINT_VALUE

    trades, live = [], []
    eq_hist = []            # closed-trade equity of ALL trades (shadow + live), for the kill switch
    net_all = 0.0
    live_net = 0.0
    peak_eod = 0.0
    pos = 0
    entry = stop = tgt = risk_pts = best = 0.0
    contracts = 0
    eng = 0
    entry_i = -1
    is_live = True
    pending = None
    day, day_pnl, day_trades = -1, 0.0, 0
    risk_now = risk_fixed
    part_qty, part_px, part_pnl, contracts0 = 0, 0.0, 0.0, 0

    def close(i, px, reason):
        nonlocal pos, day_pnl, net_all, live_net, part_qty, part_pnl
        pnl = (px - entry) * pos * POINT_VALUE * contracts - comm_rt * contracts
        day_pnl += pnl
        pnl += part_pnl                     # realized scale-out leg, if any (already in day_pnl)
        r = pnl / (risk_pts * POINT_VALUE * contracts0 + cost_pc * contracts0)
        t = Trade(idx[entry_i], idx[i], tvals[entry_i], pos, entry, px, entry - pos * risk_pts, contracts0, pnl, r,
                  reason)
        part_qty, part_pnl = 0, 0.0
        trades.append(t)
        if is_live:
            live.append(t)
            live_net += pnl
        net_all += pnl
        eq_hist.append(net_all)
        pos = 0

    def scale_out(i):
        nonlocal part_qty, part_pnl, contracts, day_pnl
        if part_qty and ((pos == 1 and h[i] >= part_px) or (pos == -1 and lo[i] <= part_px)):
            leg = (part_px - entry) * pos * POINT_VALUE * part_qty - comm_rt * part_qty
            part_pnl += leg
            day_pnl += leg
            contracts -= part_qty
            part_qty = 0

    for i in range(n):
        if tday[i] != day:
            if day != -1:
                peak_eod = max(peak_eod, live_net)   # Lucid EOD high-water mark (live account)
            day, day_pnl, day_trades = tday[i], 0.0, 0
        cfg = eng_cfg.get(eng)
        if pos != 0 and (pending == "exit" or tday[i] != tday[entry_i] or cfg["flat"][i]):
            close(i, o[i] - pos * slip, "signal" if pending == "exit" else "flatten")
        if isinstance(pending, tuple) and pos == 0 and i > 0 and tday[i] == tday[i - 1]:
            d, dist, e, q, lv = pending
            if not eng_cfg[e]["flat"][i]:
                pos, eng, risk_pts, contracts, is_live = d, e, dist, q, lv
                entry = o[i] + pos * slip
                stop = entry - pos * risk_pts
                tgt = entry + pos * risk_pts * eng_cfg[e]["target"] if eng_cfg[e]["target"] else np.nan
                entry_i, best = i, entry
                contracts0 = contracts
                if a_scale_r and e == 1 and contracts >= 2:
                    part_qty = contracts // 2
                    part_px = entry + pos * risk_pts * a_scale_r
        pending = None
        cfg = eng_cfg.get(eng)

        if pos != 0:
            if pos == 1:
                if o[i] <= stop:
                    close(i, o[i] - slip, "stop")
                elif lo[i] <= stop:
                    close(i, stop - slip, "stop")
                else:
                    scale_out(i)
                if pos != 0 and cfg["target"] and h[i] >= tgt:
                    close(i, tgt, "target")
            else:
                if o[i] >= stop:
                    close(i, o[i] + slip, "stop")
                elif h[i] >= stop:
                    close(i, stop + slip, "stop")
                else:
                    scale_out(i)
                if pos != 0 and cfg["target"] and lo[i] <= tgt:
                    close(i, tgt, "target")

        if pos != 0:
            best = max(best, h[i]) if pos == 1 else min(best, lo[i])
            if cfg["be"] and (best - entry) * pos >= cfg["be"] * risk_pts:
                be = entry + pos * 2 * TICK
                stop = max(stop, be) if pos == 1 else min(stop, be)
            if cfg["trail"]:
                t = best - pos * cfg["trail"] * atr[i]
                stop = max(stop, t) if pos == 1 else min(stop, t)
            if cfg["rsi_exit"] and ((pos == 1 and rsi[i] > cfg["rsi_exit"]) or
                                    (pos == -1 and rsi[i] < 100 - cfg["rsi_exit"])):
                pending = "exit"
            if cfg["max_bars"] and i - entry_i + 1 >= cfg["max_bars"]:
                pending = "exit"

        if cushion:
            frac, rlo, rhi = cushion
            line = min(peak_eod - 2000.0, 0.0)
            risk_now = min(rhi, max(rlo, frac * (live_net - line)))
        if pos == 0 and pending is None and i + 1 < n and day_trades < max_trades and \
                -day_pnl < daily_loss_mult * risk_now and day_pnl < profit_lock:
            cand = None
            if winA[i] and (sa["long"][i] or sa["short"][i]):
                cand = (1 if sa["long"][i] else -1, float(sa["stop_dist"][i]), 1)
            elif winB[i] and (sb["long"][i] or sb["short"][i]):
                cand = (1 if sb["long"][i] else -1, float(sb["stop_dist"][i]), 2)
            if cand and np.isfinite(cand[1]) and 0 < cand[1] <= max_stop:
                dist = max(cand[1], 3 * TICK)
                q = int(min(max_contracts, risk_now // (dist * POINT_VALUE + cost_pc)))
                if q >= 1:
                    lv = (ecf_len is None or len(eq_hist) < ecf_len or
                          eq_hist[-1] >= float(np.mean(eq_hist[-ecf_len:])))
                    pending = (cand[0], dist, cand[2], q, lv)
                    day_trades += 1
    if pos != 0:
        close(n - 1, c[-1], "end")
    return trades, live


def save_recommended():
    """Store the recommended configuration's Pine-parity results for the dashboard hero section."""
    import datetime as dt
    from datetime import datetime, timezone
    from .metrics import equity_points
    from .optimize import load_registry, save_registry
    df = load_bars("15min")
    days = sorted(set(df["trade_day"]))
    split = dt.date(2026, 3, 20)
    _, live = run(df, ecf_len=20, a_scale_r=1.0)
    seg = lambda f: compute([t for t in live if f(t.trade_day)], [d for d in days if f(d)])
    reg = load_registry()
    mc = {r.get("rule"): r for r in reg.get("montecarlo_full", {}).get("dynamic", [])}
    reg["recommended"] = {
        "name": "MCL Trend Dip (15m) — tradingview/mcl_trend_dip.pine",
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "engines": [ENGINE_A, ENGINE_B], "split": str(split), "start": str(days[0]), "end": str(days[-1]),
        "full": seg(lambda d: True), "fit": seg(lambda d: d < split), "unseen": seg(lambda d: d >= split),
        "equity": equity_points(live), "sizing_note": "fixed $300 risk, kill switch on, Engine A half off at +1R",
        "montecarlo_full": reg.get("montecarlo_full", {}),
    }
    save_registry(reg)


def main():
    import datetime as dt
    df = load_bars("15min")
    days = sorted(set(df["trade_day"]))
    split = dt.date(2026, 3, 20)
    for label, kw in [("fixed $300", {}), ("fixed $300 + kill switch", {"ecf_len": 20}),
                      ("cushion 25% $200-600 + kill switch", {"cushion": (0.25, 200, 600), "ecf_len": 20}),
                      ("cushion 20% $150-500", {"cushion": (0.20, 150, 500)})]:
        _, live = run(df, **kw)
        for seg, f in [("full", lambda d: True), ("Oct-Mar", lambda d: d < split), ("Mar20-Aug", lambda d: d >= split)]:
            m = compute([t for t in live if f(t.trade_day)], [d for d in days if f(d)])
            print(f"{label:36} {seg:9} n={m['trades']:>3} wr={m['win_rate']} pf={m['profit_factor']} "
                  f"net={m['net']} dd={m['max_dd']} eod_dd={m.get('eod_dd')}")


if __name__ == "__main__":
    main()
    save_recommended()
