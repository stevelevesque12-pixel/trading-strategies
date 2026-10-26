"""Tradeify Growth 50K: which strategy set to run on each account.

  A    Saty Phase (SBJ v2) + PQX Halyard + Gold BO
  B    Saty + Opening Drive (one script, Saty priority)
  MIX  all four on one account

Trade lists are the user's TradingView exports from Oct 18 2026 (tradeify_mix/data/). Every setup gets the same rules,
costs and dates, and each family is compared at its own best eval/funded sizes. Splitting accounts between A and B is
worth the average of A and B per account.

Rules (third-party summaries, Oct 2026 - verify on Tradeify's site):
  Evaluation : +$3,000 target, $2,000 end-of-day trailing drawdown (touching it intraday fails), $1,250 soft daily
               loss limit (positions closed for the day), ~$145 per attempt.
  Funded     : same drawdown; the floor locks at +$100 once the end-of-day high reaches +$2,100.
  Payouts    : balance >= +$3,000; 5 days > $150 since the last payout; best day <= 35% of the profit since the last
               payout; minimum $500; caps $1,500 / $2,000 / $2,500 / $3,000; $2,600 left in the account; 90% split.
Costs: $2.00 round trip per micro plus 2 ticks of slippage per side on every list. Each trade's adverse excursion is applied
inside the trade, so the drawdown and the daily loss limit see intraday dips. Days are drawn by a 5-day block bootstrap
from the window; one simulated year = 250 days, buying a new evaluation the day after any fail or blown account.

  python -m tradeify_mix.compare            # size grid per family -> results/compare_best.csv
  python -m tradeify_mix.compare --robust   # neighbours of the best sizes -> results/robustness.csv
"""
import argparse
import itertools
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
DATA, RESULTS = HERE / "data", HERE / "results"
FILES = dict(S="saty_sbj_v2_3mnq.csv", H="halyard_5mnq.csv", G="gold_bo_1mgc.csv", SO="saty_opening_drive_combo.csv")
LIST_SIZE = dict(S=3, CS=6, OD=750, H=5, G=1)          # size each list was exported at (OD = $ risk)
# Extra $ per contract to bring every list to the user's cost rule: $2.00 round trip ($1.00/side) + 2 ticks of slippage
# per side. The Saty, Halyard and Gold lists were exported with $3.64 (MNQ) / $4.24 (MGC) round trip - Tradeify's
# round-trip fee entered per side - so commission is taken back out. The Saty list carries 1 tick of slippage (its
# Pine script); Halyard and Gold are assumed to carry none. The combo list already matches ($2.00 + 2 ticks).
EXTRA_COST = dict(S=-0.64, H=0.36, G=1.76, SO=0.0)

TARGET, DD, DLL, FEE, LOCK, LOCK_FLOOR = 3000, 2000, 1250, 145, 2100, 100
PAY_BAL, KEEP, WIN, NWIN, CONS, MINPAY, CAPS, SPLIT = 3000, 2600, 150, 5, 0.35, 500, (1500, 2000, 2500, 3000), 0.9
WINDOWS = {"Nov 2023 - Oct 2026": ("2023-11-01", "2026-10-16"), "last 12 months": ("2025-10-17", "2026-10-16")}
STEPS = 32                                             # max equity events per day


def load(k):
    """One TradingView 'List of trades' export -> one row per trade (times ET, CME trade day)."""
    d = pd.read_csv(DATA / FILES[k], encoding="utf-8-sig")
    en = d[d.Type.str.startswith("Entry")].set_index("Trade number")
    ex = d[d.Type.str.startswith("Exit")].set_index("Trade number").loc[en.index]
    qty = en["Size (qty)"]
    t = pd.DataFrame(dict(entry=pd.to_datetime(en["Date and time"]), exit=pd.to_datetime(ex["Date and time"]),
                          pnl=en["Net PnL USD"] - EXTRA_COST[k] * qty,
                          mae=np.minimum(en["Adverse excursion USD"] - EXTRA_COST[k] * qty, 0),
                          dir=np.where(en.Type.str.contains("long"), 1, -1), signal=en.Signal, qty=qty))
    t["day"] = (t.exit + pd.Timedelta(hours=7)).dt.normalize()          # CME day starts 18:00 ET
    t["key"] = ((t.exit - pd.Timedelta(hours=18)).dt.hour * 60 + t.exit.dt.minute).astype(float)
    return t.reset_index(drop=True)


def load_all():
    L = {k: load(k) for k in FILES}
    L["CS"] = L["SO"][L["SO"].signal.str.startswith("SATY")]
    L["OD"] = L["SO"][L["SO"].signal.str.startswith("OD")]
    return L


def nq_book(frames):
    """One Nasdaq account: a trade opposite to an open Nasdaq trade is skipped (positions net on Tradovate)."""
    allt = pd.concat([t.assign(q=q) for t, q in frames if q], ignore_index=True).sort_values("entry")
    keep, open_ = [], []
    for i, r in allt.iterrows():
        open_ = [o for o in open_ if o[0] > r.entry]
        if any(d != r.dir for _, d in open_):
            continue
        open_.append((r.exit, r.dir)); keep.append(i)
    return allt.loc[keep]


def events(t, q):
    """Two equity events per trade: the dip inside it, then the rest of its result at the exit."""
    a = pd.DataFrame(dict(day=t.day, key=t.key - 0.5, pnl=t.mae * q))
    b = pd.DataFrame(dict(day=t.day, key=t.key, pnl=(t.pnl - t.mae) * q))
    return pd.concat([a, b])


def events_for(L, cfg):
    """cfg: S (Saty MNQ, standalone list), CS (Saty MNQ in the combo list), OD (Opening Drive $ risk), H (Halyard MNQ),
    G (Gold BO MGC). Sizes scale each list linearly."""
    fr = [(L[k], cfg[k] / LIST_SIZE[k]) for k in ("S", "CS", "OD", "H") if cfg.get(k)]
    parts = []
    if fr:
        nq = nq_book(fr); parts.append(events(nq, nq.q))
    if cfg.get("G"):
        parts.append(events(L["G"], cfg["G"] / LIST_SIZE["G"]))
    return pd.concat(parts)


def prefix(ev, days):
    """(len(days), STEPS) closed-equity path per day, starting at 0 and padded with the day's total."""
    P = np.zeros((len(days), STEPS))
    pos = {d: i for i, d in enumerate(days)}
    for d, g in ev.sort_values(["day", "key"]).groupby("day"):
        if d not in pos:
            continue
        cs = np.r_[0.0, g.pnl.cumsum().to_numpy()][:STEPS]
        P[pos[d], :len(cs)] = cs; P[pos[d], len(cs):] = cs[-1]
    return P


def boot(n_pool, n_sims, days, rng, block=5):
    idx = np.empty((n_sims, days), dtype=int)
    cur = rng.integers(n_pool, size=n_sims)
    for k in range(days):
        jump = rng.random(n_sims) < 1 / block
        cur = np.where(jump, rng.integers(n_pool, size=n_sims), cur)
        idx[:, k] = cur
        cur = (cur + 1) % n_pool
    return idx


def day_dll(seq, dll):
    """Liquidate at -dll the first time the day's equity path reaches it; returns (total, intraday min)."""
    hit = seq <= -dll
    first = np.where(hit.any(1), hit.argmax(1), seq.shape[1] - 1)
    cols = np.arange(seq.shape[1])[None, :]
    s = np.where(cols <= first[:, None], seq, np.take_along_axis(seq, first[:, None], 1))
    s = np.maximum(s, -dll)
    return s[:, -1], s.min(1)


def run(Pe, Pf, ix, days=250):
    """One year per path with eval day paths Pe and funded day paths Pf; returns per-account stats."""
    n = ix.shape[0]
    mode = np.zeros(n, int); bal = np.zeros(n); peak = np.zeros(n); wins = np.zeros(n, int); best = np.zeros(n)
    npay = np.zeros(n, int); fees = np.zeros(n); paid = np.zeros(n); evals = np.zeros(n, int)
    first = np.full(n, np.nan); since = np.zeros(n)
    for k in range(days):
        new = mode == 0; fees[new] += FEE; evals += new; mode[new] = 1
        bal[new] = peak[new] = 0; wins[new] = 0; npay[new] = 0; best[new] = 0; since[new] = 0
        e = mode == 1
        te, me = day_dll(Pe[ix[:, k]], DLL); tf, mf = day_dll(Pf[ix[:, k]], DLL)
        tot = np.where(e, te, tf); mn = np.where(e, me, mf)
        floor = np.where(~e & (peak >= LOCK), np.minimum(peak - DD, LOCK_FLOOR), peak - DD)
        dead = bal + mn <= floor; mode[dead] = 0; al = ~dead
        bal[al] += tot[al]; peak[al] = np.maximum(peak[al], bal[al])
        ok = e & al & (bal >= TARGET); mode[ok] = 2; bal[ok] = peak[ok] = 0
        fa = (mode == 2) & al & ~ok
        wins[fa] += tot[fa] > WIN; since[fa] += tot[fa]; best[fa] = np.maximum(best[fa], tot[fa])
        cap = np.array(CAPS)[np.minimum(npay, len(CAPS) - 1)]
        amt = np.minimum(cap, bal - KEEP)
        go = fa & (bal >= PAY_BAL) & (wins >= NWIN) & (best <= CONS * np.maximum(since, 1e-9)) & (amt >= MINPAY)
        paid[go] += amt[go] * SPLIT; npay += go; first = np.where(go & np.isnan(first), k + 1, first)
        bal[go] -= amt[go]; wins[go] = 0; best[go] = 0; since[go] = 0
    net = paid - fees
    return dict(net=net.mean(), p10=np.percentile(net, 10), p90=np.percentile(net, 90), evals=evals.mean(),
                payouts=npay.mean(), first_pay=np.nanmedian(np.where(np.isnan(first), 999, first)),
                any_pay=(~np.isnan(first)).mean())


# Tradeify plans, third-party summaries (same values as AI-personal-hedge-fund scripts/saty_tradeify150k_risk.py and
# research/account_structure/run_flex50_final.py) - verify on Tradeify's site. dll=None: no daily loss limit.
# Select Flex: 40% eval consistency + 3 days; funded payout every 5 days > day_min, up to 50% of profit, floor locks
# at +$100 on the first payout (or once the high reaches DD + $100). Growth: as in the module docstring.
PLANS = {
    "Growth 50K": dict(fee=145, target=3000, dd=2000, dll=1250, ev_cons=1.0, ev_days=1, lock=2100, lock_on_pay=False,
                       pay_bal=3000, keep=2600, day_min=150, cons=0.35, frac=1.0, min_pay=500,
                       caps=(1500, 2000, 2500, 3000)),
    "Select Flex 50K": dict(fee=165, target=3000, dd=2000, dll=None, ev_cons=0.40, ev_days=3, lock=2100,
                            lock_on_pay=True, pay_bal=0, keep=0, day_min=150, cons=1.0, frac=0.5, min_pay=250,
                            caps=(2500,)),
    "Growth 150K": dict(fee=369, target=9000, dd=5000, dll=3750, ev_cons=1.0, ev_days=1, lock=5100, lock_on_pay=False,
                        pay_bal=6500, keep=5100, day_min=250, cons=0.35, frac=1.0, min_pay=1500,
                        caps=(2500, 3000, 4000, 5000)),
    "Select Flex 150K": dict(fee=369, target=9000, dd=4500, dll=None, ev_cons=0.40, ev_days=3, lock=4600,
                             lock_on_pay=True, pay_bal=0, keep=0, day_min=250, cons=1.0, frac=0.5, min_pay=500,
                             caps=(4500,)),
    # LucidFlex 150K (third-party summaries, Oct 2026, as in AI-personal-hedge-fund scripts/lucid_fleet.py): no daily
    # loss limit, 50% eval consistency, 2 days minimum; funded floor locks at +$100 at +$4,600 or on the first payout;
    # payouts every 5 days of $250+, up to 50% of profit, capped at $3,000; the 5th payout ends the funded account
    # (it moves to a live review, not modelled), so a new evaluation is bought.
    "LucidFlex 150K": dict(fee=345, target=9000, dd=4500, dll=None, ev_cons=0.50, ev_days=2, lock=4600,
                           lock_on_pay=True, pay_bal=0, keep=0, day_min=250, cons=1.0, frac=0.5, min_pay=500,
                           caps=(3000,), max_payouts=5),
}


def run_plan(Pe, Pf, ix, R, days=250, return_net=False, return_cash=False):
    """run() for any plan in PLANS; return_net=True also returns each path's net, return_cash=True the
    (paths, days) cash flow: payouts after the split minus evaluation fees, on the day they happen."""
    n = ix.shape[0]
    cash = np.zeros((n, days)) if return_cash else None
    mode = np.zeros(n, int); bal = np.zeros(n); peak = np.zeros(n); wins = np.zeros(n, int); best = np.zeros(n)
    npay = np.zeros(n, int); fees = np.zeros(n); paid = np.zeros(n); evals = np.zeros(n, int)
    first = np.full(n, np.nan); since = np.zeros(n); nd = np.zeros(n, int); locked = np.zeros(n, bool)
    caps = np.array(R["caps"], float)
    tpay = np.zeros(n, int)          # payouts over the whole year, across accounts
    for k in range(days):
        new = mode == 0; fees[new] += R["fee"]; evals += new; mode[new] = 1
        if return_cash: cash[new, k] -= R["fee"]
        bal[new] = peak[new] = 0; wins[new] = 0; npay[new] = 0; best[new] = 0; since[new] = 0; nd[new] = 0
        locked[new] = False
        e = mode == 1
        if R["dll"]:
            te, me = day_dll(Pe[ix[:, k]], R["dll"]); tf, mf = day_dll(Pf[ix[:, k]], R["dll"])
        else:
            se, sf = Pe[ix[:, k]], Pf[ix[:, k]]; te, me, tf, mf = se[:, -1], se.min(1), sf[:, -1], sf.min(1)
        tot = np.where(e, te, tf); mn = np.where(e, me, mf)
        floor = np.where(~e & locked, np.minimum(peak - R["dd"], 100.0), peak - R["dd"])
        dead = bal + mn <= floor; mode[dead] = 0; al = ~dead
        bal[al] += tot[al]; peak[al] = np.maximum(peak[al], bal[al])
        ea = e & al; nd[ea] += tot[ea] != 0; best[ea] = np.maximum(best[ea], tot[ea])
        ok = ea & (bal >= R["target"]) & (nd >= R["ev_days"]) & (best <= R["ev_cons"] * bal)
        mode[ok] = 2; bal[ok] = peak[ok] = 0; best[ok] = 0
        fa = (mode == 2) & al & ~ok
        locked |= fa & (peak >= R["lock"])
        wins[fa] += tot[fa] > R["day_min"]; since[fa] += tot[fa]; best[fa] = np.maximum(best[fa], tot[fa])
        amt = np.minimum(caps[np.minimum(npay, len(caps) - 1)], np.minimum(bal - R["keep"], R["frac"] * bal))
        go = fa & (bal >= max(R["pay_bal"], 1e-9)) & (wins >= NWIN) & (best <= R["cons"] * np.maximum(since, 1e-9)) \
            & (amt >= R["min_pay"])
        paid[go] += amt[go] * SPLIT; npay += go; first = np.where(go & np.isnan(first), k + 1, first)
        if return_cash: cash[go, k] += amt[go] * SPLIT
        tpay += go
        bal[go] -= amt[go]; wins[go] = 0; best[go] = 0; since[go] = 0
        if R["lock_on_pay"]: locked |= go
        if R.get("max_payouts"):
            mode[go & (npay >= R["max_payouts"])] = 0
    net = paid - fees
    out = dict(net=net.mean(), p10=np.percentile(net, 10), p90=np.percentile(net, 90), evals=evals.mean(),
               fees=fees.mean(), payouts=tpay.mean(),
               first_pay=np.nanmedian(np.where(np.isnan(first), 999, first)), any_pay=(~np.isnan(first)).mean())
    if return_cash:
        return out, cash
    return (out, net) if return_net else out


def mix_grid(big):
    if big:
        return [dict(CS=s, OD=o, H=h, G=g) for s, o, h, g in
                itertools.product((3, 5, 7, 9), (500, 1000, 1500), (0, 4, 8), (2, 4, 6))]
    return FAMILIES["MIX all four"]


def plans(sims):
    """Best eval/funded sizes of the four-strategy mix on each plan, then each plan's stats at those sizes."""
    sim, w0 = Sim(sims, 7), list(WINDOWS)[0]
    out = []
    for pn, R in PLANS.items():
        configs = mix_grid("150K" in pn)
        st = lambda ce, cf, w: run_plan(sim.P(ce, w), sim.P(cf, w), sim.ix[w], R)
        s1 = sorted(((st(c, c, w0)["net"], i) for i, c in enumerate(configs)), reverse=True)
        best = None
        for _, fi in s1[:6]:
            for ce in configs:
                r = st(ce, configs[fi], w0)
                if best is None or r["net"] > best[0]["net"]:
                    best = (r, ce, configs[fi])
        _, ce, cf = best
        row = dict(plan=pn, eval=name(ce), funded=name(cf))
        for w in WINDOWS:
            row.update({f"{w}|{k}": v for k, v in st(ce, cf, w).items()})
        out.append(row); print(row, flush=True)
    df = pd.DataFrame(out); df.to_csv(RESULTS / "plans_best.csv", index=False)
    print(df.set_index("plan").T.to_string())


def name(c):
    return " ".join(f"{k}{v}" for k, v in c.items() if v)


FAMILIES = {
    "A  Saty+Halyard+Gold": [dict(S=s, H=h, G=g) for s, h, g in itertools.product((2, 3, 4, 5), (0, 2, 4), (1, 2, 3, 4))],
    "B  Saty+Opening Drive": [dict(CS=s, OD=o) for s, o in itertools.product((1, 2, 3, 4, 5), (0, 250, 500, 750, 1000))
                              if s or o],
    "MIX all four": [dict(CS=s, OD=o, H=h, G=g) for s, o, h, g in
                     itertools.product((2, 3, 4, 5), (250, 500, 750), (0, 2, 4), (1, 2, 3))],
}
# (eval, funded) sizes for the robustness table
CASES = {
    "A best": (dict(S=5, H=2, G=4), dict(S=2, H=2, G=1)),
    "A earlier plan (H4/S5/G3 -> H2/S5/G3)": (dict(S=5, H=4, G=3), dict(S=5, H=2, G=3)),
    "B best": (dict(CS=5, OD=1000), dict(CS=2, OD=1000)),
    "B earlier plan (S3+$800 -> S2+$500)": (dict(CS=3, OD=800), dict(CS=2, OD=500)),
    "MIX best": (dict(CS=5, OD=750, H=4, G=3), dict(CS=2, OD=750, H=4, G=1)),
    "MIX Halyard 2": (dict(CS=5, OD=750, H=2, G=3), dict(CS=2, OD=750, H=2, G=1)),
    "MIX OD $500 (recommended)": (dict(CS=5, OD=500, H=4, G=3), dict(CS=2, OD=500, H=4, G=1)),
    "MIX eval Saty 3": (dict(CS=3, OD=750, H=4, G=3), dict(CS=2, OD=750, H=4, G=1)),
    "MIX funded Gold 2": (dict(CS=5, OD=750, H=4, G=3), dict(CS=2, OD=750, H=4, G=2)),
    "MIX Halyard 6": (dict(CS=5, OD=750, H=6, G=3), dict(CS=2, OD=750, H=6, G=1)),
    "MIX same size in both (S2 OD750 H4 G1)": (dict(CS=2, OD=750, H=4, G=1),) * 2,
}


class Sim:
    def __init__(self, sims, seed):
        self.L = load_all()
        self.spans = {w: pd.bdate_range(*v) for w, v in WINDOWS.items()}
        self.ix = {w: boot(len(d), sims, 250, np.random.default_rng(seed)) for w, d in self.spans.items()}
        self.cache = {}

    def P(self, c, w):
        key = (tuple(sorted(c.items())), w)
        if key not in self.cache:
            self.cache[key] = prefix(events_for(self.L, c), self.spans[w])
        return self.cache[key]

    def stats(self, ce, cf, w):
        return run(self.P(ce, w), self.P(cf, w), self.ix[w])


def grid(sims):
    sim, w0 = Sim(sims, 7), list(WINDOWS)[0]
    out = []
    for fam, configs in FAMILIES.items():
        # stage 1: same size in eval and funded; stage 2: best 6 funded sizes x every eval size
        s1 = sorted(((sim.stats(c, c, w0)["net"], i) for i, c in enumerate(configs)), reverse=True)
        best = None
        for _, fi in s1[:6]:
            for ce in configs:
                r = sim.stats(ce, configs[fi], w0)
                if best is None or r["net"] > best[0]["net"]:
                    best = (r, ce, configs[fi])
        _, ce, cf = best
        row = dict(family=fam, eval=name(ce), funded=name(cf))
        for w in WINDOWS:
            row.update({f"{w}|{k}": v for k, v in sim.stats(ce, cf, w).items()})
        out.append(row); print(row, flush=True)
    df = pd.DataFrame(out); df.to_csv(RESULTS / "compare_best.csv", index=False)
    print(df.set_index("family").T.to_string())


def robust(sims):
    sim = Sim(sims, 99)
    rows = []
    for n, (ce, cf) in CASES.items():
        r = dict(setup=n, eval=name(ce), funded=name(cf))
        for w in WINDOWS:
            s = sim.stats(ce, cf, w)
            r.update({f"{w}|net": s["net"], f"{w}|p10": s["p10"], f"{w}|evals": s["evals"],
                      f"{w}|first_pay": s["first_pay"]})
        rows.append(r)
    df = pd.DataFrame(rows); df.to_csv(RESULTS / "robustness.csv", index=False)
    with pd.option_context("display.width", 250):
        print(df.to_string(index=False, float_format=lambda v: f"{v:,.0f}"))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--sims", type=int)
    ap.add_argument("--robust", action="store_true"); ap.add_argument("--plans", action="store_true")
    a = ap.parse_args()
    if a.plans:
        plans(a.sims or 2000)
    else:
        robust(a.sims or 3000) if a.robust else grid(a.sims or 2000)


if __name__ == "__main__":
    main()
