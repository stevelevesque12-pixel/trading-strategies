"""Scale-out study: K copy-traded Tradeify Select Flex accounts on the same signals.

Each account slot runs its own eval -> funded -> (blown) -> new eval loop on the SAME bootstrapped
trade path; slots start a few trading days apart so they don't sit in identical states.
Tradeify caps funded accounts at 5 per person, so K <= 5.
Run from repo root: python3 prop_sim/multi_account.py
"""
import sys
from multiprocessing import Pool

sys.path.insert(0, "prop_sim")
from tradeify_flex import *  # noqa: E402,F403

CSV = "prop_sim/data/vantage_nq_20m_mnq_1ct_trades.csv"
YEARS, PATHS, STAGGER = 3, 600, 4
DAYS = [d for d in load_days(CSV) if d[0] >= pd.Timestamp("2022-01-01")]


def slot_events(path, start, eval_n, fund_n, minp, acct):
    ev, i = [], start
    while i < len(path) - 1:
        res, j = run_eval(path, i, eval_n, acct.target, acct=acct)
        if acct.monthly:
            m = max(1, math.ceil((path[j][0] - path[i][0]).days / 30))
            ev += [(path[i][0] + pd.Timedelta(days=30 * k), "fee", acct.fee) for k in range(m)]
        else:
            ev.append((path[i][0], "fee", acct.fee))
        if res != "pass":
            i = j + 1
            continue
        p, k, _ = run_funded(path, j + 1, fund_n, minp, acct)
        ev += [(d, "payout", a) for d, a in p]
        i = k + 1
    return ev


def study(args):
    label, acct, k_accts, eval_n, fund_n, minp = args
    rng = np.random.default_rng(3)
    rows = []
    for pi in range(PATHS):
        path = bootstrap_days(DAYS, 140 * YEARS, rng)
        t0 = path[0][0]
        for s in range(k_accts):
            for d, kind, a in slot_events(path, s * STAGGER, eval_n, fund_n, minp, acct):
                yr = (d - t0).days // 365 + 1
                if yr <= YEARS:
                    rows.append((pi, yr, kind, a))
    df = pd.DataFrame(rows, columns=["path", "yr", "kind", "amt"])
    idx = pd.MultiIndex.from_product([range(PATHS), range(1, YEARS + 1)], names=["path", "yr"])
    pay, fee = df[df.kind == "payout"], df[df.kind == "fee"]
    g = pd.DataFrame({"payouts": pay.groupby(["path", "yr"]).size(),
                      "cash": pay.groupby(["path", "yr"]).amt.sum() * 0.9,
                      "fees": fee.groupby(["path", "yr"]).amt.sum()}).reindex(idx).fillna(0)
    g["net"] = g.cash - g.fees
    out = dict(plan=label, accts=k_accts, evalN=eval_n, fundN=fund_n, payAt=minp)
    for yr in (1, 2):
        y = g.xs(yr, level="yr")
        out.update({f"y{yr}_payouts": y.payouts.mean(), f"y{yr}_cash": y.cash.mean(),
                    f"y{yr}_fees": y.fees.mean(), f"y{yr}_net": y.net.mean(),
                    f"y{yr}_net_p50": y.net.median(), f"y{yr}_loss%": (y.net < 0).mean() * 100})
    tot = g.groupby(level="path").sum()
    out.update({f"{YEARS}y_net_mean": tot.net.mean(), f"{YEARS}y_net_p10": tot.net.quantile(.1),
                f"{YEARS}y_loss%": (tot.net < 0).mean() * 100})
    return out


if __name__ == "__main__":
    jobs = [("50K baseline", A50, 1, 3, 2, 3000)]
    for en in (3, 4):
        for fn in (2, 3, 4):
            for mp in (3000, 6000):
                jobs.append(("50K", A50, 5, en, fn, mp))
    for en in (6, 8):
        for fn in (4, 5, 6):
            for mp in (4500, 9000):
                jobs.append(("150K", A150, 5, en, fn, mp))
    with Pool() as p:
        res = pd.DataFrame(p.map(study, jobs)).round(1)
    pd.set_option("display.width", 250)
    print(res.sort_values(f"{YEARS}y_net_mean", ascending=False).to_string(index=False))
