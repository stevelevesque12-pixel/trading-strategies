"""Monte Carlo: Failed 2 + ribbon, 4m, 1 NQ, Tradeify Select 50K eval -> Select Flex funded."""
import numpy as np
import pandas as pd
import bt_failed2 as b

RT_NQ = 4.0
rng = np.random.default_rng(7)

m1 = b.load("real_mnq_1m_2026-08-02_2026-08-25.parquet")
d4 = b.resample(m1, 4)
t = b.run(d4)
t["nq"] = (t.pnl + 1.24) * 10 - RT_NQ
sessions = sorted(set(d4.between_time("09:30", "15:59").index.normalize()))
t["day"] = t.time.dt.normalize()
day_pool = [t.loc[t.day == s, "nq"].to_list() for s in sessions]
print("sessions", len(day_pool), "trades", len(t), "trade P&L:", [round(x) for x in t.nq])
print("daily P&L:", [round(sum(x)) for x in day_pool])


def sample_day():
    return day_pool[rng.integers(len(day_pool))]


def eval_run(dd=2000, target=3000, cons=0.40, max_days=150):
    bal, peak_eod, day_pnls = 0.0, 0.0, []
    for n in range(1, max_days + 1):
        floor = peak_eod - dd
        dp = 0.0
        for p in sample_day():
            dp += p
            if bal + dp <= floor:
                return "fail", n
        bal += dp; day_pnls.append(dp)
        peak_eod = max(peak_eod, bal)
        if bal >= target and n >= 3 and max(day_pnls) <= cons * bal:
            return "pass", n
    return "timeout", max_days


def funded_run(dd=2000, days=120, cap=2500):
    bal, peak_eod, locked = 0.0, 0.0, False
    floor = -dd
    win_days, cycle_start, paid, n_pay = 0, 0.0, 0.0, 0
    for n in range(1, days + 1):
        dp = 0.0
        for p in sample_day():
            dp += p
            if bal + dp <= floor:
                return paid, n_pay, True, n
        bal += dp
        if dp >= 150: win_days += 1
        if not locked:
            peak_eod = max(peak_eod, bal)
            floor = peak_eod - dd
            if bal >= dd + 100:
                locked, floor = True, 100.0
        if win_days >= 5 and bal > 0 and (n_pay == 0 or bal - cycle_start > 0):
            amt = min(0.5 * bal, cap)
            if amt >= 250:
                bal -= amt; paid += amt; n_pay += 1
                if not locked:
                    locked, floor = True, 100.0  # DD locks on payout request
                win_days, cycle_start = 0, bal
    return paid, n_pay, False, days


N = 20000
for dd in (2000, 2500):
    ev = [eval_run(dd=dd) for _ in range(N)]
    res = pd.Series([e[0] for e in ev]).value_counts(normalize=True)
    days_pass = [e[1] for e in ev if e[0] == "pass"]
    fr = [funded_run(dd=dd) for _ in range(N)]
    paid = np.array([f[0] for f in fr]); blown = np.array([f[2] for f in fr]); npay = np.array([f[1] for f in fr])
    print(f"\nDD ${dd}: eval pass {res.get('pass',0):.1%} fail {res.get('fail',0):.1%} timeout {res.get('timeout',0):.1%}; "
          f"median days to pass {np.median(days_pass):.0f}")
    print(f"  funded 120d: blown {blown.mean():.1%}, >=1 payout {(npay>=1).mean():.1%}, "
          f"median paid ${np.median(paid):,.0f}, mean paid ${paid.mean():,.0f}, p10 ${np.percentile(paid,10):,.0f}, p90 ${np.percentile(paid,90):,.0f}")
