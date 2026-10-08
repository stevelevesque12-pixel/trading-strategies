import numpy as np, pandas as pd, mc_tradeify as m
base=[list(x) for x in m.day_pool]; mean=m.t.nq.mean()
N=20000
for label,shift in [("as tested",0),("half edge",-mean/2),("no edge",-mean)]:
    m.day_pool=[[p+shift for p in d] for d in base]
    ev=[m.eval_run() for _ in range(N)]; r=pd.Series([e[0] for e in ev]).value_counts(normalize=True)
    dp=[e[1] for e in ev if e[0]=="pass"]
    fr=[m.funded_run() for _ in range(N)]; paid=np.array([f[0] for f in fr]); bl=np.array([f[2] for f in fr]); npay=np.array([f[1] for f in fr])
    print(f"{label}: pass {r.get('pass',0):.0%} fail {r.get('fail',0):.0%} timeout {r.get('timeout',0):.0%} medDays {np.median(dp) if dp else 0:.0f} | funded blown {bl.mean():.0%} >=1pay {(npay>=1).mean():.0%} medPaid {np.median(paid):,.0f} p10 {np.percentile(paid,10):,.0f} p90 {np.percentile(paid,90):,.0f}")
