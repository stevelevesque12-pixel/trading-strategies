import numpy as np, pandas as pd, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import year as y
import bt_failed2 as b, multi as mu
# gross vs net check on NQ early years
d=b.load('real_nq_15m_2016-05-29_2026-08-25.parquet'); b.TICK,b.PV,b.COMM=0.25,20.0,0.0
t=b.run(d,tmode=0.5); b.COMM=0.62
g=t.groupby(t.time.dt.year<2021).pnl.apply(mu.pf)
print('NQ 15m 0.5R PF with zero costs: 2021+ %.2f | 2016-2020 %.2f'%(g[False],g[True]))
b.TICK,b.PV=0.25,2.0
comps={'MNQ 5m':[('mnq','5m')],'MNQ 5m + 15m':[('mnq','5m'),('mnq','15m')],'MNQ 5m + 15m + MGC 5m':[('mnq','5m'),('mnq','15m'),('mgc','5m')]}
lo,hi=pd.Timestamp('2026-05-11',tz=b.NY),pd.Timestamp('2026-08-26',tz=b.NY)
for name,parts in comps.items():
    tt=[]; 
    for s,tf in parts:
        t,sess=mu.trades(s,tf); tt.append(t)
    t=pd.concat(tt).sort_values('time'); t=t[(t.time>=lo)&(t.time<hi)]
    S=[s for s in sorted(set(b.load('real_mnq_5m_2026-05-10_2026-08-25.parquet').between_time('09:30','15:59').index.normalize())) if s>=lo]
    t['day']=t.time.dt.normalize(); y.base=[t.loc[t.day==s,'pnl'].to_list() for s in S]; mean=t.pnl.mean()
    k=len(parts); print(f"\n{name}: {len(t)} trades / {len(S)} sessions = {len(t)/len(S):.2f} per session")
    best=None
    for ek in (10,20,40//k if k>1 else 30):
        for fk in (10,20//k if k>1 else 20):
            if ek*k>40 or fk*k>20: continue
            rh=y.year(1,ek,fk,-mean/2,N=1500); rt=y.year(1,ek,fk,0,N=1500)
            print(f"  {ek}/{fk} micros per strategy: tested ${rt[0]:,.0f} | half-edge ${rh[0]:,.0f} p10 ${rh[2]:,.0f} P(loss) {rh[3]:.0%}")
