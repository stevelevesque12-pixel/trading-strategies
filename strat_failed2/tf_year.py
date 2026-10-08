import numpy as np, pandas as pd, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import year as y
import bt_failed2 as b
m1=b.load('real_mnq_1m_2026-08-02_2026-08-25.parquet'); m5=b.load('real_mnq_5m_2026-05-10_2026-08-25.parquet'); m15=b.load('real_mnq_15m_2025-09-30_2026-08-25.parquet')
sets={'4m (Aug, 17 sess)':b.resample(m1,4),'5m (May-Aug, 77 sess)':m5,'15m (Sep25-Aug26)':m15}
for name,d in sets.items():
    sess=sorted(set(d.between_time('09:30','15:59').index.normalize()))
    t=b.run(d,tmode=0.5); t['u']=(t.pnl+1.24)-0.4   # per-MNQ-equivalent, NQ-rate commission
    t['day']=t.time.dt.normalize()
    y.base=[t.loc[t.day==s,'u'].to_list() for s in sess]; mean=t.u.mean()
    dd=((t.u.cumsum()-t.u.cumsum().cummax().clip(lower=0)).min())
    print(f"\n{name}: {len(t)} trades, {len(sess)} sessions, maxDD per MNQ ${dd:,.0f}")
    best=None
    for ek in (5,10,20,30):
        for fk in (5,10,15,20):
            r=y.year(1,ek,fk,-mean/2,N=1500)
            if best is None or r[0]>best[0][0]: best=(r,ek,fk)
            print(f"  eval {ek:2d} / fund {fk:2d} MNQ  half-edge: mean ${r[0]:,.0f} p10 ${r[2]:,.0f} P(loss) {r[3]:.0%} lost {r[4]:.1f}")
    r,ek,fk=best; rt=y.year(1,ek,fk,0,N=1500)
    print(f"  BEST eval {ek}/fund {fk}: half ${r[0]:,.0f} | tested ${rt[0]:,.0f} p10 ${rt[2]:,.0f}")
