import numpy as np, pandas as pd, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import year as y, mc_tradeify as m
import bt_failed2 as b
d4=b.resample(b.load('real_mnq_1m_2026-08-02_2026-08-25.parquet'),4)
sess=sorted(set(d4.between_time('09:30','15:59').index.normalize()))
for lab,kw in [('magnitude',{}),('0.5R',{'tmode':0.5}),('0.75R',{'tmode':0.75}),('1R',{'tmode':1.0}),('2R',{'tmode':2.0}),('3R',{'tmode':3.0})]:
    t=b.run(d4,**kw); t['nq']=(t.pnl+1.24)*10-4.0; t['day']=t.time.dt.normalize()
    y.base=[t.loc[t.day==s,'nq'].to_list() for s in sess]; mean=t.nq.mean()
    for sc,sh in [('tested',0),('half',-mean/2)]:
        r=y.year(1,3,2,sh,N=3000)
        print(f"{lab:9s} {sc:6s}: mean ${r[0]:,.0f} p10 ${r[2]:,.0f} P(loss) {r[3]:.0%} lost/yr {r[4]:.1f}")
