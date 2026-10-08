import bt_failed2 as b, pandas as pd, numpy as np, glob, os
SPEC={'mnq':(0.25,2.0),'mes':(0.25,5.0),'mym':(1.0,0.5),'m2k':(0.1,5.0),'mgc':(0.1,10.0),'mcl':(0.01,100.0),'sil':(0.005,1000.0)}
def trades(sym, tf, tmode=0.5):
    f=glob.glob(b.DATA+f'real_{sym}_{tf}_*.parquet')[0]; d=b.load(os.path.basename(f))
    b.TICK,b.PV=SPEC[sym]; t=b.run(d,tmode=tmode); b.TICK,b.PV=0.25,2.0
    t['sym']=sym; t['tf']=tf
    sess=sorted(set(d.between_time('09:30','15:59').index.normalize()))
    return t,sess
def pf(p):
    l=-p[p<=0].sum(); return p[p>0].sum()/l if l else 9
if __name__=='__main__':
    rows=[]
    for tf,split in [('5m','2026-07-02'),('15m','2026-03-01')]:
        h=pd.Timestamp(split,tz=b.NY)
        for sym in SPEC:
            t,sess=trades(sym,tf); p=t.pnl
            # risk per trade on stops
            sl=t[t.why=='SL']; risk=(abs(sl.entry-sl.exit)*SPEC[sym][1]+1.24).median() if len(sl) else np.nan
            eq=p.cumsum(); dd=(eq-eq.cummax().clip(lower=0)).min()
            rows.append(dict(tf=tf,sym=sym,sess=len(sess),n=len(t),wr=round(100*(p>0).mean()),pf=round(pf(p),2),
                 H1=round(pf(p[t.time<h]),2),H2=round(pf(p[t.time>=h]),2),net=round(p.sum()),dd=round(dd),med_risk=round(risk)))
    print(pd.DataFrame(rows).to_string(index=False))
