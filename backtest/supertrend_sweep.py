"""Replica of tradingview/supertrend_long.pine on 15m bars: multiplier sweep,
2016-2019 check, long+short variant and buy-and-hold comparison.

Usage: python backtest/supertrend_sweep.py <path to real_nq_15m_*.parquet>
(the NQ file lives on the claude/compassionate-hamilton-w5dvhm branch)."""
import pandas as pd, numpy as np, sys
NQ=sys.argv[1]
files={'MNQ (NQ data)':(NQ,2.0,0.25),
       'MGC (GC data)':('sample_data/real_multi_instrument/real_gc_15m_2016-05-26_2026-08-25.parquet',10.0,0.1),
       'MES (ES data)':('sample_data/real_multi_instrument/real_es_15m_2016-05-29_2026-08-25.parquet',5.0,0.25)}
def run(d,per,mult,pv,tick,short=False,start='2020-01-01'):
    h,l,c=d.high.values,d.low.values,d.close.values
    pc=np.r_[np.nan,c[:-1]]
    tr=np.nanmax(np.c_[h-l,np.abs(h-pc),np.abs(l-pc)],axis=1)
    atr=pd.Series(tr).rolling(per).mean().values
    src=(h+l)/2; n=len(c)
    up=src-mult*atr; dn=src+mult*atr; tr_=np.ones(n,int)
    for i in range(1,n):
        u1=up[i-1] if not np.isnan(up[i-1]) else up[i]; d1=dn[i-1] if not np.isnan(dn[i-1]) else dn[i]
        if not np.isnan(up[i]) and c[i-1]>u1: up[i]=max(up[i],u1)
        if not np.isnan(dn[i]) and c[i-1]<d1: dn[i]=min(dn[i],d1)
        t=tr_[i-1]
        if t==-1 and c[i]>d1: t=1
        elif t==1 and c[i]<u1: t=-1
        tr_[i]=t
    ok=d.index>=pd.Timestamp(start,tz='UTC')
    pos=0; entry=0; trades=[]; eq=[]; cost=2*1.0+2*tick*pv; realized=0
    for i in range(1,n):
        if ok[i]:
            if tr_[i]==1 and tr_[i-1]==-1:
                if pos==-1: trades.append((entry-c[i])*pv-cost); realized+=trades[-1]
                pos,entry=1,c[i]
            elif tr_[i]==-1 and tr_[i-1]==1:
                if pos==1: trades.append((c[i]-entry)*pv-cost); realized+=trades[-1]
                pos=-1 if short else 0; entry=c[i]
        eq.append(realized+(pos*(c[i]-entry)*pv if pos else 0))
    eq=np.array(eq); dd=(np.maximum.accumulate(eq)-eq).max()
    t=np.array(trades); yrs=(d.index[-1]-max(d.index[0],pd.Timestamp(start,tz='UTC'))).days/365.25
    pf=t[t>0].sum()/-t[t<0].sum() if (t<0).any() else np.inf
    return dict(net=round(t.sum()),per_yr=round(t.sum()/yrs),maxDD=round(dd),trades=len(t),win=round((t>0).mean()*100,1),PF=round(pf,2))
for name,(f,pv,tick) in files.items():
    d=pd.read_parquet(f)
    s=d[d.index>='2020-01-01']
    bh=(s.close.iloc[-1]-s.close.iloc[0])*pv; bhdd=((s.close.cummax()-s.close)*pv).max()
    print(f'\n== {name}: buy&hold 1 contract since 2020: {bh:,.0f}  maxDD {bhdd:,.0f}')
    for mult in [3,5,7,8.5,10,12]:
        print(f'  mult {mult:>4} long-only 2020+ ', run(d,10,mult,pv,tick))
    print('  8.5 long-only 2016-2019     ', run(d[d.index<'2020-01-01'],10,8.5,pv,tick,start='2016-01-01'))
    print('  8.5 long+short 2020+        ', run(d,10,8.5,pv,tick,short=True))
