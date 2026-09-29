"""Flat-by-close variants of tradingview/supertrend_long.pine on 15m bars:
hold overnight vs flatten at 16:00 ET with (a) fresh flips only, (b) re-entry
at the 18:00 reopen, (c) re-entry at 09:30. No safety stop modelled.

Usage: python backtest/supertrend_flat_by_close.py <path to real_nq_15m_*.parquet>
(the NQ file lives on the claude/compassionate-hamilton-w5dvhm branch)."""
import pandas as pd, numpy as np, sys
NQ=sys.argv[1]
files={'MNQ':(NQ,2.0,0.25),'MGC':('sample_data/real_multi_instrument/real_gc_15m_2016-05-26_2026-08-25.parquet',10.0,0.1),
       'MES':('sample_data/real_multi_instrument/real_es_15m_2016-05-29_2026-08-25.parquet',5.0,0.25)}
def trend(d,per,mult):
    h,l,c=d.high.values,d.low.values,d.close.values
    pc=np.r_[np.nan,c[:-1]]
    tr=np.nanmax(np.c_[h-l,np.abs(h-pc),np.abs(l-pc)],axis=1)
    atr=pd.Series(tr).rolling(per).mean().values
    src=(h+l)/2;n=len(c);up=src-mult*atr;dn=src+mult*atr;t=np.ones(n,int)
    for i in range(1,n):
        u1=up[i-1] if not np.isnan(up[i-1]) else up[i]; d1=dn[i-1] if not np.isnan(dn[i-1]) else dn[i]
        if not np.isnan(up[i]) and c[i-1]>u1: up[i]=max(up[i],u1)
        if not np.isnan(dn[i]) and c[i-1]<d1: dn[i]=min(dn[i],d1)
        x=t[i-1]
        if x==-1 and c[i]>d1: x=1
        elif x==1 and c[i]<u1: x=-1
        t[i]=x
    return t
def run(d,t,pv,tick,flat=None,resume=None,start='2020-01-01'):
    c=d.close.values; ny=d.index.tz_convert('America/New_York')
    om=ny.hour*60+ny.minute; cm=(om+15)%1440   # bar close minute (15m bars)
    ok=d.index>=pd.Timestamp(start,tz='UTC')
    cost=2*1.0+2*tick*pv; pos=0; e=0; trades=[]; real=0; eq=[]
    F=None if flat is None else flat//100*60+flat%100
    R=None if resume is None else resume//100*60+resume%100
    def allowed(m):  # may hold/enter at this bar close
        if F is None: return True
        # blocked from flatten time until resume/reopen (18:00) ... treat window [F, R) as no-trade, wrapping midnight
        r=R if R is not None else 18*60
        return not ((m>=F and m<r) if F<r else (m>=F or m<r))
    for i in range(1,len(c)):
        if ok[i]:
            m=cm[i]
            if pos and (t[i]==-1 or not allowed(m)):
                trades.append((c[i]-e)*pv-cost); real+=trades[-1]; pos=0
            elif not pos and allowed(m) and t[i]==1 and (t[i-1]==-1 or resume is not None):
                pos,e=1,c[i]
        eq.append(real+(pos*(c[i]-e)*pv))
    eq=np.array(eq);dd=(np.maximum.accumulate(eq)-eq).max();tt=np.array(trades)
    yrs=(d.index[-1]-pd.Timestamp(start,tz='UTC')).days/365.25
    pf=tt[tt>0].sum()/-tt[tt<0].sum()
    return f"net {tt.sum():>8,.0f}  /yr {tt.sum()/yrs:>7,.0f}  maxDD {dd:>7,.0f}  trades {len(tt):>5}  PF {pf:.2f}"
for name,(f,pv,tick) in files.items():
    d=pd.read_parquet(f); print(f'\n== {name}')
    for mult in [5,7,8.5]:
        t=trend(d,10,mult)
        print(f' x{mult} hold overnight          ',run(d,t,pv,tick))
        print(f' x{mult} flat 16:00, flips only   ',run(d,t,pv,tick,flat=1600))
        print(f' x{mult} flat 16:00, resume 18:00 ',run(d,t,pv,tick,flat=1600,resume=1800))
        print(f' x{mult} flat 16:00, resume 09:30 ',run(d,t,pv,tick,flat=1600,resume=930))
