"""Replica of tradingview/triple_supertrend_tsv.pine on 15m bars (1 contract,
$1/side + 1 tick slippage, 1:2 R:R). Compares the article's RSI<20 rule
(never fires) with Stochastic RSI variants and four stop placements.

Usage: python backtest/triple_supertrend_tsv.py <path to real_nq_15m_*.parquet>
(the NQ file lives on the claude/compassionate-hamilton-w5dvhm branch)."""
import pandas as pd, numpy as np, sys
NQ=sys.argv[1]
files={'MNQ':(NQ,2.0,0.25),'MGC':('sample_data/real_multi_instrument/real_gc_15m_2016-05-26_2026-08-25.parquet',10.0,0.1),
       'MES':('sample_data/real_multi_instrument/real_es_15m_2016-05-29_2026-08-25.parquet',5.0,0.25)}
def rma(x,n):
    return pd.Series(x).ewm(alpha=1/n,adjust=False).mean().values
def supertrend(h,l,c,per,mult):
    pc=np.r_[c[0],c[:-1]]
    tr=np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))); atr=rma(tr,per)
    hl2=(h+l)/2; ub=hl2+mult*atr; lb=hl2-mult*atr; n=len(c)
    fu=ub.copy(); fl=lb.copy(); d=np.ones(n); st=np.zeros(n)   # d=1 up
    for i in range(1,n):
        fl[i]=lb[i] if (lb[i]>fl[i-1] or c[i-1]<fl[i-1]) else fl[i-1]
        fu[i]=ub[i] if (ub[i]<fu[i-1] or c[i-1]>fu[i-1]) else fu[i-1]
        if d[i-1]==1: d[i]= -1 if c[i]<fl[i] else 1
        else: d[i]= 1 if c[i]>fu[i] else -1
        st[i]=fl[i] if d[i]==1 else fu[i]
    return d,st

def stoch(x,n):
    s=pd.Series(x); lo=s.rolling(n).min(); hi=s.rolling(n).max(); return (100*(s-lo)/(hi-lo)).values
def prep(d,mode):
    h,l,c,v=d.high.values,d.low.values,d.close.values,d.volume.values
    sts=[supertrend(h,l,c,p,m) for p,m in [(10,1),(11,2),(12,3)]]
    ema=pd.Series(c).ewm(span=200,adjust=False).mean().values; ch=np.r_[0,np.diff(c)]
    g=rma(np.maximum(ch,0),14); lo=rma(np.maximum(-ch,0),14); rsi=100-100/(1+g/np.where(lo==0,1e-12,lo))
    k=pd.Series(stoch(rsi,14)).rolling(3).mean().values; dd=pd.Series(k).rolling(3).mean().values
    tsv=pd.Series(v*ch).rolling(13).sum().values
    up=sum((s[0]==1).astype(int) for s in sts); dn=3-up
    if mode=='rsi': lo_s=rsi<20; hi_s=rsi>80
    elif mode=='below': lo_s=k<20; hi_s=k>80
    else:
        kp=np.r_[np.nan,k[:-1]]; dp=np.r_[np.nan,dd[:-1]]
        lo_s=(k>dd)&(kp<=dp)&(k<20); hi_s=(k<dd)&(kp>=dp)&(k>80)
    pc=np.r_[c[0],c[:-1]]; atr=rma(np.maximum(h-l,np.maximum(abs(h-pc),abs(l-pc))),14)
    L=(c>ema)&(up>=2)&lo_s&(tsv>0); Sh=(c<ema)&(dn>=2)&hi_s&(tsv<0)
    return h,l,c,d.open.values,sts,atr,L,Sh
def run(d,P,pv,tick,stop='st2',rr=1.5,shorts=True,start='2016-06-01'):
    h,l,c,o,sts,atr,L,Sh=P; n=len(c); ok=d.index>=pd.Timestamp(start,tz='UTC')
    cost=2+2*tick*pv; pos=0; trades=[]; real=0; eq=[]
    for i in range(1,n-1):
        if pos:
            # check stop first (conservative), then target, on bar i
            if (pos==1 and l[i]<=sp) or (pos==-1 and h[i]>=sp):
                px=sp if (pos==1 and o[i]>sp) or (pos==-1 and o[i]<sp) else o[i]
                trades.append(pos*(px-e)*pv-cost); pos=0
            elif (pos==1 and h[i]>=tp) or (pos==-1 and l[i]<=tp):
                px=tp if (pos==1 and o[i]<tp) or (pos==-1 and o[i]>tp) else o[i]
                trades.append(pos*(px-e)*pv-cost); pos=0
        if not pos and ok[i]:
            for side,sig in ((1,L[i]),(-1,Sh[i] and shorts)):
                if not sig: continue
                e=o[i+1]
                if stop=='atr': sp=e-side*1.5*atr[i]
                else:
                    # nearest supertrend line on the protective side, from the chosen one
                    k={'st1':0,'st2':1,'st3':2}[stop]; sp=sts[k][1][i]
                risk=side*(e-sp)
                if risk<=0: break
                tp=e+side*rr*risk; pos=side; break
            # entry fills at i+1 open; stop/target checked from bar i+1
        eq.append(real+sum(trades)-real)
    t=np.array(trades)
    if len(t)==0: return 'no trades'
    yrs=(d.index[-1]-pd.Timestamp(start,tz='UTC')).days/365.25
    eqc=np.cumsum(t); dd=(np.maximum.accumulate(np.r_[0,eqc])-np.r_[0,eqc]).max()
    pf=t[t>0].sum()/max(-t[t<0].sum(),1e-9)
    return f"net {t.sum():>8,.0f} /yr {t.sum()/yrs:>6,.0f} DD {dd:>6,.0f} trades {len(t):>4} win {100*(t>0).mean():4.1f}% PF {pf:.2f}"

for name,(f,pv,tick) in files.items():
    d=pd.read_parquet(f)
    for mode in ['rsi','below','cross']:
        P=prep(d,mode); print(f'\n== {name} {mode}: {P[6].sum()} long / {P[7].sum()} short signal bars')
        if P[6].sum()+P[7].sum()==0: continue
        for stop in ['st1','st2','st3','atr']:
            print(f'  stop {stop:4} 2R ',run(d,P,pv,tick,stop,2.0))
