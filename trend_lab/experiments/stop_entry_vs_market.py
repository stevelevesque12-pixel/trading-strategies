import datetime as dt, numpy as np, pandas as pd
from trend_lab.data import load_bars
from trend_lab import indicators as ind
from trend_lab.metrics import compute
from trend_lab.sim import Trade
df=load_bars("15min"); days=sorted(set(df.trade_day))
o,h,l,c=(df[k].to_numpy() for k in ("open","high","low","close")); idx=df.index; n=len(df)
a=ind.atr(df,14).to_numpy(); f=ind.ema(df.close,13).to_numpy(); e=ind.ema(df.close,100).to_numpy()
he=ind.htf(df,"60min",lambda x: ind.ema(x.close,50)).to_numpy()
mod=(idx.hour*60+idx.minute).to_numpy(); cm=(mod+15)%1440; wed=idx.weekday==2; tday=pd.factorize(df.trade_day)[0]; tv=df.trade_day.to_numpy()
up=(c>he)&(he-np.roll(he,12)>0)&(c>e); dn=(c<he)&(he-np.roll(he,12)<0)&(c<e)
dipL=pd.Series(l<=f-a).rolling(6,min_periods=1).max().to_numpy()>0; dipS=pd.Series(h>=f+a).rolling(6,min_periods=1).max().to_numpy()>0
lo8=pd.Series(l).rolling(8,min_periods=1).min().to_numpy(); hi8=pd.Series(h).rolling(8,min_periods=1).max().to_numpy()
win=(cm>=480)&(cm<=870)&~((mod>=960)&(mod<1080))&~(wed&(cm>=615)&(cm<=645)); flat=(mod>=990)&(mod<1080)
T,PV,SL,RISK=0.01,100.0,0.01,300.0
def run(mode):
    trades=[]; i=1
    while i<n-1:
        ent=None
        if mode=="market":
            j=i-1  # signal bar
            for d_ in (1,-1):
                cond=(up[j] and dipL[j] and c[j]>h[j-1]) if d_==1 else (dn[j] and dipS[j] and c[j]<l[j-1])
                if cond and win[j] and tday[i]==tday[j] and not flat[i]:
                    dist=(c[j]-lo8[j] if d_==1 else hi8[j]-c[j])+0.1*a[j]; ent=(d_,o[i]+d_*SL,dist); break
        else:
            j=i-1  # setup known at close of j; stop order at high[j]/low[j] works during bar i
            for d_ in (1,-1):
                setup=(up[j] and dipL[j]) if d_==1 else (dn[j] and dipS[j])
                lvl=h[j]+T if d_==1 else l[j]-T
                hit=(h[i]>=lvl) if d_==1 else (l[i]<=lvl)
                if setup and win[j] and hit and tday[i]==tday[j] and not flat[i]:
                    px=max(o[i],lvl) if d_==1 else min(o[i],lvl)
                    dist=(px-min(lo8[j],l[i]) if d_==1 else max(hi8[j],h[i])-px)+0.1*a[j]
                    ent=(d_,px+d_*SL,dist); break
        if ent is None or not (0<ent[2]<=1.5): i+=1; continue
        d_,en,dist=ent; q=int(min(20,RISK//(dist*PV+3.24)))
        if q<1: i+=1; continue
        stop=en-d_*dist; tgt=en+d_*2*dist; part=en+d_*dist; pq=q//2 if q>=2 else 0; best=en; pnl=0.0; rem=q; k=i; reason=None
        # entry bar: for stop-entry, only check stop after fill (conservative: stop first)
        while k<n:
            if k>i and (flat[k] or tday[k]!=tday[i]): pnl+=(o[k]-d_*SL-en)*d_*PV*rem; reason="flat"; break
            if (d_==1 and l[k]<=stop) or (d_==-1 and h[k]>=stop): pnl+=(stop-d_*SL-en)*d_*PV*rem; reason="stop"; break
            if pq and ((d_==1 and h[k]>=part) or (d_==-1 and l[k]<=part)): pnl+=(part-en)*d_*PV*pq; rem-=pq; pq=0
            if (d_==1 and h[k]>=tgt) or (d_==-1 and l[k]<=tgt): pnl+=(tgt-en)*d_*PV*rem; reason="target"; break
            best=max(best,h[k]) if d_==1 else min(best,l[k])
            if (best-en)*d_>=dist: stop=max(stop,en+2*T) if d_==1 else min(stop,en-2*T)
            t_=best-d_*2.5*a[k]; stop=max(stop,t_) if d_==1 else min(stop,t_)
            k+=1
        pnl-=1.24*q
        trades.append(Trade(idx[i],idx[min(k,n-1)],tv[i],d_,en,0,stop,q,pnl,0,reason or "end")); i=k+1
    return trades
split=dt.date(2026,3,20)
for mode in ("market","stop"):
    tr=run(mode)
    for seg,fn in [("Oct-Mar",lambda d:d<split),("Mar20-Aug",lambda d:d>=split),("full",lambda d:True)]:
        m=compute([t for t in tr if fn(t.trade_day)],[d for d in days if fn(d)])
        print(f"{mode:6} {seg:9} n={m['trades']:>3} wr={m['win_rate']:5} pf={m['profit_factor']:5} net={m['net']:7} dd={m['max_dd']}")
