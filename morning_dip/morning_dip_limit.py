"""Morning Dip Limit: reference implementation (research lead, not a proven edge).

Input: one file per trading day of 1-second NQ OHLC bars with columns t (UTC epoch seconds, bar start), o, h, l, c.
Feed each day from 08:25 CT to 15:00 CT; indicators warm up inside the day, which is part of the rule as tested.
Output: one row per order (filled or not); trades are rows with exit_i >= 0.

Rules (defaults below): 3-minute candles on a UTC grid (candle start = t // 180 * 180 + offset), ATR = simple mean of the last 14
true ranges, ER = |close change over 15 candles| / sum of |close changes|. From 09:00 to 11:00 CT, when ER <= 0.35, one second after
a candle closes rest a buy limit at close - 1.0 x ATR (rounded down to a tick). Fill only if price trades 1 tick through the limit.
Cancel after 9 minutes. Target = signal candle high-low midpoint; stop = limit - 1.5 x ATR (min 4 ticks); exit after 15 minutes;
flat at 11:00 CT. One order or position at a time. Stops and market exits slip 1 tick; the fill second can stop but not target;
inside each second the adverse move is assumed first. $2.25 per side per NQ.

Usage: python morning_dip_limit.py DATA_DIR [--offset SECONDS] [--tf 180] [--out trades.csv]
DATA_DIR holds YYYY-MM-DD.csv or .parquet files.
"""
import sys, argparse
from pathlib import Path
import numpy as np, pandas as pd
from dataclasses import dataclass, asdict, replace
try:
    from numba import njit
except ImportError:          # pure-python fallback (slow)
    def njit(*a,**k):
        return (lambda f:f) if not (a and callable(a[0])) else a[0]
CT='America/Chicago'


@dataclass(frozen=True)
class Config:
    tf:int=180
    atr_n:int=14
    er_n:int=15
    k:float=1.
    er_threshold:float=.35
    stop_atr:float=1.5
    reward_scale:float=1.
    ttl_s:float=540.
    hold_s:float=900.
    start_minute:float=540.
    end_minute:float=660.
    latency:float=1.
    entry_through:float=1.
    target_through:float=1.
    stop_ticks:float=1.
    market_ticks:float=1.
    fee:float=2.25
    pv:float=20.
    side:int=1
    gate:int=1  # 1 ER; 0 open; 2 provided random mask, ER warmup retained
    min_stop_ticks:float=4.
    min_reward_ticks:float=2.
    stall_after:float=0.
    stall_progress:float=.25
    causal_filter:int=0
    cancel_latency:float=0.
    entry_mode:int=0  # 1: random-entry market control; fixed bracket distances

def candles(raw,offset=0,cfg=Config(),resolution=1):
    a=raw[['t','o','h','l','c']].to_numpy(float);t=a[:,0].astype(np.int64)
    buckets=(t-offset)//cfg.tf*cfg.tf+offset
    starts=np.r_[0,1+np.flatnonzero(np.diff(buckets))];ends=np.r_[starts[1:],len(t)]
    q=np.zeros((len(starts),14));prev_good=False;seg_start=0
    gaps=np.r_[resolution,np.diff(t)]
    for j,(s,e) in enumerate(zip(starts,ends)):
        start=buckets[s];end=start+cfg.tf
        op=a[s,1];hi=a[s:e,2].max();lo=a[s:e,3].min();close=a[e-1,4]
        fresh=end-t[e-1]-resolution
        good=(fresh<=5 if resolution==1 else fresh==0) and gaps[s:e].max()<=max(30,resolution)
        good=good and start>=t[0] and end<=t[-1]+resolution
        if resolution==60:good=good and e-s==cfg.tf//60
        if j==0 or start!=q[j-1,0] or not good or not prev_good:seg_start=j
        tr=hi-lo if j==seg_start else max(hi-lo,abs(hi-q[j-1,1]),abs(lo-q[j-1,1]))
        # columns end,close,high,low,atr,ct_s,er,healthy,tr,segment_start,open,count,first,last
        q[j]=[end,close,hi,lo,np.nan,0,np.nan,good,tr,seg_start,op,e-s,t[s],t[e-1]]
        if j-seg_start+1>=cfg.atr_n:q[j,4]=q[j-cfg.atr_n+1:j+1,8].mean()
        if j-seg_start>=cfg.er_n:
            closes=q[j-cfg.er_n:j+1,1];den=np.abs(np.diff(closes)).sum()
            if den>0:q[j,6]=abs(closes[-1]-closes[0])/den
        prev_good=good
    clock=pd.to_datetime(q[:,0],unit='s',utc=True).tz_convert(CT)
    q[:,5]=clock.hour*3600+clock.minute*60+clock.second
    return q

COLS=['signal_end','arm','expiry','entry_i','exit_i','entry_t','exit_t','level','stop','target',
      'side','entry','exit','reason','gross','net','mae','mfe','heat','peak','open_dd','free_at',
      'atr','er','reward','stop_distance','fill_gap','fillbar_stop','origin','signal_close',
      'signal_h','signal_l']
REASONS={0:'no_fill',1:'stop',2:'target',3:'time',4:'schedule',5:'stop_gap',6:'target_gap',7:'feed_gap_exit',8:'unresolved',9:'gap_cancel',10:'structural_time'}

@njit(cache=True)
def simulate(a,b,p,allow,flat,resolution):
    (k,threshold,stop_atr,reward_scale,ttl,hold,startminute,endminute,latency,entry_ticks,
     target_ticks,stop_ticks,market_ticks,fee,pv,side,gate,min_stop,min_reward,
     stall_after,stall_progress,causal_filter,cancel_latency,entry_mode)=p
    out=np.full((len(b),32),np.nan);n=0;free=-1.;eligible=0;pre_er=0;pass_er=0
    for i in range(len(b)):
        end,C,H,L,atr,ct,er=b[i,:7]
        if not b[i,7] or not np.isfinite(atr) or atr<=0:continue
        if ct<startminute*60 or ct>=endminute*60:continue
        if gate!=0 and not np.isfinite(er):continue
        if side==1:level=np.floor((C-k*atr)*4+1e-8)/4
        else:level=np.ceil((C+k*atr)*4-1e-8)/4
        if side*(C-level)<.25-1e-8:continue
        reward=np.ceil(side*((H+L)/2-level)*reward_scale*4-1e-8)/4
        if reward<min_reward*.25-1e-8:continue
        pre_er+=1
        if np.isfinite(er) and er<=threshold:pass_er+=1
        if gate==1 and (not np.isfinite(er) or er>threshold):continue
        if not allow[i]:continue
        if causal_filter==1 and C<b[i,10]-.5*atr:continue
        eligible+=1
        if end<free:continue
        arm=end+(latency if resolution==1 else 0)
        expiry=min(end+ttl,flat)
        if arm>=expiry:continue
        stopdist=max(min_stop*.25,np.ceil(stop_atr*atr*4-1e-8)/4)
        stop=level-side*stopdist;target=level+side*reward
        entry=-1;exitidx=-1;why=0;px=np.nan;fillgap=0.;fillstop=0.
        available=expiry;begin=np.searchsorted(a[:,0],arm)
        for j in range(begin,len(a)):
            now=a[j,0]
            if now>=min(expiry+cancel_latency,flat):break
            if resolution==1 and ((j>0 and now-a[j-1,0]>30) or (j==begin and now-arm>30)):
                why=9;available=min(now,expiry);break
            if entry_mode==1:
                entry=j;level=a[j,1]+side*market_ticks*.25
                stop=level-side*stopdist;target=level+side*reward;break
            if (side==1 and a[j,3]<=level-entry_ticks*.25) or (side==-1 and a[j,2]>=level+entry_ticks*.25):
                entry=j;fillgap=1. if side*(a[j,1]-level)<0 else 0.;break
        mae=0.;mfe=0.;mine=-fee;maxe=-fee;peak=0.;dd=fee;structural=np.inf
        if entry>=0:
            due=a[entry,0]+hold
            for j in range(entry,len(a)):
                now,op,hi,lo,cl=a[j]
                first=j==entry
                opening=side*(op-level)
                if not first:
                    if side*(op-target)>=target_ticks*.25-1e-9:opening=reward
                    eq=opening*pv-fee;mine=min(mine,eq);maxe=max(maxe,eq);dd=max(dd,peak-eq);peak=max(peak,eq)
                    mae=max(mae,-opening);mfe=max(mfe,opening)
                # Resting limit may receive a pessimistic fill at its level on a gap.
                if side*(op-stop)<=0:
                    why=5;px=op-side*stop_ticks*.25
                elif not first and side*(op-target)>=target_ticks*.25-1e-9:
                    why=6;px=target
                elif not first and (now>=flat or now>=due or now>=structural or (resolution==1 and now-a[j-1,0]>30)):
                    why=4 if now>=flat else (3 if now>=due else (10 if now>=structural else 7))
                    px=op-side*market_ticks*.25
                elif (side==1 and lo<=stop) or (side==-1 and hi>=stop):
                    why=1;px=stop-side*stop_ticks*.25;fillstop=1. if first else 0.
                else:
                    bad=side*((lo if side==1 else hi)-level)
                    eq=bad*pv-fee;mine=min(mine,eq);maxe=max(maxe,eq);dd=max(dd,peak-eq);peak=max(peak,eq)
                    mae=max(mae,-bad)
                    if not first and ((side==1 and hi>=target+target_ticks*.25) or (side==-1 and lo<=target-target_ticks*.25)):
                        why=2;px=target
                    else:
                        good=side*((cl if first else (hi if side==1 else lo))-level)
                        eq=good*pv-fee;maxe=max(maxe,eq);peak=max(peak,eq);mfe=max(mfe,good)
                        eq=side*(cl-level)*pv-fee;mine=min(mine,eq);dd=max(dd,peak-eq)
                        if stall_after>0 and now+resolution>=a[entry,0]+stall_after and mfe<reward*stall_progress:
                            structural=min(structural,now+resolution+(latency if resolution==1 else 0))
                        continue
                exitidx=j;break
            if exitidx<0:why=8;available=np.inf
            else:available=a[exitidx,0]+resolution
        gross=side*(px-level)*pv if exitidx>=0 else np.nan
        net=gross-2*fee
        if exitidx>=0:
            mae=max(mae,-side*(px-level));mfe=max(mfe,side*(px-level))
            mine=min(mine,net);maxe=max(maxe,gross-fee);dd=max(dd,peak-net)
        out[n,:]=np.array([end,arm,expiry,entry,exitidx,a[entry,0] if entry>=0 else np.nan,
            a[exitidx,0] if exitidx>=0 else np.nan,level,stop if entry>=0 else np.nan,
            target if entry>=0 else np.nan,side,level if entry>=0 else np.nan,px,why,gross,net,
            mae,mfe,max(0.,-mine),maxe,dd,available,atr,er,reward,stopdist,fillgap,fillstop,0.,C,H,L])
        free=available;n+=1
    return out[:n],eligible,pre_er,pass_er

def run(raw,b,date,cfg=Config(),resolution=1,allow=None,offset=0):
    vals=[cfg.k,cfg.er_threshold,cfg.stop_atr,cfg.reward_scale,cfg.ttl_s,cfg.hold_s,cfg.start_minute,cfg.end_minute,
          cfg.latency,cfg.entry_through,cfg.target_through,cfg.stop_ticks,cfg.market_ticks,cfg.fee,cfg.pv,cfg.side,cfg.gate,
          cfg.min_stop_ticks,cfg.min_reward_ticks,cfg.stall_after,cfg.stall_progress,cfg.causal_filter,cfg.cancel_latency,cfg.entry_mode]
    flat=pd.Timestamp(date,tz=CT).timestamp()+cfg.end_minute*60
    a=raw[['t','o','h','l','c']].to_numpy(float)
    if allow is None:allow=np.ones(len(b),np.bool_)
    q,eligible,pre,passed=simulate(a,b,np.array(vals),allow,flat,resolution)
    f=pd.DataFrame(q,columns=COLS);f['date']=date;f['origin']=offset;
    f['reason_label']=f.reason.map(REASONS)
    f['R']=f.net/(f.stop_distance*cfg.pv+2*cfg.fee)
    f['hold_s']=f.exit_t-f.entry_t
    f['mae_stop_fraction']=f.mae/f.stop_distance
    trades=f[f.exit_i>=0]
    stats=dict(date=date,orders=len(f),eligible=eligible,pre_er=pre,passed_er=passed,
               fills=len(trades),net=trades.net.sum(),gross=trades.gross.sum(),wins=int((trades.net>0).sum()),
               gain=trades.net.clip(lower=0).sum(),loss=-trades.net.clip(upper=0).sum(),
               unresolved=int((f.reason==8).sum()),first_signal=float(f.signal_end.min()) if len(f) else np.nan)
    return f,stats

def metrics(trades,dates):
    f=trades[trades.exit_i>=0].sort_values(['exit_t','origin']);x=f.net.to_numpy();eq=np.r_[0,np.cumsum(x)]
    dd=np.max(np.maximum.accumulate(eq)-eq) if len(eq) else 0
    daily=f.groupby('date').net.sum().reindex(dates,fill_value=0)
    return dict(dates=len(dates),trade_dates=f.date.nunique(),fills=len(f),net=x.sum(),gross=f.gross.sum(),
        fees=(f.gross-f.net).sum(),PF=x[x>0].sum()/-x[x<0].sum() if (x<0).any() else np.nan,
        mean_net=x.mean() if len(x) else np.nan,win_rate=np.mean(x>0) if len(x) else np.nan,
        average_R=f.R.mean(),max_closed_DD=dd,daily_sd=daily.std(),daily_variance=daily.var(),
        dollars_per_252_sessions=x.sum()/len(dates)*252 if len(dates) else np.nan,
        fills_per_session=len(f)/len(dates) if len(dates) else np.nan,
        net_without_top3_trades=x.sum()-f.net.nlargest(3).sum(),
        net_without_top3_days=x.sum()-daily.nlargest(3).sum())

def load_day(p):
    f=pd.read_parquet(p) if p.suffix=='.parquet' else pd.read_csv(p)
    return f[['t','o','h','l','c']].sort_values('t').reset_index(drop=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser(); ap.add_argument('data_dir'); ap.add_argument('--offset',type=int,default=0)
    ap.add_argument('--tf',type=int,default=180); ap.add_argument('--out',default='trades.csv')
    a=ap.parse_args(); cfg=Config(tf=a.tf); rows=[]; days=[]
    for p in sorted(Path(a.data_dir).glob('*.*')):
        if p.suffix not in ('.csv','.parquet'): continue
        raw=load_day(p); date=p.stem; days.append(date)
        b=candles(raw,a.offset,cfg); f,_=run(raw,b,date,cfg,offset=a.offset); rows.append(f)
    T=pd.concat(rows); T.to_csv(a.out,index=False)
    m=metrics(T,days); print({k:(round(v,2) if isinstance(v,float) else v) for k,v in m.items()})
