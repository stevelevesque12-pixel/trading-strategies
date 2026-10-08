"""1m-source MNQ study: TF 1-5m, split halves, EMA-fast, bar-alignment and target sensitivity."""
import sys
import pandas as pd
from bt_failed2 import *
m1=load("real_mnq_1m_2026-08-02_2026-08-25.parquet")
days=sorted(set(m1.index[(m1.index.hour>=10)&(m1.index.hour<16)].date)); print(len(days),"RTH sessions",days[0],days[-1])
mid=pd.Timestamp(days[len(days)//2],tz=NY)
pd.set_option("display.width",200); rows=[]
for n in (1,2,3,4,5):
  for tm in ("mag",0.5):
    t=run(resample(m1,n),tmode=tm)
    rows.append(dict(tf=n,tgt=tm,**stats(t),pfH1=stats(t[t.time<mid]).get("pf"),pfH2=stats(t[t.time>=mid]).get("pf"),nH1=int((t.time<mid).sum())))
print(pd.DataFrame(rows).to_string(index=False))
print("\nEMA fast sensitivity, 0.5R, PF (trades)")
rows=[]
for n in (1,2,3,4,5):
  r={"tf":n}
  for f in (6,7,8,9,10):
    s_=stats(run(resample(m1,n),tmode=0.5,fast=f)); r[f"ema{f}"]=f'{s_.get("pf")} ({s_["trades"]})'
  rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("\nBar-alignment sensitivity (offset minutes), 0.5R, PF (trades)")
rows=[]
for n in (2,3,4,5):
  r={"tf":n}
  for off in range(n):
    s_=stats(run(resample(m1,n,off),tmode=0.5)); r[f"off{off}"]=f'{s_.get("pf")} ({s_["trades"]}, ${s_.get("net",0)})'
  rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
print("\nTarget sweep 4m / 2m: PF (trades, net)")
rows=[]
for n in (2,4):
  r={"tf":n}
  for tm in (0.3,0.5,0.75,1.0,1.5):
    s_=stats(run(resample(m1,n),tmode=tm)); r[f"{tm}R"]=f'{s_.get("pf")} ({s_["trades"]}, wr {s_.get("wr")})'
  rows.append(r)
print(pd.DataFrame(rows).to_string(index=False))
