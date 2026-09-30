"""Payouts per year for the recommended plan (eval 3 MNQ, funded 2 MNQ, payout once profit >= $3k):
historical replay + 5-year block-bootstrap Monte Carlo. Run from repo root."""
import sys, math; sys.path.insert(0,'prop_sim')
from tradeify_flex import *
days=load_days('prop_sim/data/vantage_nq_20m_mnq_1ct_trades.csv')
EN,FN,MP=3,2,3000

def campaign_events(days, start_idx=0):
    """Same as campaign() but returns dated events: fees (per month started) and payouts."""
    ev=[]; i=start_idx; accts=[]
    while i < len(days)-1:
        res,j=run_eval(days,i,EN,3000)
        m=max(1,math.ceil((days[j][0]-days[i][0]).days/30))
        for k in range(m): ev.append((days[i][0]+pd.Timedelta(days=30*k),'fee',159.0))
        if res!='pass': i=j+1; continue
        p,k,st=run_funded(days,j+1,FN,MP)
        accts.append((days[i][0].date(),days[j][0].date(),days[k][0].date(),st,len(p),sum(a for _,a in p)))
        for d,a in p: ev.append((d,'payout',a))
        i=k+1
    return ev,accts

 # 1) Historical replay, continuous from May 2019
ev,accts=campaign_events(days)
e=pd.DataFrame(ev,columns=['date','kind','amt']); e['year']=e.date.dt.year
y=e.pivot_table(index='year',columns='kind',values='amt',aggfunc=['count','sum'],fill_value=0)
out=pd.DataFrame({'evals_months':y[('count','fee')],'fees':y[('sum','fee')],
                 'payouts':y[('count','payout')] if ('count','payout') in y else 0,
                 'gross_payout':y[('sum','payout')] if ('sum','payout') in y else 0})
out['to_you_90pct']=out.gross_payout*0.9; out['net']=out.to_you_90pct-out.fees
print('=== Historical replay (one account at a time, 2019-05 -> 2026-09) ===')
print(out.round(0).to_string()); print('TOTAL', out.sum().round(0).to_dict())
print('\nFunded accounts (eval start, pass, end, status, #payouts, gross):')
for a in accts: print(' ',a)

# 1b) Each calendar year started fresh on Jan 1 (independent years)
print('\n=== Each year started fresh on Jan 1 (payouts within that year only) ===')
rows=[]
for yr in range(2020,2027):
   sub=[d for d in days if d[0].year==yr]
   ev,_=campaign_events(sub)
   pay=[a for d,k,a in ev if k=='payout']; fee=sum(a for d,k,a in ev if k=='fee')
   rows.append((yr,len(pay),sum(pay)*0.9,fee,sum(pay)*0.9-fee))
print(pd.DataFrame(rows,columns=['year','payouts','to_you','fees','net']).round(0).to_string(index=False))

# 2) Monte Carlo from 2022+ days, 5-year horizon, per-year stats
D=[d for d in days if d[0].year>=2022]
rng=np.random.default_rng(11); per=[]
for pi in range(2000):
    path=bootstrap_days(D,720,rng)
    ev,accts=campaign_events(path)
    t0=path[0][0]
    for d,k,a in ev: per.append((pi, int((d-t0).days//365)+1, k, a))
p=pd.DataFrame(per,columns=['path','yr','kind','amt']); p=p[p.yr<=5]

idx=pd.MultiIndex.from_product([range(2000),range(1,6)],names=['path','yr'])
pay=p[p.kind=='payout']; fee=p[p.kind=='fee']
g=pd.DataFrame({'payouts':pay.groupby(['path','yr']).size(),'gross':pay.groupby(['path','yr']).amt.sum(),
                'fees':fee.groupby(['path','yr']).amt.sum()}).reindex(idx,fill_value=0).fillna(0)
g['net']=g.gross*0.9-g.fees
print('\n=== Monte Carlo, 2000 paths bootstrapped from 2022+ trades, plan 3/2/$3000 ===')
r=g.groupby(level='yr').agg(payouts_mean=('payouts','mean'),p10=('payouts',lambda s:s.quantile(.1)),p50=('payouts','median'),
    p90=('payouts',lambda s:s.quantile(.9)),zero_pct=('payouts',lambda s:(s==0).mean()*100),to_you_mean=('gross',lambda s:(s*.9).mean()),
    fees_mean=('fees','mean'),net_mean=('net','mean'),net_p50=('net','median'),loss_pct=('net',lambda s:(s<0).mean()*100))
print(r.round(1).to_string())
print('\npayouts-per-year distribution (% of path-years, all 5 years pooled):')
print((g.payouts.value_counts(normalize=True).sort_index()*100).round(1).to_string())
c=g.groupby(level='path').sum()
print('\n5-year cumulative: payouts mean %.1f p10 %d p50 %d p90 %d; net mean %.0f p10 %.0f p50 %.0f; P(net<0) %.0f%%'%(
 c.payouts.mean(),c.payouts.quantile(.1),c.payouts.median(),c.payouts.quantile(.9),c.net.mean(),c.net.quantile(.1),c.net.median(),(c.net<0).mean()*100))
