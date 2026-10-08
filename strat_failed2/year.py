"""1-year sim: 4m Failed2 + 8/21 strict ribbon, NQ, Tradeify Select 50K -> Flex. K copy-traded slots share the same daily trades."""
import numpy as np, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import mc_tradeify as m
base=[list(x) for x in m.day_pool]; mean=m.t.nq.mean()
FEE, SPLIT, DAYS = 159, 0.9, 252
rng=np.random.default_rng(3)

class Slot:
    def __init__(s): s.new()
    def new(s):
        s.phase='eval'; s.bal=0.0; s.peak=0.0; s.dp=[]; s.edays=0; s.locked=False; s.floor=-2000
        s.wd=0; s.cyc=0.0; s.npay=0
    def day(s, trades, ek, fk):
        """returns (payout, fee) for the day"""
        fee=0.0
        if s.phase=='eval':
            if s.edays%21==0: fee=FEE
            s.edays+=1
            floor=s.peak-2000; d=0.0
            for p in trades:
                d+=p*ek
                if s.bal+d<=floor: s.new(); return 0.0, fee   # failed -> buy new eval next day
            s.bal+=d; s.dp.append(d); s.peak=max(s.peak,s.bal)
            if s.bal>=3000 and len(s.dp)>=3 and max(s.dp)<=0.4*s.bal:
                s.phase='fund'; s.bal=0.0; s.peak=0.0; s.floor=-2000; s.locked=False; s.wd=0; s.npay=0
            return 0.0, fee
        d=0.0
        for p in trades:
            d+=p*fk
            if s.bal+d<=s.floor: s.new(); return 0.0, fee      # blown -> new eval
        s.bal+=d
        if d>=150: s.wd+=1
        if not s.locked:
            s.peak=max(s.peak,s.bal); s.floor=s.peak-2000
            if s.bal>=2100: s.locked=True; s.floor=100.0
        if s.wd>=5 and s.bal>0 and (s.npay==0 or s.bal-s.cyc>0):
            amt=min(0.5*s.bal,2500)
            if amt>=250:
                s.bal-=amt; s.npay+=1; s.wd=0; s.cyc=s.bal
                if not s.locked: s.locked=True; s.floor=100.0
                return amt*SPLIT, fee
        return 0.0, fee

def year(K, ek, fk, shift, N=4000):
    out=[]; resets=[]
    for _ in range(N):
        slots=[Slot() for _ in range(K)]; net=0.0; r=0
        for _ in range(DAYS):
            tr=[p+shift for p in base[rng.integers(len(base))]]
            for s in slots:
                ph=s.phase; pay,fee=s.day(tr,ek,fk); net+=pay-fee
                if ph!=s.phase and s.phase=='eval' and s.edays==0: r+=1
        out.append(net); resets.append(r)
    o=np.array(out); return o.mean(), np.median(o), np.percentile(o,10), (o<0).mean(), np.mean(resets)

for label,shift in [("as tested",0),("half edge",-mean/2)]:
    print(label)
    for ek,fk in [(1,1),(1,2),(2,2),(3,2)]:
        r=year(1,ek,fk,shift); print(f"  1 acct eval {ek} / fund {fk}: mean ${r[0]:,.0f} med ${r[1]:,.0f} p10 ${r[2]:,.0f} P(loss) {r[3]:.0%} accts lost/yr {r[4]:.1f}")
    for K in (3,5):
        r=year(K,2,2,shift); print(f"  {K} accts copy, 2/2: mean ${r[0]:,.0f} med ${r[1]:,.0f} p10 ${r[2]:,.0f} P(loss) {r[3]:.0%} accts lost/yr {r[4]:.1f}")
