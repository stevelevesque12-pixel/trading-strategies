import numpy as np, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    import mc_tradeify as m
base=[list(x) for x in m.day_pool]; mean=m.t.nq.mean()
rng=np.random.default_rng(11); FEE=159; SPLIT=0.9; N=8000
def pool(k,shift): return [[(p+shift)*k for p in d] for d in base]
def life(ek,fk,shift,fund_days=120):
    m.day_pool=pool(ek,shift); r,days=m.eval_run(max_days=150)
    fees=FEE*int(np.ceil(days/21))
    if r!="pass": return -fees,0,days
    m.day_pool=pool(fk,shift); paid,npay,blown,fd=m.funded_run(days=fund_days)
    return paid*SPLIT-fees,1,days+fd
for label,shift in [("as tested",0),("half edge",-mean/2)]:
    print(label)
    for ek in (1,2,3,4):
        for fk in (1,2):
            res=np.array([life(ek,fk,shift) for _ in range(N)])
            net,ps,d=res[:,0],res[:,1],res[:,2]
            print(f"  eval {ek}NQ fund {fk}NQ: pass {ps.mean():.0%} | net/account mean ${net.mean():,.0f} med ${np.median(net):,.0f} | $/trading day {net.sum()/d.sum():,.0f}")
