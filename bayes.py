import numpy as np, json, csv, itertools
ns={}
exec(open('build3.py').read(), ns)
F=ns['F']; TEAMS=ns['TEAMS']; G=ns['G']; idx={t:i for i,t in enumerate(TEAMS)}; NAME2AB=ns['NAME2AB']
# a-priori stat directions (football logic only, no fitting to outcomes)
OFF=[('net-yards-per-pass-play',1),('yards-per-rush',1),('third-down-pct',1),('sack-rate',-1),('interception-rate',-1),('pass-td-rate',1),('yards-per-play',1),('first-downs-per-play',1),('giveaways-per-game',-1),('qb-blend-passer-rating',1),('qb-blend-sack-rate',-1)]
DEF=[('opp-net-yards-per-pass-play',-1),('opp-yards-per-rush',-1),('opp-third-down-pct',-1),('opp-sack-rate',1),('opp-interception-rate',1),('opp-pass-td-rate',-1),('opp-yards-per-play',-1),('opp-first-downs-per-play',-1),('takeaways-per-game',1)]
def z(s):
    v=np.array([F[s][t] for t in TEAMS],float); return (v-v.mean())/v.std()
off=np.mean([d*z(s) for s,d in OFF],axis=0); de=np.mean([d*z(s) for s,d in DEF],axis=0)
comp=(off+de)/2
# games
A=[];B=[];M=[]
for r in G:
    a,b=idx[r['w']],idx[r['l']]; m=int(r['wp'])-int(r['lp'])
    A.append(a);B.append(b);M.append(m)
A=np.array(A);B=np.array(B);M=np.array(M,float); n=len(M)
X=np.zeros((n,32)); X[np.arange(n),A]=1; X[np.arange(n),B]=-1
# winner is listed first; randomize orientation to avoid sign bias: use both orientations (symmetric). Fit as is (m>=0 always) would bias intercept-free model? no intercept, margins signed by listing -> all positive: bad. Flip half.
rng=np.random.RandomState(0); flip=rng.rand(n)<0.5
X[flip]*=-1; M[flip]*=-1
def fit(Xs,Ms,mu,lam):
    return mu+np.linalg.solve(Xs.T@Xs+lam*np.eye(32),Xs.T@(Ms-Xs@mu))
def loo(s,lam):
    mu=s*comp; err=0; hit=0; tot=0
    for g in range(n):
        keep=np.arange(n)!=g
        r=fit(X[keep],M[keep],mu,lam); p=X[g]@r
        err+=(M[g]-p)**2
        if M[g]!=0: tot+=1; hit+= (p>0)==(M[g]>0)
    return err/n, hit/tot
print("games",n)
base=np.mean(M**2); print("predict 0 margin MSE",round(base,1))
res={}
for s in [0,2,4,6,8,10]:
    for lam in [0.5,1,2,4,8,16]:
        res[(s,lam)]=loo(s,lam)
best=min(res,key=lambda k:res[k][0])
for k in sorted(res,key=lambda k:res[k][0])[:8]: print(k,[round(x,3) for x in res[k]])
print("results only (s=0) best:",min(((k,v) for k,v in res.items() if k[0]==0),key=lambda kv:kv[1][0]))
print("stat prior only, huge shrink:",loo(6,1e6) if True else None)
s,lam=best; mu=s*comp; r=fit(X,M,mu,lam)
order=np.argsort(-r)
print("best",best)
for i,t in enumerate(order[:32],1): print(i,NAME2AB[TEAMS[t]],f"{r[t]:+.1f}", "stats-prior",f"{mu[t]:+.1f}", "old pos", [e['pos'] for e in json.load(open('league_data.json')) if e['team']==TEAMS[t]][0])
json.dump({TEAMS[i]:float(r[i]) for i in range(32)},open('bayes_ratings.json','w'))
