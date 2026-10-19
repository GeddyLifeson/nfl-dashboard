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

STALE={'opp-net-yards-per-pass-play','opp-sack-rate','opp-interception-rate','opp-pass-td-rate','opp-yards-per-play'}
def comp_of(drop):
    o=np.mean([d*z(x) for x,d in OFF],axis=0); de=np.mean([d*z(x) for x,d in DEF if x not in drop],axis=0); return (o+de)/2
C={'all':comp_of(set()),'fresh-only':comp_of(STALE)}
A=[];B=[];M=[]
for r in G:
    A.append(idx[r['w']]);B.append(idx[r['l']]);M.append(int(r['wp'])-int(r['lp']))
A=np.array(A);B=np.array(B);M=np.array(M,float);n=len(M)
X=np.zeros((n,32));X[np.arange(n),A]=1;X[np.arange(n),B]=-1
flip=np.random.RandomState(0).rand(n)<0.5;X[flip]*=-1;M[flip]*=-1
def fit(Xs,Ms,mu,lam): return mu+np.linalg.solve(Xs.T@Xs+lam*np.eye(32),Xs.T@(Ms-Xs@mu))
def loo(mu,lam):
    e=0;h=0;t=0
    for g in range(n):
        k=np.arange(n)!=g;p=X[g]@fit(X[k],M[k],mu,lam);e+=(M[g]-p)**2
        if M[g]!=0:t+=1;h+=(p>0)==(M[g]>0)
    return e/n,h/t
print("games",n,"| guess-zero MSE",round(np.mean(M**2),1))
out={}
for name,c in C.items():
    best=None
    for s_ in [4,6,8,10,12,14]:
        for lam in [4,8,16,32,64,1e6]:
            m,h=loo(s_*c,lam)
            if best is None or m<best[0]: best=(m,h,s_,lam)
    out[name]=best; print(name,"best LOO MSE %.1f winners %.1f%% at s=%s lam=%s"%(best[0],100*best[1],best[2],best[3]))
m0=min((loo(0*C['all'],l) for l in [4,8,16,32,64]),key=lambda x:x[0]); print("results only: MSE %.1f winners %.1f%%"%(m0[0],100*m0[1]))
# blend of fresh-only prior with results, final ratings
m,h,s_,lam=out['fresh-only']; mu=s_*C['fresh-only']; r=fit(X,M,mu,lam)
json.dump({TEAMS[i]:float(r[i]) for i in range(32)},open('ledger_ratings.json','w'))
old={e['team']:e['pos'] for e in json.load(open('league_data.json'))}
for i,t in enumerate(np.argsort(-r),1): print(i,NAME2AB[TEAMS[t]],"%+.1f"%r[t],"was",old[TEAMS[t]])
