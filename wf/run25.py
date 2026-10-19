import sys;sys.path.insert(0,'wf')
from walk2 import *
import glob
G=load(sorted(glob.glob(N+'y25_[1-8].csv')),dt.date(2025,9,4),18)
print(len(G),sum(x['st'] is not None for x in G),[sum(x['w']==w for x in G) for w in range(1,19)])
teams=sorted({x['a'] for x in G}|{x['h'] for x in G});ti={t:i for i,t in enumerate(teams)};print(len(teams))
hf=np.mean([x['ph']-x['pa'] for x in G]);print('home margin avg',hf,'home win%',np.mean([x['ph']>x['pa'] for x in G]))
def feats(x,prior,kR,hfa):
    R=ratings(G,teams,ti,prior,kR,hfa);S=statrat(G,teams,ti,prior)
    a,h=ti[x['a']],ti[x['h']];return np.concatenate([[R[h]-R[a]],S[h]-S[a]])
# precompute features per game with honest prior (weeks before its own week), per kR
FE={}
for kR in (3,6,12):
    for x in G:
        if x['w']>=2:
            FE[(kR,x['s'])]=feats(x,[p for p in G if p['w']<x['w']],kR,2.0)
def run(kR,lam,cols,w0=6):
    se=0;n=0;ok=0;nn=0
    for k in range(w0,19):
        tr=[x for x in G if 2<=x['w']<k]
        X=np.array([FE[(kR,x['s'])][cols] for x in tr]);y=np.array([x['ph']-x['pa'] for x in tr],float)
        b,mu,ym=ridge(X,y,lam)
        for x in G:
            if x['w']!=k:continue
            p=ym+(FE[(kR,x['s'])][cols]-mu)@b;yv=x['ph']-x['pa'];se+=(p-yv)**2;n+=1
            if yv!=0:nn+=1;ok+=(p>0)==(yv>0)
    return se/n,ok/nn,n
base=[];
for k in range(6,19):
    h=np.mean([x['ph']-x['pa'] for x in G if x['w']<k])
    for x in G:
        if x['w']==k:base.append(((h-(x['ph']-x['pa']))**2,(h>0)==(x['ph']>x['pa']) if x['ph']!=x['pa'] else None))
print('HFA-only walk MSE %.1f winners %.3f n=%d'%(np.mean([a for a,_ in base]),np.mean([b for _,b in base if b is not None]),len(base)))
out=[]
for nm,cols in (('results',[0]),('stats',[1,2,3,4,5]),('both',[0,1,2,3,4,5]),('ypp',[1]),('ypp+to',[1,3])):
    for kR in (3,6,12):
        for lam in (1,10,100,1000):
            m,w,n=run(kR,lam,cols);out.append((m,w,nm,kR,lam))
out.sort()
for o in out[:10]:print('%.1f %.3f %s kR=%d lam=%g'%o)
print('--best per kind')
for nm in ('results','stats','both','ypp','ypp+to'):
    o=[x for x in out if x[2]==nm][0];print('%.1f %.3f %s kR=%d lam=%g'%o)
# coefficients on all 2025 (honest features)
best=out[0]
import pickle
