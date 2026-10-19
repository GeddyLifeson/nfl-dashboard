import sys;sys.path.insert(0,'wf')
from walk2 import *
import glob
G25=load(sorted(glob.glob(N+'y25_[1-8].csv')),dt.date(2025,9,4),18)
G26=load(sorted(glob.glob(N+'box_[0-3].csv')),dt.date(2026,9,9),18)
# add 3 box-less 2026 games from games.csv? use only games with box rows (89).
print(len(G25),len(G26),sorted({x['w'] for x in G26}))
teams=sorted({x['a'] for x in G25}|{x['h'] for x in G25}|{x['a'] for x in G26}|{x['h'] for x in G26});ti={t:i for i,t in enumerate(teams)}
print(len(teams))
# 2025 fit of single scale b on honest results-rating diff
kR=6;hfa=2.0
X=[];Y=[]
for x in G25:
    if x['w']<2:continue
    R=ratings(G25,teams,ti,[p for p in G25 if p['w']<x['w']],kR,hfa)
    X.append(R[ti[x['h']]]-R[ti[x['a']]]);Y.append(x['ph']-x['pa'])
X=np.array(X);Y=np.array(Y);b=(X-X.mean())@(Y-Y.mean())/((X-X.mean())@(X-X.mean()));a0=Y.mean()-b*X.mean()
print('2025 fit: intercept %.2f slope %.2f'%(a0,b))
R25=ratings(G25,teams,ti,G25,kR,hfa)  # end-of-2025 ratings
top=np.argsort(-R25)[:8];print([ (teams[i],round(R25[i],1)) for i in top])
def ev(c,kR2,w0=2,use_fit=True):
    se=0;n=0;ok=0;nn=0
    for x in G26:
        if x['w']<w0:continue
        prior=[p for p in G26 if p['w']<x['w']]
        R=ratings(G26,teams,ti,prior,kR2,hfa,R25,c)
        d=R[ti[x['h']]]-R[ti[x['a']]];p=a0+b*d;y=x['ph']-x['pa']
        se+=(p-y)**2;n+=1
        if y!=0:nn+=1;ok+=(p>0)==(y>0)
    return se/n,ok/nn,n
print('HFA only',np.mean([(2-(x['ph']-x['pa']))**2 for x in G26 if x['w']>=2]))
for c in (0,0.25,0.5,0.75,1.0):
    for k2 in (4,6,8):
        print('carry',c,'k',k2,'MSE %.1f winners %.3f n=%d'%ev(c,k2))
