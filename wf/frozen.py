import sys;sys.path.insert(0,'wf')
from tune import *
from math import erf,sqrt
p=dict(k=6,hfa=2.0,cap=None,decay=1.0,cb=0.6,sl=1.0)
# slope fit on tuning seasons
P1,Y1=season_eval(2024,p);P2,Y2=season_eval(2023,p,use_carry=False)
P=np.concatenate([P1,P2])-2;Y=np.concatenate([Y1,Y2])-2
sl=(P@Y)/(P@P);print('fitted slope',round(sl,2))
p['sl']=sl
def report(y,w0,carry=True):
    P,Y=season_eval(y,p,w0,carry);m,a=metrics(P,Y)
    hf=np.mean((2-Y)**2);ha=np.mean(Y>0)
    print(y,'w>=%d n=%d MSE %.1f (home-only %.1f) acc %.3f (home-only %.3f)'%(w0,len(Y),m,hf,a,ha))
    return P,Y
for y,w in ((2025,4),(2025,2),(2026,2)):report(y,w)
# pooled out-of-sample 2025+2026 for tiers
Pa,Ya=report(2025,3);Pb,Yb=report(2026,3)
P=np.concatenate([Pa,Pb]);Y=np.concatenate([Ya,Yb])
sig=np.std(Y-P);print('residual sd',round(sig,1))
nt=Y!=0;pr=np.array([0.5*(1+erf(((q-0)/sig)/sqrt(2))) for q in P])  # home win prob
win=(Y>0).astype(float)
print('Brier %.4f (coin .25) logloss %.4f (coin .693)'%(np.mean((pr-win)[nt]**2),-np.mean(np.log(np.where(win[nt]==1,pr[nt],1-pr[nt])))))
print('confidence tiers (pooled 2025 wk3+ and 2026 wk3+):')
conf=np.abs(P-2)
for q in (0,.25,.5,.75,.9):
    th=np.quantile(conf,q);sel=nt&(conf>=th)
    print(' top %3d%% of games by |pred margin|>=%.1f: n=%d acc %.3f'%(round((1-q)*100),th,sel.sum(),np.mean((P[sel]>0)==(Y[sel]>0))))
