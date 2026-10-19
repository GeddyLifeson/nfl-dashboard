import csv,glob,re,numpy as np
N='n3/'
slugs=[l.strip().split('/')[-1] for l in open(N+'box_urls.txt')]
rows={}
for f in sorted(glob.glob(N+'box_[0-3].csv')):
    for r in csv.DictReader(open(f)): rows.setdefault(r['url_slug'],{})[r['side']]=r
g=list(csv.DictReader(open(N+'games.csv')))
norm=lambda s:s.lower().replace(' ','-')
sc={}
for x in g: sc[(norm(x['w']),norm(x['l']))]=(int(x['wp']),int(x['lp']))
def week(s):
    d=int(s[-6:-2]);return 1+sum(d>b for b in (915,922,929,1006,1013))
games=[]
for s in slugs:
    m=re.match(r'(.+)-vs-(.+)-(\d{10})$',s);a,h=m.group(1),m.group(2)
    st=None
    if s in rows and rows[s]['away']['total_yds']:
        def P(r):
            f=lambda k:float(r[k]);plays=f('rush_att')+f('pass_att')+f('sacks_taken')
            return dict(pts=f('pts'),ypp=f('total_yds')/plays,fd=f('first_downs'),to=f('pass_int')+f('fumbles_lost'),
              t3=f('third_made')/max(f('third_att'),1),sk=f('sacks_taken')/(f('pass_att')+f('sacks_taken')))
        st=(P(rows[s]['away']),P(rows[s]['home']))
    if s in rows: pa,ph=int(rows[s]['away']['pts']),int(rows[s]['home']['pts'])
    else:
        if (a,h) in sc: pa,ph=sc[(a,h)]
        elif (h,a) in sc: ph,pa=sc[(h,a)]
        else: raise SystemExit('missing '+s)
    # games.csv winner order check for fallback: ambiguity resolved by name only
    games.append(dict(w=week(s),a=a,h=h,pa=pa,ph=ph,st=st,s=s))
print(len(games),sum(x['st'] is not None for x in games))
# fix missing-game score orientation: (a,h)->winner a means pa=wp. only valid if a won; handle by name
for x in games:
    if x['s'] not in rows:
        for (w,l),(wp,lp) in sc.items():
            if {w,l}=={x['a'],x['h']}:
                x['pa'],x['ph']=(wp,lp) if w==x['a'] else (lp,wp)
teams=sorted({x['a'] for x in games}|{x['h'] for x in games})
ti={t:i for i,t in enumerate(teams)}
def ratings(prior,k=4,it=30):
    r=np.zeros(len(teams))
    for _ in range(it):
        num=np.zeros(len(teams));n=np.zeros(len(teams))
        for x in prior:
            m=x['ph']-x['pa'];a,h=ti[x['a']],ti[x['h']];m-=2.0
            num[h]+=m+r[a];num[a]+=-m+r[h];n[h]+=1;n[a]+=1
        r=num/(n+k)
    return r
KEYS=['ypp','fd','to','t3','sk']
def statrat(prior):
    # per team: off avg minus def avg of each stat
    acc={t:{k:[0,0,0] for k in KEYS} for t in teams}
    cnt={t:0 for t in teams}
    for x in prior:
        if not x['st']:continue
        A,H=x['st']
        for t,o,d in ((x['a'],A,H),(x['h'],H,A)):
            cnt[t]+=1
            for k in KEYS:
                acc[t][k][0]+=o[k];acc[t][k][1]+=d[k]
    out=np.zeros((len(teams),len(KEYS)))
    for t in teams:
        for j,k in enumerate(KEYS):
            n=cnt[t]
            if n: out[ti[t],j]=(acc[t][k][0]-acc[t][k][1])/n*(n/(n+1.0))
    return out
def feats(x,prior,kR):
    R=ratings(prior,kR);S=statrat(prior)
    a,h=ti[x['a']],ti[x['h']]
    return np.concatenate([[R[h]-R[a]],S[h]-S[a]])
def ridge(X,y,lam,mask=None):
    mu=X.mean(0);Xc=X-mu;yc=y-y.mean()
    b=np.linalg.solve(Xc.T@Xc+lam*np.eye(X.shape[1]),Xc.T@yc)
    return b,mu,y.mean()
import itertools,json
res={}
def run(kR,lam,cols,wmin=3):
    se=0;n=0;ok=0;nn=0
    for k in range(wmin,7):
        tr=[x for x in games if 2<=x['w']<k]
        Xtr=np.array([feats(x,[p for p in games if p['w']<x['w']],kR)[cols] for x in tr])
        ytr=np.array([x['ph']-x['pa'] for x in tr],float)
        b,mu,ym=ridge(Xtr,ytr,lam)
        prior=[p for p in games if p['w']<k]
        for x in games:
            if x['w']!=k:continue
            f=feats(x,prior,kR)[cols];pred=ym+(f-mu)@b;y=x['ph']-x['pa']
            se+=(pred-y)**2;n+=1
            if y!=0:nn+=1;ok+=(pred>0)==(y>0)
    return se/n,ok/nn,n
base=0
# baselines
allm=[]
for k in range(3,7):
    tr=[x['ph']-x['pa'] for x in games if x['w']<k]
    hf=np.mean(tr)
    for x in games:
        if x['w']==k:allm.append(((x['ph']-x['pa']-hf)**2,(hf>0)==(x['ph']>x['pa']) if x['ph']!=x['pa'] else None))
print('HFA only MSE',np.mean([a for a,_ in allm]),len(allm))
out=[]
for name,cols in (('results',[0]),('stats',[1,2,3,4,5]),('both',[0,1,2,3,4,5])):
    for kR in (2,4,8):
        for lam in (1,10,100,1000):
            m,w,n=run(kR,lam,cols);out.append((m,w,name,kR,lam))
out.sort()
for o in out[:12]:print('%.1f %.3f %s kR=%d lam=%g'%o)
print('--best per kind')
for nm in ('results','stats','both'):
    o=[x for x in out if x[2]==nm][0];print('%.1f %.3f %s kR=%d lam=%g'%o)
print('=== fixed-structure predictors (no fitted coefs), weeks 3-6, n=60')
# precompute per-week feats
cache={}
for k in range(3,7):
    prior=[p for p in games if p['w']<k]
    for kR in (2,4,8,16):
        R=ratings(prior,kR);cache[(k,kR)]=R
    cache[(k,'S')]=statrat(prior)
def ev(fn):
    se=0;n=0;ok=0;nn=0
    for x in games:
        if x['w']<3:continue
        p=fn(x);y=x['ph']-x['pa'];se+=(p-y)**2;n+=1
        if y!=0:nn+=1;ok+=(p>0)==(y>0)
    return se/n,ok/nn
print('HFA2 only',ev(lambda x:2.0))
for kR in (2,4,8,16):
  for c in (0.5,1.0):
    print('results kR',kR,'c',c,ev(lambda x:2+c*(cache[(x['w'],kR)][ti[x['h']]]-cache[(x['w'],kR)][ti[x['a']]])))
# stats single-scalar: ypp diff, to diff
for j,k in enumerate(KEYS):
  for c in (1,2,4,8,16,30):
    S=lambda x:(cache[(x['w'],'S')][ti[x['h']],j]-cache[(x['w'],'S')][ti[x['a']],j])
    r=ev(lambda x:2+c*S(x))
    print(k,c,'%.1f %.3f'%r)
