import csv,glob,re,datetime as dt,numpy as np
N='n3/'
KEYS=['ypp','fd','to','t3','sk']
def num(v):
    try:return float(v)
    except:return None
def load(files,start,maxwk):
    rows={}
    for f in files:
        for r in csv.DictReader(open(f)): rows.setdefault(r['url_slug'],{})[r['side']]=r
    G=[]
    for s,v in rows.items():
        if 'away' not in v or 'home' not in v: continue
        m=re.match(r'(.+)-vs-(.+)-(\d{10})$',s);a,h=m.group(1),m.group(2)
        d=dt.date(int(s[-10:-6]),int(s[-6:-4]),int(s[-4:-2]))
        w=(d-start).days//7+1
        if w>maxwk:continue
        A,H=v['away'],v['home']
        pa,ph=num(A['pts']),num(H['pts'])
        if pa is None or ph is None:continue
        st=None
        try:
            def P(r):
                f=lambda k:float(r[k]);plays=f('rush_att')+f('pass_att')+f('sacks_taken')
                return dict(ypp=f('total_yds')/plays,fd=f('first_downs'),to=f('pass_int')+f('fumbles_lost'),
                  t3=f('third_made')/max(f('third_att'),1),sk=f('sacks_taken')/(f('pass_att')+f('sacks_taken')))
            st=(P(A),P(H))
        except Exception: st=None
        G.append(dict(w=w,a=a,h=h,pa=pa,ph=ph,st=st,s=s))
    G.sort(key=lambda x:(x['w'],x['s']));return G
def ratings(games,teams,ti,prior,k,hfa,r0=None,c0=0.0,it=30):
    r=np.zeros(len(teams)) if r0 is None else r0*c0
    base=r.copy()
    for _ in range(it):
        num_=np.zeros(len(teams));n=np.zeros(len(teams))
        for x in prior:
            m=x['ph']-x['pa']-hfa;a,h=ti[x['a']],ti[x['h']]
            num_[h]+=m+r[a];num_[a]+=-m+r[h];n[h]+=1;n[a]+=1
        r=(num_+k*base)/(n+k)
    return r
def statrat(games,teams,ti,prior,sh=1.0):
    acc={t:np.zeros((len(KEYS),2)) for t in teams};cnt={t:0 for t in teams}
    for x in prior:
        if not x['st']:continue
        A,H=x['st']
        for t,o,d in ((x['a'],A,H),(x['h'],H,A)):
            cnt[t]+=1
            for j,k in enumerate(KEYS):acc[t][j]+=(o[k],d[k])
    out=np.zeros((len(teams),len(KEYS)))
    for t in teams:
        n=cnt[t]
        if n:
            out[ti[t]]=(acc[t][:,0]-acc[t][:,1])/n*(n/(n+sh))
    return out
def ridge(X,y,lam):
    mu=X.mean(0);Xc=X-mu;ym=y.mean()
    b=np.linalg.solve(Xc.T@Xc+lam*np.eye(X.shape[1]),Xc.T@(y-ym));return b,mu,ym
