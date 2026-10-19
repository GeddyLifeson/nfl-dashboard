import csv,glob,re,datetime as dt,numpy as np,math
N='n3/'
def season_games(rows,start,maxwk=18):
    G=[]
    for s,a,h,pa,ph in rows:
        d=dt.date(int(s[-10:-6]),int(s[-6:-4]),int(s[-4:-2]));w=(d-start).days//7+1
        if w<1 or w>maxwk:continue
        G.append(dict(w=w,a=a,h=h,pa=float(pa),ph=float(ph),s=s))
    G.sort(key=lambda x:(x['w'],x['s']));return G
def read_scores(f):
    return [(r['url_slug'],r['away'],r['home'],r['pa'],r['ph']) for r in csv.DictReader(open(f))]
def read_box(files):
    rows={}
    for f in files:
        for r in csv.DictReader(open(f)):rows.setdefault(r['url_slug'],{})[r['side']]=r
    out=[]
    for s,v in rows.items():
        if 'away' in v and 'home' in v:
            m=re.match(r'(.+)-vs-(.+)-(\d{10})$',s)
            try:out.append((s,m.group(1),m.group(2),float(v['away']['pts']),float(v['home']['pts'])))
            except:pass
    return out
def margin_tf(m,cap):  # compress blowouts
    if cap is None:return m
    return float(np.sign(m)*min(abs(m),cap)) if cap>0 else float(np.sign(m)*math.log1p(abs(m))*(-cap))
def rate(prior,idx,nT,k,hfa,cap=None,decay=1.0,wmax=None,base=None,cb=0.0,it=25):
    base_=np.zeros(nT) if base is None else base*cb
    r=base_.copy()
    for _ in range(it):
        num=np.zeros(nT);n=np.zeros(nT)
        for x in prior:
            wt=decay**(wmax-x['w']) if wmax is not None else 1.0
            m=margin_tf(x['ph']-x['pa'],cap)-hfa;a,h=idx[x['a']],idx[x['h']]
            num[h]+=wt*(m+r[a]);num[a]+=wt*(-m+r[h]);n[h]+=wt;n[a]+=wt
        r=(num+k*base_)/(n+k)
    return r
def walk(G,idx,nT,w0,params,carry=None):
    """returns arrays of (pred margin, actual margin) for weeks>=w0, using only prior weeks"""
    P=[];Y=[]
    for k in range(w0,19):
        prior=[x for x in G if x['w']<k]
        if not prior and carry is None:continue
        r=rate(prior,idx,nT,params['k'],params['hfa'],params.get('cap'),params.get('decay',1.0),k,carry,params.get('cb',0.0))
        for x in G:
            if x['w']==k:
                P.append(params['sl']*(r[idx[x['h']]]-r[idx[x['a']]])+params['hfa']);Y.append(x['ph']-x['pa'])
    return np.array(P),np.array(Y)
def metrics(P,Y,sig=None):
    nt=Y!=0
    acc=np.mean((P[nt]>0)==(Y[nt]>0));mse=np.mean((P-Y)**2)
    return mse,acc
