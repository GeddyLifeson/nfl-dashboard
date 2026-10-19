import sys;sys.path.insert(0,'wf')
from lab import *
import glob,itertools,json
S={}
S[2023]=season_games(read_scores(N+'y23_scores.csv'),dt.date(2023,9,7))
S[2024]=season_games(read_scores(N+'y24_scores.csv'),dt.date(2024,9,5))
S[2025]=season_games(read_box(sorted(glob.glob(N+'y25_[1-8].csv'))),dt.date(2025,9,4))
S[2026]=season_games(read_box(sorted(glob.glob(N+'box_[0-3].csv'))),dt.date(2026,9,9))
# add 3 box-less 2026 games
sc={(r['w'].lower().replace(' ','-'),r['l'].lower().replace(' ','-')):(int(r['wp']),int(r['lp'])) for r in csv.DictReader(open(N+'games.csv'))}
have={x['s'] for x in S[2026]}
for l in open(N+'box_urls.txt'):
    s=l.strip().split('/')[-1]
    if s in have:continue
    m=re.match(r'(.+)-vs-(.+)-(\d{10})$',s);a,h=m.group(1),m.group(2)
    for (w,lo),(wp,lp) in sc.items():
        if {w,lo}=={a,h}:
            pa,ph=(wp,lp) if w==a else (lp,wp);S[2026].append(dict(w=6,a=a,h=h,pa=pa,ph=ph,s=s))
print({y:len(g) for y,g in S.items()})
teams=sorted({x['a'] for g in S.values() for x in g}|{x['h'] for g in S.values() for x in g});idx={t:i for i,t in enumerate(teams)};nT=len(teams);print(nT)
def endrat(y,p,carry):
    return rate(S[y],idx,nT,p['k'],p['hfa'],p.get('cap'),p.get('decay',1.0),19,carry,p.get('cb',0.0))
def season_eval(y,p,w0=4,use_carry=True):
    carry=None
    if use_carry and (y-1) in S:
        pc=dict(p);carry=endrat(y-1,pc,None)
    P,Y=walk(S[y],idx,nT,w0,p,carry);return P,Y
if __name__=='__main__':
    res=[]
    for k,cap,dec,cb in itertools.product((4,6,8,12),(None,21,14),(1.0,0.95),(0.0,0.3,0.5,0.7)):
        p=dict(k=k,hfa=2.0,cap=cap,decay=dec,cb=cb,sl=1.0)
        # tune on 2024 (carry from 2023) and 2023 (no carry) combined
        P1,Y1=season_eval(2024,p);P2,Y2=season_eval(2023,p,use_carry=False)
        P=np.concatenate([P1,P2]);Y=np.concatenate([Y1,Y2]);m,a=metrics(P,Y)
        res.append((m,a,k,cap,dec,cb))
    res.sort(key=lambda r:r[0])
    for r in res[:8]:print('tune MSE %.1f acc %.3f k=%s cap=%s decay=%s carry=%s'%r)
    json.dump(res[:20],open('wf/tune_top.json','w'))
