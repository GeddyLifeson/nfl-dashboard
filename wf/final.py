import sys;sys.path.insert(0,'wf')
from walk2 import *
import glob,json
G25=load(sorted(glob.glob(N+'y25_[1-8].csv')),dt.date(2025,9,4),18)
G26=load(sorted(glob.glob(N+'box_[0-3].csv')),dt.date(2026,9,9),18)
sc={(r['w'].lower().replace(' ','-'),r['l'].lower().replace(' ','-')):(int(r['wp']),int(r['lp'])) for r in csv.DictReader(open(N+'games.csv'))}
have={x['s'] for x in G26}
for l in open(N+'box_urls.txt'):
    s=l.strip().split('/')[-1]
    if s in have:continue
    m=re.match(r'(.+)-vs-(.+)-(\d{10})$',s);a,h=m.group(1),m.group(2)
    for (w,lo),(wp,lp) in sc.items():
        if {w,lo}=={a,h}:
            pa,ph=(wp,lp) if w==a else (lp,wp)
            G26.append(dict(w=6,a=a,h=h,pa=pa,ph=ph,st=None,s=s))
print(len(G26))
teams=sorted({x['a'] for x in G25}|{x['h'] for x in G25});ti={t:i for i,t in enumerate(teams)}
R25=ratings(G25,teams,ti,G25,6,2.0)
cur=json.load(open('league_data.json'))
ab={t['abbr']:t for t in cur} if isinstance(cur,list) else None
R=ratings(G26,teams,ti,G26,8,2.0,R25,0.5)
Rnc=ratings(G26,teams,ti,G26,8,2.0)
order=np.argsort(-R)
nm={t['team'].lower().replace(' ','-'):t for t in cur}
rows=[]
for i,t in enumerate(order):
    c=nm.get(teams[t]);rows.append((i+1,teams[t],round(R[t],1),round(Rnc[t],1),c['pos'] if c else None))
for r in rows:print(r)
json.dump(rows,open('wf/trained_rank.json','w'))
