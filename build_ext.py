import csv, json, os, collections
E='ext/'
ABBR={"ARI":"Arizona Cardinals","ATL":"Atlanta Falcons","BAL":"Baltimore Ravens","BUF":"Buffalo Bills","CAR":"Carolina Panthers","CHI":"Chicago Bears","CIN":"Cincinnati Bengals","CLE":"Cleveland Browns","DAL":"Dallas Cowboys","DEN":"Denver Broncos","DET":"Detroit Lions","GNB":"Green Bay Packers","HOU":"Houston Texans","IND":"Indianapolis Colts","JAX":"Jacksonville Jaguars","KAN":"Kansas City Chiefs","LAC":"Los Angeles Chargers","LAR":"Los Angeles Rams","LVR":"Las Vegas Raiders","MIA":"Miami Dolphins","MIN":"Minnesota Vikings","NWE":"New England Patriots","NOR":"New Orleans Saints","NYG":"New York Giants","NYJ":"New York Jets","PHI":"Philadelphia Eagles","PIT":"Pittsburgh Steelers","SFO":"San Francisco 49ers","SEA":"Seattle Seahawks","TAM":"Tampa Bay Buccaneers","TEN":"Tennessee Titans","WAS":"Washington Commanders"}
TEAMS=sorted(ABBR.values())
def rd(n): return list(csv.DictReader(open(E+n)))
def f(x): return float(x)
out={}   # slug -> {team: value or None}
# defense / pressure
dfn={r['Tm']:r for r in rd('raw_def.csv')}; adv={r['Tm']:r for r in rd('raw_adv.csv')}
def pg(d,k,g='G'): return {t:f(d[t][k])/f(d[t][g]) for t in TEAMS}
def col(d,k): return {t:f(d[t][k]) for t in TEAMS}
out['pfr-def-pass-net-yards-per-attempt-allowed']=col(dfn,'PassNYA')
out['pfr-def-rush-yards-per-attempt-allowed']=col(dfn,'RushYA')
out['pfr-def-yards-per-play-allowed']=col(dfn,'TotYP')
out['pfr-def-scoring-drive-pct-allowed']=col(dfn,'ScPct')
out['pfr-def-turnover-drive-pct']=col(dfn,'TOPct')
out['pfr-def-expected-points-added']=col(dfn,'EXP')
out['pfr-def-opponent-first-downs-per-game']=pg(dfn,'TotFirstD')
out['pfr-def-opponent-pass-first-downs-per-game']=pg(dfn,'PassFirstD')
out['pfr-def-opponent-rush-first-downs-per-game']=pg(dfn,'RushFirstD')
out['pfr-def-opponent-fumbles-lost-per-game']=pg(dfn,'OFL')
out['pfr-def-interceptions-per-game']=pg(dfn,'PassInt')
out['pfr-def-opponent-pass-td-per-game']=pg(dfn,'PassTD')
out['pfr-def-opponent-completion-pct']={t:100*f(dfn[t]['PassCmp'])/f(dfn[t]['PassAtt']) for t in TEAMS}
out['pfr-def-penalty-first-downs-given-per-game']=pg(dfn,'PenFirstPy')
out['pfr-def-blitz-pct']=col(adv,'BltzP')
out['pfr-def-hurry-pct']=col(adv,'HrryP')
out['pfr-def-qb-knockdown-pct']=col(adv,'QBKDP')
out['pfr-def-pressure-pct']=col(adv,'PrssP')
out['pfr-def-sacks-per-game']=pg(adv,'Sk')
out['pfr-def-pressures-per-game']=pg(adv,'Prss')
out['pfr-def-missed-tackles-forced-per-game']=pg(adv,'MTkl')
out['pfr-def-average-depth-of-target-allowed']=col(adv,'DADOT')
out['pfr-def-yac-allowed-per-game']=pg(adv,'YAC')
out['pfr-def-air-yards-allowed-per-game']=pg(adv,'Air')
out['pfr-def-yac-share-allowed']={t:100*f(adv[t]['YAC'])/(f(adv[t]['YAC'])+f(adv[t]['Air'])) for t in TEAMS}
out['pfr-def-sack-rate-of-pass-plays']={t:100*f(adv[t]['Sk'])/(f(adv[t]['Att'])+f(adv[t]['Sk'])) for t in TEAMS}
# QBs (yardage-weighted blend of all QBs who played; captures QB changes)
qb=collections.defaultdict(list)
for r in rd('raw_qb.csv'): qb[ABBR[r['Team']]].append(r)
def qbw(k):
    o={}
    for t in TEAMS:
        rs=qb[t]; w=sum(f(r['Yds']) for r in rs); o[t]=sum(f(r[k])*f(r['Yds']) for r in rs)/w
    return o
for k,s in [('Rate','qb-passer-rating'),('ANYA','qb-adjusted-net-yards-per-attempt'),('QBR','qb-espn-qbr'),('YA','qb-yards-per-attempt'),('CmpP','qb-completion-pct')]:
    out[s]=qbw(k)
out['qb-interceptions-per-game']={t:sum(f(r['Int']) for r in qb[t])/sum(f(r['G']) for r in qb[t]) for t in TEAMS}
out['qb-sacks-taken-per-game']={t:sum(f(r['Sk']) for r in qb[t])/sum(f(r['G']) for r in qb[t]) for t in TEAMS}
out['qb-touchdown-passes-per-game']={t:sum(f(r['TD']) for r in qb[t])/sum(f(r['G']) for r in qb[t]) for t in TEAMS}
out['qb-clutch-4th-qtr-comebacks-and-game-winners']={t:sum(f(r['QBC4'])+f(r['GWD']) for r in qb[t]) for t in TEAMS}
out['qb-starter-continuity']={t:max(f(r['Yds']) for r in qb[t])/sum(f(r['Yds']) for r in qb[t]) for t in TEAMS}
# rushers (non-QB list only; attempt-weighted)
ru=collections.defaultdict(list)
for r in rd('raw_rush.csv'): ru[ABBR[r['Team']]].append(r)
def wavg(rs,k,w):
    tw=sum(f(r[w]) for r in rs)
    return sum(f(r[k])*f(r[w]) for r in rs)/tw if tw else None
out['rb-listed-success-rate']={t:wavg(ru[t],'Succ','Att') if ru[t] else None for t in TEAMS}
out['rb-listed-yards-per-carry']={t:wavg(ru[t],'YA','Att') if ru[t] else None for t in TEAMS}
out['rb-listed-first-down-rate']={t:(100*sum(f(r['D1']) for r in ru[t])/sum(f(r['Att']) for r in ru[t])) if ru[t] else None for t in TEAMS}
out['rb-lead-back-yards-per-game']={t:max(f(r['YG']) for r in ru[t]) if ru[t] else None for t in TEAMS}
out['rb-listed-fumbles-per-game']={t:(sum(f(r['Fmb']) for r in ru[t])/max(f(r['G']) for r in ru[t])) if ru[t] else None for t in TEAMS}
# receivers
rc=collections.defaultdict(list)
for r in rd('raw_rec.csv'): rc[ABBR[r['Team']]].append(r)
out['wr-listed-yards-per-target']={t:wavg(rc[t],'YTgt','Tgt') if rc[t] else None for t in TEAMS}
out['wr-listed-success-rate']={t:wavg(rc[t],'Succ','Tgt') if rc[t] else None for t in TEAMS}
out['wr-listed-catch-pct']={t:wavg(rc[t],'CtchP','Tgt') if rc[t] else None for t in TEAMS}
out['wr-top-receiver-yards-per-game']={t:max(f(r['Yds'])/f(r['G']) for r in rc[t]) if rc[t] else None for t in TEAMS}
out['wr-listed-touchdowns']={t:sum(f(r['TD']) for r in rc[t]) if rc[t] else 0.0 for t in TEAMS}
# honors (talent) and availability
PTS={'1':3,'2':2,'3':1}
GRP={'QB':'off','RB':'off','WR':'off','TE':'off','FB':'off','T':'ol','G':'ol','C':'ol','DL':'dl','LB':'lb','DB':'db','ST':'st'}
H=rd('raw_honors.csv')
tal=collections.defaultdict(lambda:collections.defaultdict(float)); namepts={}
for r in H:
    t=ABBR[r['Team']]; p=PTS[r['Level']]; tal[t][GRP[r['Pos']]]+=p; tal[t]['all']+=p; namepts[r['Player']]=(t,p)
for g,s in [('all','talent-honors-points-total'),('off','talent-honors-points-offense-skill'),('ol','talent-honors-points-offensive-line'),('dl','talent-honors-points-front-seven-rushers'),('lb','talent-honors-points-linebackers'),('db','talent-honors-points-secondary'),('st','talent-honors-points-special-teams')]:
    out[s]={t:tal[t][g] for t in TEAMS}
OUT={"CAR":["AJ Dillon","Monroe Freeling"],"CHI":["Keyshaun Elliott","Kyle Monangai","Dayo Odeyingbo","T.J. Edwards"],"ATL":["C.J. Henderson"],"CLE":["Dillon Gabriel","Michael Hall","Teven Jenkins","Kalia Davis","Dylan Sampson"],"IND":["Ashton Dulin","Jaylon Carlies"],"ARI":["Jeremiyah Love","Andrew Wingard","Wydett Williams"],"DAL":["Cobie Durant","Drew Shelton"],"DEN":["Talanoa Hufanga","Patrick Surtain II"],"GNB":["Javon Hargrave","Benjamin St-Juste","Jonathan Ford","Skyy Moore","Jordon Riley"],"HOU":["Jalen Pitre","Tank Dell","David Montgomery"],"JAX":["Ross Matiscik"],"KAN":["Rashee Rice","Josh Simmons"],"NOR":["Anfernee Jennings","Kaden Elliss"],"NWE":["Karon Prunty","Carlton Davis"],"NYG":["Colton Hood","Tyler Nubin","Theo Johnson"],"NYJ":["Adonai Mitchell","Dylan Parham","Breece Hall","Joseph Ossai"],"TEN":["Kevin Winston","Jordan Elliott"],"PHI":["Will Shipley","A.J. Epenesa","Jalyx Hunt","Michael Carter II","Jalen Hurts","Saquon Barkley"],"LVR":["Tonka Hemingway","Dont'e Thornton","Carter Runyon"],"BAL":["Broderick Washington Jr.","Lamar Jackson","Matthew Hibner","Cam Jurgens"],"LAC":["Joe Alt","Kayode Awosika","Tre Harris","Rodney Shelley","Donte Jackson","Derwin James"],"SEA":["Chazz Surratt","Drake Thomas","Zach Charbonnet"],"SFO":["Garret Wallow","Deebo Samuel","Nick Bosa","Marques Sigle"],"TAM":["Baker Mayfield","Antoine Winfield Jr.","David Sills V","Benjamin Morrison"],"WAS":["Charles Omenihu","Marcus Mariota","Jacory Croskey-Merritt","Javontae Jean-Baptiste"]}
NODATA={"CIN":0,"DET":0,"MIA":0,"MIN":0}
NODATA={ABBR[a] for a in NODATA}
lost={};cnt={};qbo={}
for a,names in OUT.items():
    t=ABBR[a]; lost[t]=sum(namepts[n][1] for n in names if n in namepts and namepts[n][0]==t); cnt[t]=len(names)
QBOUT={"Dillon Gabriel":"CLE","Lamar Jackson":"BAL","Jalen Hurts":"PHI","Baker Mayfield":"TAM","Marcus Mariota":"WAS"}
for t in TEAMS:
    qbo[t]=0.0
for n,a in QBOUT.items(): qbo[ABBR[a]]=1.0
# a team that is not in OUT or NODATA reported no Out/Doubtful players
def avail(d): return {t:(None if t in NODATA else d.get(t,0.0)) for t in TEAMS}
out['availability-honors-points-out']=avail(lost)
out['availability-players-out-or-doubtful']=avail(cnt)
out['availability-qb-out']=avail(qbo)
out['talent-honors-points-net-of-injuries']={t:(None if t in NODATA else tal[t]['all']-lost.get(t,0.0)) for t in TEAMS}
# PFR standings extras
OSRS={"Buffalo Bills":(11.3,-7.2),"New England Patriots":(-1.3,8.8),"New York Jets":(-4.8,3.4),"Miami Dolphins":(-7.3,1.2),"Baltimore Ravens":(2.6,-1.9),"Pittsburgh Steelers":(0.3,2.0),"Cleveland Browns":(-1.9,1.8),"Cincinnati Bengals":(5.9,-0.8),"Jacksonville Jaguars":(3.9,8.6),"Indianapolis Colts":(1.4,-4.3),"Houston Texans":(-5.6,3.0),"Tennessee Titans":(-10.9,-2.3),"Kansas City Chiefs":(5.1,3.4),"Denver Broncos":(-1.4,3.6),"Las Vegas Raiders":(5.0,-5.1),"Los Angeles Chargers":(-9.9,0.3),"New York Giants":(-6.7,-0.2),"Dallas Cowboys":(3.5,-7.0),"Philadelphia Eagles":(-4.2,-0.5),"Washington Commanders":(-2.4,-7.6),"Minnesota Vikings":(-0.9,9.3),"Green Bay Packers":(1.7,-0.6),"Detroit Lions":(5.4,-3.7),"Chicago Bears":(2.3,3.9),"Carolina Panthers":(7.9,-2.8),"Atlanta Falcons":(-1.0,1.4),"New Orleans Saints":(0.4,-4.8),"Tampa Bay Buccaneers":(-2.7,2.5),"San Francisco 49ers":(6.9,2.8),"Seattle Seahawks":(1.1,1.3),"Los Angeles Rams":(0.2,0.4),"Arizona Cardinals":(-1.5,-9.8)}
out['pfr-opponent-adjusted-offense-osrs']={t:OSRS[t][0] for t in TEAMS}
out['pfr-opponent-adjusted-defense-dsrs']={t:OSRS[t][1] for t in TEAMS}
os.makedirs('new2',exist_ok=True)
import shutil
for fn in os.listdir('new'):
    if fn.startswith('stat_'): shutil.copy('new/'+fn,'new2/'+fn)
for s,d in out.items():
    vals=[(t,d[t]) for t in TEAMS]
    with open(f'new2/stat_{s}.csv','w') as fh:
        fh.write('rank,team,value\n')
        for i,(t,v) in enumerate(sorted(vals,key=lambda x:(x[1] is None,-(x[1] or 0))),1):
            fh.write(f"{i},{t},{'' if v is None else round(v,4)}\n")
slugs=open('allslugs.txt').read().split()
have=set(slugs)
allnew=list(out)
open('allslugs2.txt','w').write("\n".join([s for s in slugs if os.path.exists(f'new2/stat_{s}.csv')]+allnew)+"\n")
json.dump(dict(lost=lost,cnt=cnt,qbout=QBOUT,nodata=sorted(NODATA),talent={t:dict(tal[t]) for t in TEAMS}),open('ext_notes.json','w'),indent=1)
print(len(allnew),'new stats written;',len(open('allslugs2.txt').read().split()),'total')
