import csv, json, os, collections, numpy as np
from scipy.stats import rankdata
os.chdir(os.path.dirname(os.path.abspath(__file__))+'/n3') if False else None
D='n3/'
src=open('build_ext.py').read(); ns={}; exec(src[src.index('ABBR='):src.index('TEAMS=')],ns); ABBR=ns['ABBR']; NAME2AB={v:k for k,v in ABBR.items()}
TEAMS=sorted(ABBR.values())
def rd(n): return {r.get('Team') or r.get('Player'):r for r in csv.DictReader(open(D+n))}
def num(x): return float(str(x).replace(',',''))
G=list(csv.DictReader(open(D+'games.csv')))
pf=collections.Counter();pa=collections.Counter();W=collections.Counter();L=collections.Counter();Tt=collections.Counter();gp=collections.Counter()
opps=collections.defaultdict(list)
for r in G:
    a,b,x,y=r['w'],r['l'],int(r['wp']),int(r['lp'])
    pf[a]+=x;pa[a]+=y;pf[b]+=y;pa[b]+=x;gp[a]+=1;gp[b]+=1;opps[a].append((b,x-y));opps[b].append((a,y-x))
    if x==y: Tt[a]+=1;Tt[b]+=1
    else: W[a]+=1;L[b]+=1
mov={t:(pf[t]-pa[t])/gp[t] for t in TEAMS}
srs=dict(mov)
for _ in range(500):
    srs={t:mov[t]+np.mean([srs[o] for o,_ in opps[t]]) for t in TEAMS}
sos={t:float(np.mean([srs[o] for o,_ in opps[t]])) for t in TEAMS}
record={t:f"{W[t]}-{L[t]}"+(f"-{Tt[t]//1}" if Tt[t] else "") for t in TEAMS}
winpct=np.array([(W[t]+0.5*Tt[t])/gp[t] for t in TEAMS])
F={}   # slug -> {team: val}
def put(s,d): F[s]=d
def T_(name): return rd(name)
def pg(d,k,gk='Gms',scale=1.0): return {t:num(d[t][k])/num(d[t][gk]) for t in TEAMS}
def col(d,k): return {t:num(d[t][k]) for t in TEAMS}
for side,sfx in (('off',''),('def','opp-')):
    tot=T_(f'tot_{side}.csv'); ps=T_(f'pass_{side}.csv'); ru=T_(f'rush_{side}.csv'); dn=T_(f'downs_{side}.csv'); sc=T_(f'score_{side}.csv'); pu=T_(f'punt_{side}.csv')
    g={t:num(tot[t]['Gms']) for t in TEAMS}
    p=lambda s,d: put(sfx+s,d)
    p('points-per-game',col(tot,'Pts/G')); p('yards-per-game',col(tot,'Yds/G'))
    p('rush-yards-per-game',col(tot,'RYds/G')); p('net-pass-yards-per-game',col(tot,'PYds/G'))
    plays={t:num(ps[t]['Att'])/num(ps[t]['Gms'])+num(ps[t]['Sack'])/num(ps[t]['Gms'])+num(ru[t]['Att'])/num(ru[t]['Gms']) for t in TEAMS}
    p('plays-per-game',plays)
    p('yards-per-play',{t:num(tot[t]['Yds/G'])/plays[t] for t in TEAMS}); p('points-per-play',{t:num(tot[t]['Pts/G'])/plays[t] for t in TEAMS})
    p('yards-per-point',{t:num(tot[t]['Yds/G'])/num(tot[t]['Pts/G']) for t in TEAMS})
    for k,n in (('Att','pass-attempts-per-game'),('Cmp','completions-per-game'),('Yds','gross-pass-yards-per-game'),('TD','pass-touchdowns-per-game'),('Int','interceptions-per-game'),('Sack','sacks-per-game'),('Loss','sack-yards-per-game')):
        p(n,pg(ps,k))
    p('completion-pct',col(ps,'Pct')); p('yards-per-pass-attempt',col(ps,'YPA')); p('passer-rating',col(ps,'Rate'))
    p('net-yards-per-pass-play',{t:(num(ps[t]['Yds'])-num(ps[t]['Loss']))/(num(ps[t]['Att'])+num(ps[t]['Sack'])) for t in TEAMS})
    p('sack-rate',{t:100*num(ps[t]['Sack'])/(num(ps[t]['Att'])+num(ps[t]['Sack'])) for t in TEAMS})
    p('pass-td-rate',{t:100*num(ps[t]['TD'])/num(ps[t]['Att']) for t in TEAMS}); p('interception-rate',{t:100*num(ps[t]['Int'])/num(ps[t]['Att']) for t in TEAMS})
    p('pass-share-pct',{t:100*(num(ps[t]['Att'])/num(ps[t]['Gms']))/plays[t] for t in TEAMS})
    for k,n in (('Att','rush-attempts-per-game'),('Yds','rush-yards-per-game-2'),('TD','rush-touchdowns-per-game'),('FD','rush-first-downs-per-game')): p(n,pg(ru,k))
    p('yards-per-rush',col(ru,'Avg')); p('rush-first-down-rate',{t:100*num(ru[t]['FD'])/num(ru[t]['Att']) for t in TEAMS})
    p('rush-share-pct',{t:100*(num(ru[t]['Att'])/num(ru[t]['Gms']))/plays[t] for t in TEAMS})
    for k,n in (('Rush','rush-first-downs-per-game-b'),('Pass','pass-first-downs-per-game'),('Pen','penalty-first-downs-per-game'),('Tot','first-downs-per-game'),('A3','third-down-attempts-per-game'),('M3','third-down-conversions-per-game'),('A4','fourth-down-attempts-per-game'),('M4','fourth-down-conversions-per-game')): p(n,pg(dn,k))
    p('third-down-pct',col(dn,'P3')); p('fourth-down-pct',col(dn,'P4'))
    p('first-downs-per-play',{t:num(dn[t]['Tot'])/num(dn[t]['Gms'])/plays[t] for t in TEAMS})
    p('touchdowns-per-game',{t:num(sc[t]['Tot'])/g[t] for t in TEAMS})
    p('noffense-td-per-game' if False else 'scrimmage-td-per-game',{t:(num(sc[t]['R'])+num(sc[t]['P']))/g[t] for t in TEAMS})
    p('special-and-defensive-td-per-game',{t:sum(num(sc[t][k]) for k in ('KR','PR','IR','FR','BK','BP','FGR'))/g[t] for t in TEAMS})
    fg=lambda t:[num(x) for x in sc[t]['FG'].split('/')]; pt=lambda t:[num(x) for x in sc[t]['PAT'].split('/')]
    p('field-goals-made-per-game',{t:fg(t)[0]/g[t] for t in TEAMS}); p('field-goal-attempts-per-game',{t:fg(t)[1]/g[t] for t in TEAMS})
    p('field-goal-pct',{t:100*fg(t)[0]/fg(t)[1] for t in TEAMS}); p('extra-point-pct',{t:100*pt(t)[0]/pt(t)[1] for t in TEAMS})
    p('two-point-conversions-per-game',{t:num(sc[t]['Conv'])/g[t] for t in TEAMS}); p('safeties-per-game',{t:num(sc[t]['Saf'])/g[t] for t in TEAMS})
    p('punts-per-game',pg(pu,'Punts')); p('gross-yards-per-punt',col(pu,'Avg')); p('net-yards-per-punt',col(pu,'Net'))
    p('punts-inside-20-per-game',pg(pu,'In20')); p('punt-inside-20-rate',{t:100*num(pu[t]['In20'])/num(pu[t]['Punts']) for t in TEAMS}); p('punt-touchback-rate',{t:100*num(pu[t]['TB'])/num(pu[t]['Punts']) for t in TEAMS})
to=T_('to.csv'); pe=T_('pen.csv'); pr=T_('pr.csv'); kr=T_('kr.csv'); tg={t:num(rd('tot_off.csv')[t]['Gms']) for t in TEAMS}
for k,n in (('TkInt','takeaway-interceptions-per-game'),('TkFum','takeaway-fumbles-per-game'),('TkTot','takeaways-per-game'),('GvInt','giveaway-interceptions-per-game'),('GvFum','giveaway-fumbles-per-game'),('GvTot','giveaways-per-game')): put(n,pg(to,k))
put('turnover-margin-per-game',{t:num(to[t]['Diff'])/num(to[t]['Gms']) for t in TEAMS})
put('penalties-per-game',{t:num(pe[t]['Num'])/tg[t] for t in TEAMS}); put('penalty-yards-per-game',{t:num(pe[t]['Yds'])/tg[t] for t in TEAMS}); put('penalty-yards-per-penalty',{t:num(pe[t]['Yds'])/num(pe[t]['Num']) for t in TEAMS})
for k,n in (('FS','false-starts-per-game'),('OH','offensive-holding-per-game'),('DH','defensive-holding-per-game'),('DPI','defensive-pass-interference-per-game'),('DG','delay-of-game-per-game'),('OFF','offsides-per-game'),('UR','unnecessary-roughness-per-game')): put(n,{t:num(pe[t][k])/tg[t] for t in TEAMS})
put('punt-return-yards-per-game',col(pr,'YdsG')); put('yards-per-punt-return',col(pr,'Avg')); put('kick-return-yards-per-game',col(kr,'YdsG')); put('yards-per-kick-return',col(kr,'Avg'))
put('average-scoring-margin',mov)
# player layer: QB
Q=list(csv.DictReader(open(D+'qb.csv')))
AB={'LA':'LAR','NO':'NOR','GB':'GNB','LV':'LVR','NE':'NWE','KC':'KAN','SF':'SFO','TB':'TAM'}
byteam=collections.defaultdict(list)
for r in Q: byteam[ABBR[AB.get(r['Team'],r['Team'])] if AB.get(r['Team'],r['Team']) in ABBR else None].append(r)
def wavg(rows,k,w='Att'): return sum(num(r[k])*num(r[w]) for r in rows)/sum(num(r[w]) for r in rows)
qb1={t:max(byteam[t],key=lambda r:num(r['Att'])) for t in TEAMS}
put('qb1-passer-rating',{t:num(qb1[t]['Rate']) for t in TEAMS}); put('qb1-yards-per-attempt',{t:num(qb1[t]['YPA']) for t in TEAMS})
put('qb1-td-rate',{t:num(qb1[t]['TDp']) for t in TEAMS}); put('qb1-interception-rate',{t:num(qb1[t]['Intp']) for t in TEAMS})
put('qb-blend-passer-rating',{t:wavg(byteam[t],'Rate') for t in TEAMS})
put('qb-blend-yards-per-attempt',{t:wavg(byteam[t],'YPA') for t in TEAMS})
put('qb-blend-sack-rate',{t:100*sum(num(r['Sack']) for r in byteam[t])/sum(num(r['Att'])+num(r['Sack']) for r in byteam[t]) for t in TEAMS})
# honors (2025, static)
H=list(csv.DictReader(open('ext/raw_honors.csv'))); PTS={'1':3,'2':2,'3':1}
OFFP={'QB','RB','WR','TE','FB','T','G','C'}
tal=collections.Counter(); talo=collections.Counter(); tald=collections.Counter()
for r in H:
    t=ABBR[r['Team']]; p=PTS[r['Level']]; tal[t]+=p
    (talo if r['Pos'] in OFFP else tald)[t]+=p
put('honors-points-total',{t:float(tal[t]) for t in TEAMS}); put('honors-points-offense',{t:float(talo[t]) for t in TEAMS}); put('honors-points-defense-and-special-teams',{t:float(tald[t]) for t in TEAMS})
# weights
def spearman(x,y): return float(np.corrcoef(rankdata(x),rankdata(y))[0,1])
rho={}
for s,d in F.items():
    x=np.array([d[t] for t in TEAMS],float)
    rho[s]=0.0 if np.std(x)==0 else spearman(x,winpct)
CUT=0.35
Wt={s:(1 if r>0 else -1)*r*r for s,r in rho.items() if abs(r)>=CUT}
Z={}
for s in Wt:
    v=np.array([F[s][t] for t in TEAMS],float); Z[s]={t:(F[s][t]-v.mean())/v.std() for t in TEAMS}
comp={t:sum(w*Z[s][t] for s,w in Wt.items())/sum(abs(w) for w in Wt.values()) for t in TEAMS}
c=np.array([comp[t] for t in TEAMS]); m=np.array([mov[t] for t in TEAMS])
b,a=np.polyfit(c,m,1); fit=a+b*c; r2=float(np.corrcoef(c,m)[0,1]**2)
rating={t:float(fit[i]+sos[t]) for i,t in enumerate(TEAMS)}
order=sorted(TEAMS,key=lambda t:(-rating[t],t))
sosrank={t:i for i,t in enumerate(sorted(TEAMS,key=lambda t:-sos[t]),1)}
# displayed stats
DISP=[('points-per-game','Points per game','O',''),('yards-per-game','Yards per game','O',''),('points-per-play','Points per play','O',''),('touchdowns-per-game','Touchdowns per game','O',''),('rush-attempts-per-game','Rushes per game','O',''),('rush-yards-per-game','Rush yards per game','O',''),('rush-touchdowns-per-game','Rush TDs per game','O',''),('pass-attempts-per-game','Pass attempts per game','O',''),('net-pass-yards-per-game','Net pass yards per game','O',''),('third-down-pct','3rd-down conversion %','O','%'),('fourth-down-pct','4th-down conversion %','O','%'),('first-downs-per-game','First downs per game','O',''),('giveaways-per-game','Giveaways per game','O',''),('penalties-per-game','Penalties per game','O',''),('penalty-yards-per-game','Penalty yards per game','O','')]
DISPD=[('opp-points-per-game','Points allowed per game','D',''),('opp-yards-per-game','Yards allowed per game','D',''),('opp-points-per-play','Points allowed per play','D',''),('opp-touchdowns-per-game','TDs allowed per game','D',''),('opp-rush-attempts-per-game','Opponent rushes per game','D',''),('opp-rush-yards-per-game','Rush yards allowed per game','D',''),('opp-rush-touchdowns-per-game','Rush TDs allowed per game','D',''),('opp-pass-attempts-per-game','Opponent pass attempts per game','D',''),('opp-net-pass-yards-per-game','Net pass yards allowed per game','D',''),('opp-third-down-pct','Opponent 3rd-down %','D','%'),('opp-fourth-down-pct','Opponent 4th-down %','D','%'),('opp-first-downs-per-game','First downs allowed per game','D',''),('takeaways-per-game','Takeaways per game','D',''),('sacks-per-game','Sacks per game (defense)','D',''),('takeaway-interceptions-per-game','Interceptions per game','D','')]
LOW={'giveaways-per-game','penalties-per-game','penalty-yards-per-game'}
def rk(slug,t,low):
    d=F[slug]; vals=sorted(d.values(),reverse=not low)
    # competition rank
    return 1+sum(1 for v in vals if (v<d[t] if low else v>d[t]))
def fmt(s,v,unit):
    if s=='points-per-play' or s=='opp-points-per-play': return f"{v:.3f}"
    return f"{v:.1f}"
# defense sacks: use opp-side? sacks by defense = opp-sacks-per-game
F['sacks-per-game-def']=F['opp-sacks-per-game']
out=[]
for t in order:
    stats=[]
    for sl,lab,sd,un in DISP+DISPD+[('turnover-margin-per-game','Turnover margin per game','T','')]:
        s2=sl
        if sl=='sacks-per-game': s2='sacks-per-game-def'
        low=(sl in LOW) or sl.startswith('opp-') and not sl.startswith('opp-sacks') 
        if sl=='opp-first-downs-per-game' or sl.startswith('opp-'): low=True
        if sl in ('takeaways-per-game','takeaway-interceptions-per-game') or sl=='sacks-per-game' or sl=='turnover-margin-per-game': low=False
        v=F[s2][t]; stats.append(dict(slug=sl,label=lab,side=sd,unit=un,value=fmt(sl,v,un),rank=rk(s2,t,low)))
    O=[s for s in stats if s['side']=='O']; De=[s for s in stats if s['side']=='D']
    q=qb1[t]
    out.append(dict(team=t,abbr={'GNB':'GB','KAN':'KC','LVR':'LV','NWE':'NE','NOR':'NO','SFO':'SF','TAM':'TB'}.get(NAME2AB[t],NAME2AB[t]),record=record[t],sos=round(sos[t],1),sos_rank=sosrank[t],overall=rating[t],off_avg=float(np.mean([s['rank'] for s in O])),def_avg=float(np.mean([s['rank'] for s in De])),n_stats=len(F),pos=order.index(t)+1,prev_pos=order.index(t)+1,move=0,stats=stats,avg_rank=float(np.mean([s['rank'] for s in stats])),rating=rating[t],fitted=float(fit[TEAMS.index(t)]),margin=round(mov[t],1),composite=comp[t],
      player=dict(qb1=q['Player'],qb1rate=num(q['Rate']),qb1ypa=num(q['YPA']),qbrate=F['qb-blend-passer-rating'][t],qbypa=F['qb-blend-yards-per-attempt'][t],qbsack=F['qb-blend-sack-rate'][t],hon=F['honors-points-total'][t],gp=gp[t])))
json.dump(out,open('league_data.json','w'))
json.dump(dict(weights=Wt,rhos=rho,slope=b,r2=r2,n=len(F),nw=len(Wt)),open('weights4_out.json','w'),indent=1)
print(f"inputs {len(F)} weighted {len(Wt)} R2 {r2:.2f} slope {b:.2f}")
for i,t in enumerate(order,1): print(i,NAME2AB[t],record[t],f"{rating[t]:+.1f}",f"fit {fit[TEAMS.index(t)]:+.1f} mov {mov[t]:+.1f} sos {sos[t]:+.1f}")
print(sorted(((s,round(w,2)) for s,w in Wt.items()),key=lambda x:-abs(x[1]))[:14])
