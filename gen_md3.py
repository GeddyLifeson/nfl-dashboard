import json,sys
D=json.load(open('league_data.json')); W=json.load(open('weights4_out.json'))
def ordn(n):
    s="th" if 10<=n%100<=20 else {1:"st",2:"nd",3:"rd"}.get(n%10,"th"); return f"{n}{s}"
def cell(s): return f"{s['label']}: {s['value']}{'%' if s['unit']=='%' else ''} ({ordn(s['rank'])})"
def team(e):
    O=[s for s in e['stats'] if s['side']=='O']; De=[s for s in e['stats'] if s['side']=='D']; T=[s for s in e['stats'] if s['side']=='T'][0]
    P=e['player']
    out=f"## {e['pos']}. {e['team']} ({e['record']})\n\n"
    out+=f"Rating {e['rating']:+.1f} · Avg scoring margin {e['margin']:+.1f} · Move — · SoS {e['sos']:+.1f} ({ordn(e['sos_rank'])} toughest) · Offense avg rank {e['off_avg']:.1f} · Defense avg rank {e['def_avg']:.1f} · {cell(T)}\n\n"
    out+=f"Player layer: Lead QB {P['qb1']} (rating {P['qb1rate']:.1f}, {P['qb1ypa']:.1f} yards/att) · QB rating across all QBs used {P['qbrate']:.1f} · QB sack rate {P['qbsack']:.1f}% · 2025 honors points {P['hon']:.0f}\n\n"
    out+="| Offense | Defense |\n| --- | --- |\n"
    for a,b in zip(O,De): out+=f"| {cell(a)} | {cell(b)} |\n"
    return out
def head():
    t=D[:]
    top=", ".join(e['team'] for e in t[1:5])
    s=f"{t[0]['team']} hold No. 1 on the rebuilt rating ({t[0]['rating']:+.1f}), followed by {top}. The model now measures {W['n']} inputs per team from one source, all on the same set of games: offense, defense, passing, rushing, first downs, scoring, turnovers, penalties, special teams, quarterback production and 2025 honors. Each is weighted by how closely it tracks winning this season, then the rating is adjusted for strength of schedule computed from every score. Positions were reset with this method, so every movement arrow starts at neutral. All games through Sunday, Oct 18 are in; Monday night's game (Washington at San Francisco) is not.\n\n## League overview\n\n| Rank | Team | Record | Rating | Avg margin | SoS (rank) | Lead QB rating | 2025 honors pts | Move |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n"
    for e in t: s+=f"| {e['pos']} | {e['team']} | {e['record']} | {e['rating']:+.1f} | {e['margin']:+.1f} | {e['sos']:+.1f} ({ordn(e['sos_rank'])}) | {e['player']['qb1rate']:.1f} | {e['player']['hon']:.0f} | — |\n"
    return s
if sys.argv[1]=='head': print(head())
else:
    a,b=int(sys.argv[1]),int(sys.argv[2]); print("\n".join(team(e) for e in D[a:b]))
