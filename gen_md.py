"""Markdown for the stats document. Run: python gen_md.py head   or   python gen_md.py <from> <to>  (team slice, e.g. 0 8)"""
import json, sys

sys.stdout.reconfigure(encoding='utf-8')
D = json.load(open('league_data.json', encoding='utf-8'))


def ordn(n):
    s = 'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')
    return f'{n}{s}'


def move(e):
    return f"▲{e['move']}" if e['move'] > 0 else f"▼{-e['move']}" if e['move'] < 0 else '—'


def cell(s):
    flag = '*' if s['behind'] else ''
    return f"{s['label']}: {s['value']}{s['unit']}{flag} ({ordn(s['rank'])})"


def team(e):
    O = [s for s in e['stats'] if s['side'] == 'O']
    De = [s for s in e['stats'] if s['side'] == 'D']
    T = [s for s in e['stats'] if s['side'] == 'T'][0]
    P = e['player']
    out = f"## {e['pos']}. {e['team']} ({e['record']})\n\n"
    out += (f"Rating {e['rating']:+.1f} · Avg scoring margin {e['margin']:+.1f} · Move {move(e)} · "
            f"SoS {e['sos']:+.1f} ({ordn(e['sos_rank'])} toughest) · Offense avg rank {e['off_avg']:.1f} · "
            f"Defense avg rank {e['def_avg']:.1f} · {cell(T)}\n\n")
    out += (f"Player layer: Lead QB {P['qb1']} (rating {P['qb1rate']:.1f}, {P['qb1ypa']:.1f} yards/att) · "
            f"QB rating across all QBs used {P['qbrate']:.1f} · QB sack rate {P['qbsack']:.1f}% · 2025 honors points {P['hon']}\n\n")
    out += '| Offense | Defense |\n| --- | --- |\n'
    for a, b in zip(O, De):
        out += f'| {cell(a)} | {cell(b)} |\n'
    if any(s['behind'] for s in e['stats']):
        out += '\n\\* Source table is missing at least one game this team has played.\n'
    return out


def head():
    t = D['teams']
    m = D['model']
    later = ' and '.join(D['not_yet_played'])
    pending = f" {later} {'is' if len(D['not_yet_played']) == 1 else 'are'} not in yet." if later else ''
    s = (f"{t[0]['team']} are No. 1 ({t[0]['rating']:+.1f}), followed by {', '.join(e['team'] for e in t[1:5])}. "
         f"A rating is points per game better than an average team on a neutral field, built from every score this season "
         f"and {m['carry']:.0%} of last season's final rating. All games through {D['through']} are in.{pending}\n\n")
    s += '## League overview\n\n| Rank | Team | Record | Rating | Avg margin | SoS (rank) | Lead QB rating | 2025 honors pts | Move |\n| --- | --- | --- | --- | --- | --- | --- | --- | --- |\n'
    for e in t:
        s += (f"| {e['pos']} | {e['team']} | {e['record']} | {e['rating']:+.1f} | {e['margin']:+.1f} | "
              f"{e['sos']:+.1f} ({ordn(e['sos_rank'])}) | {e['player']['qb1rate']:.1f} | {e['player']['hon']} | {move(e)} |\n")
    if D['upcoming']:
        s += '\n## Picks\n\n| Week | Game | Pick | Win chance | Line |\n| --- | --- | --- | --- | --- |\n'
        for u in D['upcoming']:
            home_fav = u['home_margin'] > 0
            pick = f"{u['home'] if home_fav else u['away']} by {abs(u['home_margin']):.1f}"
            chance = u['home_win'] if home_fav else 1 - u['home_win']
            line = '' if u['line'] is None else f"{u['home']} {-u['line']:+g}"
            s += f"| {u['week']} | {u['away']} at {u['home']}{' (neutral)' if u['neutral'] else ''} | {pick} | {chance:.0%} | {line} |\n"
    r = D['pick_record']
    if r['right'] + r['wrong']:
        s += f"\nPublished picks so far: {r['right']}-{r['wrong']}.\n"
    return s


if __name__ == '__main__':
    if sys.argv[1:2] == ['head']:
        print(head())
    else:
        a, b = int(sys.argv[1]), int(sys.argv[2])
        print('\n'.join(team(e) for e in D['teams'][a:b]))
