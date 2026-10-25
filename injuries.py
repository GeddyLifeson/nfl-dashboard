"""Injury adjustment for predictions, every position except quarterback (see qb_adjust.py).

Who is out: the latest official injury report in nflverse (Out, Doubtful, Questionable), plus
regulars who have vanished from today's depth chart without a report entry (injured reserve
drops players from both).

How much each absence costs, in points per game:
    position weight x role x star factor x chance he misses x share of games he has played
  * position weight: rough points an average starter at that spot is worth over a replacement
    (edge rushers, tackles and corners highest; running backs, guards and specialists lower).
    Judgment-based, not fitted.
  * role: his average snap share (offense, defense, or special teams for kickers) in games played.
  * star factor: 2025 All-Pro / Pro Bowl honors (stats/honors_2025.csv) add 25-75%.
  * chance he misses: Out 100%, Doubtful 85%, Questionable 25%, vanished regular 90%.
  * share of games played: the team's results already include the games he missed, so only the
    part of his value that is baked into the rating is taken away.
A team's total is capped at MAX_TEAM points. Not walk-forward tested.
"""
import csv, os, re, time, urllib.request
from collections import defaultdict

BASE = 'https://github.com/nflverse/nflverse-data/releases/download/'
WEIGHT = {'T': 0.8, 'OT': 0.8, 'G': 0.45, 'OG': 0.45, 'C': 0.5, 'OL': 0.5,
          'WR': 0.65, 'TE': 0.45, 'RB': 0.35, 'FB': 0.1,
          'DE': 0.85, 'EDGE': 0.85, 'OLB': 0.7, 'DT': 0.55, 'NT': 0.4, 'DL': 0.55,
          'LB': 0.45, 'ILB': 0.45, 'MLB': 0.45, 'CB': 0.75, 'S': 0.45, 'FS': 0.45, 'SS': 0.45, 'DB': 0.45,
          'K': 0.35, 'P': 0.15, 'LS': 0.05}
P_OUT = {'Out': 1.0, 'Doubtful': 0.85, 'Questionable': 0.25}
P_VANISHED = 0.9
STAR = {'1': 1.75, '2': 1.5, '3': 1.25}       # honors level: All-Pro 1st, 2nd, Pro Bowl
MAX_TEAM = 6.0


def _norm(n):
    n = re.sub(r"[^a-z ]", "", n.lower().replace('-', ' '))
    return re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n).strip()


def _short(n):
    """(last name, first initial): survives nicknames like Cam/Camryn, Greg/Gregory, Dee/DeAundre."""
    parts = _norm(n).split()
    return (parts[-1], parts[0][0]) if parts else ('', '')


def _cached(name, hours=6):
    path = f'cache/{name}'
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > hours * 3600:
        os.makedirs('cache', exist_ok=True)
        folder = name.rsplit('_', 1)[0]
        data = urllib.request.urlopen(BASE + f'{folder}/{name}', timeout=180).read()
        with open(path + '.tmp', 'wb') as f:
            f.write(data)
        os.replace(path + '.tmp', path)
    return path


def roles(season):
    """{(team, name): dict(pos, share, games)} and {team: games played} from snap counts."""
    snaps = list(csv.DictReader(open(_cached(f'snap_counts_{season}.csv'), encoding='utf-8')))
    team_games = defaultdict(set)
    per = defaultdict(list)
    for r in snaps:
        if r['game_type'] != 'REG':
            continue
        team_games[r['team']].add(r['game_id'])
        pct = max(float(r['offense_pct'] or 0), float(r['defense_pct'] or 0))
        if r['position'] in ('K', 'P', 'LS'):
            pct = float(r['st_pct'] or 0)
        per[(r['team'], _norm(r['player']))].append((r['week'], r['position'], pct, r['player']))
    out = {}
    for key, games in per.items():
        played = [g for g in games if g[2] > 0]
        if not played:
            continue
        out[key] = dict(pos=played[-1][1], name=played[-1][3], games=len(played),
                        share=sum(g[2] for g in played) / len(played),
                        last_week=max(int(g[0]) for g in played))
    return out, {t: len(g) for t, g in team_games.items()}


def adjustments(season):
    """{team: dict(points, players=[(name, pos, status, cost)])}; points are negative."""
    role, team_games = roles(season)
    inj = list(csv.DictReader(open(_cached(f'injuries_{season}.csv'), encoding='utf-8')))
    week = max(int(r['week']) for r in inj)
    status = {(r['team'], _norm(r['full_name'])): r['report_status'] for r in inj
              if int(r['week']) == week and r['report_status'] in P_OUT}
    reported = {(r['team'], _norm(r['full_name'])) for r in inj if int(r['week']) == week}
    depth = csv.DictReader(open(_cached(f'depth_charts_{season}.csv'), encoding='utf-8'))
    latest, listed = {}, defaultdict(set)
    rows = list(depth)
    for r in rows:
        latest[r['team']] = max(latest.get(r['team'], ''), r['dt'])
    for r in rows:
        if r['dt'] == latest[r['team']]:
            listed[r['team']].add(_short(r['player_name']))
    stars = {}
    for r in csv.DictReader(open('stats/honors_2025.csv', encoding='utf-8')):
        stars[_norm(r['Player'])] = max(stars.get(_norm(r['Player']), 1.0), STAR[r['Level']])

    out: dict = defaultdict(lambda: {'points': 0.0, 'players': []})
    for (team, key), info in role.items():
        pos = info['pos']
        if pos == 'QB' or pos not in WEIGHT or info['share'] < 0.25:
            continue
        st = status.get((team, key))
        if st:
            p = P_OUT[st]
        elif (team, key) not in reported and _short(info['name']) not in listed.get(team, set()) and info['share'] >= 0.5:
            st, p = 'off depth chart', P_VANISHED
        else:
            continue
        frac = info['games'] / max(team_games.get(team, 1), 1)
        cost = WEIGHT[pos] * info['share'] * stars.get(key, 1.0) * p * frac
        out[team]['points'] -= cost
        out[team]['players'].append((info['name'], pos, st, round(cost, 2)))
    for t in out:
        out[t]['points'] = round(max(out[t]['points'], -MAX_TEAM), 1)
        out[t]['players'].sort(key=lambda x: -x[3])
    return dict(out)


if __name__ == '__main__':
    adj = adjustments(2026)
    for t, a in sorted(adj.items(), key=lambda kv: kv[1]['points']):
        top = ', '.join(f"{n} ({p}, {s})" for n, p, s, c in a['players'][:4])
        print(f"{t:4s} {a['points']:+.1f}  {top}")
