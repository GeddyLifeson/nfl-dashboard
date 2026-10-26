"""Team offense and defense stats with league ranks, computed from nflverse play-by-play and scores.
Replaces the hand-collected FootballDB tables, so every number updates itself."""
import csv, gzip
from collections import defaultdict
import nflv
from lineup import FIX
from teams import ABBRS

# (key, label, side, higher is better, format)
STATS = [
    ('pts', 'Points per game', 'O', True, '.1f'), ('yds', 'Yards per game', 'O', True, '.1f'),
    ('ypp', 'Yards per play', 'O', True, '.2f'), ('pass_yds', 'Net pass yards per game', 'O', True, '.1f'),
    ('rush_yds', 'Rush yards per game', 'O', True, '.1f'), ('ypc', 'Yards per carry', 'O', True, '.2f'),
    ('epa', 'EPA per play', 'O', True, '+.3f'), ('success', 'Success rate %', 'O', True, '.1f'),
    ('third', '3rd-down conversion %', 'O', True, '.1f'), ('redzone', 'Red-zone TD %', 'O', True, '.1f'),
    ('explosive', 'Explosive plays (20+) per game', 'O', True, '.2f'), ('sacked', 'Sacks allowed per game', 'O', False, '.2f'),
    ('giveaways', 'Giveaways per game', 'O', False, '.2f'), ('pen_yds', 'Penalty yards per game', 'O', False, '.1f'),
    ('pts_a', 'Points allowed per game', 'D', False, '.1f'), ('yds_a', 'Yards allowed per game', 'D', False, '.1f'),
    ('ypp_a', 'Yards per play allowed', 'D', False, '.2f'), ('pass_yds_a', 'Net pass yards allowed per game', 'D', False, '.1f'),
    ('rush_yds_a', 'Rush yards allowed per game', 'D', False, '.1f'), ('epa_a', 'EPA per play allowed', 'D', False, '+.3f'),
    ('success_a', 'Success rate allowed %', 'D', False, '.1f'), ('third_a', 'Opponent 3rd-down %', 'D', False, '.1f'),
    ('redzone_a', 'Opponent red-zone TD %', 'D', False, '.1f'), ('sacks', 'Sacks per game', 'D', True, '.2f'),
    ('takeaways', 'Takeaways per game', 'D', True, '.2f'), ('explosive_a', 'Explosive plays allowed per game', 'D', False, '.2f'),
    ('to_margin', 'Turnover margin per game', 'T', True, '+.2f'),
]


def compute(season, games_csv='data/games.csv'):
    """{team: {key: value}} plus {team: {key: rank}}."""
    games = defaultdict(set)
    pts, pts_a = defaultdict(float), defaultdict(float)
    for r in csv.DictReader(open(games_csv)):
        if r['season'] == str(season) and r['home_score']:
            h, a, hs, as_ = r['home_team'], r['away_team'], int(r['home_score']), int(r['away_score'])
            for t, f, ag in ((h, hs, as_), (a, as_, hs)):
                games[t].add(r['game_id']); pts[t] += f; pts_a[t] += ag
    o = defaultdict(lambda: defaultdict(float))
    d = defaultdict(lambda: defaultdict(float))
    drives = {}
    with gzip.open(nflv.get('pbp', season), 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['season_type'] != 'REG' or not r['posteam']:
                continue
            off, de = FIX.get(r['posteam'], r['posteam']), FIX.get(r['defteam'], r['defteam'])
            if r['penalty'] == '1' and r['penalty_team']:
                o[FIX.get(r['penalty_team'], r['penalty_team'])]['pen_yds'] += float(r['penalty_yards'] or 0)
            if r['play_type'] not in ('pass', 'run'):
                continue
            y = float(r['yards_gained'] or 0)
            epa = float(r['epa']) if r['epa'] not in ('', 'NA') else 0.0
            dropback = r['qb_dropback'] == '1'
            to = r['interception'] == '1' or r['fumble_lost'] == '1'
            for side, team in (('o', off), ('d', de)):
                s = o[team] if side == 'o' else d[team]
                s['plays'] += 1; s['yds'] += y; s['epa'] += epa; s['success'] += float(r['success'] or 0)
                s['pass_yds' if dropback else 'rush_yds'] += y
                if not dropback:
                    s['carries'] += 1
                s['explosive'] += y >= 20
                s['third_att'] += r['third_down_converted'] == '1' or r['third_down_failed'] == '1'
                s['third_conv'] += r['third_down_converted'] == '1'
                s['sacks'] += r['sack'] == '1'
                s['to'] += to
            key = (r['game_id'], r['drive'])
            dv = drives.setdefault(key, [off, de, 100.0, ''])
            dv[2] = min(dv[2], float(r['yardline_100'] or 100))
            dv[3] = r['fixed_drive_result'] or dv[3]
    rz = defaultdict(lambda: [0, 0]); rz_a = defaultdict(lambda: [0, 0])
    for off, de, best, result in drives.values():
        if best <= 20:
            td = result == 'Touchdown'
            rz[off][0] += 1; rz[off][1] += td
            rz_a[de][0] += 1; rz_a[de][1] += td
    val = {}
    for t in ABBRS:
        g = max(len(games[t]), 1)
        a, b = o[t], d[t]
        pct = lambda n, k: 100 * n / k if k else 0.0
        val[t] = dict(
            pts=pts[t] / g, yds=a['yds'] / g, ypp=a['yds'] / max(a['plays'], 1), pass_yds=a['pass_yds'] / g,
            rush_yds=a['rush_yds'] / g, ypc=a['rush_yds'] / max(a['carries'], 1), epa=a['epa'] / max(a['plays'], 1),
            success=pct(a['success'], a['plays']), third=pct(a['third_conv'], a['third_att']), redzone=pct(rz[t][1], rz[t][0]),
            explosive=a['explosive'] / g, sacked=a['sacks'] / g, giveaways=a['to'] / g, pen_yds=a['pen_yds'] / g,
            pts_a=pts_a[t] / g, yds_a=b['yds'] / g, ypp_a=b['yds'] / max(b['plays'], 1), pass_yds_a=b['pass_yds'] / g,
            rush_yds_a=b['rush_yds'] / g, epa_a=b['epa'] / max(b['plays'], 1), success_a=pct(b['success'], b['plays']),
            third_a=pct(b['third_conv'], b['third_att']), redzone_a=pct(rz_a[t][1], rz_a[t][0]), sacks=b['sacks'] / g,
            takeaways=b['to'] / g, explosive_a=b['explosive'] / g, to_margin=(b['to'] - a['to']) / g, games=len(games[t]))
    rank = {t: {} for t in ABBRS}
    for key, _, _, hi, _ in STATS:
        for t in ABBRS:
            rank[t][key] = 1 + sum(1 for u in ABBRS if (val[u][key] > val[t][key] if hi else val[u][key] < val[t][key]))
    return val, rank


def table(season):
    """Per team: list of dict(key, label, side, value, rank) ready for display."""
    val, rank = compute(season)
    return {t: [dict(key=k, label=lab, side=side, value=format(val[t][k], fmt), rank=rank[t][k])
                for k, lab, side, _, fmt in STATS] for t in ABBRS}


if __name__ == '__main__':
    tab = table(nflv.current_season())
    for t in ('CHI', 'SEA'):
        print(t, ', '.join(f"{s['label']} {s['value']} ({s['rank']})" for s in tab[t]))
