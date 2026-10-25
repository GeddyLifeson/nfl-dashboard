"""Starting-quarterback adjustment for predictions.

The rating model only sees scores, so it assumes each team keeps the quarterback its results came
from. This reads every team's current QB1 from the nflverse depth chart (refreshed several times a
day; injured-reserve players drop off it) and compares him with the quarterbacks behind the team's
season so far:

    adjustment (points per game) = (QB1's EPA per dropback - team's season EPA per dropback) x dropbacks per game

QB1's EPA per dropback is shrunk toward backup level (PRIOR_EPA) with the weight of PRIOR_DROPBACKS,
so a quarterback with few snaps counts as roughly a backup. Not walk-forward tested; it is a
correction for known lineup changes, applied only when QB1 is not the team's main passer.
"""
import csv, gzip, io, os, time, urllib.request
from collections import defaultdict

DEPTH_URL = 'https://github.com/nflverse/nflverse-data/releases/download/depth_charts/depth_charts_{}.csv'
PBP = 'cache/pbp_{}.csv.gz'
PRIOR_EPA = -0.10        # typical backup quarterback, EPA per dropback
PRIOR_DROPBACKS = 150
MAX_ADJ = 7.0            # points; a guard against small-sample extremes


def _cached(url, path, hours=6):
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > hours * 3600:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = urllib.request.urlopen(url, timeout=120).read()
        with open(path + '.tmp', 'wb') as f:
            f.write(data)
        os.replace(path + '.tmp', path)
    return path


def starters(season):
    """{team: (name, gsis_id, date)} for the latest depth chart's QB1."""
    path = _cached(DEPTH_URL.format(season), f'cache/depth_{season}.csv')
    latest, qb1 = {}, {}
    rows = [r for r in csv.DictReader(open(path, encoding='utf-8')) if r['pos_abb'] == 'QB']
    for r in rows:
        latest[r['team']] = max(latest.get(r['team'], ''), r['dt'])
    for r in rows:
        if r['dt'] == latest[r['team']] and r['pos_rank'] == '1':
            qb1[r['team']] = (r['player_name'], r['gsis_id'], r['dt'][:10])
    return qb1


def passing(season):
    """Per team: dropbacks and EPA by passer id, and games played."""
    if not os.path.exists(PBP.format(season)):
        import units
        list(units.plays(season))                    # downloads and caches the play-by-play
    by = defaultdict(lambda: defaultdict(lambda: [0, 0.0]))
    games = defaultdict(set)
    with gzip.open(PBP.format(season), 'rt', encoding='utf-8') as f:
        for r in csv.DictReader(f):
            if r['season_type'] != 'REG' or r['qb_dropback'] != '1' or r['qb_epa'] in ('', 'NA') or not r['passer_player_id']:
                continue
            q = by[r['posteam']][r['passer_player_id']]
            q[0] += 1
            q[1] += float(r['qb_epa'])
            games[r['posteam']].add(r['game_id'])
    return by, games


def adjustments(season):
    """{team: dict(points, qb1, main, note)} for teams whose QB1 is not their main passer."""
    qb1 = starters(season)
    by, games = passing(season)
    out = {}
    for team, (name, gid, _) in qb1.items():
        qbs = by.get(team)
        if not qbs:
            continue
        main = max(qbs, key=lambda k: qbs[k][0])
        if gid == main:
            continue
        db = sum(v[0] for v in qbs.values())
        team_epa = sum(v[1] for v in qbs.values()) / db
        mine = qbs.get(gid, [0, 0.0])
        shrunk = (mine[1] + PRIOR_EPA * PRIOR_DROPBACKS) / (mine[0] + PRIOR_DROPBACKS)
        per_game = db / max(len(games[team]), 1)
        pts = max(-MAX_ADJ, min(MAX_ADJ, (shrunk - team_epa) * per_game))
        out[team] = dict(points=round(pts, 1), qb1=name, qb1_dropbacks=mine[0],
                         note=f"{name} starting ({mine[0]} dropbacks this season), not the team's main passer")
    return out


if __name__ == '__main__':
    for t, a in sorted(adjustments(2026).items()):
        print(f"{t:4s} {a['points']:+.1f} pts  {a['note']}")
