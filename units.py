"""Offense, defense and quarterback rankings from nflverse play-by-play.

Offense and defense: EPA per play (expected points added) on runs and dropbacks, adjusted for the
opponents faced and shrunk slightly toward average so a single blowout does not dominate; garbage
time (win probability under 5% or over 95%) is left out. Quarterbacks: EPA per dropback including
scrambles and sacks, with completion % over expected; minimum 100 dropbacks.
Run: python units.py   (writes units.json)
"""
import csv, gzip, io, json, urllib.request
import numpy as np
from teams import ABBRS, IDX, display_abbr, name

URL = 'https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_{}.csv.gz'
SHRINK_PLAYS = 60    # one game's worth of plays pulls each team toward average
MIN_DROPBACKS = 100


def plays(season):
    raw = urllib.request.urlopen(URL.format(season), timeout=120).read()
    for r in csv.DictReader(io.TextIOWrapper(gzip.GzipFile(fileobj=io.BytesIO(raw)), encoding='utf-8')):
        if r['season_type'] == 'REG' and r['play_type'] in ('pass', 'run') and r['epa'] not in ('', 'NA'):
            yield r


def adjusted(rows):
    """Solve offense[t] + defense[opp] = EPA per play for every team-game, weighted by plays."""
    n = len(ABBRS)
    M = np.diag(np.full(2 * n, float(SHRINK_PLAYS)))
    b = np.zeros(2 * n)
    games = {}
    for r in rows:
        g = games.setdefault((r['game_id'], r['posteam'], r['defteam']), [0, 0.0])
        g[0] += 1
        g[1] += float(r['epa'])
    mean = sum(e for _, e in games.values()) / sum(c for c, _ in games.values())
    for (_, off, de), (cnt, epa) in games.items():
        o, d = IDX[off], n + IDX[de]
        M[o, o] += cnt; M[d, d] += cnt; M[o, d] += cnt; M[d, o] += cnt
        b[o] += epa - cnt * mean; b[d] += epa - cnt * mean
    x = np.linalg.solve(M, b)
    return x[:n], x[n:]


def quarterbacks(rows):
    qb = {}
    for r in rows:
        if r['qb_dropback'] != '1' or not r['passer_player_name'] or r['qb_epa'] in ('', 'NA'):
            continue
        q = qb.setdefault(r['passer_player_id'], dict(name=r['passer_player_name'], team=r['posteam'], db=0, epa=0.0, cp=[]))
        q['db'] += 1
        q['epa'] += float(r['qb_epa'])
        q['team'] = r['posteam']
        if r['cpoe'] not in ('', 'NA'):
            q['cp'].append(float(r['cpoe']))
    out = [dict(name=q['name'], team=display_abbr(q['team']), dropbacks=q['db'], epa_per_db=round(q['epa'] / q['db'], 3),
                cpoe=round(sum(q['cp']) / len(q['cp']), 1) if q['cp'] else None)
           for q in qb.values() if q['db'] >= MIN_DROPBACKS]
    return sorted(out, key=lambda q: -q['epa_per_db'])


def main(season=2026):
    rows = [r for r in plays(season) if r['wp'] not in ('', 'NA') and 0.05 <= float(r['wp']) <= 0.95]
    off, de = adjusted(rows)
    offense = sorted(ABBRS, key=lambda t: -off[IDX[t]])
    defense = sorted(ABBRS, key=lambda t: de[IDX[t]])          # lower EPA allowed is better
    qbs = quarterbacks(rows)
    weeks = max(int(r['week']) for r in rows)
    out = dict(season=season, through_week=weeks,
               offense=[dict(rank=i, team=name(t), abbr=display_abbr(t), epa_per_play=round(float(off[IDX[t]]), 3))
                        for i, t in enumerate(offense, 1)],
               defense=[dict(rank=i, team=name(t), abbr=display_abbr(t), epa_per_play_allowed=round(float(de[IDX[t]]), 3))
                        for i, t in enumerate(defense, 1)],
               quarterbacks=[dict(rank=i, **q) for i, q in enumerate(qbs, 1)])
    json.dump(out, open('units.json', 'w'), indent=1)
    print(f"{season} through week {weeks} (EPA per play vs an average opponent; garbage time excluded)")
    print(f"\n{'#':>2} {'Offense':5s} {'EPA/play':>8}   {'Defense':7s} {'EPA/play':>8}")
    for o, d in zip(out['offense'], out['defense']):
        print(f"{o['rank']:2d} {o['abbr']:5s} {o['epa_per_play']:+8.3f}   {d['abbr']:7s} {d['epa_per_play_allowed']:+8.3f}")
    print(f"\nQuarterbacks (min {MIN_DROPBACKS} dropbacks)")
    for q in out['quarterbacks']:
        cpoe = '' if q['cpoe'] is None else f"  CPOE {q['cpoe']:+.1f}"
        print(f"{q['rank']:2d} {q['name']:22s} {q['team']:4s} {q['epa_per_db']:+.3f} EPA/dropback  ({q['dropbacks']} dropbacks){cpoe}")


if __name__ == '__main__':
    main()
