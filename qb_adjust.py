"""Each team's current starting quarterback, from today's nflverse depth chart (injured-reserve
players drop off it). lineup.py turns a change of starter into points."""
import csv
import nflv


def starters(season):
    """{team: (name, gsis_id, date)} for the latest depth chart's QB1."""
    latest, qb1 = {}, {}
    rows = [r for r in csv.DictReader(open(nflv.get('depth', season), encoding='utf-8')) if r['pos_abb'] == 'QB']
    for r in rows:
        latest[r['team']] = max(latest.get(r['team'], ''), r['dt'])
    for r in rows:
        if r['dt'] == latest[r['team']] and r['pos_rank'] == '1':
            qb1[r['team']] = (r['player_name'], r['gsis_id'], r['dt'][:10])
    return qb1


if __name__ == '__main__':
    for t, (n, _, dt) in sorted(starters(nflv.current_season()).items()):
        print(t, n, dt)
