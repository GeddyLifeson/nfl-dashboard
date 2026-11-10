"""9: SAM vs the earliest line the dashboard archived (history/ commits) vs the closing line, 2026 games only.
6: wind and totals. Both use data the dashboard already has."""
import csv, json, subprocess
import numpy as np
games = {r['game_id']: r for r in csv.DictReader(open('cache/games_2012.csv'))}
first = {}
for f in ['history/2026-w06.json', 'history/2026-w07.json', 'history/2026-w08.json', 'history/2026-w09.json', 'history/2026-w10.json']:
    shas = subprocess.run(['git', 'log', '--reverse', '--format=%H', 'origin/main', '--', f], capture_output=True, text=True).stdout.split()
    for sha in shas:
        try:
            h = json.loads(subprocess.run(['git', 'show', f'{sha}:{f}'], capture_output=True, text=True).stdout)
        except Exception:
            continue
        for g in h.get('upcoming', []):
            if g.get('line') is not None and g['id'] not in first:
                first[g['id']] = g['line']
rows = [(first[i], float(games[i]['spread_line']), int(games[i]['home_score']) - int(games[i]['away_score']))
        for i in first if i in games and games[i]['home_score'] and games[i]['spread_line']]
e, c, y = map(np.array, zip(*rows))
nt = y != 0
print(f'== 9 earliest archived line vs close, 2026 ({len(y)} games)')
print('   earliest: winners', round(float(np.mean((e[nt] > 0) == (y[nt] > 0))), 3), 'mse', round(float(np.mean((e - y) ** 2)), 1))
print('   closing : winners', round(float(np.mean((c[nt] > 0) == (y[nt] > 0))), 3), 'mse', round(float(np.mean((c - y) ** 2)), 1))
print('   mean |move| open->close', round(float(np.mean(np.abs(c - e))), 2))
json.dump({k: v for k, v in first.items()}, open('cache/first_lines.json', 'w'))

print('\n== 6 wind and totals (outdoor games): actual total minus closing total')
pts = []
for r in games.values():
    if not r['home_score'] or not r['total_line'] or int(r['season']) < 2014 or r['roof'] not in ('outdoors', 'open'):
        continue
    pts.append((int(r['season']), float(r['wind'] or 0), int(r['home_score']) + int(r['away_score']) - float(r['total_line'])))
for lab, ss in (('train 2014-24', range(2014, 2025)), ('test 2025-26', (2025, 2026))):
    d = [(w, x) for s, w, x in pts if s in ss]
    for lo, hi in ((0, 10), (10, 15), (15, 99)):
        v = [x for w, x in d if lo <= w < hi]
        if v:
            under = np.mean([x < 0 for x in v if x != 0])
            print(f'   {lab} wind {lo}-{hi} mph: {len(v)} games, actual minus line {np.mean(v):+.2f} pts, under hit {under:.3f}')
