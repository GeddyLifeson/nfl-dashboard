"""Do referee crews matter? Crew differences in flags, home-team penalty edge, margin vs the spread and points vs the
total, tested for being beyond chance (permutation), then out of sample: crew effects estimated on 2014-24 (shrunk),
applied to 2025-26."""
import csv, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nflv
from lineup import FIX

G = [r for r in csv.DictReader(open(sys.argv[1])) if r['game_type'] == 'REG' and int(r['season']) >= 2014 and r['home_score'] and r['referee']]
pen = {}
for s in sorted({int(r['season']) for r in G}):
    df = pd.read_csv(nflv.get('pbp', s), compression='gzip', low_memory=False,
                     usecols=['game_id', 'season_type', 'penalty', 'penalty_team', 'penalty_yards', 'home_team', 'away_team'])
    df = df[(df.season_type == 'REG') & (df.penalty == 1)]
    for gid, d in df.groupby('game_id'):
        h = d.home_team.iloc[0]
        hy = d.penalty_yards[d.penalty_team == h].sum(); ay = d.penalty_yards[d.penalty_team != h].sum()
        pen[gid] = (len(d), float(ay - hy))                 # flags, penalty-yard edge to the home team
rows = []
for r in G:
    if r['game_id'] not in pen or not r['spread_line'] or not r['total_line']:
        continue
    m = int(r['home_score']) - int(r['away_score']); t = int(r['home_score']) + int(r['away_score'])
    rows.append(dict(season=int(r['season']), ref=r['referee'], flags=pen[r['game_id']][0], pen_edge=pen[r['game_id']][1],
                     ats=m - float(r['spread_line']), ou=t - float(r['total_line'])))
d = pd.DataFrame(rows)
print(f"{len(d)} games with a referee, {d.ref.nunique()} referees, seasons {d.season.min()}-{d.season.max()}")


def spread_beyond_chance(x, groups, n=2000, seed=1):
    """Variance of crew means (games-weighted) vs the same with crews shuffled. Returns (observed, p)."""
    rng = np.random.default_rng(seed)
    def stat(g):
        means = pd.Series(x).groupby(g).agg(['mean', 'size'])
        return float(np.average((means['mean'] - x.mean()) ** 2, weights=means['size']))
    obs = stat(groups)
    sims = [stat(rng.permutation(groups)) for _ in range(n)]
    return obs, float(np.mean([s >= obs for s in sims]))

train, test = d[d.season <= 2024], d[d.season >= 2025]
print('\n== Do crews differ beyond chance? (2014-24, permutation test; p < 0.05 means real differences)')
for col, lab in [('flags', 'flags per game'), ('pen_edge', 'penalty-yard edge to home team'), ('ats', 'home margin minus spread'), ('ou', 'total points minus the total line')]:
    obs, p = spread_beyond_chance(train[col].values, train.ref.values)
    top = train.groupby('ref')[col].agg(['mean', 'size']).query('size >= 80').sort_values('mean')
    print(f'  {lab:34s} p = {p:.3f}   crew range (80+ games): {top["mean"].iloc[0]:+.2f} to {top["mean"].iloc[-1]:+.2f}')

print('\n== Out of sample: crew effect from 2014-24 (shrunk to 0 by k games), applied to 2025-26')
for col, lab, k in [('flags', 'flags', 40), ('pen_edge', 'home penalty edge', 40), ('ats', 'home vs spread', 80), ('ou', 'points vs total', 80)]:
    g = train.groupby('ref')[col].agg(['sum', 'size'])
    eff = ((g['sum'] - g['size'] * train[col].mean()) / (g['size'] + k)).to_dict()
    t = test[test.ref.isin(eff)]
    x = t.ref.map(eff).values; y = t[col].values - train[col].mean()
    corr = float(np.corrcoef(x, y)[0, 1]) if len(t) > 2 else float('nan')
    line = ''
    if col in ('ats', 'ou'):
        sel = np.abs(x) >= 0.5
        side = np.sign(x[sel]); res = np.sign(t[col].values[sel])
        w, l = int((side == res).sum()), int((side == -res).sum())
        line = f' | betting the crew lean (|effect| >= 0.5 pts): {w}-{l}' + (f' ({w / (w + l):.3f})' if w + l else '')
    print(f'  {lab:20s} test games {len(t)}, correlation with what happened {corr:+.3f}{line}')
