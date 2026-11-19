"""Would the new team rating make the game picks better? Same games, same extra factors (QB swap, injuries, rest,
travel, wind, coaching), only the team-strength part swapped. Printed for the record; nothing is written.

  python research/predictions_check.py
"""
import os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratings_lib as L, ratings_eval as E, teamrate, multifactor as mf
from model import load_games

gl = [g for g in load_games(teamrate.ALL_GAMES) if g['margin'] is not None and g['season'] >= 2014]
old = {r['id']: r for r in pickle.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'cache', 'old_rows.pkl'), 'rb'))}
sp = teamrate.load(); qb, _ = teamrate.qb_values()
_, rs = E.build(sp['k'], sp['carry'], sp['decay'], [g for g in gl if g['id'] in old], qb)
for r in rs:
    r['f'].update({'x_' + a: b for a, b in old[r['id']]['f'].items()})
T = tuple(range(2014, 2025))
extra = ['x_' + n for n in ['qb538', 'qb'] + mf.LINEUP[1:] + mf.SITU[1:] + mf.COACH]
sets = {'Current picks model (uses book win totals)': ['x_' + n for n in mf.NAMES],
        'Current picks model, win totals removed': ['x_' + n for n in mf.NAMES if n not in ('wt', 'wt_early')],
        'New team rating + the same extra factors': E.NAMES + extra}
keep = {}
for nm, names in sets.items():
    _, lam = min((float(np.mean((P - np.array([r['margin'] for r in q])) ** 2)), l_) for l_ in (30, 100, 300, 1000) for P, q in [L.loso(rs, names, l_, T)])
    Pc, cv = L.loso(rs, names, lam, T)
    te = [r for r in rs if r['season'] in L.TEST]
    Pt = L.pred(L.fit([r for r in rs if r['season'] in T], names, lam), te)
    keep[nm] = (Pc, Pt)
    a, b = L.score(Pc, cv), L.score(Pt, te)
    print(f"{nm:44s} | 2014-24 out-of-season: error {a['mse']:.2f}, winners {100 * a['winners']:.1f}%, log loss {a['log_loss']:.4f} | 2025-26: error {b['mse']:.2f}, winners {100 * b['winners']:.1f}%, log loss {b['log_loss']:.4f}")
k = list(sets)
print('new minus current error (95% interval): 2014-24', L.paired(keep[k[2]][0], keep[k[0]][0], cv), ' 2025-26', L.paired(keep[k[2]][1], keep[k[0]][1], te))
