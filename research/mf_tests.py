"""Tests of the ten changes from research/how_books_set_nfl_odds.md against the multi-factor model.
Train 2014-24, frozen test on every 2025-26 game with a closing line. Run after features.py."""
import csv, json, math, os, sys
from collections import defaultdict
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault('TRAIN_FROM', '2014')
import mf_backtest as m

rs = [r for r in json.load(open('cache/mf_rows.json')) if r['season'] >= 2013]
TR, TE = m.TRAIN, m.TEST
BASEF = m.SETS['E + coaching changes']
out = {}


def show(label, names, rows=rs):
    lam, f, sig, met, te = m.run(rows, names)
    print(f'{label:44s} {met}')
    out[label] = met
    return f, sig


# team-specific home field: shrunk mean of the scores model's home residual, from earlier seasons only
def add_team_hfa(rows, k=40.0):
    home_res, road_res = defaultdict(list), defaultdict(list)   # scores-model residual at home / away team's outperformance on the road
    for s in sorted({r['season'] for r in rows}):
        cur = [r for r in rows if r['season'] == s]
        for r in cur:
            hh, ra = home_res[r['home']], road_res[r['away']]
            r['f']['team_hfa'] = (sum(hh) / (len(hh) + k) - sum(ra) / (len(ra) + k)) if r['f']['home'] else 0.0
        for r in cur:
            if r['f']['home']:
                e = r['margin'] - r['f']['base']
                home_res[r['home']].append(e); road_res[r['away']].append(-e)
add_team_hfa(rs)

print('== baseline')
fE, sigE = show('E all factors (current best)', BASEF)
print('\n== 1 QB as starter-vs-backup gap plus starter level')
show('E + qb_level', BASEF + ['qb_level'])
print('\n== 2 team-specific home field (shrunk)')
show('E + team_hfa', BASEF + ['team_hfa'])
print('\n== 3 recency weighting of the scores rating')
show('E with 0.9/week decay', [n if n != 'base' else 'base_d90' for n in BASEF])
show('E with 0.8/week decay', [n if n != 'base' else 'base_d80' for n in BASEF])
show('E + both base and 0.9 decay', BASEF + ['base_d90'])
print('\n== 5 non-QB position values: fitted points per fully missing starter-equivalent')
lam = m.choose_lam(rs, BASEF)
tr = [r for r in rs if r['season'] in TR]
X, y = m.mat(tr, BASEF)
Z1 = np.hstack([(X - X.mean(0)) / (X.std(0) + 1e-9), np.ones((len(X), 1))])
P = lam * np.eye(Z1.shape[1]); P[-1, -1] = 0
b = np.linalg.solve(Z1.T @ Z1 + P, Z1.T @ y)
raw = {n: b[i] / (X[:, i].std() + 1e-9) for i, n in enumerate(BASEF)}
print('  ', {n: round(raw[n], 2) for n in BASEF if n.startswith('miss_') or n == 'qb'})
print('\n== 10 rest/travel/coaching combined weight (points, typical game = 1 sd of their combined effect)')
idx = [i for i, n in enumerate(BASEF) if n in m.SITU + m.COACH and n != 'home']
contrib = (Z1[:, idx] @ b[idx])
print('   sd', round(float(contrib.std()), 2), 'max abs', round(float(np.abs(contrib).max()), 2),
      {n: round(raw[n], 2) for n in m.SITU + m.COACH})

# ---- 7 spread -> win chance: normal curve vs NFL's real margin distribution (key numbers)
print('\n== 7 win chance from projected margin: normal vs empirical margin distribution')
te = [r for r in rs if r['season'] in TE and r['spread'] is not None]
Xt, yt = m.mat(te, BASEF); pt = fE(Xt)
Xr, yr = m.mat(tr, BASEF); pr = fE(Xr)
resid = yr - pr
def p_emp(pred, resid=resid, bw=1.0):
    # P(margin > 0) with residuals placed on actual integer margins, smoothed slightly; ties count half
    vals = np.round(pred + resid)
    return float(np.mean(vals > 0) + 0.5 * np.mean(vals == 0))
# NFL margins cluster: compare to residuals of the closing line from training seasons, re-centred at our projection
spr = np.array([r['spread'] for r in tr if r['spread'] is not None]); ysp = np.array([r['margin'] for r in tr if r['spread'] is not None], float)
line_resid = ysp - spr
def ll(p, y):
    nt = y != 0; w = (y > 0).astype(float); p = np.clip(p, 1e-4, 1 - 1e-4)
    return round(float(-np.mean((w * np.log(p) + (1 - w) * np.log(1 - p))[nt])), 4)
pn = 0.5 * (1 + np.vectorize(math.erf)(pt / sigE / math.sqrt(2)))
pe = np.array([p_emp(x) for x in pt])
def p_key(pred):
    # line-residual shape (key numbers live in the actual margin, not the projection): P(round(pred)+resid > 0)
    vals = np.round(pred * 2) / 2 + line_resid * (sigE / line_resid.std())
    return float(np.mean(np.round(vals) > 0) + 0.5 * np.mean(np.round(vals) == 0))
pk = np.array([p_key(x) for x in pt])
print('   log loss normal', ll(pn, yt), '| empirical model residuals', ll(pe, yt), '| key-number (line residual) shape', ll(pk, yt))
for lo, hi in [(2, 4), (6, 8)]:
    sel = (np.abs(pt) >= lo) & (np.abs(pt) < hi)
    fav = np.where(pt > 0, yt > 0, yt < 0)[sel]
    print(f'   projected margin {lo}-{hi}: games {int(sel.sum())}, favorite won {fav.mean():.3f}, normal says {np.where(pt>0,pn,1-pn)[sel].mean():.3f}, empirical {np.where(pt>0,pe,1-pe)[sel].mean():.3f}')

# ---- 8 SAM vs the closing line: when they disagree by X points, who covered?
print('\n== 8 SAM vs closing line: games where SAM differs from the close by 3+ points (test seasons)')
sp = np.array([r['spread'] for r in te])
for X_ in (1.5, 3, 5):
    sel = np.abs(pt - sp) >= X_
    side = np.sign(pt - sp)[sel]; cover = np.sign(yt - sp)[sel]
    w = int((side == cover).sum()); l = int((side == -cover).sum())
    print(f'   gap {X_}+: {int(sel.sum())} games, SAM side covered {w}-{l} ({w / max(w + l, 1):.3f}; 52.4% breaks even at -110)')
json.dump(out, open('research/mf_tests.json', 'w'), indent=1)
