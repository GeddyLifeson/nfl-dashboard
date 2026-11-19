"""Fit and test SAM's team ratings (teamrate.py). No sportsbook number is used anywhere in the fit; the closing
line is printed at the end only as a yardstick.

  python research/ratings_eval.py          # tunes, tests, writes rating_params.json and research/ratings_results.txt

Method: ratings are rebuilt week by week from 1999 using only earlier games. The weights are fitted on
2000-2024 and judged two ways on games they never saw:
  1. leave-one-season-out: each season 2000-2024 predicted by weights fitted on the other seasons;
  2. frozen test: weights from 2000-2024 applied to 2025-2026.
Measures: error of the predicted margin from the rating gap alone (rating home - rating away + home field),
share of winners, and how well a team's rating after weeks 4, 8 and 12 lines up with its results the rest of the way.
"""
import json, os, pickle, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ratings_lib as L
import teamrate
from model import Engine, load_games, load_params
from teams import ABBRS, IDX

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN = tuple(range(2000, 2025))
SEASONS = range(teamrate.FIRST, 2027)
NAMES = teamrate.FEATURES + ['home']


def build(k, carry, decay, gl, qb):
    R = teamrate.Ratings(SEASONS, k, carry, decay)
    memo, out = {}, []
    for g in gl:
        key = (g['season'], g['week'])
        if key not in memo:
            memo[key] = {t: teamrate.derive(f) for t, f in R.features(*key).items()}
        fh, fa = memo[key][g['home']], memo[key][g['away']]
        f = {n: fh[n] - fa[n] for n in teamrate.FEATURES if n != 'qb'}
        qh, qa = qb.get(g['id'], (0.0, 0.0))
        f['qb'], f['home'] = qh - qa, 0.0 if g['neutral'] else 1.0
        out.append(dict(id=g['id'], season=g['season'], week=g['week'], margin=g['margin'], spread=g['spread'], home=g['home'], away=g['away'], f=f))
    return R, out


def rest_of_season(rating_at, gl, seasons, checkpoints=(5, 9, 13)):
    """Correlation between each team's rating before week w and its average margin from week w on."""
    out = {}
    for w in checkpoints:
        xs, ys = [], []
        for s in seasons:
            r = rating_at(s, w)
            fut = {t: [] for t in ABBRS}
            for g in gl:
                if g['season'] == s and g['week'] >= w:
                    fut[g['home']].append(g['margin']); fut[g['away']].append(-g['margin'])
            for t in ABBRS:
                if len(fut[t]) >= 4:
                    xs.append(r[t]); ys.append(np.mean(fut[t]))
        out[w] = round(float(np.corrcoef(xs, ys)[0, 1]), 3)
    return out


def main():
    lines = []
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)
    teamrate.ensure_all_games()
    gl = [g for g in load_games(teamrate.ALL_GAMES) if g['margin'] is not None and g['season'] >= 2000]
    qb, _ = teamrate.qb_values()
    say(f'{len(gl)} regular-season games 2000-2026; train {TRAIN[0]}-{TRAIN[-1]}, frozen test {L.TEST}')
    best = None
    for k in (6, 9, 12):
        for carry in (0.5, 0.65):
            for decay in (0.94, 0.97, 1.0):
                _, rs = build(k, carry, decay, gl, qb)
                mse, lam = min((float(np.mean((P - np.array([r['margin'] for r in rr])) ** 2)), lam)
                               for lam in (30, 100, 300, 1000) for P, rr in [L.loso(rs, NAMES, lam, TRAIN)])
                say(f'  prior games {k:2d}, carry {carry}, weekly decay {decay}: leave-one-season-out error {mse:.2f} (ridge {lam})')
                if best is None or mse < best[0]:
                    best = (mse, k, carry, decay, lam)
    _, k, carry, decay, lam = best
    say(f'chosen: prior games {k}, carry {carry}, decay {decay}, ridge {lam}')
    R, rs = build(k, carry, decay, gl, qb)
    train = [r for r in rs if r['season'] in TRAIN]; test = [r for r in rs if r['season'] in L.TEST]
    coef = L.fit(train, NAMES, lam)
    P_cv, rr = L.loso(rs, NAMES, lam, TRAIN)
    sigma = float(np.std(P_cv - np.array([r['margin'] for r in rr])))
    P_te = L.pred(coef, test)

    # yardsticks on the same games: scores only (the old results rating), and the old SAM rating with and without win totals
    eng = Engine(load_games(teamrate.ALL_GAMES), load_params(os.path.join(HERE, '..', 'params.json')))
    p0 = eng.p
    def scores_pred(r):
        x = eng.ratings(r['season'], r['week'])
        return p0['slope'] * (x[IDX[r['home']]] - x[IDX[r['away']]]) + p0['hfa'] * r['f']['home']
    res = {'SAM rating (new, no book input)': dict(cv=L.score(P_cv, rr, sigma), test=L.score(P_te, test, sigma))}
    S_cv, S_te = np.array([scores_pred(r) for r in rr]), np.array([scores_pred(r) for r in test])
    res['Scores only (old results rating)'] = dict(cv=L.score(S_cv, rr, sigma), test=L.score(S_te, test, sigma))
    say('\nRating gap + home field only. Lower error is better.')
    say(f"{'':38s} | 2000-24 out-of-season: games error winners | 2025-26 frozen: games error winners")
    for nm, v in res.items():
        say(f"{nm:38s} | {v['cv']['n']:5d} {v['cv']['mse']:7.2f} {100 * v['cv']['winners']:5.1f}% | {v['test']['n']:4d} {v['test']['mse']:7.2f} {100 * v['test']['winners']:5.1f}%")
    say('new minus scores-only error (95% interval): out-of-season', L.paired(P_cv, S_cv, rr), ' frozen test', L.paired(P_te, S_te, test))

    old_path = os.path.join(HERE, '..', 'cache', 'old_rows.pkl')
    if not os.path.exists(old_path):
        import multifactor
        pickle.dump(multifactor.build_rows(list(range(2013, 2027)), teamrate.ALL_GAMES), open(old_path, 'wb'))
    import multifactor as mf
    old = {r['id']: r for r in pickle.load(open(old_path, 'rb'))}
    sub = [r for r in rs if r['id'] in old and r['season'] >= 2014]
    for r in sub:
        r['f'].update({'x_' + a: b for a, b in old[r['id']]['f'].items()})
    T14 = tuple(range(2014, 2025))
    sets = {'New SAM rating': NAMES,
            'Old SAM rating (with book win totals)': ['x_' + n for n in mf.TEAM_LEVEL] + ['home'],
            'Old SAM rating, win totals removed': ['x_' + n for n in mf.TEAM_LEVEL if n not in ('wt', 'wt_early')] + ['home'],
            'Scores only': None}
    say('\nSame comparison on 2014-2026, where the old SAM rating can be rebuilt (all refit on 2014-24 the same way):')
    keep = {}
    for nm, names in sets.items():
        te = [r for r in sub if r['season'] in L.TEST]
        if names is None:
            cvr = [r for r in sub if r['season'] in T14]
            Pc, Pt = np.array([scores_pred(r) for r in cvr]), np.array([scores_pred(r) for r in te])
        else:
            _, lm = min((float(np.mean((P - np.array([r['margin'] for r in q])) ** 2)), l_) for l_ in (30, 100, 300, 1000) for P, q in [L.loso(sub, names, l_, T14)])
            Pc, cvr = L.loso(sub, names, lm, T14)
            Pt = L.pred(L.fit([r for r in sub if r['season'] in T14], names, lm), te)
        keep[nm] = (Pc, Pt)
        a, b = L.score(Pc, cvr, sigma), L.score(Pt, te, sigma)
        say(f"{nm:38s} | 2014-24: {a['n']} games, error {a['mse']:.2f}, winners {100 * a['winners']:.1f}% | 2025-26: {b['n']} games, error {b['mse']:.2f}, winners {100 * b['winners']:.1f}%")
    say('new minus old-with-win-totals error (95% interval): 2014-24', L.paired(keep['New SAM rating'][0], keep['Old SAM rating (with book win totals)'][0], cvr),
        ' 2025-26', L.paired(keep['New SAM rating'][1], keep['Old SAM rating (with book win totals)'][1], te))
    line_cv = np.array([r['spread'] for r in cvr if r['spread'] is not None]); yy = [r for r in cvr if r['spread'] is not None]
    say(f"(yardstick only, never used in the fit) closing line 2014-24: error {L.score(line_cv, yy)['mse']}, winners {100 * L.score(line_cv, yy)['winners']:.1f}%")

    # rest of season
    qteam = {}
    for g in load_games(teamrate.ALL_GAMES):
        if g['id'] in qb:
            qteam[(g['season'], g['week'], g['home'])], qteam[(g['season'], g['week'], g['away'])] = qb[g['id']]
    sp = dict(coef=coef)
    def new_at(s, w):
        q = {t: next((qteam[(s, x, t)] for x in (w, w + 1, w - 1, w + 2) if (s, x, t) in qteam), 0.0) for t in ABBRS}
        return {t: v['rating'] for t, v in teamrate.team_points(R.features(s, w), sp, q).items()}
    def old_at(s, w):
        x = eng.ratings(s, w)
        return {t: float(x[IDX[t]]) for t in ABBRS}
    say('\nRating before week w against average margin from week w to the end of the season (correlation, 1 = perfect):')
    for label, seasons in (('2002-2024', range(2002, 2025)), ('2025 (held out)', (2025,))):
        say(f'  {label}: new {rest_of_season(new_at, gl, seasons)}   scores only {rest_of_season(old_at, gl, seasons)}')
    say('\nWeights (points of future margin per unit; o_ offense, d_ defense with bigger = better, n_ net):')
    say(' ', {a: round(b, 3) for a, b in coef.items()})
    out = dict(made=str(__import__('datetime').date.today()), trained_on=[TRAIN[0], TRAIN[-1]], tested_on=list(L.TEST), k=k, carry=carry, decay=decay, ridge=lam,
               coef=coef, hfa=coef['home'], sigma=round(sigma, 2), book_inputs='none', test=res)
    json.dump(out, open(teamrate.PARAMS, 'w'), indent=1)
    open(os.path.join(HERE, 'ratings_results.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
