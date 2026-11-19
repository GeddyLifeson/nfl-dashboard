"""Shared pieces for the ratings research: game rows, ridge fit, leave-one-season-out, scoring."""
import math, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import teamrate
from model import load_games

TRAIN, TEST = tuple(range(2014, 2025)), (2025, 2026)


def games(path=teamrate.ALL_GAMES):
    return [g for g in load_games(path) if g['margin'] is not None and g['season'] >= 2014]


def rows(R, gl):
    """One dict per game: home-minus-away value of every bucket feature, from games before that week only."""
    out, memo = [], {}
    for g in gl:
        key = (g['season'], g['week'])
        if key not in memo:
            memo[key] = R.features(*key)
        fh, fa = memo[key][g['home']], memo[key][g['away']]
        f = {k: fh[k] - fa[k] for k in fh}
        f['home'] = 0.0 if g['neutral'] else 1.0
        out.append(dict(id=g['id'], season=g['season'], week=g['week'], margin=g['margin'], spread=g['spread'], f=f))
    return out


def fit(rs, names, lam):
    """Ridge on standardised features; 'home' is never penalised and there is no other intercept
    (swapping home and away must flip the prediction)."""
    X = np.array([[r['f'][n] for n in names] for r in rs], float); y = np.array([r['margin'] for r in rs], float)
    sd = np.sqrt((X ** 2).mean(0)) + 1e-9
    Z = X / sd
    P = lam * np.eye(len(names))
    if 'home' in names:
        P[names.index('home'), names.index('home')] = 0
    b = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    return dict(zip(names, (b / sd).tolist()))


def pred(coef, rs):
    return np.array([sum(c * r['f'][n] for n, c in coef.items()) for r in rs])


def loso(rs, names, lam, seasons=TRAIN):
    """Out-of-season predictions for every game in `seasons` (each season predicted by a fit on the others)."""
    P = np.zeros(len(rs)); idx = {s: [i for i, r in enumerate(rs) if r['season'] == s] for s in seasons}
    for s in seasons:
        c = fit([r for r in rs if r['season'] in seasons and r['season'] != s], names, lam)
        P[idx[s]] = pred(c, [rs[i] for i in idx[s]])
    keep = [i for s in seasons for i in idx[s]]
    return P[keep], [rs[i] for i in keep]


def wp(x, sigma):
    return 0.5 * (1 + math.erf(x / sigma / math.sqrt(2)))


def score(P, rs, sigma=13.2):
    y = np.array([r['margin'] for r in rs], float); nt = y != 0
    p = np.clip([wp(x, sigma) for x in P], 1e-6, 1 - 1e-6); win = (y > 0).astype(float)
    return dict(n=len(y), mse=round(float(np.mean((P - y) ** 2)), 2), mae=round(float(np.mean(np.abs(P - y))), 3),
                winners=round(float(np.mean((P[nt] > 0) == (y[nt] > 0))), 4),
                log_loss=round(float(-np.mean((win * np.log(p) + (1 - win) * np.log(1 - p))[nt])), 4))


def paired(Pa, Pb, rs, n=4000, seed=1):
    """Is model A's squared error lower than B's? Mean difference (A - B) with a 95% bootstrap interval over games."""
    y = np.array([r['margin'] for r in rs], float)
    d = (Pa - y) ** 2 - (Pb - y) ** 2
    rng = np.random.default_rng(seed)
    bs = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)]
    return round(float(d.mean()), 2), round(float(np.percentile(bs, 2.5)), 2), round(float(np.percentile(bs, 97.5)), 2)


def best_lam(rs, names, lams=(1, 3, 10, 30, 100, 300, 1000)):
    res = []
    for lam in lams:
        P, rr = loso(rs, names, lam)
        res.append((float(np.mean((P - np.array([r['margin'] for r in rr])) ** 2)), lam))
    return min(res)
