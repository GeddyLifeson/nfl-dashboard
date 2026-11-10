"""Fit and test the multi-factor model: ridge regression of the final margin on standardized pre-game features.
Lambda chosen by leave-one-season-out on the training seasons; then frozen and tested on later seasons."""
import json, math, sys
import numpy as np

import os
TRAIN = tuple(range(int(os.environ.get('TRAIN_FROM', 2022)), 2025)); TEST = (2025, 2026)
EFF = ['off_epa', 'def_epa', 'off_pass_epa', 'def_pass_epa', 'off_rush_epa', 'def_rush_epa', 'off_success', 'def_success', 'st', 'to_margin']
LINEUP = ['qb', 'miss_OL', 'miss_WR', 'miss_TE', 'miss_RB', 'miss_EDGE', 'miss_DL', 'miss_LB', 'miss_CB', 'miss_S']
SITU = ['home', 'rest', 'bye', 'tz_shift', 'west_early', 'wind_x_pass']
NET = ['net_epa', 'net_success', 'st', 'to_margin']
COACH = ['new_coach', 'mid_change']
SETS = {
    'A current model (scores only)': ['base'],
    'B scores + efficiency': ['base'] + EFF,
    'C + QB and injuries': ['base'] + EFF + LINEUP,
    'D + rest, travel, weather': ['base'] + EFF + LINEUP + SITU,
    'E + coaching changes': ['base'] + EFF + LINEUP + SITU + COACH,
    'F everything except scores': EFF + LINEUP + SITU + COACH,
    'G scores + QB and injuries': ['base'] + LINEUP,
    'H scores + QB/injuries + rest/travel': ['base'] + LINEUP + SITU,
    'I compact: scores + net efficiency + lineup + situ + coach': ['base'] + NET + LINEUP + SITU + COACH,
}
LAMS = [1, 3, 10, 30, 100, 300, 1000, 3000, 10000]


def mat(rs, names):
    return np.array([[r['f'][n] for n in names] for r in rs], float), np.array([r['margin'] for r in rs], float)


def fit(X, y, lam):
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    Z1 = np.hstack([Z, np.ones((len(Z), 1))])
    P = lam * np.eye(Z1.shape[1]); P[-1, -1] = 0
    b = np.linalg.solve(Z1.T @ Z1 + P, Z1.T @ y)
    return lambda X2: np.hstack([(X2 - mu) / sd, np.ones((len(X2), 1))]) @ b, dict(zip(range(len(b)), b))


def metrics(pred, y, sigma):
    nt = y != 0
    p = 0.5 * (1 + np.vectorize(math.erf)(pred / sigma / math.sqrt(2)))
    p = np.clip(p, 1e-6, 1 - 1e-6); win = (y > 0).astype(float)
    ll = -np.mean((win * np.log(p) + (1 - win) * np.log(1 - p))[nt])
    return dict(games=int(len(y)), winners=round(float(np.mean((pred[nt] > 0) == (y[nt] > 0))), 4),
                mse=round(float(np.mean((pred - y) ** 2)), 1), log_loss=round(float(ll), 4))


def choose_lam(rs, names):
    best = None
    for lam in LAMS:
        err = []
        for s in TRAIN:
            tr = [r for r in rs if r['season'] in TRAIN and r['season'] != s]; va = [r for r in rs if r['season'] == s]
            f, _ = fit(*mat(tr, names), lam); X, y = mat(va, names); err += list((f(X) - y) ** 2)
        e = float(np.mean(err))
        if best is None or e < best[1]:
            best = (lam, e)
    return best[0]


def run(rs, names):
    lam = choose_lam(rs, names)
    tr = [r for r in rs if r['season'] in TRAIN]; te = [r for r in rs if r['season'] in TEST and r['spread'] is not None]
    f, b = fit(*mat(tr, names), lam)
    Xtr, ytr = mat(tr, names); sigma = float(np.std(f(Xtr) - ytr))
    X, y = mat(te, names)
    return lam, f, sigma, metrics(f(X), y, sigma), te


if __name__ == '__main__':
    rs = [r for r in json.load(open('cache/mf_rows.json')) if r['season'] in TRAIN + TEST]
    print('train seasons', TRAIN[0], '-', TRAIN[-1], 'games', sum(r['season'] in TRAIN for r in rs))
    out = {}
    for label, names in SETS.items():
        lam, f, sigma, m, te = run(rs, names)
        out[label] = dict(lam=lam, **m)
        print(f'{label:32s} lam={lam:<5} {m}')
    te = [r for r in rs if r['season'] in TEST and r['spread'] is not None]
    y = np.array([r['margin'] for r in te], float); sp = np.array([r['spread'] for r in te], float)
    print(f'{"closing line":32s}           {metrics(sp, y, 13.0)}')
    full = SETS['E + coaching changes']
    print('\nDrop one group from E:')
    for g, cols in [('efficiency', EFF), ('lineup', LINEUP), ('situational', SITU), ('coaching', COACH)]:
        names = [n for n in full if n not in cols]
        print(f'  without {g:12s}', run(rs, names)[3])
    for s in TEST:
        sub = [r for r in rs if r['season'] == s and r['spread'] is not None]
        ys = np.array([r['margin'] for r in sub], float)
        for label in ['A current model (scores only)', 'E + coaching changes']:
            _, f, sig, _, _ = run(rs, SETS[label]); X, _ = mat(sub, SETS[label])
            print(s, label, metrics(f(X), ys, sig))
        print(s, 'closing line', metrics(np.array([r['spread'] for r in sub]), ys, 13.0))
