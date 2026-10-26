"""Does knowing the lineup improve the results-based predictions, and by how much per position?

For every regular-season game from 2022 on, the results model predicts the margin from earlier
games only. lineup.py then measures, also from pre-game information only, the starting QB's edge
and the missing players per position group. A ridge regression of the model's error on those
features is fitted on 2022-2024 and tested, frozen, on 2025-2026. Writes lineup_weights.json,
which build.py uses only if the test seasons improve.
Run: python lineup_backtest.py
"""
import datetime, json
import numpy as np
from lineup import GROUPS, Season
from model import Engine, load_games, load_params

TRAIN, TEST = (2022, 2023, 2024), (2025, 2026)
NAMES = ['qb'] + GROUPS
LAMBDA = 30.0


def rows(seasons):
    games, p = load_games(), load_params()
    e = Engine(games, p)
    starters = {}
    import csv
    for r in csv.DictReader(open('data/games.csv')):
        starters[r['game_id']] = (r.get('home_qb_id'), r.get('away_qb_id'))
    X, Y, M, info = [], [], [], []
    for s in seasons:
        ss = Season(s)
        for pred, g in e.walk(s, 1):
            hq, aq = starters.get(g['id'], (None, None))
            mh, ma = ss.missing(g['home'], g['week']), ss.missing(g['away'], g['week'])
            x = [ss.qb(g['home'], hq, g['week']) - ss.qb(g['away'], aq, g['week'])]
            x += [mh[k] - ma[k] for k in GROUPS]
            X.append(x); Y.append(g['margin']); M.append(pred); info.append((s, g['id']))
    return np.array(X), np.array(Y, float), np.array(M), info


def fit(X, r):
    return np.linalg.solve(X.T @ X + LAMBDA * np.eye(X.shape[1]), X.T @ r)


def score(pred, y):
    nt = y != 0
    return dict(games=int(len(y)), mse=round(float(np.mean((pred - y) ** 2)), 2),
                acc=round(float(np.mean((pred[nt] > 0) == (y[nt] > 0))), 4))


def main():
    Xtr, Ytr, Mtr, _ = rows(TRAIN)
    w = fit(Xtr, Ytr - Mtr)
    Xte, Yte, Mte, info = rows(TEST)
    out = dict(made=datetime.date.today().isoformat(), trained_on=list(TRAIN), tested_on=list(TEST), ridge_lambda=LAMBDA,
               weights={n: round(float(v), 3) for n, v in zip(NAMES, w)},
               train=dict(results_only=score(Mtr, Ytr), with_lineup=score(Mtr + Xtr @ w, Ytr)),
               test=dict(results_only=score(Mte, Yte), with_lineup=score(Mte + Xte @ w, Yte)))
    # one feature at a time on the test seasons: which parts carry the improvement
    out['test_by_feature'] = {}
    for i, n in enumerate(NAMES):
        wi = np.zeros_like(w); wi[i] = w[i]
        out['test_by_feature'][n] = score(Mte + Xte @ wi, Yte)['mse']
    out['use'] = out['test']['with_lineup']['mse'] < out['test']['results_only']['mse']
    json.dump(out, open('lineup_weights.json', 'w'), indent=1)
    print(json.dumps(out, indent=1))


if __name__ == '__main__':
    main()
