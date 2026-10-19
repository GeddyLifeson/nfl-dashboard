"""Tune the rating settings on 2022-2024, freeze them, then test on 2025 and 2026 without changing anything.
Every prediction uses only games played before that week. Writes params.json for build.py.
Run: python backtest.py"""
import datetime, itertools, json
import numpy as np
from model import Engine, load_games, score, win_prob

TUNE = (2022, 2023, 2024)   # 2021 only seeds the carry-over into 2022
TEST = (2025, 2026)
GRID = dict(k=(3, 4, 6, 8, 12), carry=(0.0, 0.3, 0.5, 0.6, 0.7, 0.8), hfa=(1.0, 1.5, 2.0, 2.5))


def walk(games, p, seasons, first_week=1):
    e = Engine(games, p)
    return [x for s in seasons for x in e.walk(s, first_week)]


def home_edge(g, p):
    return 0.0 if g['neutral'] else p['hfa']


def tune(games):
    best = None
    for k, carry, hfa in itertools.product(*GRID.values()):
        p: dict = dict(k=k, carry=carry, hfa=hfa, slope=1.0)
        s = score(walk(games, p, TUNE))
        if best is None or s['mse'] < best[1]['mse']:
            best = (p, s)
    assert best is not None
    p = best[0]
    # Scale rating gaps so predicted margins are unbiased on the tuning seasons.
    pr = walk(games, p, TUNE)
    x = np.array([pred - home_edge(g, p) for pred, g in pr])
    y = np.array([g['margin'] - home_edge(g, p) for _, g in pr], float)
    p['slope'] = round(float(x @ y / (x @ x)), 3)
    pr = walk(games, p, TUNE)
    p['sigma'] = round(float(np.std([g['margin'] - pred for pred, g in pr])), 2)
    return p


def report(preds, sigma):
    nt = [(pred, g) for pred, g in preds if g['margin'] != 0]
    win = np.array([g['margin'] > 0 for _, g in nt], float)
    prob = np.clip([win_prob(pred, sigma) for pred, _ in nt], 1e-6, 1 - 1e-6)
    lined = [(pred, g) for pred, g in nt if g['spread'] is not None]
    s = score(preds)
    return dict(
        games=s['n'], model_acc=round(s['acc'], 3), model_mse=round(s['mse'], 1),
        home_acc=round(float(win.mean()), 3),
        line_acc=round(float(np.mean([(g['spread'] > 0) == (g['margin'] > 0) for _, g in lined if g['spread'] != 0])), 3),
        line_mse=round(float(np.mean([(g['margin'] - g['spread']) ** 2 for _, g in lined])), 1),
        brier=round(float(np.mean((prob - win) ** 2)), 4),
        logloss=round(float(-np.mean(win * np.log(prob) + (1 - win) * np.log(1 - prob))), 4))


def tiers(preds, p):
    conf = np.array([abs(pred - home_edge(g, p)) for pred, g in preds])
    out = []
    for share in (1.0, 0.5, 0.25, 0.1):
        th = np.quantile(conf, 1 - share)
        sel = [(pred, g) for (pred, g), c in zip(preds, conf) if c >= th and g['margin'] != 0]
        out.append(dict(top_share=share, games=len(sel), acc=round(float(np.mean([(pr > 0) == (g['margin'] > 0) for pr, g in sel])), 3)))
    return out


def main():
    games = load_games()
    p = tune(games)
    print('tuned on', TUNE, p)
    tested = {}
    for s in TEST:
        for w0 in (1, 4):
            key = f'{s} from week {w0}'
            tested[key] = report(walk(games, p, (s,), w0), p['sigma'])
            print(key, tested[key])
    pooled = walk(games, p, TEST, 3)
    t = tiers(pooled, p)
    for row in t:
        print('most confident {:.0%}: {} games, {:.1%}'.format(row['top_share'], row['games'], row['acc']))
    p.update(tuned_on=list(TUNE), tested=tested, confidence_tiers=t, made=datetime.date.today().isoformat())
    json.dump(p, open('params.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
