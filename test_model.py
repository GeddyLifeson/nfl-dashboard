"""Run: python test_model.py"""
import numpy as np
from model import Engine, load_games, load_params, rate, score, win_prob
from teams import IDX


def game(home, away, margin, neutral=False):
    return dict(home=home, away=away, margin=margin, neutral=neutral)


def test_rate():
    r = rate([game('KC', 'BUF', 10, neutral=True)], k=1e-6, hfa=2)
    assert abs((r[IDX['KC']] - r[IDX['BUF']]) - 10) < 1e-3, r
    r = rate([game('KC', 'BUF', 2)], k=1e-6, hfa=2)
    assert abs(r[IDX['KC']] - r[IDX['BUF']]) < 1e-9, 'winning by exactly home field means equal teams'
    base = np.zeros(len(IDX)); base[IDX['KC']] = 8
    r = rate([], k=6, hfa=2, base=base, carry=0.5)
    assert abs(r[IDX['KC']] - 4) < 1e-9, 'no games yet: rating is the carried share of last season'
    assert abs(win_prob(0, 13) - 0.5) < 1e-12 and win_prob(7, 13) > 0.5


def test_beats_home_team_on_2025():
    games = load_games()
    preds = Engine(games, load_params()).walk(2025, first_week=2)
    s = score(preds)
    home = np.mean([g['margin'] > 0 for _, g in preds if g['margin'] != 0])
    assert s['n'] > 250 and s['acc'] > home + 0.03, (s, home)


if __name__ == '__main__':
    test_rate()
    test_beats_home_team_on_2025()
    print('ok')
