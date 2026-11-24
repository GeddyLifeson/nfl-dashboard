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


def test_strength_of_wins():
    from build import strength_of_wins
    from teams import ABBRS
    r = np.zeros(len(ABBRS)); r[IDX['KC']] = 8; r[IDX['MIA']] = -8
    played = [game('KC', 'BUF', -3, neutral=True), game('MIA', 'DEN', -3, neutral=True), game('NYJ', 'DAL', 0, neutral=True),
              game('SF', 'SEA', 7), game('SF', 'LAC', -7)]
    s = strength_of_wins(played, r, hfa=2, sigma=13)
    assert s['BUF']['score'] > s['DEN']['score'] > 0, 'beating a good team is worth more than beating a bad one'
    assert s['KC']['score'] == -0.5 and s['MIA']['score'] == -0.5, 'an even game on a neutral field is a coin flip for an average team'
    assert s['NYJ']['score'] == 0 and s['DAL']['score'] == 0, 'a tie in an even game is worth nothing either way'
    assert s['LAC']['score'] > 0.5 > 1 - win_prob(2, 13), 'a road win is worth more than the same win at home'
    assert s['BUF']['best'][0] == 'KC' and s['SF']['worst'][0] == 'LAC'


if __name__ == '__main__':
    test_rate()
    test_strength_of_wins()
    test_beats_home_team_on_2025()
    print('ok')
