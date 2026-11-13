"""Checks for the fantasy scoring and ridge helper. Run: python test_fantasy.py"""
import numpy as np
import fantasy, fantasy_kdst

row = {k: '0' for k in ['passing_yards', 'passing_tds', 'passing_interceptions', 'passing_2pt_conversions', 'rushing_yards',
                        'rushing_tds', 'rushing_2pt_conversions', 'receptions', 'receiving_yards', 'receiving_tds',
                        'receiving_2pt_conversions', 'special_teams_tds', 'sack_fumbles_lost', 'rushing_fumbles_lost',
                        'receiving_fumbles_lost']}
row.update(passing_yards='300', passing_tds='2', passing_interceptions='1', rushing_yards='20', receptions='0')
assert abs(fantasy.points(row, fantasy.SCORING['league']) - (12 + 8 - 1 + 2)) < 1e-9
wr = dict(row, passing_yards='0', passing_tds='0', passing_interceptions='0', rushing_yards='0', receptions='6',
          receiving_yards='80', receiving_tds='1', receiving_fumbles_lost='1')
assert abs(fantasy.points(wr, fantasy.SCORING['league']) - (6 + 8 + 6 - 2)) < 1e-9
assert abs(fantasy.points(wr, fantasy.SCORING['half']) - (3 + 8 + 6 - 2)) < 1e-9
assert fantasy_kdst.allow_pts(0) == 10 and fantasy_kdst.allow_pts(17) == 1 and fantasy_kdst.allow_pts(40) == -4
X = np.random.RandomState(0).randn(200, 3)
y = X @ np.array([1.0, -2.0, 0.5]) + 3
assert np.abs(fantasy.Ridge(1e-6).fit(X, y).predict(X) - y).max() < 1e-6
print('ok')
