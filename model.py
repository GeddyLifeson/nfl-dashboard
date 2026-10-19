"""Results-based team ratings: each team's rating is its average opponent-adjusted scoring margin,
shrunk toward a share of last season's final rating. Ratings are in points per game against an average team."""
import csv, json, math
from functools import lru_cache
import numpy as np
from teams import IDX

N = len(IDX)


def load_games(path='data/games.csv'):
    games = []
    for r in csv.DictReader(open(path)):
        played = r['home_score'] != ''
        games.append(dict(
            id=r['game_id'], season=int(r['season']), week=int(r['week']), day=r['gameday'],
            away=r['away_team'], home=r['home_team'], neutral=r['location'] == 'Neutral',
            margin=int(r['home_score']) - int(r['away_score']) if played else None,
            spread=float(r['spread_line']) if r['spread_line'] else None))
    return games


def load_params(path='params.json'):
    return json.load(open(path))


def rate(games, k, hfa, base=None, carry=0.0):
    """Solves, for every team, rating = (sum over its games of margin + opponent rating,
    after removing home field, + k * carry * base) / (games + k). k > 0 is required."""
    prior = np.zeros(N) if base is None else carry * base
    M = np.diag(np.full(N, float(k)))
    b = k * prior
    for g in games:
        h, a = IDX[g['home']], IDX[g['away']]
        m = g['margin'] - (0.0 if g['neutral'] else hfa)
        M[h, h] += 1; M[a, a] += 1; M[h, a] -= 1; M[a, h] -= 1
        b[h] += m; b[a] -= m
    return np.linalg.solve(M, b)


def predict(r, g, p):
    """Predicted home margin."""
    return p['slope'] * (r[IDX[g['home']]] - r[IDX[g['away']]]) + (0.0 if g['neutral'] else p['hfa'])


def win_prob(margin, sigma):
    return 0.5 * (1 + math.erf(margin / sigma / math.sqrt(2)))


class Engine:
    """Ratings for any point in time, carrying each season's final rating into the next."""

    def __init__(self, games, p):
        self.games, self.p = games, p
        self.first = min(g['season'] for g in games)
        self._end = lru_cache(maxsize=None)(self._end_of_season)

    def played(self, season, before_week=99):
        return [g for g in self.games if g['season'] == season and g['week'] < before_week and g['margin'] is not None]

    def _end_of_season(self, season):
        return self.ratings(season, 99)

    def ratings(self, season, before_week):
        base = None if season <= self.first else self._end(season - 1)
        return rate(self.played(season, before_week), self.p['k'], self.p['hfa'], base, self.p['carry'])

    def walk(self, season, first_week=1):
        """Predict every played game from first_week on using only games from earlier weeks."""
        out = []
        weeks = sorted({g['week'] for g in self.played(season)})
        for w in weeks:
            if w < first_week:
                continue
            r = self.ratings(season, w)
            for g in self.games:
                if g['season'] == season and g['week'] == w and g['margin'] is not None:
                    out.append((predict(r, g, self.p), g))
        return out


def score(preds):
    """Winner accuracy (ties skipped) and mean squared error of predicted margins."""
    P = np.array([p for p, _ in preds])
    Y = np.array([g['margin'] for _, g in preds], float)
    nt = Y != 0
    return dict(n=len(Y), acc=float(np.mean((P[nt] > 0) == (Y[nt] > 0))), mse=float(np.mean((P - Y) ** 2)))
