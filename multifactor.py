"""SAM's multi-factor model: the scores rating plus play-by-play efficiency, the starting QB, missing starters,
rest, travel, wind and coaching changes, combined by a ridge regression fitted on 2014-2024 and tested, frozen,
on 2025-2026 (research/mf_backtest.py, research/mf_tests.py). Every feature uses only what was known before kickoff.

  python multifactor.py fit    # rebuild features since 2013 and refit -> sam_params.json (a few minutes)
  build.py calls predict_upcoming() and team_ratings() using sam_params.json.
"""
import csv, json, math, os, sys, urllib.request
import numpy as np
import pandas as pd
import nflv
from model import Engine, load_games, load_params, predict
from teams import ABBRS, IDX
from lineup import FIX, GROUPS, Season

N = len(ABBRS)
TZ = dict(ARI=-7, ATL=-5, BAL=-5, BUF=-5, CAR=-5, CHI=-6, CIN=-5, CLE=-5, DAL=-6, DEN=-7, DET=-5, GB=-6, HOU=-6,
          IND=-5, JAX=-5, KC=-6, LA=-8, LAC=-8, LV=-8, MIA=-5, MIN=-6, NE=-5, NO=-6, NYG=-5, NYJ=-5, PHI=-5, PIT=-5,
          SEA=-8, SF=-8, TB=-5, TEN=-6, WAS=-5)
METRICS = ['epa', 'pass_epa', 'rush_epa', 'success']
EFF = ['off_epa', 'def_epa', 'off_pass_epa', 'def_pass_epa', 'off_rush_epa', 'def_rush_epa', 'off_success', 'def_success', 'st', 'to_margin']
LINEUP = ['qb'] + ['miss_' + g for g in GROUPS]
SITU = ['home', 'rest', 'bye', 'tz_shift', 'west_early', 'wind_x_pass']
COACH = ['new_coach', 'mid_change']
NAMES = ['base'] + EFF + LINEUP + SITU + COACH
TEAM_LEVEL = ['base'] + EFF + COACH          # features that describe the team itself (for the power rating)
PRIOR_PLAYS, CARRY, SHRINK = 200.0, 0.6, 60.0
TRAIN, TEST = tuple(range(2014, 2025)), (2025, 2026)
PARAMS = 'sam_params.json'
ALL_GAMES = 'cache/games_all.csv'
COLS = ['game_id', 'season', 'season_type', 'week', 'posteam', 'defteam', 'home_team', 'away_team', 'home_coach', 'away_coach',
        'play_type', 'epa', 'success', 'pass', 'rush', 'wp', 'interception', 'fumble_lost']


def team_games(season):
    """One row per team-game: offensive plays and per-metric sums (garbage time out), special-teams EPA, giveaways, coach."""
    df = pd.read_csv(nflv.get('pbp', season), compression='gzip', usecols=lambda c: c in COLS, low_memory=False)
    df = df[(df.season_type == 'REG') & df.posteam.notna() & df.epa.notna()].copy()
    for c in ('posteam', 'defteam', 'home_team', 'away_team'):
        df[c] = df[c].map(lambda t: FIX.get(t, t))
    sc = df[df.play_type.isin(['pass', 'run']) & (df.wp >= 0.05) & (df.wp <= 0.95)].copy()
    sc['pass_e'] = sc.epa * (sc['pass'] == 1); sc['rush_e'] = sc.epa * (sc['rush'] == 1)
    g = sc.groupby(['game_id', 'week', 'posteam', 'defteam'])
    out = pd.DataFrame({'n': g.size(), 'epa': g.epa.sum(), 'success': g.success.sum(), 'pass_n': g['pass'].sum(),
                        'pass_epa': g.pass_e.sum(), 'rush_n': g['rush'].sum(), 'rush_epa': g.rush_e.sum()}).reset_index()
    st = df[df.play_type.isin(['kickoff', 'punt', 'field_goal', 'extra_point'])].groupby(['game_id', 'posteam']).epa.sum()
    gv = df.groupby(['game_id', 'posteam'])[['interception', 'fumble_lost']].sum().sum(axis=1)
    out['st'] = [st.get((a, b), 0.0) for a, b in zip(out.game_id, out.posteam)]
    out['give'] = [gv.get((a, b), 0.0) for a, b in zip(out.game_id, out.posteam)]
    out['take'] = [gv.get((a, b), 0.0) for a, b in zip(out.game_id, out.defteam)]
    coach = {}
    for r in df.drop_duplicates('game_id').itertuples():
        coach[(r.game_id, r.home_team)] = r.home_coach; coach[(r.game_id, r.away_team)] = r.away_coach
    out['coach'] = [coach.get((a, b)) for a, b in zip(out.game_id, out.posteam)]
    return out


def adjusted(tg, metric, prior_off=None, prior_def=None, prior_plays=0.0):
    """Opponent-adjusted offense and defense per play for metric, as deviations from the league mean."""
    count = {'epa': 'n', 'success': 'n', 'pass_epa': 'pass_n', 'rush_epa': 'rush_n'}[metric]
    M = np.diag(np.full(2 * N, SHRINK + prior_plays)); b = np.zeros(2 * N)
    if prior_off is not None:
        b[:N] += prior_plays * prior_off; b[N:] += prior_plays * prior_def
    tot = tg[count].sum()
    mean = tg[metric].sum() / tot if tot else 0.0
    for o_, d_, c, v in zip(tg.posteam, tg.defteam, tg[count], tg[metric]):
        if c > 0:
            o, d = IDX[o_], N + IDX[d_]
            M[o, o] += c; M[d, d] += c; M[o, d] += c; M[d, o] += c
            b[o] += v - c * mean; b[d] += v - c * mean
    x = np.linalg.solve(M, b)
    return x[:N], x[N:]


class Context:
    """Everything known before week `week` of `season`."""

    def __init__(self, season, week, tg, prev_tg, eng, p, ss):
        past = tg[tg.week < week]
        prior = {m: adjusted(prev_tg, m) for m in METRICS}
        self.adj = {m: adjusted(past, m, CARRY * prior[m][0], CARRY * prior[m][1], PRIOR_PLAYS) for m in METRICS}
        by = past.groupby('posteam')
        self.ng = by.size().to_dict()
        self.st = by['st'].mean().to_dict()
        self.to = (by['take'].mean() - by['give'].mean()).to_dict()
        ordered = past.sort_values('week').groupby('posteam').coach
        self.cur_coach, self.first_coach = ordered.last().to_dict(), ordered.first().to_dict()
        self.prev_coach = prev_tg.sort_values('week').groupby('posteam').coach.last().to_dict()
        self.r_scores, self.p, self.ss, self.week = eng.ratings(season, week), p, ss, week

    def team(self, t):
        """Team-level values (vs an average team) for the TEAM_LEVEL features."""
        i = IDX[t]
        v = {'base': self.p['slope'] * self.r_scores[i]}
        for m in METRICS:
            v['off_' + m], v['def_' + m] = self.adj[m][0][i], self.adj[m][1][i]
        v['st'] = self.st.get(t, 0.0) if self.ng.get(t) else 0.0
        v['to_margin'] = self.to.get(t, 0.0) if self.ng.get(t) else 0.0
        cur = self.cur_coach.get(t, self.prev_coach.get(t))
        v['new_coach'] = float(cur != self.prev_coach.get(t))
        v['mid_change'] = float(t in self.first_coach and self.cur_coach.get(t) != self.first_coach.get(t))
        return v

    def features(self, g, r, home_qb=None, away_qb=None):
        h, a, w = g['home'], g['away'], self.week
        th, ta = self.team(h), self.team(a)
        f = {k: th[k] - ta[k] for k in TEAM_LEVEL}
        f['base'] = predict(self.r_scores, g, self.p)
        f['home'] = 0.0 if g['neutral'] else 1.0
        f['qb'] = self.ss.qb(h, home_qb or r.get('home_qb_id'), w) - self.ss.qb(a, away_qb or r.get('away_qb_id'), w)
        mh, ma = self.ss.missing(h, w), self.ss.missing(a, w)
        for k in GROUPS:
            f['miss_' + k] = mh[k] - ma[k]
        hr, ar = int(r.get('home_rest') or 7), int(r.get('away_rest') or 7)
        f['rest'] = max(min(hr - ar, 7), -7)
        f['bye'] = float(hr >= 13) - float(ar >= 13)
        tz = 0.0 if g['neutral'] else TZ[h] - TZ[a]
        f['tz_shift'] = tz
        f['west_early'] = float(tz >= 2 and bool(r.get('gametime')) and r['gametime'] < '14:00')
        outdoor = r.get('roof') in ('outdoors', 'open')
        wind = float(r.get('wind') or 0) if outdoor else 0.0
        f['wind_x_pass'] = wind * (th['off_pass_epa'] - ta['off_pass_epa'])
        return f


def build_rows(seasons, games_path):
    """Features and outcomes for every played regular-season game in seasons."""
    games, p = load_games(games_path), load_params()
    eng = Engine(games, p)
    info = {r['game_id']: r for r in csv.DictReader(open(games_path))}
    rows, prev_tg = [], team_games(min(seasons) - 1)
    for s in seasons:
        tg, ss = team_games(s), Season(s)
        for w in sorted({g['week'] for g in games if g['season'] == s and g['margin'] is not None}):
            ctx = Context(s, w, tg, prev_tg, eng, p, ss)
            for g in games:
                if g['season'] == s and g['week'] == w and g['margin'] is not None:
                    rows.append(dict(season=s, week=w, id=g['id'], margin=g['margin'], spread=g['spread'], f=ctx.features(g, info[g['id']])))
        prev_tg = tg
    return rows


def _fit(rows, lam):
    X = np.array([[r['f'][n] for n in NAMES] for r in rows], float); y = np.array([r['margin'] for r in rows], float)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = np.hstack([(X - mu) / sd, np.ones((len(X), 1))])
    P = lam * np.eye(Z.shape[1]); P[-1, -1] = 0
    b = np.linalg.solve(Z.T @ Z + P, Z.T @ y)
    return dict(names=NAMES, mu=mu.tolist(), sd=sd.tolist(), b=b[:-1].tolist(), intercept=float(b[-1]), lam=lam)


def margin(m, f):
    return m['intercept'] + sum((f[n] - mu) / sd * b for n, mu, sd, b in zip(m['names'], m['mu'], m['sd'], m['b']))


def win_prob(x, sigma):
    return 0.5 * (1 + math.erf(x / sigma / math.sqrt(2)))


def score(m, rows, sigma):
    P = np.array([margin(m, r['f']) for r in rows]); y = np.array([r['margin'] for r in rows], float); nt = y != 0
    p = np.clip([win_prob(x, sigma) for x in P], 1e-6, 1 - 1e-6); win = (y > 0).astype(float)
    return dict(games=len(rows), winners=round(float(np.mean((P[nt] > 0) == (y[nt] > 0))), 4), mse=round(float(np.mean((P - y) ** 2)), 1),
                log_loss=round(float(-np.mean((win * np.log(p) + (1 - win) * np.log(1 - p))[nt])), 4))


def fit():
    os.makedirs('cache', exist_ok=True)
    import fetch
    try:
        raw = urllib.request.urlopen(fetch.URL, timeout=120).read().decode('utf-8')
        rs = [r for r in csv.DictReader(raw.splitlines()) if r['game_type'] == 'REG' and int(r['season']) >= 2012]
        for r in rs:
            r['home_team'], r['away_team'] = FIX.get(r['home_team'], r['home_team']), FIX.get(r['away_team'], r['away_team'])
        with open(ALL_GAMES, 'w', newline='') as f:
            w = csv.DictWriter(f, fetch.KEEP, extrasaction='ignore'); w.writeheader(); w.writerows(rs)
    except Exception as e:
        if not os.path.exists(ALL_GAMES):
            raise
        print(f'(schedule download failed, using {ALL_GAMES}: {e})')
    rows = build_rows(list(range(2013, nflv.current_season() + 1)), ALL_GAMES)
    train = [r for r in rows if r['season'] in TRAIN]
    best = None
    for lam in (10, 30, 100, 300, 1000):                       # leave-one-season-out on the training seasons
        err = []
        for s in TRAIN:
            m = _fit([r for r in train if r['season'] != s], lam)
            err += [(margin(m, r['f']) - r['margin']) ** 2 for r in train if r['season'] == s]
        if best is None or np.mean(err) < best[1]:
            best = (lam, float(np.mean(err)))
    m = _fit(train, best[0])
    sigma = float(np.std([margin(m, r['f']) - r['margin'] for r in train]))
    test = [r for r in rows if r['season'] in TEST and r['spread'] is not None]
    line = np.array([r['spread'] for r in test]); y = np.array([r['margin'] for r in test], float); nt = y != 0
    out = dict(made=str(pd.Timestamp.today().date()), trained_on=[TRAIN[0], TRAIN[-1]], tested_on=list(TEST), model=m, sigma=round(sigma, 2),
               test=dict(sam=score(m, test, sigma), closing_line=dict(games=len(test), winners=round(float(np.mean((line[nt] > 0) == (y[nt] > 0))), 4),
                                                                     mse=round(float(np.mean((line - y) ** 2)), 1)),
                         scores_only=score(dict(names=['base'], mu=[0.0], sd=[1.0], b=[1.0], intercept=0.0), test, load_params()['sigma'])))
    json.dump(out, open(PARAMS, 'w'), indent=1)
    print(json.dumps(out['test']), 'lambda', best[0], 'sigma', out['sigma'])


def load():
    return json.load(open(PARAMS)) if os.path.exists(PARAMS) else None


def live_context(season, week, games, p):
    eng = Engine(games, p)
    return Context(season, week, team_games(season), team_games(season - 1), eng, p, Season(season))


def predict_upcoming(ctx, sp, g, r, home_qb=None, away_qb=None):
    """(home margin, home win chance, features) for an unplayed game."""
    f = ctx.features(g, r, home_qb, away_qb)
    x = margin(sp['model'], f)
    return x, win_prob(x, sp['sigma']), f


def team_ratings(ctx, sp):
    """Points better than an average team on a neutral field, nobody missing: the team-level part of the model."""
    m = sp['model']
    coef = {n: b / sd for n, b, sd in zip(m['names'], m['b'], m['sd'])}
    vals = {t: ctx.team(t) for t in ABBRS}
    raw = {t: sum(coef[k] * v[k] for k in TEAM_LEVEL) for t, v in vals.items()}
    mean = sum(raw.values()) / len(raw)
    return {t: x - mean for t, x in raw.items()}


if __name__ == '__main__':
    if sys.argv[1:] == ['fit']:
        fit()
    else:
        print(__doc__)
