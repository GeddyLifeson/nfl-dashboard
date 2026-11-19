"""SAM team ratings: built to rate teams accurately, with no sportsbook input of any kind.

Every play of every game is put in one of a few buckets (clean dropbacks, designed runs, sacks, interceptions,
lost fumbles, penalties, field goals, punts, kickoffs), each measured in expected points added per game. Each
bucket is adjusted for the opponents faced, weighted toward recent games, and started from a share of last
season. A regression fitted on past seasons then says how many points of FUTURE scoring margin each bucket is
worth (lucky buckets such as lost fumbles get little weight, repeatable ones such as clean passing get a lot).
A team's rating is the sum: points better than an average team on a neutral field, split into parts
(pass offense, run offense, pass defense, ...), plus the value of the current starting quarterback.

  python teamrate.py fit     # rebuild from play-by-play since 2013 and refit -> rating_params.json
  build.py calls current() for the live ratings.
Research and tests: research/ratings_eval.py, research/RATINGS.md.
"""
import json, os, sys
import numpy as np
import pandas as pd
import nflv
from lineup import FIX
from teams import ABBRS, IDX

N = len(ABBRS)
HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, 'rating_params.json')
ALL_GAMES = os.path.join(HERE, 'cache', 'games_all.csv')
CATS = ['pass', 'rush', 'sack', 'int', 'fuml', 'pen_o', 'pen_d', 'fg', 'punt', 'ko', 'succ', 'expl', 'pts', 'cap', 'win']
MARGIN_CAP = 21.0
COLS = ['game_id', 'season_type', 'week', 'posteam', 'defteam', 'home_team', 'away_team', 'play_type', 'epa', 'success', 'wp',
        'qb_dropback', 'sack', 'interception', 'fumble_lost', 'penalty', 'penalty_team', 'yards_gained', 'location',
        'home_score', 'away_score', 'pass_oe']
FIRST = 1999                                          # nflverse play-by-play starts here
# The fitted features. o_ = the team's offense, d_ = what its defense allows, n_ = offense minus defense.
FEATURES = ['o_drop', 'd_drop', 'o_rush', 'd_rush', 'o_succ', 'd_succ', 'o_int', 'd_int', 'o_pts', 'd_pts', 'o_cap', 'd_cap', 'o_st', 'n_pen', 'qb']
# which features make up each part of the rating shown on the dashboard
PARTS = {'Offense': ['o_drop', 'o_rush', 'o_succ', 'o_int', 'o_pts', 'o_cap'], 'Defense': ['d_drop', 'd_rush', 'd_succ', 'd_int', 'd_pts', 'd_cap'],
         'Special teams': ['o_st'], 'Penalties': ['n_pen'], 'Quarterback': ['qb']}


def team_games(season, garbage=(0.03, 0.97)):
    """One row per offense per game: expected points added per bucket (from the offense's side), plus counts."""
    path = os.path.join(HERE, 'cache', f'tg_{season}.pkl')
    src = nflv.get('pbp', season)
    if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(src):
        return pd.read_pickle(path)
    df = pd.read_csv(src, compression='gzip', usecols=lambda c: c in COLS, low_memory=False)
    df = df[(df.season_type == 'REG') & df.posteam.notna() & df.epa.notna()].copy()
    for c in ('posteam', 'defteam', 'home_team', 'away_team', 'penalty_team'):
        df[c] = df[c].map(lambda t: FIX.get(t, t))
    live = df.wp.between(*garbage) | df.wp.isna()
    scr = df.play_type.isin(['pass', 'run'])
    db = scr & (df.qb_dropback == 1)
    fl, it, sk = df.fumble_lost == 1, df.interception == 1, df.sack == 1
    pen = (df.play_type == 'no_play') & (df.penalty == 1)
    clean = scr & ~fl & ~it & ~sk
    league_sr = df[clean & live].success.mean()
    bucket = {
        'pass': db & clean & live, 'rush': scr & ~db & clean & live, 'sack': db & sk & ~fl & live, 'int': it & scr, 'fuml': fl & scr,
        'pen_o': pen & (df.penalty_team == df.posteam), 'pen_d': pen & (df.penalty_team == df.defteam),
        'fg': df.play_type.isin(['field_goal', 'extra_point']), 'punt': df.play_type == 'punt', 'ko': df.play_type == 'kickoff'}
    key = [df.game_id, df.week, df.posteam, df.defteam]
    out = {c: df.epa.where(m, 0.0).groupby(key).sum() for c, m in bucket.items()}
    out['succ'] = (df.success - league_sr).where(clean & live, 0.0).groupby(key).sum()         # successful plays above average
    out['expl'] = ((df.yards_gained >= 20) & clean & live).astype(float).groupby(key).sum()
    out['proe'] = (df.pass_oe.fillna(0.0) if 'pass_oe' in df else df.epa * 0).where(scr, 0.0).groupby(key).sum()        # passes above expected, in % x plays
    out['n_proe'] = (df.pass_oe.notna() if 'pass_oe' in df else df.epa < -99).where(scr, False).astype(float).groupby(key).sum()
    for c in ('pass', 'rush', 'sack', 'int', 'fuml'):
        out['n_' + c] = bucket[c].astype(float).groupby(key).sum()
    t = pd.DataFrame(out).reset_index()
    t.columns = ['game_id', 'week', 'team', 'opp'] + list(t.columns[4:])
    g = df.drop_duplicates('game_id').set_index('game_id')
    t['home'] = np.where(g.location.reindex(t.game_id).values == 'Neutral', 0.0, np.where(g.home_team.reindex(t.game_id).values == t.team, 1.0, -1.0))
    hs, as_ = g.home_score.reindex(t.game_id).values, g.away_score.reindex(t.game_id).values
    is_home = g.home_team.reindex(t.game_id).values == t.team
    t['pts'] = np.where(is_home, hs, as_).astype(float)
    opp_pts = np.where(is_home, as_, hs).astype(float)
    t['cap'] = np.clip(t.pts - opp_pts, -MARGIN_CAP, MARGIN_CAP) / 2      # half the capped margin, so offense minus defense adds up to it
    t['win'] = np.sign(t.pts - opp_pts) / 2
    t['season'] = season
    t.to_pickle(path)
    return t


def solve(tg, weights, prior_o=None, prior_d=None, k=6.0, cats=CATS):
    """Opponent-adjusted offense and defense per game for every bucket at once.
    value = league average + home edge + offense[team] + defense[opponent]; each team is pulled toward its prior
    (a share of last season) with the weight of k games. Returns (off, def): arrays teams x buckets, as
    differences from an average team. For defense, a positive number means it ALLOWS more than average."""
    C = len(cats)
    M = np.zeros((2 * N, 2 * N)); B = np.zeros((2 * N, C))
    M[np.arange(2 * N), np.arange(2 * N)] = k
    if prior_o is not None:
        B[:N] += k * prior_o; B[N:] += k * prior_d
    if len(tg):
        Y = tg[cats].values.astype(float)
        w = np.asarray(weights, float)
        h = tg.home.values
        A = np.column_stack([np.ones(len(tg)), h])                       # league average and home edge, removed first
        WA = A * w[:, None]
        coef = np.linalg.solve(A.T @ WA + 1e-6 * np.eye(2), WA.T @ Y)
        R = Y - A @ coef
        o = tg.team.map(IDX).values; d = N + tg.opp.map(IDX).values
        np.add.at(M, (o, o), w); np.add.at(M, (d, d), w); np.add.at(M, (o, d), w); np.add.at(M, (d, o), w)
        np.add.at(B, o, R * w[:, None]); np.add.at(B, d, R * w[:, None])
    X = np.linalg.solve(M, B)
    return X[:N], X[N:]


class Ratings:
    """Bucket ratings for any week, walking forward through the seasons so nothing from the future is used."""

    def __init__(self, seasons, k=6.0, carry=0.6, decay=0.97, cats=CATS):
        self.k, self.carry, self.decay, self.cats = k, carry, decay, cats
        self.tg = {s: team_games(s) for s in seasons}
        self.seasons = sorted(seasons)
        self._end, self._cache = {}, {}

    def _prior(self, season):
        if season - 1 not in self.tg:
            return None, None
        if season - 1 not in self._end:
            self._end[season - 1] = self.at(season - 1, 99)
        o, d = self._end[season - 1]
        return self.carry * o, self.carry * d

    def at(self, season, week):
        """(off, def) from games before `week` of `season`."""
        if (season, week) not in self._cache:
            tg = self.tg[season]
            past = tg[tg.week < week]
            last = past.week.max() if len(past) else 0
            w = self.decay ** (last - past.week.values) if len(past) else []
            po, pd_ = self._prior(season)
            self._cache[(season, week)] = solve(past, w, po, pd_, self.k, self.cats)
        return self._cache[(season, week)]

    def features(self, season, week):
        """{team: {feature: value}}: o_<bucket> is the offense; d_<bucket> is the defense with the sign flipped,
        so for every feature a bigger number is better."""
        o, d = self.at(season, week)
        return {t: dict({'o_' + c: o[IDX[t], j] for j, c in enumerate(self.cats)}, **{'d_' + c: -d[IDX[t], j] for j, c in enumerate(self.cats)})
                for t in ABBRS}


def derive(f):
    """Add the combined features to one team's bucket values."""
    f = dict(f)
    f['o_drop'], f['d_drop'] = f['o_pass'] + f['o_sack'], f['d_pass'] + f['d_sack']                # every dropback that did not end in a turnover
    f['o_st'] = f['o_fg'] + f['o_punt'] + f['o_ko'] - f['d_punt'] - f['d_ko']                     # own kicks, punts and returns, minus what opponents' units get
    f['n_pen'] = f['o_pen_o'] + f['o_pen_d'] - f['d_pen_o'] - f['d_pen_d']
    return f


def qb_values(games_path=ALL_GAMES, first=FIRST, qb_override=None, a=0.1):
    """Quarterback value from box-score stats (the public 538 formula), adjusted for the defenses faced, as a running
    average over each QB's starts. Returns ({game_id: (home level, away level)}, {team: level now}) in points per game
    against a league-average starter. qb_override: {team: player id} for games not played yet."""
    games = pd.read_csv(games_path)
    games = games[games.season >= first].sort_values(['gameday', 'game_id'])
    fx = lambda t: FIX.get(t, t)
    pw = pd.concat([pd.read_csv(nflv.get('player_week', s), low_memory=False,
                                usecols=['player_id', 'season_type', 'position', 'team', 'game_id', 'season', 'week', 'attempts', 'completions', 'passing_yards', 'passing_tds',
                                         'passing_interceptions', 'sacks_suffered', 'carries', 'rushing_yards', 'rushing_tds'])
                    for s in sorted(set(games.season))])
    pw = pw[(pw.season_type == 'REG') & (pw.position == 'QB')].fillna(0)
    pw['team'] = pw.team.map(fx)
    pw['game_id'] = [f"{s}_{int(w):02d}" for s, w in zip(pw.season, pw.week)]                      # matched to the schedule by season, week and team
    pw['value'] = (-2.2 * pw.attempts + 3.7 * pw.completions + pw.passing_yards / 5 + 11.3 * pw.passing_tds - 14.1 * pw.passing_interceptions
                   - 8 * pw.sacks_suffered - 1.1 * pw.carries + 0.6 * pw.rushing_yards + 15.9 * pw.rushing_tds)
    pw['plays'] = pw.attempts + pw.carries + pw.sacks_suffered
    top = pw.sort_values('plays').groupby(['game_id', 'team']).tail(1)
    main = {(g, t): (p, v) for g, t, p, v in zip(top.game_id, top.team, top.player_id, top.value)}
    qb_val, team_val, allowed, hist, feat = {}, {}, {}, [], {}
    qb_override = qb_override or {}
    for g in games.itertuples():
        h, aw = fx(g.home_team), fx(g.away_team)
        league = float(np.mean(hist[-1000:])) if hist else 0.0
        repl = float(np.percentile(hist[-1000:], 25)) if len(hist) > 50 else league
        played = not pd.isna(g.home_score)

        def level(team, qb_id):
            if not played and qb_override.get(team):
                qb_id = qb_override[team]
            return (qb_val.get(qb_id, repl) if isinstance(qb_id, str) and qb_id else team_val.get(team, repl)) - league
        feat[g.game_id] = (level(h, g.home_qb_id) * 3.3 / 25, level(aw, g.away_qb_id) * 3.3 / 25)
        if not played:
            continue
        key = f"{g.season}_{int(g.week):02d}"
        for team, opp in ((h, aw), (aw, h)):
            if (key, team) not in main:
                continue
            qid, val = main[(key, team)]
            d = allowed.get(opp)
            adj = val - ((d - league) if d is not None else 0.0)
            allowed[opp] = val if d is None else (1 - a) * d + a * val
            qb_val[qid] = (1 - a) * qb_val.get(qid, repl) + a * adj          # a new QB starts at backup level, not at his first game
            team_val[team] = adj if team not in team_val else (1 - a) * team_val[team] + a * adj
            hist.append(adj)
    league = float(np.mean(hist[-1000:])) if hist else 0.0
    repl = float(np.percentile(hist[-1000:], 25)) if hist else 0.0
    now = {t: (qb_val.get(qb_override.get(t), team_val.get(t, repl)) - league) * 3.3 / 25 for t in ABBRS}
    return feat, now


TRAIT_KEYS = ['o_drop', 'o_rush', 'o_expl', 'o_succ', 'o_int', 'd_sack', 'd_drop', 'd_rush', 'd_int', 'd_expl', 'n_pen', 'o_st']


def team_traits_input(feats, tg):
    """{team: {key: z-score across the league}} for the team tags: opponent-adjusted buckets (bigger = better)
    plus `proe`, how much more a team passes than the down, distance and score call for."""
    f = {t: derive(v) for t, v in feats.items()}
    by = tg.groupby('team')[['proe', 'n_proe']].sum() if len(tg) else None
    proe = {t: (by.proe.get(t, 0.0) / by.n_proe.get(t, 0.0) if by is not None and by.n_proe.get(t, 0.0) > 0 else 0.0) for t in f}
    out = {t: {} for t in f}
    for k in TRAIT_KEYS + ['proe']:
        v = np.array([proe[t] if k == 'proe' else f[t][k] for t in f])
        sd = v.std() or 1.0
        for t, x in zip(f, (v - v.mean()) / sd):
            out[t][k] = float(x)
    return out


def ensure_all_games():
    """The full schedule since 1999 (cache/games_all.csv), downloaded once; needed only for fitting and the all-time lists."""
    if not os.path.exists(ALL_GAMES):
        import csv, io, urllib.request, fetch
        raw = urllib.request.urlopen(fetch.URL, timeout=120).read().decode('utf-8')
        rows = [r for r in csv.DictReader(io.StringIO(raw)) if r['game_type'] == 'REG']
        for r in rows:
            r['home_team'], r['away_team'] = FIX.get(r['home_team'], r['home_team']), FIX.get(r['away_team'], r['away_team'])
        os.makedirs(os.path.dirname(ALL_GAMES), exist_ok=True)
        with open(ALL_GAMES, 'w', newline='') as f:
            w = csv.DictWriter(f, fetch.KEEP + ['game_type'], extrasaction='ignore'); w.writeheader(); w.writerows(rows)
    return ALL_GAMES


def load():
    return json.load(open(PARAMS)) if os.path.exists(PARAMS) else None


def team_points(feats, sp, qb=None):
    """{team: dict(rating, parts={part: points}, off, def)} from the fitted weights: points per game better than an
    average team on a neutral field. feats: Ratings.features(); qb: {team: quarterback level in points}."""
    coef = sp['coef']
    out = {}
    for t, f in feats.items():
        f = derive(f)
        f['qb'] = (qb or {}).get(t, 0.0)
        out[t] = dict(parts={p: sum(coef.get(k, 0.0) * f[k] for k in keys) for p, keys in PARTS.items()})
    for p in PARTS:                                                       # each part is shown against the league average
        m = float(np.mean([out[t]['parts'][p] for t in out]))
        for t in out:
            out[t]['parts'][p] -= m
    for t in out:
        x = out[t]['parts']
        out[t].update(rating=sum(x.values()), off=x['Offense'] + x['Quarterback'], **{'def': x['Defense']})
    return out


def ovr(points, per_point=2.2):
    """Points vs average on the 60-99 card scale: an average team is 78."""
    return int(round(float(np.clip(78 + per_point * points, 60, 99))))


def current(season, week, games_path, qb_override=None):
    """Live ratings after `week` games weeks: {team: dict(rating, prev, off, def, parts, buckets, traits)}.
    Needs only the last few seasons of play-by-play. games_path: schedule with scores and starting QBs."""
    import traits
    sp = load()
    first = max(FIRST, season - 3)
    R = Ratings(range(first, season + 1), sp['k'], sp['carry'], sp['decay'])
    sched = pd.read_csv(games_path)
    _, qb_now = qb_values(games_path, first=max(first, int(sched.season.min())), qb_override=qb_override)
    feats = R.features(season, week + 1)
    now = team_points(feats, sp, qb_now)
    prev = team_points(R.features(season, week), sp, qb_now) if week > 1 else None
    tg = R.tg[season]
    z = team_traits_input(feats, tg[tg.week <= week])
    out = {}
    for t, v in now.items():
        f = derive(feats[t])
        out[t] = dict(rating=float(v['rating']), prev=float(prev[t]['rating']) if prev else None, off=float(v['off']), **{'def': float(v['def'])},
                      parts={k: round(float(x), 2) for k, x in v['parts'].items()},
                      buckets={k: round(float(f[k]), 2) for k in ('o_drop', 'o_rush', 'd_drop', 'd_rush', 'o_st', 'n_pen', 'o_int', 'd_int')},
                      traits=traits.team(z[t]))
    return out


if __name__ == '__main__':
    if sys.argv[1:] == ['fit']:
        sys.path.insert(0, os.path.join(HERE, 'research'))
        import ratings_eval
        ratings_eval.main()
    else:
        print(__doc__)
