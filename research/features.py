"""Per-game features for the multi-factor model, each built only from information available before kickoff.

All team features are home minus away. Efficiency ratings are opponent-adjusted per play (offense + opponent
defense = observed), using this season's earlier games plus last season's final rating as a prior worth
PRIOR_PLAYS plays at CARRY strength. Garbage time (win probability outside 5-95%) is dropped.
"""
import csv, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import nflv
from model import Engine, load_games, load_params, predict
from teams import IDX, ABBRS
from lineup import GROUPS, Season

N = len(ABBRS)
TZ = dict(ARI=-7, ATL=-5, BAL=-5, BUF=-5, CAR=-5, CHI=-6, CIN=-5, CLE=-5, DAL=-6, DEN=-7, DET=-5, GB=-6, HOU=-6,
          IND=-5, JAX=-5, KC=-6, LA=-8, LAC=-8, LV=-8, MIA=-5, MIN=-6, NE=-5, NO=-6, NYG=-5, NYJ=-5, PHI=-5, PIT=-5,
          SEA=-8, SF=-8, TB=-5, TEN=-6, WAS=-5)
METRICS = ['epa', 'pass_epa', 'rush_epa', 'success']
COLS = ['game_id', 'season', 'season_type', 'week', 'posteam', 'defteam', 'home_team', 'away_team', 'home_coach', 'away_coach',
        'play_type', 'epa', 'success', 'pass', 'rush', 'wp', 'interception', 'fumble_lost', 'special_teams_play']


def team_games(season):
    """One row per team-game: offensive plays and sums for each metric, special-teams EPA, giveaways."""
    df = pd.read_csv(nflv.get('pbp', season), compression='gzip', usecols=lambda c: c in COLS, low_memory=False)
    df = df[(df.season_type == 'REG') & df.posteam.notna() & df.epa.notna()]
    sc = df[df.play_type.isin(['pass', 'run'])]
    sc = sc[(sc.wp >= 0.05) & (sc.wp <= 0.95)]
    g = sc.groupby(['game_id', 'week', 'posteam', 'defteam'])
    out = pd.DataFrame({
        'n': g.size(), 'epa': g.epa.sum(), 'success': g.success.sum(),
        'pass_n': g['pass'].sum(), 'pass_epa': g.apply(lambda x: x.epa[x['pass'] == 1].sum(), include_groups=False),
        'rush_n': g['rush'].sum(), 'rush_epa': g.apply(lambda x: x.epa[x['rush'] == 1].sum(), include_groups=False)}).reset_index()
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


def adjusted(tg, metric, prior_off=None, prior_def=None, prior_plays=0.0, shrink=60.0):
    """Opponent-adjusted per-play offense and defense for metric (per-play deviation from league mean)."""
    count = {'epa': 'n', 'success': 'n', 'pass_epa': 'pass_n', 'rush_epa': 'rush_n'}[metric]
    M = np.diag(np.full(2 * N, shrink + prior_plays)); b = np.zeros(2 * N)
    if prior_off is not None:
        b[:N] += prior_plays * prior_off; b[N:] += prior_plays * prior_def
    tot = tg[count].sum()
    mean = tg[metric].sum() / tot if tot else 0.0
    for o_, d_, c, v in zip(tg.posteam, tg.defteam, tg[count], tg[metric]):
        if c <= 0:
            continue
        o, d = IDX[o_], N + IDX[d_]
        M[o, o] += c; M[d, d] += c; M[o, d] += c; M[d, o] += c
        b[o] += v - c * mean; b[d] += v - c * mean
    x = np.linalg.solve(M, b)
    return x[:N], x[N:]


def build(seasons, prior_plays=200.0, carry=0.6, games_path='cache/games_2012.csv'):
    games, p = load_games(games_path), load_params()
    eng = Engine(games, p)
    info = {r['game_id']: r for r in csv.DictReader(open(games_path))}
    rows = []
    prev_tg = team_games(min(seasons) - 1)
    for s in seasons:
        tg = team_games(s)
        prior = {m: adjusted(prev_tg, m) for m in METRICS}
        prev_coach = prev_tg.sort_values('week').groupby('posteam').coach.last().to_dict()
        ss = Season(s)
        last_played = {}
        weeks = sorted({g['week'] for g in games if g['season'] == s and g['margin'] is not None})
        for w in weeks:
            past = tg[tg.week < w]
            adj = {m: adjusted(past, m, carry * prior[m][0], carry * prior[m][1], prior_plays) for m in METRICS}
            ng = past.groupby('posteam').size()
            st = past.groupby('posteam')['st'].mean(); to = (past.groupby('posteam')['take'].mean() - past.groupby('posteam')['give'].mean())
            cur_coach = past.sort_values('week').groupby('posteam').coach.last().to_dict()
            first_coach = past.sort_values('week').groupby('posteam').coach.first().to_dict()
            r_scores = eng.ratings(s, w)
            for g in games:
                if g['season'] != s or g['week'] != w or g['margin'] is None:
                    continue
                h, a = g['home'], g['away']; i, j = IDX[h], IDX[a]
                r = info[g['id']]
                f = {'base': predict(r_scores, g, p), 'home': 0.0 if g['neutral'] else 1.0}
                for m in METRICS:
                    f['off_' + m] = adj[m][0][i] - adj[m][0][j]
                    f['def_' + m] = adj[m][1][i] - adj[m][1][j]
                f['net_epa'] = f['off_epa'] - f['def_epa']
                f['net_success'] = f['off_success'] - f['def_success']
                f['st'] = (st.get(h, 0.0) if ng.get(h, 0) else 0.0) - (st.get(a, 0.0) if ng.get(a, 0) else 0.0)
                f['to_margin'] = (to.get(h, 0.0) if ng.get(h, 0) else 0.0) - (to.get(a, 0.0) if ng.get(a, 0) else 0.0)
                hq, aq = r.get('home_qb_id'), r.get('away_qb_id')
                f['qb'] = ss.qb(h, hq, w) - ss.qb(a, aq, w)
                mh, ma = ss.missing(h, w), ss.missing(a, w)
                for k in GROUPS:
                    f['miss_' + k] = mh[k] - ma[k]
                hr, ar = int(r['home_rest'] or 7), int(r['away_rest'] or 7)
                f['rest'] = max(min(hr - ar, 7), -7)
                f['bye'] = float(hr >= 13) - float(ar >= 13)
                f['short_week'] = float(hr <= 5) - float(ar <= 5)
                tz = 0.0 if g['neutral'] else TZ[h] - TZ[a]                   # + means the away team travelled east
                f['tz_shift'] = tz
                early = r['gametime'] and r['gametime'] < '14:00'
                f['west_early'] = float(tz >= 2 and bool(early))
                outdoor = r['roof'] in ('outdoors', 'open')
                wind = float(r['wind'] or 0) if outdoor else 0.0
                temp = float(r['temp'] or 60) if outdoor else 60.0
                f['wind_x_pass'] = wind * (adj['pass_epa'][0][i] - adj['pass_epa'][0][j])
                f['cold_x_dome'] = 0.0
                f['div'] = float(r['div_game'] == '1')
                hc = lambda t: cur_coach.get(t, prev_coach.get(t))
                f['new_coach'] = float(hc(h) != prev_coach.get(h)) - float(hc(a) != prev_coach.get(a))
                f['mid_change'] = float(cur_coach.get(h) != first_coach.get(h) and h in first_coach) - float(cur_coach.get(a) != first_coach.get(a) and a in first_coach)
                rows.append(dict(season=s, week=w, id=g['id'], home=h, away=a, margin=g['margin'], spread=g['spread'], f=f))
        prev_tg = tg
    return rows


if __name__ == '__main__':
    import json, time
    t = time.time()
    rs = build(list(range(2013, 2027)))
    os.makedirs('cache', exist_ok=True)
    json.dump(rs, open('cache/mf_rows.json', 'w'))
    print(len(rs), 'rows', round(time.time() - t), 's')
