"""Pieces borrowed from nfelo (github.com/greerreNFL) for SAM's multi-factor model, all known before kickoff.

qb538, qb538_level  538-style QB value (the formula nfelo's QB model uses), opponent-adjusted, an EWMA over each QB's
                    starts and each team's starts. qb538 = starter minus the team's recent QB value (the backup swap),
                    qb538_level = starter minus league average. Points = value x 3.3 / 25. Built from nflverse stats.
wepa                nfelo's weighted EPA per game (point-in-time weights), as an opponent-adjusted season rating with
                    last season as a prior, from games before each week.
wt, wt_early        preseason rating from betting season win totals (nfelosrs wt_ratings.csv, points vs average);
                    wt_early fades it out over the first 8 weeks.
Research and test: research/nfelo_pieces.py.
"""
import math, os, time, urllib.request
from collections import defaultdict
import numpy as np, pandas as pd
import nflv
from model import rate
from teams import ABBRS, IDX

WEPA_URL = 'https://raw.githubusercontent.com/greerreNFL/wepa/main/data/wepa_by_game_pit.csv'
WT_URL = 'https://raw.githubusercontent.com/greerreNFL/nfelosrs/main/wt_ratings.csv'
FIXT = {'LAR': 'LA', 'STL': 'LA', 'OAK': 'LV', 'SD': 'LAC'}
NAMES = ['qb538', 'qb538_level', 'wepa', 'wt', 'wt_early']
fx = lambda t: FIXT.get(t, t)
A = 0.1


def _cached(url, name, hours=6):
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', name)
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > hours * 3600:
        try:
            data = urllib.request.urlopen(url, timeout=120).read()
            os.makedirs(os.path.dirname(path), exist_ok=True)
            with open(path + '.tmp', 'wb') as f:
                f.write(data)
            os.replace(path + '.tmp', path)
        except Exception:
            if not os.path.exists(path):
                raise
    return path


def compute(games_path, first=2012, qb_override=None):
    """({game_id: {feature: value}} for every game in games_path from `first` on, played or not, and
    {team: {'wepa', 'wt', 'qb538_level'}} as of now). qb_override: {team: gsis id} for unplayed games."""
    games = pd.read_csv(games_path)
    games = games[(games.season >= first) & (games.game_type == 'REG' if 'game_type' in games else True)].sort_values(['gameday', 'game_id'])
    seasons = sorted(set(games.season))
    played = lambda g: not (isinstance(g.home_score, float) and math.isnan(g.home_score)) and g.home_score != ''

    # QB value
    pw = pd.concat([pd.read_csv(nflv.get('player_week', s), low_memory=False) for s in seasons])
    pw = pw[(pw.season_type == 'REG') & (pw.position == 'QB')].fillna(0)
    pw['team'] = pw.team.map(fx)
    pw['value'] = (-2.2 * pw.attempts + 3.7 * pw.completions + pw.passing_yards / 5 + 11.3 * pw.passing_tds - 14.1 * pw.passing_interceptions
                   - 8 * pw.sacks_suffered - 1.1 * pw.carries + 0.6 * pw.rushing_yards + 15.9 * pw.rushing_tds)
    pw['plays'] = pw.attempts + pw.carries + pw.sacks_suffered
    main = {k: (r.player_id, r.value) for k, r in pw.sort_values('plays').groupby(['game_id', 'team']).tail(1).set_index(['game_id', 'team']).iterrows()}
    qb_val, team_val, def_allowed, hist, feat = {}, {}, {}, [], {}
    qb_override = qb_override or {}
    for g in games.itertuples():
        h, a = fx(g.home_team), fx(g.away_team)
        league = float(np.mean(hist[-1000:])) if hist else 0.0
        repl = float(np.percentile(hist[-1000:], 25)) if len(hist) > 50 else league
        is_played = played(g)

        def pre(team, qb_id):
            if not is_played and qb_override.get(team):
                qb_id = qb_override[team]
            sv = qb_val.get(qb_id, repl) if isinstance(qb_id, str) and qb_id else repl
            return sv - team_val.get(team, sv), sv - league
        (dh, lh), (da, la) = pre(h, g.home_qb_id), pre(a, g.away_qb_id)
        feat[g.game_id] = dict(qb538=(dh - da) * 3.3 / 25, qb538_level=(lh - la) * 3.3 / 25)
        if not is_played:
            continue
        for team, opp in ((h, a), (a, h)):
            if (g.game_id, team) not in main:
                continue
            qid, val = main[(g.game_id, team)]
            d = def_allowed.get(opp)
            adj = val - ((d - league) if d is not None else 0.0)
            def_allowed[opp] = val if d is None else (1 - A) * d + A * val
            qb_val[qid] = adj if qid not in qb_val else (1 - A) * qb_val[qid] + A * adj
            team_val[team] = adj if team not in team_val else (1 - A) * team_val[team] + A * adj
            hist.append(adj)
    league = float(np.mean(hist[-1000:])) if hist else 0.0
    repl = float(np.percentile(hist[-1000:], 25)) if hist else 0.0
    team_now = {t: {'qb538_level': (qb_val.get(qb_override.get(t), team_val.get(t, repl)) - league) * 3.3 / 25} for t in ABBRS}

    # weighted EPA rating
    wp = pd.read_csv(_cached(WEPA_URL, 'wepa_by_game_pit.csv'))
    wp['team'] = wp.team.map(fx)
    wnet = {(r.game_id, r.team): r.wepa_net for r in wp.itertuples() if not math.isnan(r.wepa_net)}
    gl = [dict(season=int(g.season), week=int(g.week), home=fx(g.home_team), away=fx(g.away_team), neutral=g.location == 'Neutral',
               margin=wnet[(g.game_id, fx(g.home_team))]) for g in games.itertuples() if (g.game_id, fx(g.home_team)) in wnet]
    hfa_w = float(np.mean([x['margin'] for x in gl if x['season'] <= seasons[0] + 1 and not x['neutral']])) if gl else 0.0
    end, r = {}, None
    for s in seasons:
        cur = [x for x in gl if x['season'] == s]
        for w in sorted(set(games[games.season == s].week)):
            r = rate([x for x in cur if x['week'] < w], 8, hfa_w, end.get(s - 1), 0.6)
            for g in games[(games.season == s) & (games.week == w)].itertuples():
                feat[g.game_id]['wepa'] = float(r[IDX[fx(g.home_team)]] - r[IDX[fx(g.away_team)]])
        end[s] = rate(cur, 8, hfa_w, end.get(s - 1), 0.6)
    last = end[seasons[-1]]
    for t in ABBRS:
        team_now[t]['wepa'] = float(last[IDX[t]])

    # win-total prior
    wt = pd.read_csv(_cached(WT_URL, 'wt_ratings.csv', hours=24))
    wtd = {(fx(x.team), int(x.season)): x.wt_rating for x in wt.itertuples()}
    for g in games.itertuples():
        f = feat[g.game_id]
        f['wt'] = wtd.get((fx(g.home_team), int(g.season)), 0.0) - wtd.get((fx(g.away_team), int(g.season)), 0.0)
        f['wt_early'] = f['wt'] * max(0.0, 1 - (int(g.week) - 1) / 8)
    for t in ABBRS:
        team_now[t]['wt'] = wtd.get((t, seasons[-1]), 0.0)
    return feat, team_now
