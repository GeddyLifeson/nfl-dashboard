"""SAM player ratings: an estimate of how much each player adds per game, built to be accurate, not to look tidy.

How it differs from the old position scores (players.rankings):
  * Sample size. Every stat is pulled toward the position average by an amount measured from the data: a stat
    that takes 400 plays to mean something (interception rate) is trusted far less than one that settles in 80
    (sack rate, completion % over expected). A hot three-game stretch no longer reads as a 99.
  * Last season counts (at a discount), so a proven player is not rated on six games alone.
  * The weights are fitted, not hand-picked: for each position they are the mix of stats that best predicted the
    player's efficiency over the REST of past seasons (2006-2022), then checked on 2023-2025.
  * Value = (efficiency - a replacement-level player's) x expected workload, in expected points per game.
  * Defenders are credited with the measured point value of each play they make (sack, QB hit, tackle for loss,
    stop, pass defended, interception, forced fumble), each stat trusted according to how repeatable it is.
    Public data has no coverage or blocking grades, so corners and linemen are judged only on plays they make.
  * Kickers: field goals made above expected for the distance of each kick.

  python playerrate.py fit    # measure repeatability, fit weights, test -> player_params.json
  rankings(season) is what the dashboard shows.
Research and tests: research/players_eval.py.
"""
import json, os, sys
import numpy as np
import pandas as pd
import nflv
from lineup import FIX

HERE = os.path.dirname(os.path.abspath(__file__))
PARAMS = os.path.join(HERE, 'player_params.json')
FIRST = 1999
EDGE_RATIO = 0.2
GROUP = {'QB': 'QB', 'RB': 'RB', 'FB': 'RB', 'WR': 'WR', 'TE': 'TE', 'K': 'K', 'DE': 'EDGE', 'OLB': 'EDGE', 'EDGE': 'EDGE', 'DT': 'DL', 'NT': 'DL', 'DL': 'DL',
         'LB': 'LB', 'ILB': 'LB', 'MLB': 'LB', 'SAF': 'S', 'S': 'S', 'FS': 'S', 'SS': 'S', 'CB': 'CB', 'DB': 'CB'}
RAW = ['completions', 'attempts', 'passing_yards', 'passing_tds', 'passing_interceptions', 'sacks_suffered', 'passing_air_yards', 'passing_first_downs',
       'passing_epa', 'passing_cpoe', 'carries', 'rushing_yards', 'rushing_tds', 'rushing_first_downs', 'rushing_epa', 'receptions', 'targets',
       'receiving_yards', 'receiving_tds', 'receiving_air_yards', 'receiving_yards_after_catch', 'receiving_first_downs', 'receiving_epa', 'target_share',
       'air_yards_share', 'def_tackles_solo', 'def_tackles_with_assist', 'def_tackle_assists', 'def_tackles_for_loss', 'def_fumbles_forced', 'def_sacks',
       'def_qb_hits', 'def_interceptions', 'def_pass_defended', 'def_tds', 'fumbles_lost_total', 'rushing_10', 'rushing_20', 'receiving_20']
# measured from play-by-play 2014-2022: average expected points a play of this kind takes from the offense
EVENT_PTS = dict(sack=1.45, hit=0.2, tfl=1.0, stop=0.55, pd=0.75, int=3.5, ff=2.0)
# Per position: the efficiency stat (numerator, denominator), the workload unit, the stats that feed the fitted
# efficiency estimate, and the six card attributes (label, description, numerator, denominator, higher is better).
SPEC = {
    'QB': dict(eff=('epa', 'plays'), work='plays', min_work=100,
               comps={'epa': ('epa', 'plays'), 'cpoe': ('cpoe_sum', 'cpoe_n'), 'ypa': ('passing_yards', 'attempts'), 'sack': ('sacks_suffered', 'dropbacks'),
                      'int': ('passing_interceptions', 'attempts'), 'td': ('passing_tds', 'attempts'), 'fd': ('first_downs', 'plays'), 'rush': ('rushing_yards', 'games')},
               attrs=[('EFF', 'Expected points added per play', 'epa', 'plays', 1), ('ACC', 'Completion % over expected for the throws he attempts', 'cpoe_sum', 'cpoe_n', 1),
                      ('DEEP', 'Average depth of his throws (air yards per attempt)', 'passing_air_yards', 'air_att', 1), ('SAFE', 'Interceptions per attempt (fewer is better)', 'passing_interceptions', 'attempts', -1),
                      ('PKT', 'Sacks taken per dropback (fewer is better)', 'sacks_suffered', 'dropbacks', -1), ('RUN', 'Rushing yards per game', 'rushing_yards', 'games', 1)]),
    'RB': dict(eff=('rr_epa', 'opps'), work='opps', min_work=40,
               comps={'epa': ('rr_epa', 'opps'), 'ypc': ('rushing_yards', 'carries'), 'fd': ('first_downs', 'opps'), 'ypt': ('receiving_yards', 'targets'),
                      'td': ('tds', 'opps'), 'expl': ('rushing_10', 'carries'), 'tshare': ('target_share', 'games')},
               attrs=[('EFF', 'Expected points added per carry or target', 'rr_epa', 'opps', 1), ('YPC', 'Yards per carry', 'rushing_yards', 'carries', 1),
                      ('BIG', 'Share of carries that gain 10+ yards', 'rushing_10', 'carries', 1), ('WORK', 'Carries per game', 'carries', 'games', 1),
                      ('REC', 'Receiving yards per game', 'receiving_yards', 'games', 1), ('TD', 'Touchdowns per game', 'tds', 'games', 1)]),
    'WR': dict(eff=('receiving_epa', 'targets'), work='targets', min_work=20,
               comps={'epa': ('receiving_epa', 'targets'), 'ypt': ('receiving_yards', 'targets'), 'catch': ('receptions', 'targets'), 'fd': ('receiving_first_downs', 'targets'),
                      'td': ('receiving_tds', 'targets'), 'adot': ('receiving_air_yards', 'air_tgt'), 'tshare': ('target_share', 'games'), 'yac': ('receiving_yards_after_catch', 'receptions')},
               attrs=[('EFF', 'Expected points added per target', 'receiving_epa', 'targets', 1), ('TGT', 'Share of the team\'s targets', 'target_share', 'games', 1),
                      ('HNDS', 'Catch rate (receptions per target)', 'receptions', 'targets', 1), ('DEEP', 'Average depth of his targets (air yards per target)', 'receiving_air_yards', 'air_tgt', 1),
                      ('YAC', 'Yards after the catch per reception', 'receiving_yards_after_catch', 'receptions', 1), ('TD', 'Touchdowns per game', 'receiving_tds', 'games', 1)]),
    'EDGE': dict(events=['sack', 'hit', 'tfl', 'stop', 'ff', 'pd', 'int'], min_games=3,
                 attrs=[('SACK', 'Sacks per game', 'def_sacks', 'games', 1), ('PRES', 'QB hits that were not sacks, per game', 'hit', 'games', 1),
                        ('TFL', 'Tackles for loss on runs and catches, per game', 'tfl', 'games', 1), ('STOP', 'Tackles that ended a failed play for the offense, per game', 'stop', 'games', 1),
                        ('STRP', 'Forced fumbles per game', 'def_fumbles_forced', 'games', 1), ('TKL', 'Tackles per game', 'tackles', 'games', 1)]),
    'LB': dict(events=['sack', 'hit', 'tfl', 'stop', 'ff', 'pd', 'int'], min_games=3,
               attrs=[('STOP', 'Tackles that ended a failed play for the offense, per game', 'stop', 'games', 1), ('TKL', 'Tackles per game', 'tackles', 'games', 1),
                      ('TFL', 'Tackles for loss on runs and catches, per game', 'tfl', 'games', 1), ('RUSH', 'Sacks and QB hits per game', 'pressure', 'games', 1),
                      ('COV', 'Passes defended per game', 'pd', 'games', 1), ('BALL', 'Interceptions and forced fumbles per game', 'takeaway', 'games', 1)]),
    'CB': dict(events=['pd', 'int', 'stop', 'tfl', 'ff', 'sack', 'hit'], min_games=3,
               attrs=[('COV', 'Passes broken up (not intercepted) per game', 'pd', 'games', 1), ('BALL', 'Interceptions per game', 'def_interceptions', 'games', 1),
                      ('STOP', 'Tackles that ended a failed play for the offense, per game', 'stop', 'games', 1), ('TKL', 'Tackles per game', 'tackles', 'games', 1),
                      ('TFL', 'Tackles for loss per game', 'tfl', 'games', 1), ('STRP', 'Forced fumbles per game', 'def_fumbles_forced', 'games', 1)]),
    'K': dict(min_att=5,
              attrs=[('ACC', 'Field goals made above expected for their distance, per attempt', 'fgoe', 'fg_att', 1), ('LONG', 'Makes above expected on kicks of 50+ yards, per attempt', 'fgoe50', 'fg_att50', 1),
                     ('MID', 'Makes above expected on kicks of 40-49 yards, per attempt', 'fgoe40', 'fg_att40', 1), ('XP', 'Extra points made above expected, per attempt', 'xpoe', 'xp_att', 1),
                     ('VOL', 'Field-goal attempts per game', 'fg_att', 'games', 1), ('PCT', 'Field-goal %', 'fg_made', 'fg_att', 1)]),
}
SPEC['TE'] = dict(SPEC['WR'], min_work=15)
SPEC['DL'] = SPEC['EDGE']
SPEC['S'] = SPEC['CB']
EV_COL = dict(sack='def_sacks', hit='hit', tfl='tfl', stop='stop', pd='pd', int='def_interceptions', ff='def_fumbles_forced')
POSITIONS = ['QB', 'RB', 'WR', 'TE', 'EDGE', 'DL', 'LB', 'CB', 'S', 'K']


# ---------------------------------------------------------------- data
def extras(season):
    """Per player-week numbers the weekly stat file lacks, from play-by-play: stops, and kicks against expectation."""
    path = os.path.join(HERE, 'cache', f'px_{season}.pkl')
    src = nflv.get('pbp', season)
    if os.path.exists(path) and os.path.getmtime(path) >= os.path.getmtime(src):
        return pd.read_pickle(path)
    cols = ['season_type', 'week', 'play_type', 'epa', 'success', 'sack', 'tackled_for_loss', 'solo_tackle_1_player_id', 'solo_tackle_2_player_id',
            'assist_tackle_1_player_id', 'assist_tackle_2_player_id', 'tackle_with_assist_1_player_id', 'tackle_with_assist_2_player_id',
            'kicker_player_id', 'fg_prob', 'field_goal_result', 'kick_distance', 'extra_point_result', 'extra_point_prob']
    df = pd.read_csv(src, compression='gzip', usecols=lambda c: c in cols, low_memory=False)
    df = df[df.season_type == 'REG']
    st = df[df.play_type.isin(['pass', 'run']) & (df.success == 0) & (df.sack != 1) & (df.tackled_for_loss != 1)]
    parts = []
    for c, w in (('solo_tackle_1_player_id', 1.0), ('solo_tackle_2_player_id', 1.0), ('tackle_with_assist_1_player_id', 1.0), ('tackle_with_assist_2_player_id', 1.0),
                 ('assist_tackle_1_player_id', 0.5), ('assist_tackle_2_player_id', 0.5)):
        if c in st:
            x = st[st[c].notna()].groupby(['week', c]).size() * w
            x.index.names = ['week', 'player_id']
            parts.append(x.rename('stop'))
    stop = pd.concat(parts).groupby(level=[0, 1]).sum() if parts else pd.Series(dtype=float, name='stop')
    fg = df[(df.play_type == 'field_goal') & df.kicker_player_id.notna() & df.fg_prob.notna()].copy()
    fg['made'] = (fg.field_goal_result == 'made').astype(float)
    fg['oe'] = fg.made - fg.fg_prob
    d = fg.kick_distance.fillna(0)
    for nm, m in (('50', d >= 50), ('40', (d >= 40) & (d < 50))):
        fg['att' + nm] = m.astype(float); fg['oe' + nm] = fg.oe.where(m, 0.0)
    k = fg.groupby(['week', 'kicker_player_id']).agg(fg_att=('made', 'size'), fg_made=('made', 'sum'), fgoe=('oe', 'sum'), fg_att50=('att50', 'sum'),
                                                     fgoe50=('oe50', 'sum'), fg_att40=('att40', 'sum'), fgoe40=('oe40', 'sum'))
    xp = df[(df.play_type == 'extra_point') & df.kicker_player_id.notna() & df.extra_point_prob.notna()].copy()
    xp['oe'] = (xp.extra_point_result == 'good').astype(float) - xp.extra_point_prob
    x = xp.groupby(['week', 'kicker_player_id']).agg(xp_att=('oe', 'size'), xpoe=('oe', 'sum'))
    k = k.join(x, how='outer'); k.index.names = ['week', 'player_id']
    out = pd.concat([stop, k], axis=1).fillna(0.0).reset_index()
    out['season'] = season
    out.to_pickle(path)
    return out


def weekly(season):
    """One row per player per regular-season game, with the derived columns every rating uses."""
    df = pd.read_csv(nflv.get('player_week', season), low_memory=False,
                     usecols=lambda c: c in RAW + ['player_id', 'player_display_name', 'position', 'season', 'week', 'season_type', 'team'])
    df = df[df.season_type == 'REG'].copy()
    df['pos'] = df.position.map(GROUP)
    df = df[df.pos.notna()]
    for c in RAW:
        if c not in df:
            df[c] = 0.0
    df = df.merge(extras(season).drop(columns='season'), on=['week', 'player_id'], how='left')
    has_cpoe, has_tgt_air = df.passing_cpoe.notna(), df.receiving_air_yards.notna() & (season >= 2006)
    df[RAW] = df[RAW].fillna(0.0)
    x = pd.DataFrame(index=df.index)
    x['games'] = 1.0
    x['dropbacks'] = df.attempts + df.sacks_suffered
    x['plays'] = x.dropbacks + df.carries
    x['epa'] = df.passing_epa + df.rushing_epa
    x['cpoe_n'] = df.attempts.where(has_cpoe, 0.0); x['cpoe_sum'] = (df.passing_cpoe * df.attempts).where(has_cpoe, 0.0)
    x['air_att'] = df.attempts if season >= 2006 else 0.0
    x['air_tgt'] = df.targets.where(has_tgt_air, 0.0)
    x['opps'] = df.carries + df.targets
    x['rr_epa'] = df.rushing_epa + df.receiving_epa
    x['tds'] = df.rushing_tds + df.receiving_tds
    x['first_downs'] = df.passing_first_downs + df.rushing_first_downs + df.receiving_first_downs
    x['hit'] = (df.def_qb_hits - df.def_sacks).clip(lower=0)
    x['tfl'] = (df.def_tackles_for_loss - df.def_sacks).clip(lower=0)
    x['pd'] = (df.def_pass_defended - df.def_interceptions).clip(lower=0)
    x['tackles'] = df.def_tackles_solo + df.def_tackles_with_assist + 0.5 * df.def_tackle_assists
    x['pressure'] = df.def_sacks + x.hit
    x['takeaway'] = df.def_interceptions + df.def_fumbles_forced
    df = pd.concat([df.fillna({c: 0.0 for c in ('stop', 'fg_att', 'fg_made', 'fgoe', 'fg_att50', 'fgoe50', 'fg_att40', 'fgoe40', 'xp_att', 'xpoe')}), x], axis=1)
    df['team'] = df.team.map(lambda t: FIX.get(t, t))
    # The stat file labels 3-4 edge rushers and 4-3 off-ball linebackers both as LB/OLB. Sort them by what they do:
    # a linebacker with at least one sack or QB hit for every five tackles this season is rated as an edge rusher.
    lb = df.position.isin(['LB', 'OLB', 'ILB', 'MLB'])
    tot = df[lb].groupby('player_id')[['pressure', 'tackles']].sum()
    edge = set(tot.index[(tot.pressure >= EDGE_RATIO * tot.tackles) & (tot.pressure >= 2)])
    df.loc[lb, 'pos'] = np.where(df.loc[lb, 'player_id'].isin(edge), 'EDGE', 'LB')
    return df.drop(columns=['position', 'season_type'])


_ALL = {}


def all_weeks(first=FIRST, last=None):
    last = last or nflv.current_season()
    key = (first, last)
    if key not in _ALL:
        _ALL[key] = pd.concat([weekly(s) for s in range(first, last + 1)], ignore_index=True)
    return _ALL[key]


NUM = None


def totals(df, season, before_week=99, prior=0.5, from_week=1):
    """Per player: this season's sums for weeks [from_week, before_week) plus `prior` x last season's. Columns
    cur_<x> keep this season alone; name, pos and team are the latest."""
    global NUM
    if NUM is None:
        NUM = [c for c in df.columns if c not in ('player_id', 'player_display_name', 'season', 'week', 'team', 'pos')]
    cur = df[(df.season == season) & (df.week < before_week) & (df.week >= from_week)]
    a = cur.groupby('player_id')[NUM].sum()
    out = a.copy()
    for back, w in enumerate(prior if isinstance(prior, (list, tuple)) else [prior], 1):
        if w:
            b = df[df.season == season - back].groupby('player_id')[NUM].sum() * w
            out = out.add(b.reindex(out.index).fillna(0.0))
    last = cur.sort_values('week').groupby('player_id').last()
    out['name'], out['pos'], out['team'] = last.player_display_name, last.pos, last.team
    for c in ('games', 'plays', 'opps', 'targets', 'fg_att', 'carries', 'attempts'):
        out['cur_' + c] = a[c]
    return out


def rate(t, num, den, m, mu=None):
    """Stat per unit, pulled toward the group average `mu` with the weight of `m` units."""
    n, d = t[num].values.astype(float), t[den].values.astype(float)
    if mu is None:
        mu = n.sum() / d.sum() if d.sum() > 0 else 0.0
    return (n + m * mu) / (d + m), mu


def qualifies(t, pos):
    sp = SPEC[pos]
    if 'eff' in sp:
        return (t.pos == pos) & (t['cur_' + sp['work']] >= sp['min_work'])
    if pos == 'K':
        return (t.pos == pos) & (t.cur_fg_att >= sp['min_att'])
    return (t.pos == pos) & (t.cur_games >= sp['min_games'])


def zs(v, w=None):
    v = np.asarray(v, float)
    m = np.average(v, weights=w) if w is not None else v.mean()
    sd = np.sqrt(np.average((v - m) ** 2, weights=w)) if w is not None else v.std()
    return (v - m) / (sd if sd > 1e-12 else 1.0)


def estimate(t, pos, pp, all_pos=None):
    """Adds to the position's qualifying players: eff (efficiency estimate), work (per game), value (points per game
    above a replacement-level player), and a_<LABEL> attribute z-scores. pp: fitted parameters for the position."""
    sp = SPEC[pos]
    pool = t[t.pos == pos] if all_pos is None else all_pos
    q = t[qualifies(t, pos)].copy()
    if not len(q):
        return q.assign(eff=[], work=[], value=[])
    m = pp['m']
    if 'eff' in sp:
        X = []
        for c, (num, den) in sp['comps'].items():
            mu = pool[num].sum() / pool[den].sum() if pool[den].sum() > 0 else 0.0
            r, _ = rate(q, num, den, m[c], mu)
            sd = pp['sd'][c]
            X.append((r - mu) / sd if sd else np.zeros(len(q)))
        X = np.column_stack(X)
        q['eff'] = pp['eff_mean'] + X @ np.array([pp['w'][c] for c in sp['comps']])
        w = sp['work']
        q['work'] = (q[w] + pp['work_m'] * pp['work_mu']) / (q['games'] + pp['work_m']) if pos != 'QB' else pp['work_mu']
        q['value'] = (q.eff - pp['repl']) * q.work                       # repl: what bench players at the position produce per play
    elif pos == 'K':
        mu = 0.0
        r, _ = rate(q, 'fgoe', 'fg_att', m['fgoe'], mu)
        x, _ = rate(q, 'xpoe', 'xp_att', m['xpoe'], mu)
        q['eff'] = r
        q['work'] = pp['work_mu']
        q['value'] = 3.0 * r * pp['work_mu'] + 1.0 * x * pp['xp_mu']
    else:
        val = np.zeros(len(q))
        for e in sp['events']:
            col = EV_COL[e]
            mu = pool[col].sum() / pool['games'].sum() if pool['games'].sum() > 0 else 0.0
            r, _ = rate(q, col, 'games', m[e], mu)
            val += EVENT_PTS[e] * (r - mu)
        q['eff'] = val
        q['work'] = 1.0
        q['value'] = val
    for label, _, num, den, sign in sp['attrs']:
        mm = pp['attr_m'].get(label, 8.0)
        mu = pool[num].sum() / pool[den].sum() if pool[den].sum() > 0 else 0.0
        r, _ = rate(q, num, den, mm, mu)
        q['v_' + label] = n_over_d(q, num, den)
        q['a_' + label] = sign * zs(r) if q[den].sum() > 0 else 0.0
    return q


def n_over_d(q, num, den):
    return np.where(q[den].values > 0, q[num].values / np.maximum(q[den].values, 1e-9), np.nan)


def ovr(z):
    """Overall on the familiar 45-99 scale: an average qualifying player is about 72, one standard deviation is 9."""
    return int(round(float(np.clip(72 + 9 * z, 45, 99))))


def attr_score(z):
    return int(round(float(np.clip(70 + 11 * z, 40, 99))))


def load():
    return json.load(open(PARAMS)) if os.path.exists(PARAMS) else None


SHOW = {
    'QB': lambda r: dict(cmp=f"{r.completions:.0f}/{r.attempts:.0f}", pass_yds=round(r.passing_yards), td=round(r.passing_tds), int=round(r.passing_interceptions),
                         epa_play=round(r.epa / max(r.plays, 1), 3), cpoe=round(r.cpoe_sum / r.cpoe_n, 1) if r.cpoe_n else None, rush_yds=round(r.rushing_yards)),
    'RB': lambda r: dict(car=round(r.carries), rush_yds=round(r.rushing_yards), ypc=round(r.rushing_yards / max(r.carries, 1), 2), rec=round(r.receptions),
                         rec_yds=round(r.receiving_yards), td=round(r.tds)),
    'WR': lambda r: dict(tgt=round(r.targets), rec=round(r.receptions), yds=round(r.receiving_yards), td=round(r.receiving_tds),
                         yds_g=round(r.receiving_yards / max(r.games, 1), 1), epa_tgt=round(r.receiving_epa / max(r.targets, 1), 2),
                         adot=round(r.receiving_air_yards / r.air_tgt, 1) if r.air_tgt else None),
    'EDGE': lambda r: dict(sacks=r.def_sacks, qb_hits=round(r.def_qb_hits), tfl=round(r.def_tackles_for_loss), ff=round(r.def_fumbles_forced),
                           tackles=round(r.def_tackles_solo + r.def_tackles_with_assist + r.def_tackle_assists), stops=round(r.stop, 1)),
    'LB': lambda r: dict(tackles=round(r.def_tackles_solo + r.def_tackles_with_assist + r.def_tackle_assists), stops=round(r.stop, 1), tfl=round(r.def_tackles_for_loss),
                         sacks=r.def_sacks, int=round(r.def_interceptions), pd=round(r.def_pass_defended)),
    'CB': lambda r: dict(int=round(r.def_interceptions), pd=round(r.def_pass_defended), tackles=round(r.def_tackles_solo + r.def_tackles_with_assist + r.def_tackle_assists),
                         stops=round(r.stop, 1), td=round(r.def_tds)),
    'K': lambda r: dict(fg=f"{r.fg_made:.0f}/{r.fg_att:.0f}", fg_pct=round(100 * r.fg_made / max(r.fg_att, 1), 1), fg_over_exp=round(r.fgoe, 1),
                        fifty_att=round(r.fg_att50), pat_over_exp=round(r.xpoe, 1)),
}
SHOW['TE'], SHOW['DL'], SHOW['S'] = SHOW['WR'], SHOW['EDGE'], SHOW['CB']


def rankings(season, before_week=99, params=None, df=None):
    """{position: [dict(rank, name, team, games, rating, ovr, value, attrs, traits, stats...)]} best first.
    rating is 0-100 for older parts of the page; ovr, attrs and traits are what the cards show."""
    import traits
    params = params or load()
    df = df if df is not None else all_weeks(max(FIRST, season - 2), season)
    cur = totals(df, season, before_week, 0.0)
    out, memo = {}, {}
    for pos in POSITIONS:
        pr_ = params['pos'][pos]['prior']
        key = json.dumps(pr_)
        if key not in memo:
            memo[key] = totals(df, season, before_week, pr_)
        q = estimate(memo[key], pos, params['pos'][pos])
        if len(q) < 3:
            continue
        z = (q.value - q.value.mean()) / (params['pos'][pos]['value_sd'] or q.value.std() or 1.0)
        q = q.assign(z=z).sort_values('value', ascending=False)
        lo, hi = q.value.min(), q.value.max()
        rows = []
        for i, (pid, r) in enumerate(q.iterrows(), 1):
            c = cur.loc[pid]
            attrs = [[label, desc, attr_score(r['a_' + label])] for label, desc, *_ in SPEC[pos]['attrs']]
            rows.append(dict(rank=i, id=pid, name=r['name'], team=r.team, games=int(c.games), rating=int(round(100 * (r.value - lo) / (hi - lo))) if hi > lo else 50,
                             ovr=ovr(r.z), value=round(float(r.value), 2), attrs=attrs,
                             traits=traits.player(pos, {label: float(r['a_' + label]) for label, *_ in SPEC[pos]['attrs']}, params), **SHOW[pos](c)))
        out[pos] = rows
    return out


if __name__ == '__main__':
    if sys.argv[1:] == ['fit']:
        sys.path.insert(0, os.path.join(HERE, 'research'))
        import players_eval
        players_eval.main()
    else:
        print(__doc__)
