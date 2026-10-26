"""Player power rankings by position and next-game stat projections, from nflverse weekly player stats.

Rankings: a per-position score built from efficiency and volume, as a 0-100 rating (100 = best at
the position this season). Defensive and kicking scores use box-score production, since public
data has no coverage or blocking grades.

Projections: each stat is a recency-weighted average of this season's games (each older game
counts 15% less) blended with last season's per-game rate (worth 2 games), then scaled by how
much the opponent allows to that position versus the league. tune() picks each
method's per-stat settings on 2023-2024 and reports them frozen on 2025 against a season-to-date average.
"""
import csv, json, math
from collections import defaultdict
import nflv

EDGE = {'DE', 'OLB', 'EDGE'}
DL = {'DT', 'NT', 'DL'}
LB = {'LB', 'ILB', 'MLB'}
S = {'SAF', 'S', 'FS', 'SS'}
CB = {'CB', 'DB'}
DECAY, PRIOR_GAMES, OPP_WEIGHT = 0.85, 2.0, 0.5
PROJ = {'QB': ['passing_yards', 'passing_tds', 'passing_interceptions', 'rushing_yards'],
        'RB': ['rushing_yards', 'rushing_tds', 'receptions', 'receiving_yards'],
        'WR': ['receptions', 'receiving_yards', 'receiving_tds'],
        'TE': ['receptions', 'receiving_yards', 'receiving_tds']}


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def group(pos):
    if pos in ('QB', 'RB', 'WR', 'TE', 'K'):
        return pos
    if pos == 'FB':
        return 'RB'
    for name, members in (('EDGE', EDGE), ('DL', DL), ('LB', LB), ('S', S), ('CB', CB)):
        if pos in members:
            return name
    return None


def load(season):
    return [r for r in csv.DictReader(open(nflv.get('player_week', season), encoding='utf-8')) if r['season_type'] == 'REG']


def totals(rows):
    """{player_id: dict(name, pos, team, games, sums...)} for a season so far."""
    out = {}
    for r in rows:
        g = group(r['position'])
        if not g:
            continue
        p = out.setdefault(r['player_id'], dict(id=r['player_id'], name=r['player_display_name'], pos=g, team=r['team'],
                                               games=0, s=defaultdict(float), weeks=[]))
        p['games'] += 1
        p['team'] = r['team']
        p['weeks'].append(int(r['week']))
        for k, v in r.items():
            if k not in ('player_id', 'player_name', 'player_display_name', 'position', 'position_group', 'headshot_url',
                         'season', 'week', 'season_type', 'game_id', 'team', 'opponent_team') and v not in ('', 'NA'):
                p['s'][k] += f(v)
    return out


def _z(values):
    m = sum(values) / len(values)
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / len(values)) or 1.0
    return [(v - m) / sd for v in values]


def rankings(season):
    """{position: [dict(rank, name, team, rating, stats...)]} ranked best first."""
    P = totals(load(season))
    spec = {
        'QB': (lambda p: p['s']['attempts'] + p['s']['sacks_suffered'] + p['s']['carries'] >= 100,
               [(lambda p: (p['s']['passing_epa'] + p['s']['rushing_epa']) / (p['s']['attempts'] + p['s']['sacks_suffered'] + p['s']['carries']), 0.6),
                (lambda p: p['s']['passing_cpoe'] / p['games'], 0.2),
                (lambda p: (p['s']['passing_yards'] + p['s']['rushing_yards']) / p['games'], 0.2)],
               lambda p: dict(cmp=f"{p['s']['completions']:.0f}/{p['s']['attempts']:.0f}", pass_yds=round(p['s']['passing_yards']),
                              td=round(p['s']['passing_tds']), int=round(p['s']['passing_interceptions']),
                              epa_play=round((p['s']['passing_epa'] + p['s']['rushing_epa']) / (p['s']['attempts'] + p['s']['sacks_suffered'] + p['s']['carries']), 3),
                              cpoe=round(p['s']['passing_cpoe'] / p['games'], 1), rush_yds=round(p['s']['rushing_yards']))),
        'RB': (lambda p: p['s']['carries'] + p['s']['targets'] >= 40,
               [(lambda p: (p['s']['rushing_epa'] + p['s']['receiving_epa']) / (p['s']['carries'] + p['s']['targets']), 0.45),
                (lambda p: (p['s']['rushing_yards'] + p['s']['receiving_yards']) / p['games'], 0.4),
                (lambda p: (p['s']['rushing_tds'] + p['s']['receiving_tds']) / p['games'], 0.15)],
               lambda p: dict(car=round(p['s']['carries']), rush_yds=round(p['s']['rushing_yds'] if 'rushing_yds' in p['s'] else p['s']['rushing_yards']),
                              ypc=round(p['s']['rushing_yards'] / max(p['s']['carries'], 1), 2), rec=round(p['s']['receptions']),
                              rec_yds=round(p['s']['receiving_yards']), td=round(p['s']['rushing_tds'] + p['s']['receiving_tds']))),
        'WR': (lambda p: p['s']['targets'] >= 20,
               [(lambda p: p['s']['receiving_epa'] / p['s']['targets'], 0.4),
                (lambda p: p['s']['receiving_yards'] / p['games'], 0.4),
                (lambda p: p['s']['target_share'] / p['games'], 0.2)],
               lambda p: dict(tgt=round(p['s']['targets']), rec=round(p['s']['receptions']), yds=round(p['s']['receiving_yards']),
                              td=round(p['s']['receiving_tds']), yds_g=round(p['s']['receiving_yards'] / p['games'], 1),
                              epa_tgt=round(p['s']['receiving_epa'] / p['s']['targets'], 2))),
        'EDGE': (lambda p: p['games'] >= 3,
                 [(lambda p: (p['s']['def_sacks'] + 0.5 * p['s']['def_qb_hits'] + 0.5 * p['s']['def_tackles_for_loss']) / p['games'], 0.8),
                  (lambda p: p['s']['def_fumbles_forced'] / p['games'], 0.2)],
                 lambda p: dict(sacks=p['s']['def_sacks'], qb_hits=round(p['s']['def_qb_hits']), tfl=round(p['s']['def_tackles_for_loss']),
                                ff=round(p['s']['def_fumbles_forced']), tackles=round(p['s']['def_tackles_solo'] + p['s']['def_tackle_assists']))),
        'LB': (lambda p: p['games'] >= 3,
               [(lambda p: (p['s']['def_tackles_solo'] + 0.5 * p['s']['def_tackle_assists']) / p['games'], 0.4),
                (lambda p: (p['s']['def_tackles_for_loss'] + p['s']['def_sacks'] + 0.5 * p['s']['def_qb_hits']) / p['games'], 0.35),
                (lambda p: (2 * p['s']['def_interceptions'] + p['s']['def_pass_defended'] + 2 * p['s']['def_fumbles_forced']) / p['games'], 0.25)],
               lambda p: dict(tackles=round(p['s']['def_tackles_solo'] + p['s']['def_tackle_assists']), tfl=round(p['s']['def_tackles_for_loss']),
                              sacks=p['s']['def_sacks'], int=round(p['s']['def_interceptions']), pd=round(p['s']['def_pass_defended']))),
        'CB': (lambda p: p['games'] >= 3,
               [(lambda p: (3 * p['s']['def_interceptions'] + p['s']['def_pass_defended']) / p['games'], 0.75),
                (lambda p: p['s']['def_tackles_solo'] / p['games'], 0.25)],
               lambda p: dict(int=round(p['s']['def_interceptions']), pd=round(p['s']['def_pass_defended']),
                              tackles=round(p['s']['def_tackles_solo'] + p['s']['def_tackle_assists']), td=round(p['s']['def_tds']))),
        'K': (lambda p: p['s']['fg_att'] >= 5,
              [(lambda p: (p['s']['fg_made'] + 1) / (p['s']['fg_att'] + 2), 0.6),
               (lambda p: p['s']['fg_made'] / p['games'], 0.2),
               (lambda p: p['s']['fg_made_50_59'] + p['s']['fg_made_60_'], 0.2)],
              lambda p: dict(fg=f"{p['s']['fg_made']:.0f}/{p['s']['fg_att']:.0f}", fg_pct=round(100 * p['s']['fg_made'] / p['s']['fg_att'], 1),
                             long=round(max(p['s'].get('fg_long', 0), 0)), fifty_plus=round(p['s']['fg_made_50_59'] + p['s']['fg_made_60_']),
                             pat=f"{p['s']['pat_made']:.0f}/{p['s']['pat_att']:.0f}")),
    }
    spec['TE'] = (lambda p: p['s']['targets'] >= 15, spec['WR'][1], spec['WR'][2])
    spec['DL'] = spec['EDGE']
    spec['S'] = (lambda p: p['games'] >= 3,
                 [(lambda p: (3 * p['s']['def_interceptions'] + p['s']['def_pass_defended']) / p['games'], 0.6),
                  (lambda p: (p['s']['def_tackles_solo'] + 0.5 * p['s']['def_tackle_assists']) / p['games'], 0.4)],
                 spec['CB'][2])
    out = {}
    for pos, (eligible, parts, show) in spec.items():
        pool = [p for p in P.values() if p['pos'] == pos and eligible(p)]
        if len(pool) < 3:
            continue
        score = [0.0] * len(pool)
        for fn, wt in parts:
            for i, z in enumerate(_z([fn(p) for p in pool])):
                score[i] += wt * z
        lo, hi = min(score), max(score)
        ranked = sorted(zip(score, pool), key=lambda x: -x[0])
        out[pos] = [dict(rank=i, name=p['name'], team=p['team'], games=p['games'],
                         rating=round(100 * (s - lo) / (hi - lo)) if hi > lo else 50, **show(p))
                    for i, (s, p) in enumerate(ranked, 1)]
    return out


def _params():
    """Per-stat (decay, prior games, opponent weight) tuned by tune(); defaults if not tuned yet."""
    try:
        return {k: tuple(v) for k, v in json.load(open('projection_params.json'))['params'].items()}
    except (OSError, ValueError, KeyError):
        return {}


def project_from(history, prior, opp_factor, pos, params=None):
    """{stat: projection} for one player from his games this season (newest first) and last season's averages."""
    params = _params() if params is None else params
    out = {}
    for k in PROJ[pos]:
        decay, prior_games, opp_w = params.get(f"{pos} {k}", (DECAY, PRIOR_GAMES, OPP_WEIGHT))
        w = [decay ** i for i in range(len(history))]
        num, den = sum(wi * f(g[k]) for wi, g in zip(w, history)), sum(w)
        if prior is not None:
            num += prior_games * prior.get(k, 0.0); den += prior_games
        base = num / den if den else 0.0
        out[k] = round(base * (1 + opp_w * (opp_factor.get(k, 1.0) - 1)), 2)
    return out


def opponent_factors(rows, before_week=99):
    """{(defense team, position): {stat: allowed per game / league average}}."""
    allowed = defaultdict(lambda: defaultdict(float)); games = defaultdict(set)
    for r in rows:
        if int(r['week']) >= before_week:
            continue
        pos = group(r['position'])
        if pos not in PROJ:
            continue
        games[r['opponent_team']].add(r['week'])
        for k in PROJ[pos]:
            allowed[(r['opponent_team'], pos)][k] += f(r[k])
    league = defaultdict(lambda: defaultdict(list))
    per = {}
    for (team, pos), d in allowed.items():
        n = max(len(games[team]), 1)
        per[(team, pos)] = {k: v / n for k, v in d.items()}
        for k, v in per[(team, pos)].items():
            league[pos][k].append(v)
    return {key: {k: (v / (sum(league[key[1]][k]) / len(league[key[1]][k])) if sum(league[key[1]][k]) else 1.0)
                  for k, v in d.items()} for key, d in per.items()}


def projections(season, upcoming, out_players=frozenset()):
    """Next-game projections for every regular at QB/RB/WR/TE whose team plays in `upcoming`
    ({team: (opponent, week, day)}). Players in out_players (team, name) are listed as out."""
    rows = load(season)
    prev = totals(load(season - 1)) if season - 1 >= 2022 else {}
    hist = defaultdict(list)
    for r in sorted(rows, key=lambda r: -int(r['week'])):
        if group(r['position']) in PROJ:
            hist[r['player_id']].append(r)
    opp = opponent_factors(rows)
    out = []
    for pid, games in hist.items():
        latest = games[0]
        team, pos = latest['team'], group(latest['position'])
        if team not in upcoming or len(games) < 2:
            continue
        opp_team, week, day = upcoming[team]
        prior = None
        if pid in prev and prev[pid]['games'] >= 4:
            prior = {k: prev[pid]['s'][k] / prev[pid]['games'] for k in PROJ[pos]}
        proj = project_from(games, prior, opp.get((opp_team, pos), {}), pos)
        volume = proj.get('passing_yards', 0) + proj.get('rushing_yards', 0) + proj.get('receiving_yards', 0)
        if (pos == 'QB' and proj['passing_yards'] < 120) or volume < 15:
            continue
        out.append(dict(name=latest['player_display_name'], pos=pos, team=team, opp=opp_team, week=week, day=day,
                        out=(team, latest['player_display_name']) in out_players, **proj))
    return sorted(out, key=lambda p: (p['pos'], -(p.get('passing_yards', 0) + p.get('rushing_yards', 0) + p.get('receiving_yards', 0))))


def instances(season, first_week=4):
    """Walk-forward cases: (pos, history newest first, prior, opponent factors, actual row)."""
    rows = load(season)
    prev = totals(load(season - 1)) if season - 1 >= 2022 else {}
    by_week = defaultdict(list)
    for r in rows:
        by_week[int(r['week'])].append(r)
    cases = []
    for w in sorted(by_week):
        if w < first_week:
            continue
        before = [r for r in rows if int(r['week']) < w]
        hist = defaultdict(list)
        for r in sorted(before, key=lambda r: -int(r['week'])):
            hist[r['player_id']].append(r)
        opp = opponent_factors(before)
        for r in by_week[w]:
            pos = group(r['position'])
            h = hist.get(r['player_id'], [])
            if pos not in PROJ or len(h) < 2:
                continue
            prior = None
            if r['player_id'] in prev and prev[r['player_id']]['games'] >= 4:
                prior = {k: prev[r['player_id']]['s'][k] / prev[r['player_id']]['games'] for k in PROJ[pos]}
            cases.append((pos, h, prior, opp.get((r['opponent_team'], pos), {}), r))
    return cases


def evaluate(cases, params):
    """{'POS stat': (model MAE, season-average MAE, n)}."""
    err = defaultdict(lambda: [0.0, 0.0, 0])
    for pos, h, prior, opp, r in cases:
        proj = project_from(h, prior, opp, pos, params)
        for k in PROJ[pos]:
            e = err[f"{pos} {k}"]
            e[0] += abs(proj[k] - f(r[k])); e[1] += abs(sum(f(g[k]) for g in h) / len(h) - f(r[k])); e[2] += 1
    return {k: (a / n, b / n, n) for k, (a, b, n) in err.items() if n}


def tune(train=(2023, 2024), test=2025):
    """Pick each stat's (decay, prior games, opponent weight) on `train`, report it frozen on `test`."""
    import datetime, itertools
    tr, te = [c for y in train for c in instances(y)], instances(test)
    grid = list(itertools.product((0.7, 0.85, 1.0), (0.0, 2.0, 5.0), (0.0, 0.5, 1.0)))
    best = {}
    for combo in grid:
        res = evaluate(tr, {k: combo for k in (f"{p} {s}" for p in PROJ for s in PROJ[p])})
        for k, (mae, _, _) in res.items():
            if k not in best or mae < best[k][0]:
                best[k] = (mae, combo)
    params = {k: list(c) for k, (_, c) in best.items()}
    res = evaluate(te, {k: tuple(v) for k, v in params.items()})
    report = {k: dict(n=n, model_mae=round(a, 2), season_avg_mae=round(b, 2), better=a < b) for k, (a, b, n) in res.items()}
    json.dump(dict(made=datetime.date.today().isoformat(), tuned_on=list(train), tested_on=test, params=params, test=report),
              open('projection_params.json', 'w'), indent=1)
    return report


if __name__ == '__main__':
    print(json.dumps(tune(), indent=1))
