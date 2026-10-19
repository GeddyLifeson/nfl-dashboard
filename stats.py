"""Season stat tables from FootballDB (stats/*.csv), shown next to each team for context.
They do not feed the rating: walk-forward tests found box score stats added nothing beyond game results."""
import collections, csv, datetime, hashlib, json, os
from teams import ABBRS, BY_NAME, BY_PFR

D = 'stats/'


def _num(x):
    return float(str(x).replace(',', ''))


def _table(name):
    rows = {}
    for r in csv.DictReader(open(D + name + '.csv')):
        key = r['Team']
        if key in BY_NAME:
            rows[BY_NAME[key]] = r
    missing = set(ABBRS) - set(rows)
    if missing:
        raise SystemExit(f'{name}.csv is missing {sorted(missing)}')
    return rows


TABLES = ['tot_off', 'tot_def', 'pass_off', 'pass_def', 'rush_off', 'rush_def', 'downs_off', 'downs_def',
          'score_off', 'score_def', 'to', 'pen']
T = {n: _table(n) for n in TABLES}


def _g(n, t):
    return _num(T[n][t]['Gms']) if 'Gms' in T[n][t] else _num(T['tot_off'][t]['Gms'])


def _plays(side, t):
    p, r = T['pass_' + side][t], T['rush_' + side][t]
    return (_num(p['Att']) + _num(p['Sack'])) / _num(p['Gms']) + _num(r['Att']) / _num(r['Gms'])


def _pg(n, col):
    return lambda t: _num(T[n][t][col]) / _g(n, t)


def _col(n, col):
    return lambda t: _num(T[n][t][col])


def _side(side, opp):
    s = '_' + side
    return [
        ('points-per-game', 'Points allowed per game' if opp else 'Points per game', _col('tot' + s, 'Pts/G'), ['tot' + s]),
        ('yards-per-game', 'Yards allowed per game' if opp else 'Yards per game', _col('tot' + s, 'Yds/G'), ['tot' + s]),
        ('points-per-play', 'Points allowed per play' if opp else 'Points per play',
         lambda t: _num(T['tot' + s][t]['Pts/G']) / _plays(side, t), ['tot' + s, 'pass' + s, 'rush' + s]),
        ('touchdowns-per-game', 'TDs allowed per game' if opp else 'Touchdowns per game', _pg('score' + s, 'Tot'), ['score' + s, 'tot_off']),
        ('rush-attempts-per-game', 'Opponent rushes per game' if opp else 'Rushes per game', _pg('rush' + s, 'Att'), ['rush' + s]),
        ('rush-yards-per-game', 'Rush yards allowed per game' if opp else 'Rush yards per game', _col('tot' + s, 'RYds/G'), ['tot' + s]),
        ('rush-touchdowns-per-game', 'Rush TDs allowed per game' if opp else 'Rush TDs per game', _pg('rush' + s, 'TD'), ['rush' + s]),
        ('pass-attempts-per-game', 'Opponent pass attempts per game' if opp else 'Pass attempts per game', _pg('pass' + s, 'Att'), ['pass' + s]),
        ('net-pass-yards-per-game', 'Net pass yards allowed per game' if opp else 'Net pass yards per game', _col('tot' + s, 'PYds/G'), ['tot' + s]),
        ('third-down-pct', 'Opponent 3rd-down %' if opp else '3rd-down conversion %', _col('downs' + s, 'P3'), ['downs' + s]),
        ('fourth-down-pct', 'Opponent 4th-down %' if opp else '4th-down conversion %', _col('downs' + s, 'P4'), ['downs' + s]),
        ('first-downs-per-game', 'First downs allowed per game' if opp else 'First downs per game', _pg('downs' + s, 'Tot'), ['downs' + s]),
    ]


# (slug, label, side, value function, source tables, higher is better)
STATS = [(sl, lab, 'O', fn, src, True) for sl, lab, fn, src in _side('off', False)] + [
    ('giveaways-per-game', 'Giveaways per game', 'O', _pg('to', 'GvTot'), ['to'], False),
    ('penalties-per-game', 'Penalties per game', 'O', lambda t: _num(T['pen'][t]['Num']) / _g('tot_off', t), ['pen', 'tot_off'], False),
    ('penalty-yards-per-game', 'Penalty yards per game', 'O', lambda t: _num(T['pen'][t]['Yds']) / _g('tot_off', t), ['pen', 'tot_off'], False),
] + [('opp-' + sl, lab, 'D', fn, src, False) for sl, lab, fn, src in _side('def', True)] + [
    ('takeaways-per-game', 'Takeaways per game', 'D', _pg('to', 'TkTot'), ['to'], True),
    ('sacks-per-game', 'Sacks per game (defense)', 'D', _pg('pass_def', 'Sack'), ['pass_def'], True),
    ('takeaway-interceptions-per-game', 'Interceptions per game', 'D', _pg('to', 'TkInt'), ['to'], True),
    ('turnover-margin-per-game', 'Turnover margin per game', 'T', lambda t: _num(T['to'][t]['Diff']) / _g('to', t), ['to'], True),
]


def behind(played):
    """Per table, the teams whose row counts fewer games than they have played."""
    out = {}
    for n in TABLES:
        if 'Gms' in next(iter(T[n].values())):
            out[n] = {t for t in ABBRS if _num(T[n][t]['Gms']) < played[t]}
    return out


def log_tables(played, path='data/ledger.jsonl'):
    """Append one ledger record per stat table the first time its contents are seen."""
    seen = set()
    if os.path.exists(path):
        seen = {json.loads(l).get('sha256') for l in open(path)}
    late = behind(played)
    with open(path, 'a') as f:
        for n in TABLES + ['qb', 'honors_2025']:
            sha = hashlib.sha256(open(D + n + '.csv', 'rb').read()).hexdigest()[:16]
            if sha in seen:
                continue
            f.write(json.dumps(dict(table=n, source='footballdb' if n != 'honors_2025' else 'pro-football-reference',
                                    logged=datetime.date.today().isoformat(), sha256=sha,
                                    teams_behind=sorted(late.get(n, ())))) + '\n')
    return late


def team_stats(played):
    late = behind(played)
    vals = {sl: {t: fn(t) for t in ABBRS} for sl, _, _, fn, _, _ in STATS}
    out = {t: [] for t in ABBRS}
    for sl, lab, side, _, src, hi in STATS:
        v = vals[sl]
        for t in ABBRS:
            rank = 1 + sum(1 for x in v.values() if (x > v[t] if hi else x < v[t]))
            fmt = f'{v[t]:.3f}' if 'per-play' in sl else f'{v[t]:.1f}'
            out[t].append(dict(slug=sl, label=lab, side=side, unit='%' if sl.endswith('pct') else '', value=fmt, rank=rank,
                               behind=any(t in late.get(n, ()) for n in src)))
    return out


def players():
    """Lead QB (most attempts), attempt-weighted QB numbers, and 2025 All-Pro/Pro Bowl honors points."""
    byteam = collections.defaultdict(list)
    for r in csv.DictReader(open(D + 'qb.csv')):
        byteam[r['Team']].append(r)
    pts = {'1': 3, '2': 2, '3': 1}
    honors = collections.Counter()
    for r in csv.DictReader(open(D + 'honors_2025.csv')):
        honors[BY_PFR[r['Team']]] += pts[r['Level']]
    out = {}
    for t in ABBRS:
        rows = byteam[t]
        att = sum(_num(r['Att']) for r in rows)
        q = max(rows, key=lambda r: _num(r['Att']))
        out[t] = dict(qb1=q['Player'], qb1rate=_num(q['Rate']), qb1ypa=_num(q['YPA']),
                      qbrate=round(sum(_num(r['Rate']) * _num(r['Att']) for r in rows) / att, 1),
                      qbsack=round(100 * sum(_num(r['Sack']) for r in rows) / (att + sum(_num(r['Sack']) for r in rows)), 1),
                      hon=honors[t])
    return out
