"""Run: python test_ratings.py   (checks for the team ratings, player ratings, tags and all-time lists)"""
import json, os, re
import numpy as np
import pandas as pd
import playerrate as pr, teamrate, traits
from teams import ABBRS, IDX

HERE = os.path.dirname(os.path.abspath(__file__))


def test_no_book_inputs():
    """The rating code must never read a betting column or the win-total file."""
    banned = re.compile(r"spread|moneyline|total_line|_odds|wt_ratings|nfelosrs|WT_URL|vegas", re.I)
    for f in ('teamrate.py', 'playerrate.py', 'traits.py', 'alltime.py'):
        code = '\n'.join(l.split('#')[0] for l in open(os.path.join(HERE, f)) if not l.strip().startswith(('"""', '#')))
        code = re.sub(r'""".*?"""', '', code, flags=re.S)
        hit = banned.findall(code)
        assert not hit, (f, hit)
    assert 'pass_oe' in teamrate.COLS and not any(banned.search(c) for c in teamrate.COLS)
    sp = teamrate.load()
    assert sp['book_inputs'] == 'none' and not any(banned.search(k) for k in sp['coef'])


def test_solve_recovers_known_teams():
    """Two made-up buckets where KC's offense is +5 and BUF's defense allows +3: the solver must find them."""
    rng = np.random.default_rng(0); rows = []
    for g in range(400):
        a, b = rng.choice(len(ABBRS), 2, replace=False)
        for o, d, h in ((a, b, 1.0), (b, a, -1.0)):
            v = 20 + 1.0 * h + (5 if ABBRS[o] == 'KC' else 0) + (3 if ABBRS[d] == 'BUF' else 0)
            rows.append(dict(team=ABBRS[o], opp=ABBRS[d], home=h, x=v, y=2 * v))
    tg = pd.DataFrame(rows)
    o, d = teamrate.solve(tg, np.ones(len(tg)), k=1e-6, cats=['x', 'y'])
    assert abs(o[IDX['KC'], 0] - o[:, 0].mean() - 5 * 31 / 32) < 0.05, o[IDX['KC']]
    assert abs(d[IDX['BUF'], 0] - d[:, 0].mean() - 3 * 31 / 32) < 0.05
    assert np.allclose(o[:, 1], 2 * o[:, 0], atol=1e-6), 'buckets are solved independently and linearly'
    # with no games, a team is exactly its prior
    po = np.zeros((len(ABBRS), 1)); po[IDX['KC']] = 4.0
    o, _ = teamrate.solve(tg.iloc[:0], [], po, np.zeros_like(po), k=9, cats=['x'])
    assert abs(o[IDX['KC'], 0] - 4.0) < 1e-9


def test_team_points_add_up():
    sp = teamrate.load(); rng = np.random.default_rng(1)
    feats = {t: dict({p + c: float(rng.normal()) for c in teamrate.CATS for p in ('o_', 'd_')}) for t in ABBRS}
    qb = {t: float(rng.normal()) for t in ABBRS}
    pts = teamrate.team_points(feats, sp, qb)
    assert abs(sum(v['rating'] for v in pts.values())) < 1e-9, 'ratings average zero'
    for t, v in pts.items():
        assert abs(sum(v['parts'].values()) - v['rating']) < 1e-9, 'parts add up to the rating'
    # a better quarterback can only raise the rating, by the fitted weight
    qb2 = dict(qb); qb2['KC'] += 32.0 / 31
    assert abs(teamrate.team_points(feats, sp, qb2)['KC']['rating'] - pts['KC']['rating'] - sp['coef']['qb']) < 1e-6
    assert sp['coef']['qb'] > 0 and 1.0 < sp['hfa'] < 3.5 and 12 < sp['sigma'] < 15
    assert teamrate.ovr(0) == 78 and teamrate.ovr(50) == 99 and teamrate.ovr(-50) == 60


def test_shrinkage_and_scales():
    t = pd.DataFrame(dict(n=[10.0, 100.0], d=[10.0, 1000.0]))
    r, mu = pr.rate(t, 'n', 'd', 100.0, 0.2)
    assert abs(r[0] - (10 + 20) / 110) < 1e-12 and abs(r[1] - 120 / 1100) < 1e-12
    assert r[0] < 1.0 and abs(r[0] - 0.2) < abs(1.0 - 0.2), 'a small sample is pulled most of the way to average'
    assert pr.ovr(0) == 72 and pr.ovr(9) == 99 and pr.ovr(-9) == 45 and pr.attr_score(0) == 70
    P = pr.load()
    for pos in pr.POSITIONS:
        pp = P['pos'][pos]
        assert pp['value_sd'] > 0 and all(v > 0 for v in pp['m'].values()), pos
    assert P['pos']['QB']['m']['int'] > P['pos']['QB']['m']['sack'], 'interception rate takes longer to trust than sack rate'
    assert P['pos']['QB']['repl'] < P['pos']['QB']['eff_mean'], 'bench QBs are worse than the average starter'


def test_tags():
    a = dict(EFF=1.2, RUN=1.5, DEEP=0.0, SAFE=0.0, ACC=0.0, PKT=0.0)
    names = [n for n, _ in traits.player('QB', a, everything=True)]
    assert names == ['Playmaker', 'Dual-threat'], names
    assert len(traits.player('QB', a)) <= 2
    kept = traits.keep()
    for pos, rules in traits.PLAYER.items():
        assert set(kept.get(pos, [])) <= {n for n, _, _ in rules}, pos
    assert set(kept['TEAM']) <= {n for n, _, _ in traits.TEAM}
    z = {k: 0.0 for k in teamrate.TRAIT_KEYS + ['proe']}
    assert traits.team(z) == [] and traits.team(dict(z, proe=1.5), everything=True)[0][0] == 'Pass-first offense'


def test_alltime_file():
    path = os.path.join(HERE, 'site', 'alltime.json')
    if not os.path.exists(path):
        return
    a = json.load(open(path))
    ts = a['team_seasons']
    assert a['first'] == 1999 and len(ts) > 850 and ts[0]['rating'] >= ts[1]['rating']
    assert [r['rank'] for r in ts[:3]] == [1, 2, 3]
    for s in (2005, 2015):                                   # each completed season's ratings average about zero
        v = [r['rating'] for r in ts if r['season'] == s]
        assert len(v) == 32 and abs(np.mean(v)) < 0.1, (s, np.mean(v))
    best = {(r['season'], r['abbr']) for r in ts[:25]}
    assert (2007, 'NE') in best, 'the 16-0 Patriots should be near the top'
    assert (2008, 'DET') in {(r['season'], r['abbr']) for r in ts[-25:]}, 'the 0-16 Lions should be near the bottom'
    for pos in pr.POSITIONS:
        assert len(a['player_seasons'][pos]) >= 50 and len(a['player_careers'][pos]) >= 50, pos
        z = [r['z'] for r in a['player_seasons'][pos]]
        assert z == sorted(z, reverse=True)


if __name__ == '__main__':
    for name, fn in sorted(globals().items()):
        if name.startswith('test_'):
            fn(); print('  passed', name)
    print('ok')
