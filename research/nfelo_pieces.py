"""Pieces copied from nfelo, added to SAM's features and retested on the frozen 2025-26 games.

qb538 / qb538_level: 538-style QB value (the formula nfelo's QB model uses), opponent-adjusted, an EWMA over each
  QB's starts and over each team's starts. qb538 = starter's value minus the team's recent QB value (the backup swap);
  qb538_level = starter minus league average. Built here from nflverse weekly stats, walk-forward.
wepa: nfelo's weighted EPA per game (github.com/greerreNFL/wepa, point-in-time weights, no look-ahead), turned into an
  opponent-adjusted season rating with last season as a prior, from games before each week.
wt: preseason rating from betting season win totals (github.com/greerreNFL/nfelosrs wt_ratings.csv, points vs average).
Also: SAM + market, a blend of SAM and the closing line with the weight fitted on 2014-24.
"""
import json, math, os, sys
from collections import defaultdict
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT); sys.path.insert(0, os.path.join(ROOT, 'research'))
os.environ.setdefault('TRAIN_FROM', '2014')
import nflv, mf_backtest as m
from model import rate
from teams import IDX
EXT = sys.argv[1] if len(sys.argv) > 1 else '/home/claude/ext'
FIXT = {'LAR': 'LA', 'STL': 'LA', 'OAK': 'LV', 'SD': 'LAC'}
fx = lambda t: FIXT.get(t, t)
games = pd.read_csv(os.path.join(ROOT, 'cache/games_all.csv'))
games = games[games.season >= 2012].sort_values(['gameday', 'game_id'])

# ---- QB value (538 formula), walk-forward
pw = pd.concat([pd.read_csv(nflv.get('player_week', s), low_memory=False) for s in range(2012, 2027)])
pw = pw[(pw.season_type == 'REG') & (pw.position == 'QB')].fillna(0)
pw['team'] = pw.team.map(fx); pw['opponent_team'] = pw.opponent_team.map(fx)
pw['value'] = (-2.2 * pw.attempts + 3.7 * pw.completions + pw.passing_yards / 5 + 11.3 * pw.passing_tds - 14.1 * pw.passing_interceptions
               - 8 * pw.sacks_suffered - 1.1 * pw.carries + 0.6 * pw.rushing_yards + 15.9 * pw.rushing_tds)
pw['plays'] = pw.attempts + pw.carries + pw.sacks_suffered
main = pw.sort_values('plays').groupby(['game_id', 'team']).tail(1).set_index(['game_id', 'team'])   # the game's QB
A = 0.1
qb_val, team_val, def_allowed = {}, {}, defaultdict(lambda: None)
hist_vals = []
feat = {}
for g in games.itertuples():
    h, a = fx(g.home_team), fx(g.away_team)
    league = float(np.mean(hist_vals[-1000:])) if hist_vals else 0.0
    repl = float(np.percentile(hist_vals[-1000:], 25)) if len(hist_vals) > 50 else league
    def pre(team, qb_id):
        sv = qb_val.get(qb_id, repl) if isinstance(qb_id, str) and qb_id else repl
        tv = team_val.get(team, sv)
        return sv - tv, sv - league
    hq = getattr(g, 'home_qb_id'); aq = getattr(g, 'away_qb_id')
    if isinstance(g.home_score, float) and math.isnan(g.home_score):
        continue
    (dh, lh), (da, la) = pre(h, hq), pre(a, aq)
    feat[g.game_id] = dict(qb538=(dh - da) * 3.3 / 25, qb538_level=(lh - la) * 3.3 / 25)
    for team, opp in ((h, a), (a, h)):                    # update after the game
        k = (g.game_id, team)
        if k not in main.index:
            continue
        r = main.loc[k]
        da_ = def_allowed[opp]
        adj = r.value - ((da_ - league) if da_ is not None else 0.0)
        def_allowed[opp] = r.value if da_ is None else (1 - A) * da_ + A * r.value
        qid = r.player_id
        qb_val[qid] = adj if qid not in qb_val else (1 - A) * qb_val[qid] + A * adj
        team_val[team] = adj if team not in team_val else (1 - A) * team_val[team] + A * adj
        hist_vals.append(adj)

# ---- WEPA rating, walk-forward per week
wp = pd.read_csv(os.path.join(EXT, 'wepa/data/wepa_by_game_pit.csv'))
wp['team'] = wp.team.map(fx)
wnet = {(r.game_id, r.team): r.wepa_net for r in wp.itertuples()}
gl = [dict(id=g.game_id, season=int(g.season), week=int(g.week), home=fx(g.home_team), away=fx(g.away_team),
           neutral=g.location == 'Neutral', margin=wnet.get((g.game_id, fx(g.home_team)))) for g in games.itertuples()]
gl = [x for x in gl if x['margin'] is not None and not (isinstance(x['margin'], float) and math.isnan(x['margin']))]
hfa_w = float(np.mean([x['margin'] for x in gl if x['season'] <= 2013 and not x['neutral']]))
end = {}
for s in range(2012, 2027):
    cur = [x for x in gl if x['season'] == s]
    base = end.get(s - 1)
    for w in sorted({x['week'] for x in games[games.season == s].itertuples() for x in [x]} if False else set(games[games.season == s].week)):
        r = rate([x for x in cur if x['week'] < w], 8, hfa_w, base, 0.6)
        for g in games[(games.season == s) & (games.week == w)].itertuples():
            if g.game_id in feat:
                feat[g.game_id]['wepa'] = float(r[IDX[fx(g.home_team)]] - r[IDX[fx(g.away_team)]])
    end[s] = rate(cur, 8, hfa_w, base, 0.6)

# ---- win-total preseason rating
wt = pd.read_csv(os.path.join(EXT, 'nfelosrs/wt_ratings.csv'))
wtd = {(fx(r.team), int(r.season)): r.wt_rating for r in wt.itertuples()}
for g in games.itertuples():
    if g.game_id in feat:
        feat[g.game_id]['wt'] = wtd.get((fx(g.home_team), int(g.season)), 0.0) - wtd.get((fx(g.away_team), int(g.season)), 0.0)

rs = [r for r in json.load(open(os.path.join(ROOT, 'cache/mf_rows.json'))) if r['season'] >= 2013]
for r in rs:
    r['f'].update({k: 0.0 for k in ('qb538', 'qb538_level', 'wepa', 'wt')})
    r['f'].update(feat.get(r['id'], {}))
    r['f']['wt_early'] = r['f']['wt'] * max(0.0, 1 - (r['week'] - 1) / 8)     # prior that fades over 8 weeks
json.dump(rs, open(os.path.join(ROOT, 'cache/mf_rows_plus.json'), 'w'))
print('coverage', {k: round(np.mean([r['f'][k] != 0 for r in rs]), 3) for k in ('qb538', 'wepa', 'wt')})

E = m.SETS['E + coaching changes']
tests = {
    'SAM E (current)': E,
    'E + QB value (backup swap + level)': E + ['qb538', 'qb538_level'],
    'E, QB value replacing old QB gap': [n for n in E if n != 'qb'] + ['qb538', 'qb538_level'],
    'E + weighted EPA rating': E + ['wepa'],
    'E + win-total prior': E + ['wt', 'wt_early'],
    'E + all three nfelo pieces': E + ['qb538', 'qb538_level', 'wepa', 'wt', 'wt_early'],
}
res = {}
for lab, names in tests.items():
    lam, f, sig, met, te = m.run(rs, names)
    res[lab] = dict(lam=lam, **met); print(f'{lab:40s} lam={lam:<5} {met}')
best = min(res, key=lambda k: res[k]['log_loss'])
print('best by log loss:', best)
# early weeks check for the prior
for lab in ('SAM E (current)', 'E + win-total prior', 'E + all three nfelo pieces'):
    lam, f, sig, _, _ = m.run(rs, tests[lab])
    sub = [r for r in rs if r['season'] in m.TEST and r['spread'] is not None and r['week'] <= 6]
    X, y = m.mat(sub, tests[lab]); print(f'   weeks 1-6 only, {lab}: {m.metrics(f(X), y, sig)}')

# ---- SAM + market: blend weight fitted on 2014-24, frozen
names = tests[best]
lam, f, sig, _, _ = m.run(rs, names)
tr = [r for r in rs if r['season'] in m.TRAIN and r['spread'] is not None]
te = [r for r in rs if r['season'] in m.TEST and r['spread'] is not None]
# out-of-fold SAM predictions on training seasons (leave one season out) so the weight isn't flattered
oof = []
for s in m.TRAIN:
    ff, _ = m.fit(*m.mat([r for r in rs if r['season'] in m.TRAIN and r['season'] != s], names), lam)
    sub = [r for r in tr if r['season'] == s]; oof += list(zip(ff(m.mat(sub, names)[0]), [r['spread'] for r in sub], [r['margin'] for r in sub]))
P, L, Y = map(np.array, zip(*oof))
ws = np.linspace(0, 1, 101); w = float(ws[np.argmin([np.mean((wi * P + (1 - wi) * L - Y) ** 2) for wi in ws])])
sig_b = float(np.std(w * P + (1 - w) * L - Y))
X, y = m.mat(te, names); pt = f(X); lt = np.array([r['spread'] for r in te])
print(f'\nSAM + market: weight on SAM {w:.2f} (fitted on 2014-24 out-of-fold)')
print('  SAM alone   ', m.metrics(pt, y, sig))
print('  SAM + market', m.metrics(w * pt + (1 - w) * lt, y, sig_b))
print('  closing line', m.metrics(lt, y, float(np.std(L - Y))))
res['blend'] = dict(weight_on_sam=w, sam_plus_market=m.metrics(w * pt + (1 - w) * lt, y, sig_b), close=m.metrics(lt, y, float(np.std(L - Y))), features=names)
json.dump(res, open(os.path.join(ROOT, 'research/nfelo_pieces.json'), 'w'), indent=1)
