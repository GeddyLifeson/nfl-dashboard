"""Kicker and team defense (DEF) projections for Marlon's Sleeper league scoring.

Kicker features: recency-weighted kicker points (last season as a 3-game prior), the team's implied points from the
betting total and spread, dome, wind and cold.
Defense features: recency-weighted DEF points, the opponent's implied points, the opponent offense's recent
giveaways (interceptions + fumbles lost) and sacks taken per game, home, wind.
A ridge regression per type, fit on 2017-2024 and tested on 2025 and 2026 against the recency-weighted average.

Usage: python fantasy_kdst.py backtest | week [SEASON WEEK]
"""
import csv, json, math, os, sys, time
from collections import defaultdict
import numpy as np
import nflv
from fantasy import schedule, Ridge, f, HERE, DECAY, PRIOR_GAMES

K_SC = {'fg': [(19, 3), (29, 3), (39, 3), (49, 4), (59, 5), (99, 6)], 'fgmiss': -1, 'xpm': 1, 'xpmiss': -1}
D_SC = {'sack': 1, 'int': 2, 'fum_rec': 2, 'ff': 1, 'td': 6, 'safe': 2, 'blk': 2, 'st_td': 6,
        'allow': [(0, 10), (6, 7), (13, 4), (20, 1), (27, 0), (34, -1), (999, -4)]}


def k_points(r):
    p = 0.0
    for col, (hi, v) in zip(('fg_made_0_19', 'fg_made_20_29', 'fg_made_30_39', 'fg_made_40_49', 'fg_made_50_59', 'fg_made_60_'), K_SC['fg']):
        p += v * f(r[col])
    return p + K_SC['fgmiss'] * f(r['fg_missed']) + K_SC['xpm'] * f(r['pat_made']) + K_SC['xpmiss'] * f(r['pat_missed'])


def allow_pts(pa):
    for hi, v in D_SC['allow']:
        if pa <= hi:
            return v
    return -4


class TeamData:
    def __init__(self, seasons):
        self.games = schedule()
        self.k = defaultdict(list)       # kicker id -> games
        self.kmeta = {}
        self.d = defaultdict(list)       # team -> games (season, week, pts, giveaways, sacks_taken)
        self.off = defaultdict(list)     # team -> offense games (giveaways, sacks taken)
        for s in seasons:
            team = defaultdict(lambda: defaultdict(float))
            for r in csv.DictReader(open(nflv.get('player_week', s), encoding='utf-8')):
                if r['season_type'] != 'REG':
                    continue
                w, t = int(r['week']), r['team']
                if r['position'] == 'K' and (f(r['fg_att']) or f(r['pat_att'])):
                    self.k[r['player_id']].append({'season': s, 'week': w, 'team': t, 'pts': k_points(r)})
                    self.kmeta[r['player_id']] = r['player_display_name']
                a = team[(w, t)]
                a['opp'] = r['opponent_team']
                a['sack'] += f(r['def_sacks'])
                a['int'] += f(r['def_interceptions'])
                a['fum_rec'] += f(r['fumble_recovery_opp'])
                a['ff'] += f(r['def_fumbles_forced'])
                a['td'] += f(r['def_tds']) + f(r['fumble_recovery_tds'])
                a['safe'] += f(r['def_safeties'])
                a['blk'] += f(r['def_fg_blocks']) + f(r['def_punt_blocks']) + f(r['def_pat_blocks'])
                a['st_td'] += f(r['special_teams_tds'])
                a['give'] += f(r['passing_interceptions']) + f(r['sack_fumbles_lost']) + f(r['rushing_fumbles_lost']) + f(r['receiving_fumbles_lost'])
                a['sacked'] += f(r['sacks_suffered'])
            for (w, t), a in team.items():
                g = self.games.get((s, w, t))
                if not g or not g['played']:
                    continue
                pa = self._opp_score(s, w, t)
                pts = sum(D_SC[k] * a[k] for k in ('sack', 'int', 'fum_rec', 'ff', 'td', 'safe', 'blk', 'st_td')) + allow_pts(pa)
                self.d[t].append({'season': s, 'week': w, 'pts': pts})
                self.off[t].append({'season': s, 'week': w, 'give': a['give'], 'sacked': a['sacked']})
        for dd in (self.k, self.d, self.off):
            for h in dd.values():
                h.sort(key=lambda g: (g['season'], g['week']))

    def _opp_score(self, s, w, t):
        if not hasattr(self, '_scores'):
            self._scores = {}
            path = os.path.join(HERE, 'cache', 'schedule.csv')
            for g in csv.DictReader(open(path, encoding='utf-8')):
                if g['home_score'] != '':
                    self._scores[(int(g['season']), int(g['week']), g['home_team'])] = f(g['away_score'])
                    self._scores[(int(g['season']), int(g['week']), g['away_team'])] = f(g['home_score'])
        return self._scores.get((s, w, t), 21.0)

    @staticmethod
    def wavg(h, season, week, key):
        cur = [g[key] for g in h if g['season'] == season and g['week'] < week]
        prev = [g[key] for g in h if g['season'] == season - 1]
        num = den = 0.0
        for i, v in enumerate(reversed(cur)):
            num += DECAY ** i * v
            den += DECAY ** i
        if prev:
            num += PRIOR_GAMES * sum(prev) / len(prev)
            den += PRIOR_GAMES
        return num / den if den else None

    K_FEAT = ['pts', 'implied', 'dome', 'wind', 'cold']
    D_FEAT = ['pts', 'opp_implied', 'opp_give', 'opp_sacked', 'home', 'wind']

    def k_features(self, pid, s, w, team):
        g = self.games.get((s, w, team))
        p = self.wavg(self.k[pid], s, w, 'pts')
        if not g or p is None:
            return None
        return {'pts': p, 'implied': g['implied'] or 22.0, 'dome': 0.0 if g['outdoor'] else 1.0, 'wind': g['wind'], 'cold': g['cold']}

    def d_features(self, team, s, w):
        g = self.games.get((s, w, team))
        p = self.wavg(self.d[team], s, w, 'pts')
        if not g or p is None:
            return None
        og = self.games.get((s, w, g['opp']))
        give = self.wavg(self.off[g['opp']], s, w, 'give')
        sacked = self.wavg(self.off[g['opp']], s, w, 'sacked')
        return {'pts': p, 'opp_implied': (og['implied'] if og and og['implied'] is not None else 22.0),
                'opp_give': 1.3 if give is None else give, 'opp_sacked': 2.5 if sacked is None else sacked,
                'home': g['home'], 'wind': g['wind']}

    def rows(self, kind, season):
        out = []
        if kind == 'K':
            for pid, h in self.k.items():
                for gm in h:
                    if gm['season'] == season:
                        x = self.k_features(pid, season, gm['week'], gm['team'])
                        if x:
                            out.append((gm['week'], x, gm['pts']))
        else:
            for t, h in self.d.items():
                for gm in h:
                    if gm['season'] == season:
                        x = self.d_features(t, season, gm['week'])
                        if x:
                            out.append((gm['week'], x, gm['pts']))
        return out


def backtest(train=range(2017, 2025), tests=(2025, 2026)):
    td = TeamData(range(min(train) - 1, max(tests) + 1))
    out = {'made': time.strftime('%Y-%m-%d'), 'models': {}, 'report': {}}
    for kind, feats in (('K', TeamData.K_FEAT), ('DEF', TeamData.D_FEAT)):
        M = lambda rs: (np.array([[x[k] for k in feats] for _, x, _ in rs]), np.array([y for *_, y in rs]))
        tr = [r for s in train for r in td.rows(kind, s)]
        X, y = M(tr)
        m = Ridge(100).fit(X, y)
        out['models'][kind] = {'features': feats, 'b0': float(m.b0), 'mu': m.mu.tolist(), 'sd': m.sd.tolist(), 'b': m.b.tolist()}
        for s in tests:
            Xt, yt = M(td.rows(kind, s))
            for name, p in (('SAM', m.predict(Xt)), ('recency_avg', Xt[:, 0])):
                out['report'].setdefault(str(s), {}).setdefault(kind, {})[name] = {
                    'n': int(len(yt)), 'mae': round(float(np.abs(p - yt).mean()), 3),
                    'corr': round(float(np.corrcoef(p, yt)[0, 1]), 3)}
    json.dump(out, open(os.path.join(HERE, 'fantasy_kdst_params.json'), 'w'), indent=1)
    return out


def _pred(m, x):
    return m['b0'] + sum((x[k] - mu) / sd * b for k, mu, sd, b in zip(m['features'], m['mu'], m['sd'], m['b']))


def week(season, wk):
    params = json.load(open(os.path.join(HERE, 'fantasy_kdst_params.json')))['models']
    td = TeamData(range(season - 1, season + 1))
    out = []
    for pid, h in td.k.items():
        cur = [g for g in h if g['season'] == season and g['week'] < wk]
        if not cur:
            continue
        team = cur[-1]['team']
        x = td.k_features(pid, season, wk, team)
        g = td.games.get((season, wk, team))
        if g and g['played']:
            continue
        out.append({'name': td.kmeta[pid], 'pos': 'K', 'team': team, 'opp': g['opp'] if g else 'BYE',
                    'proj': round(max(0.0, _pred(params['K'], x)), 2) if x else 0.0})
    for t in td.d:
        g = td.games.get((season, wk, t))
        if g and g['played']:
            continue
        x = td.d_features(t, season, wk)
        out.append({'name': t + ' DEF', 'pos': 'DEF', 'team': t, 'opp': g['opp'] if g else 'BYE',
                    'proj': round(_pred(params['DEF'], x), 2) if x else 0.0})
    return sorted(out, key=lambda r: -r['proj'])


if __name__ == '__main__':
    if sys.argv[1:2] == ['backtest']:
        r = backtest()
        for s, by in r['report'].items():
            for kind, d in by.items():
                for name, m in d.items():
                    print(f"{s} {kind:3s} {name:12s} n={m['n']:4d} MAE {m['mae']:5.2f} corr {m['corr']:.3f}")
    else:
        s, w = (int(v) for v in sys.argv[2:4])
        for r in week(s, w)[:40]:
            print(r)
