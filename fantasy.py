"""Season-long fantasy football projections for QB, RB, WR and TE (default: Marlon's Sleeper league, full PPR).

Each projection uses only information available before the game:
  - the player's own recent production and usage, recency-weighted (each older game counts 15% less) with last
    season as a prior worth 3 games: fantasy points, targets, carries, pass attempts, target share, air-yards share,
    offensive snap share, red-zone-free EPA per game (passing + rushing + receiving EPA);
  - team context from the betting market: the team's implied points (total and spread), home or away;
  - the opponent: fantasy points it has allowed to the position this season versus the league (shrunk);
  - the starting quarterback (nflverse schedule): his passing EPA per attempt and whether he differs from the
    team's last starter;
  - injuries: the player's own Questionable tag and the targets and carries left behind by teammates listed Out
    or Doubtful;
  - weather: wind and cold for outdoor games, domes.

A ridge regression per position turns these into a point projection. backtest() fits on 2017-2024 and reports
2025 and 2026 against two simple baselines (season-to-date average and recency-weighted average), using MAE and
start/sit accuracy (how often the higher-projected of two same-position players outscored the other).

Usage:
  python fantasy.py backtest            # fit, test, write fantasy_params.json
  python fantasy.py week [SEASON WEEK]  # projections for an upcoming week -> fantasy_week.json
"""
import csv, json, math, os, sys, time, urllib.request
from collections import defaultdict
import numpy as np
import nflv
from lineup import norm

HERE = os.path.dirname(os.path.abspath(__file__))
SCHEDULE_URL = 'https://github.com/nflverse/nflverse-data/releases/download/schedules/games.csv'
POS = ('QB', 'RB', 'WR', 'TE')
DECAY, PRIOR_GAMES = 0.85, 3.0

# Sleeper-style scoring. 'ppr' is Marlon's league; league settings can override any key.
SCORING = {
    'ppr': {'pass_yd': 0.04, 'pass_td': 4, 'pass_int': -2, 'pass_2pt': 2, 'rush_yd': 0.1, 'rush_td': 6, 'rush_2pt': 2,
            'rec': 1.0, 'rec_yd': 0.1, 'rec_td': 6, 'rec_2pt': 2, 'fum_lost': -2, 'st_td': 6},
}
SCORING['half'] = dict(SCORING['ppr'], rec=0.5)
# Marlon's Sleeper league "Premier Bois Roundtable": full PPR, 4-pt pass TD, -1 per interception, -2 fumble lost
SCORING['league'] = dict(SCORING['ppr'], pass_int=-1)
SCORING['std'] = dict(SCORING['ppr'], rec=0.0)


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return 0.0


def points(r, sc):
    return (sc['pass_yd'] * f(r['passing_yards']) + sc['pass_td'] * f(r['passing_tds'])
            + sc['pass_int'] * f(r['passing_interceptions']) + sc['pass_2pt'] * f(r['passing_2pt_conversions'])
            + sc['rush_yd'] * f(r['rushing_yards']) + sc['rush_td'] * f(r['rushing_tds'])
            + sc['rush_2pt'] * f(r['rushing_2pt_conversions']) + sc['rec'] * f(r['receptions'])
            + sc['rec_yd'] * f(r['receiving_yards']) + sc['rec_td'] * f(r['receiving_tds'])
            + sc['rec_2pt'] * f(r['receiving_2pt_conversions']) + sc['st_td'] * f(r['special_teams_tds'])
            + sc['fum_lost'] * (f(r['sack_fumbles_lost']) + f(r['rushing_fumbles_lost']) + f(r['receiving_fumbles_lost'])))


def schedule(hours=6):
    path = os.path.join(HERE, 'cache', 'schedule.csv')
    if not os.path.exists(path) or time.time() - os.path.getmtime(path) > hours * 3600:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = urllib.request.urlopen(SCHEDULE_URL, timeout=120).read()
        open(path + '.tmp', 'wb').write(data)
        os.replace(path + '.tmp', path)
    games = {}
    for g in csv.DictReader(open(path, encoding='utf-8')):
        if g['game_type'] != 'REG':
            continue
        s, w = int(g['season']), int(g['week'])
        spread, total = g['spread_line'], g['total_line']
        outdoor = g['roof'] in ('outdoors', 'open', '')
        for side, other, sign in (('home', 'away', 1), ('away', 'home', -1)):
            implied = None
            if spread != '' and total != '':
                implied = (f(total) + sign * f(spread)) / 2
            games[(s, w, g[side + '_team'])] = {
                'opp': g[other + '_team'], 'home': 1.0 if side == 'home' and g['location'] == 'Home' else 0.0,
                'implied': implied, 'outdoor': outdoor, 'wind': f(g['wind']) if outdoor and g['wind'] else (8.0 if outdoor else 0.0),
                'cold': 1.0 if outdoor and g['temp'] and f(g['temp']) < 35 else 0.0,
                'qb': g[side + '_qb_id'], 'qb_name': g[side + '_qb_name'], 'played': g['home_score'] != '',
                'game_id': g['game_id'], 'gameday': g['gameday'], 'gametime': g['gametime']}
    return games


class Data:
    """Player game logs, snaps and injury reports for a range of seasons."""

    def __init__(self, seasons, sc):
        self.sc = sc
        self.games = schedule()
        self.hist = defaultdict(list)          # player_id -> list of game dicts, in date order
        self.meta = {}                         # player_id -> (name, pos)
        self.team_qb = {}                      # (season, week, team) -> QB id with most attempts
        self.allowed = defaultdict(lambda: defaultdict(float))  # (season, opp) -> {(week,pos): pts}
        self.status = {}                       # (season, week, gsis_id) -> report status
        snaps = {}
        for s in seasons:
            for r in csv.DictReader(open(nflv.get('snaps', s), encoding='utf-8')):
                if r['game_type'] == 'REG':
                    snaps[(s, int(r['week']), r['team'], norm(r['player']))] = f(r['offense_pct'])
            for r in csv.DictReader(open(nflv.get('injuries', s), encoding='utf-8')):
                if r['game_type'] == 'REG' and r['report_status']:
                    self.status[(s, int(r['week']), r['gsis_id'])] = r['report_status']
            att = {}
            for r in csv.DictReader(open(nflv.get('player_week', s), encoding='utf-8')):
                if r['season_type'] != 'REG':
                    continue
                pos = 'RB' if r['position'] == 'FB' else r['position']
                w, team = int(r['week']), r['team']
                if pos == 'QB' and f(r['attempts']) > att.get((w, team), (0, None))[0]:
                    att[(w, team)] = (f(r['attempts']), r['player_id'])
                if pos not in POS:
                    continue
                pts = points(r, sc)
                self.allowed[(s, r['opponent_team'])][(w, pos)] += pts
                self.meta[r['player_id']] = (r['player_display_name'], pos)
                self.hist[r['player_id']].append({
                    'season': s, 'week': w, 'team': team, 'opp': r['opponent_team'], 'pos': pos, 'pts': pts,
                    'targets': f(r['targets']), 'carries': f(r['carries']), 'att': f(r['attempts']),
                    'tshare': f(r['target_share']), 'ashare': f(r['air_yards_share']),
                    'snap': snaps.get((s, w, team, norm(r['player_display_name'])), float('nan')),
                    'epa': f(r['passing_epa']) + f(r['rushing_epa']) + f(r['receiving_epa']),
                    'pass_epa': f(r['passing_epa'])})
            for (w, team), (_, pid) in att.items():
                self.team_qb[(s, w, team)] = pid
        for h in self.hist.values():
            h.sort(key=lambda g: (g['season'], g['week']))
        # league average points allowed to each position per team-game, by season
        self.lg = {}
        for (s, opp), d in self.allowed.items():
            for (w, pos), v in d.items():
                a = self.lg.setdefault((s, pos), [0.0, 0])
                a[0] += v
                a[1] += 1
        self.roster = defaultdict(set)         # (season, team) -> player ids who played for team
        for pid, h in self.hist.items():
            for g in h:
                self.roster[(g['season'], g['team'])].add(pid)

    # ---------- per-player summaries from games before (season, week) ----------
    def split(self, pid, season, week):
        h = self.hist.get(pid, [])
        cur = [g for g in h if g['season'] == season and g['week'] < week]
        prev = [g for g in h if g['season'] == season - 1]
        return cur, prev

    @staticmethod
    def wavg(cur, prev, key):
        vals = [g[key] for g in cur if not math.isnan(g[key])]
        pv = [g[key] for g in prev if not math.isnan(g[key])]
        num = den = 0.0
        for i, v in enumerate(reversed(vals)):
            wt = DECAY ** i
            num += wt * v
            den += wt
        if pv:
            num += PRIOR_GAMES * sum(pv) / len(pv)
            den += PRIOR_GAMES
        return num / den if den else float('nan')

    def opp_factor(self, season, opp, pos, week):
        d = self.allowed.get((season, opp), {})
        vals = [v for (w, p), v in d.items() if p == pos and w < week]
        lg = self.lg.get((season, pos)) or self.lg.get((season - 1, pos))
        lgavg = lg[0] / lg[1] if lg else 1.0
        return ((sum(vals) + 4 * lgavg) / (len(vals) + 4)) / lgavg if lgavg else 1.0

    def qb_epa(self, qb, season, week):
        cur, prev = self.split(qb, season, week)
        vals = [(g['pass_epa'], g['att']) for g in cur + prev if g['att'] > 0]
        e = sum(v for v, _ in vals)
        a = sum(n for _, n in vals)
        return (e - 0.05 * 150) / (a + 150)   # shrink toward a below-average passer (-0.05/att) with 150 attempts

    def vacated(self, season, week, team, me):
        """Targets and carries per game of teammates listed Out/Doubtful who were playing recently."""
        t = c = 0.0
        for pid in self.roster.get((season, team), ()):
            if pid == me or self.status.get((season, week, pid)) not in ('Out', 'Doubtful'):
                continue
            cur = [g for g in self.hist[pid] if g['season'] == season and g['week'] < week and g['team'] == team]
            if cur and cur[-1]['week'] >= week - 3:
                t += sum(g['targets'] for g in cur) / len(cur)
                c += sum(g['carries'] for g in cur) / len(cur)
        return t, c

    FEATURES = ['pts', 'targets', 'carries', 'att', 'tshare', 'ashare', 'snap', 'epa', 'last3', 'season_avg',
                'implied', 'home', 'opp', 'pts_x_opp', 'pts_x_implied', 'qb_epa', 'qb_change', 'questionable',
                'vac_targets', 'vac_carries', 'new_team', 'wind', 'cold', 'outdoor', 'first_games']

    def features(self, pid, season, week, team):
        g = self.games.get((season, week, team))
        if not g:
            return None
        cur, prev = self.split(pid, season, week)
        if not cur and not prev:
            return None
        pos = self.meta[pid][1]
        W = lambda k: self.wavg(cur, prev, k)
        pts = W('pts')
        snap = W('snap')
        lg_implied = 22.5
        implied = g['implied'] if g['implied'] is not None else lg_implied
        opp = self.opp_factor(season, g['opp'], pos, week)
        last_qb = None
        for wk in range(week - 1, 0, -1):
            last_qb = self.team_qb.get((season, wk, team))
            if last_qb:
                break
        qb = g['qb'] or last_qb
        vt, vc = self.vacated(season, week, team, pid)
        last = (cur or prev)[-1]
        x = {'pts': pts, 'targets': W('targets'), 'carries': W('carries'), 'att': W('att'), 'tshare': W('tshare'),
             'ashare': W('ashare'), 'snap': 0.5 if math.isnan(snap) else snap, 'epa': W('epa'),
             'last3': sum(q['pts'] for q in (cur or prev)[-3:]) / len((cur or prev)[-3:]),
             'season_avg': sum(q['pts'] for q in (cur or prev)) / len(cur or prev),
             'implied': implied, 'home': g['home'], 'opp': opp, 'pts_x_opp': pts * (opp - 1),
             'pts_x_implied': pts * (implied - lg_implied) / lg_implied,
             'qb_epa': self.qb_epa(qb, season, week) if qb else -0.05,
             'qb_change': 1.0 if (qb and last_qb and qb != last_qb) else 0.0,
             'questionable': 1.0 if self.status.get((season, week, pid)) == 'Questionable' else 0.0,
             'vac_targets': vt, 'vac_carries': vc, 'new_team': 1.0 if last['team'] != team else 0.0,
             'wind': g['wind'], 'cold': g['cold'], 'outdoor': 1.0 if g['outdoor'] else 0.0,
             'first_games': 1.0 if len(cur) < 2 else 0.0}
        for k, v in x.items():
            if isinstance(v, float) and math.isnan(v):
                x[k] = 0.0
        return x, g

    def instances(self, season, first_week=1, last_week=18):
        """Every player-game in the season with a prior game, as (pid, week, features, actual points)."""
        out = []
        for pid, h in self.hist.items():
            for gm in h:
                if gm['season'] != season or not first_week <= gm['week'] <= last_week:
                    continue
                r = self.features(pid, season, gm['week'], gm['team'])
                if r:
                    out.append((pid, gm['week'], r[0], gm['pts']))
        return out


class Ridge:
    def __init__(self, lam):
        self.lam = lam

    def fit(self, X, y):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-9
        Z = (X - self.mu) / self.sd
        n = Z.shape[1]
        self.b0 = y.mean()
        self.b = np.linalg.solve(Z.T @ Z + self.lam * np.eye(n), Z.T @ (y - self.b0))
        return self

    def predict(self, X):
        return self.b0 + ((X - self.mu) / self.sd) @ self.b


def matrix(rows):
    return np.array([[x[k] for k in Data.FEATURES] for _, _, x, _ in rows]), np.array([y for *_, y in rows])


def startsit(rows, pred, mask):
    """Share of same-position, same-week pairs of fantasy-relevant players (mask) ordered correctly.
    Every method is scored on the same pairs."""
    by = defaultdict(list)
    for (pid, week, x, y), p, keep in zip(rows, pred, mask):
        if keep:
            by[(week, x['_pos'])].append((p, y))
    right = total = 0
    for grp in by.values():
        for i in range(len(grp)):
            for j in range(i + 1, len(grp)):
                (pa, ya), (pb, yb) = grp[i], grp[j]
                if ya == yb or pa == pb:
                    continue
                total += 1
                right += (pa > pb) == (ya > yb)
    return right / total if total else float('nan'), total


def backtest(train=range(2017, 2025), tests=(2025, 2026), sc=SCORING['league']):
    seasons = range(min(train) - 1, max(tests) + 1)
    d = Data(seasons, sc)
    rows = {s: d.instances(s) for s in list(train) + list(tests)}
    for s, rs in rows.items():
        for pid, w, x, y in rs:
            x['_pos'] = d.meta[pid][1]
    report, models = {}, {}
    for pos in POS:
        tr = [r for s in train for r in rows[s] if r[2]['_pos'] == pos]
        # pick ridge strength on the last training season, then refit on all of it
        inner = [r for r in tr if True]
        fit_rows = [r for s in list(train)[:-1] for r in rows[s] if r[2]['_pos'] == pos]
        val_rows = [r for r in rows[list(train)[-1]] if r[2]['_pos'] == pos]
        Xf, yf = matrix(fit_rows)
        Xv, yv = matrix(val_rows)
        best = min((np.abs(Ridge(l).fit(Xf, yf).predict(Xv) - yv).mean(), l) for l in (1, 10, 100, 1000, 10000))
        X, y = matrix(tr)
        m = Ridge(best[1]).fit(X, y)
        models[pos] = {'lam': best[1], 'b0': float(m.b0), 'mu': m.mu.tolist(), 'sd': m.sd.tolist(),
                       'b': dict(zip(Data.FEATURES, m.b.tolist()))}
        for s in tests:
            te = [r for r in rows[s] if r[2]['_pos'] == pos]
            Xt, yt = matrix(te)
            preds = {'SAM': m.predict(Xt), 'season_avg': Xt[:, Data.FEATURES.index('season_avg')],
                     'recency_avg': Xt[:, Data.FEATURES.index('pts')]}
            # fantasy-relevant: averaging 8+ points (season_avg), the same set for every method
            rel = preds['season_avg'] >= 8
            for name, p in preds.items():
                ss, n = startsit(te, p, rel)
                report.setdefault(str(s), {}).setdefault(pos, {})[name] = {
                    'n': int(len(yt)), 'mae': round(float(np.abs(p - yt).mean()), 3),
                    'mae_8plus': round(float(np.abs(p - yt)[rel].mean()), 3),
                    'corr': round(float(np.corrcoef(p, yt)[0, 1]), 3), 'startsit': round(ss, 4), 'pairs': n}
    out = {'made': time.strftime('%Y-%m-%d'), 'scoring': sc, 'train': [min(train), max(train)],
           'tests': list(tests), 'features': Data.FEATURES, 'models': models, 'report': report}
    json.dump(out, open(os.path.join(HERE, 'fantasy_params.json'), 'w'), indent=1)
    return out


def predict(params, pos, x):
    m = params['models'][pos]
    z = [(x[k] - mu) / sd for k, mu, sd in zip(params['features'], m['mu'], m['sd'])]
    return m['b0'] + sum(zi * m['b'][k] for zi, k in zip(z, params['features']))


def week(season=None, wk=None, sc=SCORING['league']):
    """Projections for every QB/RB/WR/TE who played this season, for the given (default: next) week."""
    params = json.load(open(os.path.join(HERE, 'fantasy_params.json')))
    season = season or nflv.current_season()
    d = Data(range(season - 1, season + 1), sc)
    if wk is None:
        done = [k[1] for k, g in d.games.items() if k[0] == season and g['played']]
        wk = max(done) if done else 1
        # the current week is the first one with unplayed games
        pending = sorted(k[1] for k, g in d.games.items() if k[0] == season and not g['played'])
        wk = pending[0] if pending else wk
    out = []
    for pid, h in d.hist.items():
        cur = [g for g in h if g['season'] == season and g['week'] < wk]
        if not cur:
            continue
        team = cur[-1]['team']
        g = d.games.get((season, wk, team))
        name, pos = d.meta[pid]
        status = d.status.get((season, wk, pid), '')
        if not g:
            out.append({'id': pid, 'name': name, 'pos': pos, 'team': team, 'opp': 'BYE', 'proj': 0.0, 'status': status})
            continue
        if g['played']:
            continue
        x, _ = d.features(pid, season, wk, team)
        p = max(0.0, predict(params, pos, x))
        if status == 'Out':
            p = 0.0
        out.append({'id': pid, 'name': name, 'pos': pos, 'team': team, 'opp': g['opp'], 'home': bool(g['home']),
                    'proj': round(p, 2), 'status': status, 'implied': g['implied'], 'snap': round(x['snap'], 2),
                    'targets': round(x['targets'], 1), 'carries': round(x['carries'], 1),
                    'opp_factor': round(x['opp'], 2), 'qb': g['qb_name'], 'gameday': g['gameday']})
    out.sort(key=lambda r: -r['proj'])
    res = {'season': season, 'week': wk, 'made': time.strftime('%Y-%m-%d %H:%M'), 'players': out}
    json.dump(res, open(os.path.join(HERE, 'fantasy_week.json'), 'w'), indent=1)
    return res


if __name__ == '__main__':
    cmd = sys.argv[1] if len(sys.argv) > 1 else 'week'
    if cmd == 'backtest':
        r = backtest()
        for s, by in r['report'].items():
            print(f'== {s} ==')
            for pos, d in by.items():
                for name, m in d.items():
                    print(f"{pos:3s} {name:12s} n={m['n']:5d} MAE {m['mae']:5.2f} MAE(8+avg) {m['mae_8plus']:5.2f} "
                          f"corr {m['corr']:.3f} start/sit {m['startsit']:.1%} ({m['pairs']} pairs)")
    elif cmd == 'week':
        a = [int(v) for v in sys.argv[2:4]]
        r = week(*a) if a else week()
        print(f"Season {r['season']} week {r['week']}: {len(r['players'])} players")
        for pos in POS:
            top = [p for p in r['players'] if p['pos'] == pos][:12]
            print(pos, ', '.join(f"{p['name']} {p['proj']:.1f}" for p in top))
