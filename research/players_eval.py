"""Fit and test SAM's player ratings (playerrate.py).

  python research/players_eval.py     # writes player_params.json and research/players_results.txt

1. Repeatability: for every stat, how many plays (or games) it takes before a player's number says as much about him
   as the position average does. Measured by comparing each player's odd weeks with his even weeks, 2006-2022.
2. Weights: the mix of (stabilised) stats that best predicted a player's efficiency over the rest of the season,
   from snapshots after weeks 4, 8 and 12 of 2006-2022.
3. Test on 2023-2025, never used above: rate everyone after week 8, then see how well the rating lines up with
   what the player did from week 9 on. The old position score is put through the same test.
"""
import json, os, sys
import numpy as np
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import playerrate as pr

HERE = os.path.dirname(os.path.abspath(__file__))
TRAIN, TEST = range(2006, 2023), (2023, 2024, 2025)
SNAPS = (5, 9, 13)
FUT_MIN = {'QB': 100, 'RB': 40, 'WR': 20, 'TE': 15}


def wcorr(a, b, w):
    a, b, w = map(np.asarray, (a, b, w))
    ma, mb = np.average(a, weights=w), np.average(b, weights=w)
    c = np.average((a - ma) * (b - mb), weights=w)
    return c / np.sqrt(np.average((a - ma) ** 2, weights=w) * np.average((b - mb) ** 2, weights=w) + 1e-18)


def stabil(df, pos, num, den):
    """Units of `den` at which a player's own rate and the position average deserve equal weight."""
    d = df[(df.pos == pos) & df.season.isin(TRAIN)]
    odd = d[d.week % 2 == 1].groupby(['season', 'player_id'])[[num, den]].sum()
    even = d[d.week % 2 == 0].groupby(['season', 'player_id'])[[num, den]].sum()
    j = odd.join(even, lsuffix='_o', rsuffix='_e', how='inner')
    j = j[(j[den + '_o'] > 0) & (j[den + '_e'] > 0)]
    if len(j) < 50:
        return 1e6, 0.0
    w = np.minimum(j[den + '_o'], j[den + '_e'])
    j = j[w >= np.quantile(w, 0.5)]; w = np.minimum(j[den + '_o'], j[den + '_e'])     # the half with real workloads
    r = wcorr(j[num + '_o'] / j[den + '_o'], j[num + '_e'] / j[den + '_e'], w)
    r = float(np.clip(r, 0.02, 0.98))
    n_half = float(np.average((j[den + '_o'] + j[den + '_e']) / 2, weights=w))
    return n_half * (1 - r) / r, r


def spearman(a, b):
    return float(pd.Series(a).rank().corr(pd.Series(b).rank()))


def main():
    lines = []
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)
    df = pr.all_weeks(pr.FIRST, 2026)
    P = dict(pos={})
    say('Repeatability (units before a player\'s own number gets half the weight; split-half correlation in brackets)')
    for pos in pr.POSITIONS:
        sp = pr.SPEC[pos]; pp = dict(m={}, attr_m={})
        items = dict(sp.get('comps', {}))
        if 'events' in sp:
            items = {e: (pr.EV_COL[e], 'games') for e in sp['events']}
        if pos == 'K':
            items = {'fgoe': ('fgoe', 'fg_att'), 'xpoe': ('xpoe', 'xp_att')}
        txt = []
        for c, (num, den) in items.items():
            m, r = stabil(df, pos, num, den)
            pp['m'][c] = round(m, 1); txt.append(f'{c} {m:.0f} {den} [{r:.2f}]')
        for label, _, num, den, _ in sp['attrs']:
            m, r = stabil(df, pos, num, den)
            pp['attr_m'][label] = round(min(m, 5000.0), 1)
        say(f'  {pos}: ' + ', '.join(txt))
        P['pos'][pos] = pp

    # workload and efficiency weights
    def snapshots(pos, seasons, prior, snaps=SNAPS):
        out = []
        for s in seasons:
            for w in snaps:
                t = pr.totals(df, s, w, prior)
                fut = pr.totals(df, s, 99, 0.0, from_week=w)
                out.append((s, w, t, fut))
        return out

    say('\nHow much should last season count? 1 - correlation between the rating\'s main number after weeks 4, 8, 12 and the rest of the season (lower is better):')
    made = lambda t: sum(pr.EVENT_PTS[e] * t[pr.EV_COL[e]] for e in pr.EVENT_PTS)
    for pos in pr.POSITIONS:
        sp = pr.SPEC[pos]; res = {}
        grid = ((1.0, 0.7), (1.0, 0.0), 0.5, 0.0) if pos == 'K' else (0.0, 0.3, 0.5, 0.7, 1.0)
        for prior in grid:
            a, b, wt = [], [], []
            for s, w, t, fut in snapshots(pos, TRAIN, prior):
                q = t[pr.qualifies(t, pos)]; f = fut.reindex(q.index)
                if 'eff' in sp:
                    num, den = sp['eff']; ok = (f[den] >= FUT_MIN[pos]).values
                    r, _ = pr.rate(q[ok], num, den, P['pos'][pos]['m']['epa'], t[t.pos == pos][num].sum() / t[t.pos == pos][den].sum())
                    a += list(r); b += list(f[ok][num] / f[ok][den]); wt += list(f[ok][den])
                elif pos == 'K':
                    ok = (f.fg_att >= 8).values
                    a += list(pr.rate(q[ok], 'fgoe', 'fg_att', P['pos'][pos]['m']['fgoe'], 0.0)[0]); b += list(f[ok].fgoe / f[ok].fg_att); wt += list(f[ok].fg_att)
                else:
                    ok = (f.games >= 4).values
                    v = pr.estimate(t, pos, dict(P['pos'][pos], value_sd=1.0)).value
                    a += list(v[ok]); b += list((made(f) / f.games)[ok]); wt += list(f.games[ok])
            res[prior] = round(1 - float(wcorr(a, b, wt)), 4)
        P['pos'][pos]['prior'] = min(res, key=res.get)
        say(f"  {pos}: {res} -> last-season weight {P['pos'][pos]['prior']}")

    for pos in ('QB', 'RB', 'WR', 'TE'):
        sp = pr.SPEC[pos]; pp = P['pos'][pos]; num, den = sp['eff']; work = sp['work']
        snaps = snapshots(pos, TRAIN, pp['prior'])
        full = [pr.totals(df, s, 99, 0.0) for s in TRAIN]
        qs = [t[pr.qualifies(t, pos)] for t in full]
        pp['work_mu'] = float(sum(q[work].sum() for q in qs) / sum(q.games.sum() for q in qs))
        pp['work_m'] = round(stabil(df, pos, work, 'games')[0], 2)
        bench = pd.concat([t[(t.pos == pos) & (t[work] >= sp['min_work'] / 4) & ~pr.qualifies(t, pos)] for t in full])
        pp['repl'] = float(bench[num].sum() / bench[den].sum())         # efficiency of players who did not earn a regular role
        # scale of each stabilised stat among qualifiers
        raw = {c: [] for c in sp['comps']}; ys, ws, groups = [], [], []
        for s, w, t, fut in snaps:
            q = t[pr.qualifies(t, pos)]; pool = t[t.pos == pos]
            f = fut.reindex(q.index); ok = (f[den] >= FUT_MIN[pos]).values
            for c, (n_, d_) in sp['comps'].items():
                mu = pool[n_].sum() / pool[d_].sum() if pool[d_].sum() > 0 else 0.0
                raw[c] += list((pr.rate(q, n_, d_, pp['m'][c], mu)[0] - mu)[ok])
            ys += list((f[num] / f[den])[ok]); ws += list(f[den][ok]); groups += [s] * int(ok.sum())
        pp['sd'] = {c: float(np.std(v)) for c, v in raw.items()}
        X = np.column_stack([np.array(raw[c]) / (pp['sd'][c] or 1.0) for c in sp['comps']])
        y, w_, groups = np.array(ys), np.array(ws), np.array(groups)
        ymean = float(np.average(y, weights=w_))

        def ridge(idx, lam):
            Xi, yi, wi = X[idx], y[idx] - ymean, w_[idx]
            A = Xi.T @ (Xi * wi[:, None]) + lam * wi.sum() * np.eye(X.shape[1])
            return np.linalg.solve(A, Xi.T @ (wi * yi))
        cv = {}
        for lam in (0.01, 0.03, 0.1, 0.3, 1.0):
            e = 0.0
            for fold in range(4):
                te = np.isin(groups, [s for s in TRAIN if s % 4 == fold])
                b = ridge(~te, lam)
                e += float(np.sum(w_[te] * (y[te] - ymean - X[te] @ b) ** 2))
            cv[lam] = e / w_.sum()
        lam = min(cv, key=cv.get)
        b = ridge(np.ones(len(y), bool), lam)
        base = float(np.average((y - ymean) ** 2, weights=w_))
        only = X[:, 0] * np.average(X[:, 0] * (y - ymean), weights=w_) / np.average(X[:, 0] ** 2, weights=w_)
        pp.update(w=dict(zip(sp['comps'], b.tolist())), eff_mean=ymean, ridge=lam)
        say(f"  {pos}: bench level {pp['repl']:.3f}, league {ymean:.3f}; rest-of-season efficiency, share of variation explained out of fold: fitted mix {100 * (1 - cv[lam] / base):.1f}%, "
            f"efficiency stat alone {100 * (1 - float(np.average((y - ymean - only) ** 2, weights=w_)) / base):.1f}% (in sample); weights {dict(zip(sp['comps'], np.round(b, 4).tolist()))}")
    k = pr.totals(df, 2022, 99, 0.0); kq = pd.concat([pr.totals(df, s, 99, 0.0) for s in TRAIN]); kq = kq[(kq.pos == 'K') & (kq.cur_fg_att >= 5)]
    P['pos']['K'].update(work_mu=float(kq.fg_att.sum() / kq.games.sum()), xp_mu=float(kq.xp_att.sum() / kq.games.sum()))
    for pos in pr.POSITIONS:                                             # spread of value among qualifying players, for the OVR scale
        v = []
        for s in TRAIN:
            q = pr.estimate(pr.totals(df, s, 99, P['pos'][pos]['prior']), pos, dict(P['pos'][pos], value_sd=1.0))
            v += list(q.value - q.value.mean())
        P['pos'][pos]['value_sd'] = float(np.std(v))

    # ---- held-out test against the old position score
    import players as old_players
    say('\nHeld-out test, 2023-2025: rated after week 8, compared with what the player did from week 9 on.')
    say('Rank correlation with the future number (1 = the rating put players in exactly the right order).')
    say(f"{'':6s}{'future yardstick':44s} {'players':>7s} {'new':>6s} {'old':>6s}")
    yard = {'QB': [('EPA per play', 'epa', 'plays', 100)],
            'RB': [('EPA per game (rush + receiving)', 'rr_epa', 'games', 4), ('yards from scrimmage per game', 'scrim', 'games', 4), ('EPA per carry or target', 'rr_epa', 'opps', 40)],
            'WR': [('receiving EPA per game', 'receiving_epa', 'games', 4), ('receiving yards per game', 'receiving_yards', 'games', 4), ('EPA per target', 'receiving_epa', 'targets', 20)],
            'TE': [('receiving EPA per game', 'receiving_epa', 'games', 4), ('receiving yards per game', 'receiving_yards', 'games', 4)],
            'EDGE': [('points of plays made per game', 'made', 'games', 4), ('sacks per game', 'def_sacks', 'games', 4)],
            'DL': [('points of plays made per game', 'made', 'games', 4), ('sacks per game', 'def_sacks', 'games', 4)],
            'LB': [('points of plays made per game', 'made', 'games', 4), ('stops per game', 'stop', 'games', 4)],
            'CB': [('points of plays made per game', 'made', 'games', 4), ('passes defended + INT per game', 'pdi', 'games', 4)],
            'S': [('points of plays made per game', 'made', 'games', 4), ('passes defended + INT per game', 'pdi', 'games', 4)],
            'K': [('field goals above expected per attempt', 'fgoe', 'fg_att', 8), ('field-goal %', 'fg_made', 'fg_att', 8)]}
    summary = {}
    for pos in pr.POSITIONS:
        acc = {y[0]: ([], [], []) for y in yard[pos]}
        for s in TEST:
            new = pr.rankings(s, 9, P, df)
            if pos not in new:
                continue
            rows_s = [r for r in csv_rows(s) if int(r['week']) < 9]
            orig = old_players.load; old_players.load = lambda season, rows_s=rows_s: rows_s
            try:
                old = {(r['name'], r['team']): r['rating'] for r in old_players.rankings(s).get(pos, [])}
            finally:
                old_players.load = orig
            fut = pr.totals(df, s, 99, 0.0, from_week=9)
            fut['scrim'] = fut.rushing_yards + fut.receiving_yards
            fut['made'] = sum(pr.EVENT_PTS[e] * fut[pr.EV_COL[e]] for e in pr.EVENT_PTS)
            fut['pdi'] = fut.pd + fut.def_interceptions
            for r in new[pos]:
                o = old.get((r['name'], raw_team(r['team'])))
                if o is None or r['id'] not in fut.index:
                    continue
                f = fut.loc[r['id']]
                for label, num, den, mn in yard[pos]:
                    if f[den] >= mn:
                        a = acc[label]; a[0].append(r['value']); a[1].append(o); a[2].append(f[num] / f[den])
        for label, (a, o, y) in acc.items():
            if len(a) > 10:
                sn, so = spearman(a, y), spearman(o, y)
                summary[f'{pos} {label}'] = dict(players=len(a), new=round(sn, 3), old=round(so, 3))
                say(f'{pos:6s}{label:44s} {len(a):7d} {sn:6.3f} {so:6.3f}')
    wins = sum(v['new'] > v['old'] for v in summary.values())
    say(f'new rating ahead on {wins} of {len(summary)} yardsticks')
    P.update(made=str(__import__('datetime').date.today()), trained_on=[TRAIN[0], TRAIN[-1]], tested_on=list(TEST), event_points=pr.EVENT_PTS, test=summary)
    json.dump(P, open(pr.PARAMS, 'w'), indent=1)
    open(os.path.join(HERE, 'players_results.txt'), 'w').write('\n'.join(lines) + '\n')


def raw_team(t):
    return t


_ROWS = {}


def csv_rows(season):
    import csv, nflv
    if season not in _ROWS:
        _ROWS[season] = [r for r in csv.DictReader(open(nflv.get('player_week', season), encoding='utf-8')) if r['season_type'] == 'REG']
    return _ROWS[season]


if __name__ == '__main__':
    main()
