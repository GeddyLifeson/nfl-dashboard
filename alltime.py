"""All-time lists since 1999 (the first season with play-by-play): the best team seasons, franchise averages,
the best player seasons and the best careers, all from SAM's own ratings. Regular season only.

  python alltime.py     # writes site/alltime.json (a few minutes; run after a season ends or when the ratings change)

Team seasons: the fitted rating (teamrate.py) from that season's games alone, no carry-over from the year before.
Player seasons: the fitted value per game (playerrate.py) from that season alone, compared with the other
qualifying players at the position THAT season, so eras with more passing do not swamp earlier ones.
Careers: season value above a replacement-level regular (the 20th percentile of qualifiers), times games, summed.
Careers that began before 1999 only count from 1999 on.
"""
import json, os
from collections import defaultdict
import numpy as np
import pandas as pd
import nflv, playerrate as pr, teamrate
from teams import ABBRS, display_abbr, name

HERE = os.path.dirname(os.path.abspath(__file__))


def teams(last):
    sp = teamrate.load()
    R = teamrate.Ratings(range(teamrate.FIRST, last + 1), sp['k'], 0.0, 1.0)
    qb, _ = teamrate.qb_values()
    g = pd.read_csv(teamrate.ALL_GAMES)
    g = g[g.home_score.notna() & (g.season >= teamrate.FIRST)]
    rows = []
    for s in range(teamrate.FIRST, last + 1):
        gs = g[g.season == s]
        if not len(gs):
            continue
        q, rec = defaultdict(list), defaultdict(lambda: [0, 0, 0, 0.0])
        for r in gs.itertuples():
            m = r.home_score - r.away_score
            for t, x, lv in ((r.home_team, m, qb[r.game_id][0]), (r.away_team, -m, qb[r.game_id][1])):
                q[t].append(lv); rec[t][0 if x > 0 else 1 if x < 0 else 2] += 1; rec[t][3] += x
        pts = teamrate.team_points(R.features(s, 99), sp, {t: float(np.mean(v)) for t, v in q.items()})
        done = len(gs) >= 16 * 15
        for t in ABBRS:
            if t not in rec:
                continue
            w, l, ties, tot = rec[t]; n = w + l + ties
            v = pts[t]
            rows.append(dict(season=s, team=name(t), abbr=display_abbr(t), record=f'{w}-{l}' + (f'-{ties}' if ties else ''), margin=round(tot / n, 1),
                             rating=round(float(v['rating']), 1), off=round(float(v['off']), 1), **{'def': round(float(v['def']), 1)},
                             st=round(float(v['parts']['Special teams']), 1), complete=bool(done)))
    rows.sort(key=lambda r: -r['rating'])
    for i, r in enumerate(rows, 1):
        r['rank'] = i
    fr = defaultdict(list)
    for r in rows:
        if r['complete']:
            fr[r['abbr']].append(r)
    franchises = sorted((dict(team=v[0]['team'], abbr=a, seasons=len(v), avg=round(float(np.mean([x['rating'] for x in v])), 2),
                              best=f"{max(v, key=lambda x: x['rating'])['season']} ({max(v, key=lambda x: x['rating'])['rating']:+.1f})",
                              worst=f"{min(v, key=lambda x: x['rating'])['season']} ({min(v, key=lambda x: x['rating'])['rating']:+.1f})",
                              top5=sum(1 for x in v if x['season_rank'] <= 5) if 'season_rank' in v[0] else None) for a, v in fr.items()), key=lambda r: -r['avg'])
    for i, r in enumerate(franchises, 1):
        r['rank'] = i; r.pop('top5')
    return rows, franchises


def players(last):
    P = pr.load(); df = pr.all_weeks(pr.FIRST, last)
    seasons, careers = {}, {}
    for pos in pr.POSITIONS:
        rows, car = [], defaultdict(lambda: dict(total=0.0, seasons=0, games=0, peak=0, peak_season=None, name='', teams=[], first=9999, last=0))
        for s in range(pr.FIRST, last + 1):
            t = pr.totals(df, s, 99, 0.0)
            q = pr.estimate(t, pos, P['pos'][pos])
            if len(q) < 8:
                continue
            games_played = df[(df.season == s)].week.nunique()
            full = games_played >= 15
            z = (q.value - q.value.mean()) / (q.value.std() or 1.0)
            floor = np.quantile(q.value, 0.2)
            for pid, r in q.assign(z=z).iterrows():
                o = pr.ovr(r.z)
                tot = max(float(r.value - floor), 0.0) * r.cur_games
                c = car[pid]
                c['total'] += tot; c['seasons'] += 1; c['games'] += int(r.cur_games); c['name'] = r['name']
                c['first'], c['last'] = min(c['first'], s), max(c['last'], s)
                if display_abbr(r.team) not in c['teams']:
                    c['teams'].append(display_abbr(r.team))
                if o > c['peak'] or (o == c['peak'] and c['peak_season'] is None):
                    c['peak'], c['peak_season'] = o, s
                if r.cur_games >= (10 if full else 99):
                    rows.append(dict(season=s, name=r['name'], team=display_abbr(r.team), games=int(r.cur_games), ovr=o, z=round(float(r.z), 2),
                                     value=round(float(r.value), 2), **{k: v for k, v in pr.SHOW[pos](t.loc[pid]).items()}))
        rows.sort(key=lambda r: -r['z'])
        seasons[pos] = [dict(r, rank=i) for i, r in enumerate(rows[:150], 1)]
        cl = sorted(car.values(), key=lambda c: -c['total'])[:100]
        careers[pos] = [dict(rank=i, name=c['name'], teams=', '.join(c['teams'][:4]), years=f"{c['first']}-{c['last']}", seasons=c['seasons'], games=c['games'],
                             total=round(c['total'], 1), peak=c['peak'], peak_season=c['peak_season']) for i, c in enumerate(cl, 1)]
    return seasons, careers


def main():
    last = nflv.current_season()
    teamrate.ensure_all_games()
    t_rows, franchises = teams(last)
    p_seasons, p_careers = players(last)
    out = dict(made=str(pd.Timestamp.today().date()), first=teamrate.FIRST, last=last, team_seasons=t_rows, franchises=franchises,
               player_seasons=p_seasons, player_careers=p_careers)
    json.dump(out, open(os.path.join(HERE, 'site', 'alltime.json'), 'w'))
    print('team seasons', len(t_rows), 'best:', [(r['season'], r['abbr'], r['record'], r['rating']) for r in t_rows[:10]])
    print('worst:', [(r['season'], r['abbr'], r['record'], r['rating']) for r in t_rows[-5:]])
    print('franchises:', [(r['abbr'], r['avg']) for r in franchises[:5]], '...', [(r['abbr'], r['avg']) for r in franchises[-3:]])
    for pos in pr.POSITIONS:
        print(pos, 'seasons:', [(r['season'], r['name'], r['ovr']) for r in p_seasons[pos][:6]])
        print(pos, 'careers:', [(r['name'], r['total']) for r in p_careers[pos][:8]])


if __name__ == '__main__':
    main()
