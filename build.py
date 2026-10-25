"""Build the weekly rankings from the tested model. Writes league_data.json and history/<season>-w<week>.json.
Run: python fetch.py && python build.py   (python backtest.py first when the settings should be re-tuned)"""
import datetime, glob, json, os
import numpy as np
from model import Engine, load_games, load_params, predict, win_prob
from stats import log_tables, players, team_stats
from teams import ABBRS, IDX, display_abbr, name


def lineup_adjustments(season):
    """{team: dict(points, notes)}: starting-QB changes plus injuries, in points per game (negative =
    weaker than the results-based rating assumes). Empty if the injury feeds can't be reached."""
    import injuries, qb_adjust
    out = {}
    try:
        qb = qb_adjust.adjustments(season)
        inj = injuries.adjustments(season)
    except Exception as e:                       # predictions still work from results alone
        print(f"(lineup adjustments skipped: {e})")
        return out
    for t in set(qb) | set(inj):
        notes = [qb[t]['note']] if t in qb else []
        notes += [f"{n} ({pos}, {st})" for n, pos, st, _ in inj.get(t, {}).get('players', [])[:5]]
        out[t] = dict(points=round(qb.get(t, {}).get('points', 0.0) + inj.get(t, {}).get('points', 0.0), 1),
                      qb=qb.get(t, {}).get('points', 0.0), injuries=inj.get(t, {}).get('points', 0.0), notes=notes)
    return out


def short_date(iso):
    d = datetime.date.fromisoformat(iso)
    return f"{d:%b} {d.day}"


def standings(played, rating):
    """Wins, losses, ties, games and point margin per team, plus strength of schedule
    as the average current rating of the opponents faced."""
    out = {t: dict(w=0, l=0, t=0, gp=0, pts=0) for t in ABBRS}
    opp = {t: [] for t in ABBRS}
    for g in played:
        for team, other, m in ((g['home'], g['away'], g['margin']), (g['away'], g['home'], -g['margin'])):
            s = out[team]
            s['gp'] += 1; s['pts'] += m
            s['w' if m > 0 else 'l' if m < 0 else 't'] += 1
            opp[team].append(rating[IDX[other]])
    return out, {t: float(np.mean(opp[t])) for t in ABBRS}


def order_of(rating):
    return sorted(ABBRS, key=lambda t: (-rating[IDX[t]], t))


def grade_history(games):
    """Score every archived pick whose game has since been played."""
    result = {g['id']: g['margin'] for g in games if g['margin'] is not None}
    seen, right, wrong = set(), 0, 0
    for path in sorted(glob.glob('history/*.json')):
        for pk in json.load(open(path)).get('upcoming', []):
            m = result.get(pk['id'])
            if pk['id'] in seen or m is None or m == 0:
                continue
            seen.add(pk['id'])
            right += (pk['home_margin'] > 0) == (m > 0)
            wrong += (pk['home_margin'] > 0) != (m > 0)
    return dict(right=right, wrong=wrong)


def main():
    games, p = load_games(), load_params()
    e = Engine(games, p)
    season = max(g['season'] for g in games if g['margin'] is not None)
    played = e.played(season)
    week = max(g['week'] for g in played)
    raw = e.ratings(season, 99)
    rating = p['slope'] * raw
    prev = order_of(p['slope'] * e.ratings(season, week)) if week > 1 else None
    order = order_of(rating)
    st, sos = standings(played, rating)
    gp = {t: st[t]['gp'] for t in ABBRS}
    log_tables(gp)
    stats, qb = team_stats(gp), players()
    sos_rank = {t: i for i, t in enumerate(sorted(ABBRS, key=lambda t: -sos[t]), 1)}

    teams = []
    for pos, t in enumerate(order, 1):
        s = st[t]
        o = [x['rank'] for x in stats[t] if x['side'] == 'O']
        d = [x['rank'] for x in stats[t] if x['side'] == 'D']
        prev_pos = prev.index(t) + 1 if prev else pos
        teams.append(dict(
            team=name(t), abbr=display_abbr(t), pos=pos, prev_pos=prev_pos, move=prev_pos - pos,
            record=f"{s['w']}-{s['l']}" + (f"-{s['t']}" if s['t'] else ''), gp=s['gp'],
            rating=round(float(rating[IDX[t]]), 2), margin=round(s['pts'] / s['gp'], 1),
            sos=round(sos[t], 1), sos_rank=sos_rank[t],
            off_avg=round(float(np.mean(o)), 1), def_avg=round(float(np.mean(d)), 1),
            stats=stats[t], player=qb[t]))

    lineup = lineup_adjustments(season)
    upcoming = []
    for g in games:
        if g['season'] == season and g['margin'] is None and g['week'] <= week + 1:
            base = predict(raw, g, p)
            ah, aa = lineup.get(g['home'], {}), lineup.get(g['away'], {})
            m = base + ah.get('points', 0.0) - aa.get('points', 0.0)
            upcoming.append(dict(id=g['id'], week=g['week'], day=g['day'], away=display_abbr(g['away']), home=display_abbr(g['home']),
                                 neutral=g['neutral'], home_margin=round(m, 1), home_win=round(win_prob(m, p['sigma']), 3),
                                 line=g['spread'], results_margin=round(base, 1),
                                 lineup={display_abbr(g['home']): ah, display_abbr(g['away']): aa}))

    out = dict(season=season, week=week, through=short_date(max(g['day'] for g in played)),
               not_yet_played=[f"{name(g['away'])} at {name(g['home'])}" for g in games
                               if g['season'] == season and g['week'] <= week and g['margin'] is None],
               model={k: p[k] for k in ('k', 'carry', 'hfa', 'slope', 'sigma', 'tuned_on', 'tested', 'confidence_tiers')},
               teams=teams, upcoming=upcoming)
    os.makedirs('history', exist_ok=True)
    path = f'history/{season}-w{week:02d}.json'
    # Rebuilding mid-week must not drop picks archived earlier (e.g. Monday night's), or they never get graded.
    archived = json.load(open(path)).get('upcoming', []) if os.path.exists(path) else []
    keep = [p for p in archived if p['id'] not in {u['id'] for u in upcoming}]
    json.dump(dict(season=season, week=week, ranks={x['abbr']: x['pos'] for x in teams},
                   ratings={x['abbr']: x['rating'] for x in teams}, upcoming=keep + upcoming),
              open(path, 'w'), indent=1)
    out['pick_record'] = grade_history(games)
    json.dump(out, open('league_data.json', 'w'), indent=1)

    print(f"{season} week {week}, games through {out['through']}")
    for x in teams:
        mv = f"{'+' if x['move'] > 0 else ''}{x['move']}" if x['move'] else '='
        print(f"{x['pos']:2d} {x['abbr']:4s} {x['record']:6s} {x['rating']:+5.1f}  {mv:>3s}  sos {x['sos']:+.1f}")
    for u in upcoming:
        fav, by = (u['home'], u['home_margin']) if u['home_margin'] > 0 else (u['away'], -u['home_margin'])
        line = '' if u['line'] is None else f"  (line: {u['home']} {-u['line']:+g})"
        print(f"wk{u['week']} {u['away']} at {u['home']}: {fav} by {by:.1f}{line}")


if __name__ == '__main__':
    main()
