"""Start/sit and waiver pickups for Marlon's Sleeper league, from SAM's projections averaged with Sleeper's.

Inputs in cache/sleeper/ (pulled read-only from api.sleeper.app on the user's PC): league_<id>.json,
league_<id>_rosters.json, league_<id>_users.json, players_compact.csv, proj_<season>_w<week>.json, trending_add.json.
SAM's own numbers come from fantasy.week() and fantasy_kdst.week().

The backtest (fantasy_vs_sleeper.py) found SAM alone slightly behind Sleeper's projections for RB and WR, about even
for QB and TE, and the average of the two about as good as or a little better than either, so the average is used.

Usage: python fantasy_league.py [WEEK]   (default: the next week with unplayed games)   -> prints the report and writes fantasy_league_w<WEEK>.json
"""
import csv, glob, json, os, re, sys
import fantasy, fantasy_kdst, sleeper
from lineup import norm

SL = os.path.join(fantasy.HERE, 'cache', 'sleeper')
LEAGUE = '1394022273928671232'
OWNER = 'geddylifeson'
TEAM_FIX = {'LAR': 'LA', 'JAC': 'JAX', 'WSH': 'WAS'}
FLEX_OK = {'RB', 'WR', 'TE'}


def load(name):
    return json.load(open(os.path.join(SL, name), encoding='utf-8'))


def run(week=None, season=None, pull=True):
    """Best lineup, bench and waiver options. pull=True refreshes the Sleeper files first; if Sleeper can't be
    reached, the files already in cache/sleeper/ are used and the reason is kept in 'warning'."""
    warning, lid = None, None
    if pull:
        try:
            _, _, lid = sleeper.refresh()
        except Exception as e:
            warning = f'Sleeper not reachable, using saved league files: {type(e).__name__}: {e}'
    if lid is None:
        found = [m.group(1) for p in glob.glob(os.path.join(SL, 'league_*.json'))
                 if (m := re.fullmatch(r'league_(\d+)\.json', os.path.basename(p)))]
        lid = found[0] if found else LEAGUE
    league = load(f'league_{lid}.json')
    rosters = load(f'league_{lid}_rosters.json')
    users = {u['user_id']: u['display_name'] for u in load(f'league_{lid}_users.json')}
    players = {r['player_id']: r for r in csv.DictReader(open(os.path.join(SL, 'players_compact.csv'), encoding='utf-8'))}
    season = season or int(league.get('season') or fantasy.nflv.current_season())
    sam = fantasy.week(season, week)
    week = sam['week']
    proj_path = os.path.join(SL, f'proj_{season}_w{week}.json')
    sl_rows = load(f'proj_{season}_w{week}.json') if os.path.exists(proj_path) else []
    if not sl_rows:
        warning = (warning + '; ' if warning else '') + f'no Sleeper projections for week {week}, SAM only'
    sl_proj = {r['player_id']: (r.get('stats') or {}).get('pts_ppr') for r in sl_rows}
    for r in sl_rows:   # injury tags from the projection file are fresher than the daily player list
        if r['player_id'] in players and (r.get('player') or {}).get('injury_status') is not None:
            players[r['player_id']]['injury_status'] = r['player']['injury_status'] or ''
    by_gsis = {p['id']: p for p in sam['players']}
    by_name = {(norm(p['name']), p['team']): p for p in sam['players']}
    kd = fantasy_kdst.week(season, week)
    k_by_name = {(norm(p['name']), p['team']): p for p in kd if p['pos'] == 'K'}
    d_by_team = {p['team']: p for p in kd if p['pos'] == 'DEF'}
    games = fantasy.schedule()

    def row(sid):
        meta = players.get(sid, {'full_name': sid, 'position': 'DEF', 'team': sid, 'gsis_id': '', 'injury_status': ''})
        pos, team = meta['position'], TEAM_FIX.get(meta['team'], meta['team'])
        g = games.get((season, week, team))
        mine = None
        if pos == 'DEF':
            mine = d_by_team.get(TEAM_FIX.get(sid, sid))
        elif pos == 'K':
            mine = k_by_name.get((norm(meta['full_name']), team))
        else:
            mine = by_gsis.get((meta['gsis_id'] or '').strip()) or by_name.get((norm(meta['full_name']), team))
        s_sam = mine['proj'] if mine else None
        s_sl = sl_proj.get(sid)
        s_sl = float(s_sl) if s_sl is not None else None
        both = [v for v in (s_sam, s_sl) if v is not None]
        status = meta.get('injury_status') or ''
        proj = sum(both) / len(both) if both else 0.0
        if not g:
            proj, opp = 0.0, 'BYE'
        else:
            opp = ('vs ' if g['home'] else '@ ') + g['opp']
        if status in ('Out', 'IR', 'PUP', 'Sus'):
            proj = 0.0
        return {'id': sid, 'name': meta['full_name'], 'pos': pos, 'team': team, 'opp': opp, 'status': status,
                'played': bool(g and g['played']), 'sam': s_sam, 'sleeper': s_sl, 'proj': round(proj, 1)}

    mine = next(r for r in rosters if users.get(r['owner_id'], '').lower() == OWNER)
    taxi = set(mine.get('taxi') or [])
    roster = [row(p) for p in mine['players'] if p not in taxi]
    current = set(mine['starters'])

    # best lineup: locked players (game already played) stay where they are if they started
    slots = [s for s in league['roster_positions'] if s != 'BN']
    used, lineup = set(), []
    for slot in slots:
        ok = FLEX_OK if slot == 'FLEX' else {slot}
        cands = [p for p in roster if p['pos'] in ok and p['id'] not in used]
        locked = [p for p in cands if p['played'] and p['id'] in current]
        pick = locked[0] if locked else max((p for p in cands if not p['played']), key=lambda p: p['proj'], default=None)
        if pick:
            used.add(pick['id'])
            lineup.append((slot, pick))
    # fill FLEX properly: re-run so FLEX takes the best leftover after fixed slots
    fixed = [(s, p) for s, p in lineup if s != 'FLEX']
    taken = {p['id'] for _, p in fixed}
    flex_pool = sorted((p for p in roster if p['pos'] in FLEX_OK and p['id'] not in taken and (not p['played'] or p['id'] in current)),
                       key=lambda p: -p['proj'])
    locked_flex = [p for p in flex_pool if p['played'] and p['id'] in current]
    flex = (locked_flex + [p for p in flex_pool if not p['played']])[:slots.count('FLEX')]
    lineup = fixed + [('FLEX', p) for p in flex]
    best_ids = {p['id'] for _, p in lineup}
    changes = {'bench': [p for p in roster if p['id'] in current and p['id'] not in best_ids],
               'start': [p for _, p in lineup if p['id'] not in current]}

    rostered = {p for r in rosters for p in (r.get('players') or [])}
    pool = []
    for sid, meta in players.items():
        if sid in rostered or meta['position'] not in ('QB', 'RB', 'WR', 'TE', 'K', 'DEF') or not meta['team']:
            continue
        r = row(sid)
        if r['proj'] > 0 and not r['played']:
            pool.append(r)
    trending = {t['player_id']: t['count'] for t in load('trending_add.json')}
    for r in pool:
        r['trending_adds'] = trending.get(r['id'], 0)
    waivers = {pos: sorted([r for r in pool if r['pos'] == pos], key=lambda r: -r['proj'])[:6] for pos in ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')}
    worst = {pos: min([p['proj'] for p in roster if p['pos'] == pos] or [0]) for pos in ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')}
    out = {'season': season, 'week': week, 'league': league.get('name'), 'scoring_note': 'full PPR, 4-pt pass TD, -1 INT',
           'record': f"{mine['settings'].get('wins', 0)}-{mine['settings'].get('losses', 0)}", 'warning': warning, 'lineup': [{'slot': s, **p} for s, p in lineup],
           'lineup_total': round(sum(p['proj'] for _, p in lineup), 1), 'changes': changes,
           'bench': sorted([p for p in roster if p['id'] not in best_ids], key=lambda p: -p['proj']),
           'taxi': [row(p) for p in taxi], 'waivers': waivers, 'worst_on_roster': worst,
           # SAM's own projection for every player this week (top 40 per position), for the rankings table
           'rankings': {pos: [{k: p.get(k) for k in ('name', 'team', 'opp', 'proj', 'status')}
                              for p in [q for q in sam['players'] + kd if q['pos'] == pos and q['opp'] != 'BYE'][:40]]
                        for pos in ('QB', 'RB', 'WR', 'TE', 'K', 'DEF')}}
    json.dump(out, open(os.path.join(fantasy.HERE, f'fantasy_league_w{week}.json'), 'w'), indent=1)
    return out


def show(o):
    fmt = lambda p: (f"{p['name']} ({p['pos']} {p['team']} {p['opp']}{', ' + p['status'] if p['status'] else ''}"
                     f"{', already played' if p['played'] else ''}) {p['proj']:.1f}"
                     f" [SAM {p['sam'] if p['sam'] is not None else '-'} / Sleeper {p['sleeper'] if p['sleeper'] is not None else '-'}]")
    print(f"Week {o['week']} best lineup, projected {o['lineup_total']}:")
    for p in o['lineup']:
        print(f"  {p['slot']:4s} {fmt(p)}")
    print('Changes: start ' + (', '.join(p['name'] for p in o['changes']['start']) or 'none')
          + ' | bench ' + (', '.join(p['name'] for p in o['changes']['bench']) or 'none'))
    print('Bench:')
    for p in o['bench']:
        print('  ' + fmt(p))
    print('Waiver options (this week):')
    for pos, rs in o['waivers'].items():
        print(f"  {pos}: " + '; '.join(f"{r['name']} {r['team']} {r['proj']:.1f}" + (f" (trending +{r['trending_adds']})" if r['trending_adds'] else '') for r in rs))


if __name__ == '__main__':
    show(run(int(sys.argv[1]) if len(sys.argv) > 1 else None))
