"""Read-only pull of Marlon's Sleeper league from Sleeper's free public API into cache/sleeper/.

Called by fantasy_league.py on every dashboard refresh. Never writes to Sleeper. The full player list (~15 MB) is
re-downloaded at most once a day. Some sandboxes block api.sleeper.app; then the files already in cache/sleeper/
are used as they are.
"""
import csv, json, os, time, urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
SL = os.path.join(HERE, 'cache', 'sleeper')
API = 'https://api.sleeper.app/v1/'
USER = 'geddylifeson'
FANTASY_POS = {'QB', 'RB', 'WR', 'TE', 'K', 'DEF'}


def get(url, timeout=60):
    req = urllib.request.Request(url, headers={'User-Agent': 'nfl-dashboard'})
    return json.loads(urllib.request.urlopen(req, timeout=timeout).read())


def save(name, obj):
    os.makedirs(SL, exist_ok=True)
    path = os.path.join(SL, name)
    with open(path + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(obj, f)
    os.replace(path + '.tmp', path)


def refresh(league_name=None):
    """Download state, league, rosters, users, players (daily), trending adds and this week's projections.
    Returns (season, week, league_id)."""
    state = get(API + 'state/nfl')
    save('state_nfl.json', state)
    season, week = int(state['season']), int(state['week'] or 1)
    user = get(API + f'user/{USER}')
    leagues = get(API + f"user/{user['user_id']}/leagues/nfl/{season}")
    save(f'leagues_{season}.json', leagues)
    if not leagues:
        raise RuntimeError(f'no Sleeper leagues for {USER} in {season}')
    league = next((l for l in leagues if l['name'] == league_name), leagues[0])
    lid = league['league_id']
    save(f'league_{lid}.json', get(API + f'league/{lid}'))
    save(f'league_{lid}_rosters.json', get(API + f'league/{lid}/rosters'))
    save(f'league_{lid}_users.json', get(API + f'league/{lid}/users'))
    save('trending_add.json', get(API + 'players/nfl/trending/add?lookback_hours=48&limit=50'))
    compact = os.path.join(SL, 'players_compact.csv')
    if not os.path.exists(compact) or time.time() - os.path.getmtime(compact) > 24 * 3600:
        players = get(API + 'players/nfl', timeout=300)
        with open(compact + '.tmp', 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['player_id', 'full_name', 'position', 'team', 'gsis_id', 'status', 'injury_status', 'age', 'years_exp'])
            for pid, p in players.items():
                if p.get('position') in FANTASY_POS:
                    name = p.get('full_name') or f"{p.get('first_name', '')} {p.get('last_name', '')}".strip()
                    w.writerow([pid, name, p.get('position'), p.get('team') or '', (p.get('gsis_id') or '').strip(),
                                p.get('status') or '', p.get('injury_status') or '', p.get('age') or '', p.get('years_exp') or ''])
        os.replace(compact + '.tmp', compact)
    pos = '&'.join(f'position[]={p}' for p in ('QB', 'RB', 'WR', 'TE', 'K', 'DEF'))
    for wk in (week, week + 1):
        try:
            save(f'proj_{season}_w{wk}.json', get(f'https://api.sleeper.app/projections/nfl/{season}/{wk}?season_type=regular&{pos}'))
        except Exception:
            if wk == week:
                raise
    return season, week, lid


if __name__ == '__main__':
    print(refresh())
