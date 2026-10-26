"""Cached downloads from nflverse releases (free, no key). Files for finished seasons are kept;
the current season is refreshed after `hours`."""
import datetime, os, time, urllib.request

BASE = 'https://github.com/nflverse/nflverse-data/releases/download/'
FILES = {'pbp': 'pbp/play_by_play_{}.csv.gz', 'injuries': 'injuries/injuries_{}.csv',
         'snaps': 'snap_counts/snap_counts_{}.csv', 'depth': 'depth_charts/depth_charts_{}.csv',
         'player_week': 'stats_player/stats_player_week_{}.csv'}


def current_season():
    today = datetime.date.today()
    return today.year if today.month >= 8 else today.year - 1


def get(kind, season, hours=6):
    """Local path to the nflverse file, downloading it if missing or (current season) stale."""
    name = FILES[kind].format(season)
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'cache', name.split('/')[1])
    stale = season >= current_season() and (not os.path.exists(path) or time.time() - os.path.getmtime(path) > hours * 3600)
    if not os.path.exists(path) or stale:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = urllib.request.urlopen(BASE + name, timeout=300).read()
        with open(path + '.tmp', 'wb') as f:
            f.write(data)
        os.replace(path + '.tmp', path)
    return path
