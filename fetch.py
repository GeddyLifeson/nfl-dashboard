"""Download the nflverse schedule (scores, home/away, neutral sites, betting lines) to data/games.csv."""
import csv, datetime, hashlib, io, json, os, sys, urllib.request
from teams import TEAMS

URL = 'https://github.com/nflverse/nfldata/raw/master/data/games.csv'
FIRST_SEASON = 2021
KEEP = ['game_id', 'season', 'week', 'gameday', 'weekday', 'gametime', 'away_team', 'away_score', 'home_team', 'home_score',
        'location', 'spread_line', 'away_rest', 'home_rest', 'div_game', 'roof', 'temp', 'wind', 'away_qb_id', 'home_qb_id']


def check(rows):
    """Refuse a download that is not a full schedule, so a bad fetch never replaces good data."""
    problems = []
    seasons = sorted({int(r['season']) for r in rows})
    for s in seasons:
        n = sum(int(r['season']) == s for r in rows)
        if n < 256:
            problems.append(f'{s} has {n} games')
    unknown = {r[k] for r in rows for k in ('home_team', 'away_team')} - set(TEAMS)
    if unknown:
        problems.append(f'unknown teams {sorted(unknown)}')
    if not seasons or seasons[0] != FIRST_SEASON:
        problems.append(f'seasons {seasons}')
    return problems


def main():
    raw = urllib.request.urlopen(URL, timeout=60).read()
    rows = [r for r in csv.DictReader(io.StringIO(raw.decode('utf-8')))
            if r['game_type'] == 'REG' and int(r['season']) >= FIRST_SEASON]
    problems = check(rows)
    if problems:
        sys.exit('refused: ' + '; '.join(problems))
    os.makedirs('data', exist_ok=True)
    with open('data/games.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, KEEP, extrasaction='ignore')
        w.writeheader()
        w.writerows(rows)
    played = sum(bool(r['home_score']) for r in rows)
    rec = dict(table='games', source=URL, fetched=datetime.date.today().isoformat(),
               sha256=hashlib.sha256(raw).hexdigest()[:16], rows=len(rows), played=played)
    with open('data/ledger.jsonl', 'a') as f:
        f.write(json.dumps(rec) + '\n')
    print(f"{len(rows)} games, {played} played")


if __name__ == '__main__':
    main()
