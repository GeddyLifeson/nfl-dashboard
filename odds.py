"""NFL odds for the dashboard's parlay builder: every major US book's price plus Pinnacle's
vig-free line as the fair chance. Writes site/odds.json.

Needs the ODDS_API_KEY secret (GitHub: Settings > Secrets and variables > Actions). Without it,
or when the account is low on credits, it leaves the last file alone. One call asks for ten named
books, which the Odds API bills as one region: 3 credits for moneylines, spreads and totals.
Fetches at most once a day Tuesday-Saturday and every 5 hours on Sunday and Monday (US Central).
Run: python odds.py [--force]
"""
import datetime, json, os, statistics, sys, urllib.request
from zoneinfo import ZoneInfo

OUT = 'site/odds.json'
BOOKS = {'pinnacle': 'Pinnacle', 'draftkings': 'DraftKings', 'fanduel': 'FanDuel', 'betmgm': 'BetMGM',
         'williamhill_us': 'Caesars', 'espnbet': 'ESPN BET', 'betrivers': 'BetRivers', 'fanatics': 'Fanatics',
         'hardrockbet': 'Hard Rock', 'ballybet': 'Bally Bet'}
URL = ('https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?apiKey={key}&markets=h2h,spreads,totals'
       '&oddsFormat=decimal&dateFormat=iso&bookmakers=' + ','.join(BOOKS))
KEEP_CREDITS = 150          # leave this many for SAM's own scheduled runs
MARKET = {'h2h': 'ML', 'spreads': 'spread', 'totals': 'total'}


def due(prev, now):
    if not prev:
        return True
    age = (now - datetime.datetime.fromisoformat(prev['updated'])).total_seconds() / 3600
    game_days = now.astimezone(ZoneInfo('America/Chicago')).weekday() in (6, 0)
    return age >= (5 if game_days else 24) and prev.get('remaining', 999) > KEEP_CREDITS


def fair(prices):
    """{side: chance} from Pinnacle's two prices, else the median vig-free chance across books."""
    if all('Pinnacle' in p for p in prices.values()):
        inv = {s: 1 / p['Pinnacle'] for s, p in prices.items()}
        z = sum(inv.values())
        return {s: v / z for s, v in inv.items()}, 'pinnacle'
    sides = list(prices)
    books = set.intersection(*(set(p) for p in prices.values())) - {'Pinnacle'}
    if not books:
        return None, None
    devig = {s: [] for s in sides}
    for b in books:
        z = sum(1 / prices[s][b] for s in sides)
        for s in sides:
            devig[s].append(1 / prices[s][b] / z)
    med = {s: statistics.median(v) for s, v in devig.items()}
    z = sum(med.values())
    return {s: v / z for s, v in med.items()}, 'consensus'


def legs(event):
    """Flat leg list: one entry per market, side and line, with fair chance and each US book's price."""
    groups = {}                                     # (market, line key) -> {side: {book: price}}
    for bk in event.get('bookmakers', []):
        book = BOOKS.get(bk['key'])
        if not book:
            continue
        for m in bk['markets']:
            kind = MARKET.get(m['key'])
            for o in m['outcomes']:
                if kind == 'spread':
                    key, side, line = (kind, abs(o['point'])), o['name'], o['point']
                elif kind == 'total':
                    key, side, line = (kind, o['point']), o['name'], o['point']
                else:
                    key, side, line = (kind, None), o['name'], None
                g = groups.setdefault(key, {})
                g.setdefault((side, line), {})[book] = o['price']
    out = []
    for (kind, _), sides in groups.items():
        if len(sides) != 2:
            continue                                # e.g. one book's -3 against another's +3.5
        f, src = fair({k: v for k, v in sides.items()})
        for (side, line), prices in sides.items():
            us = {b: p for b, p in prices.items() if b != 'Pinnacle'}
            if not us or not f:
                continue
            label = side if kind == 'ML' else f"{side} {line:+g}" if kind == 'spread' else f"{side} {line:g}"
            out.append(dict(key=f"{event['id']}|{kind}|{label}", market=kind, side=side, line=line, label=label,
                            fair=round(f[(side, line)], 4), src=src, books=us))
    return out


def main(force=False):
    key = os.getenv('ODDS_API_KEY')
    prev = json.load(open(OUT)) if os.path.exists(OUT) else None
    now = datetime.datetime.now(datetime.timezone.utc)
    if not key:
        print('odds: no ODDS_API_KEY secret, parlay builder keeps its last odds')
        return
    if not force and not due(prev, now):
        print('odds: not due yet (or credits low), keeping', prev and prev['updated'])
        return
    with urllib.request.urlopen(URL.format(key=key), timeout=60) as r:
        events = json.load(r)
        remaining = int(float(r.headers.get('x-requests-remaining') or 0))
    games = []
    for e in sorted(events, key=lambda e: e['commence_time']):
        ls = legs(e)
        if ls:
            games.append(dict(id=e['id'], home=e['home_team'], away=e['away_team'], start=e['commence_time'], legs=ls))
    data = dict(updated=now.isoformat(timespec='seconds'), remaining=remaining, games=games)
    with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))
    os.replace(OUT + '.tmp', OUT)
    print(f'odds: {len(games)} games, {sum(len(g["legs"]) for g in games)} legs, {remaining} credits left')


if __name__ == '__main__':
    main('--force' in sys.argv)
