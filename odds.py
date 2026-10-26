"""NFL odds for the dashboard's parlay builder: every major US sportsbook and US prediction
market / exchange (Kalshi, Polymarket US, Novig, ProphetX, BetOpenly), plus Pinnacle's vig-free
line as the fair chance. Writes site/odds.json.

Needs the ODDS_API_KEY secret (GitHub: Settings > Secrets and variables > Actions). Without it,
or when the account is low on credits, it leaves the last file alone. One call names 15 books,
which the Odds API bills as two regions: 6 credits for moneylines, spreads and totals. Fetches
every other day, plus every 5 hours on Sunday (US Central): about 6 calls, 36 credits a week, so
the free 500-credit month still covers SAM's own picks, closes and alerts.
Run: python odds.py [--force]
"""
import datetime, json, os, statistics, sys, urllib.request
from zoneinfo import ZoneInfo

OUT = 'site/odds.json'
BOOKS = {'pinnacle': 'Pinnacle', 'draftkings': 'DraftKings', 'fanduel': 'FanDuel', 'betmgm': 'BetMGM',
         'williamhill_us': 'Caesars', 'espnbet': 'ESPN BET', 'betrivers': 'BetRivers', 'fanatics': 'Fanatics',
         'hardrockbet': 'Hard Rock', 'ballybet': 'Bally Bet',
         # US prediction markets and exchanges (the Odds API's us_ex region); prices are before their fees
         'kalshi': 'Kalshi', 'polymarket_us': 'Polymarket US', 'novig': 'Novig', 'prophetx': 'ProphetX', 'betopenly': 'BetOpenly'}
EXCHANGES = {'Kalshi', 'Polymarket US', 'Novig', 'ProphetX', 'BetOpenly', 'Gemini'}   # Gemini comes free from its own API
URL = ('https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds?apiKey={key}&markets=h2h,spreads,totals'
       '&oddsFormat=decimal&dateFormat=iso&bookmakers=' + ','.join(BOOKS))
KEEP_CREDITS = 150          # leave this many for SAM's own scheduled runs
MARKET = {'h2h': 'ML', 'spreads': 'spread', 'totals': 'total'}


def due(prev, now):
    if not prev:
        return True
    age = (now - datetime.datetime.fromisoformat(prev['updated'])).total_seconds() / 3600
    sunday = now.astimezone(ZoneInfo('America/Chicago')).weekday() == 6
    return age >= (5 if sunday else 44) and prev.get('remaining', 999) > KEEP_CREDITS


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
        if prev:                                    # Gemini is free: refresh its prices on every run
            n = add_gemini(prev['games'])
            prev['gemini_updated'] = now.isoformat(timespec='seconds')
            save(prev)
            print(f'odds: books not due yet (or credits low); Gemini refreshed on {n} legs')
        return
    with urllib.request.urlopen(URL.format(key=key), timeout=60) as r:
        events = json.load(r)
        remaining = int(float(r.headers.get('x-requests-remaining') or 0))
    games = []
    for e in sorted(events, key=lambda e: e['commence_time']):
        ls = legs(e)
        if ls:
            games.append(dict(id=e['id'], home=e['home_team'], away=e['away_team'], start=e['commence_time'], legs=ls))
    n = add_gemini(games)
    stamp = now.isoformat(timespec='seconds')
    save(dict(updated=stamp, gemini_updated=stamp, remaining=remaining, exchanges=sorted(EXCHANGES), games=games))
    print(f'odds: {len(games)} games, {sum(len(g["legs"]) for g in games)} legs, Gemini on {n}, {remaining} credits left')


def save(data):
    with open(OUT + '.tmp', 'w', encoding='utf-8') as f:
        json.dump(data, f, separators=(',', ':'))
    os.replace(OUT + '.tmp', OUT)


GEMINI = 'https://api.gemini.com/v1/prediction-markets/events?status=active&search=NFL&limit=100&offset={}'


def gemini_events():
    """Gemini Predictions' NFL game markets (keyless public API): moneyline (-M), spread (-S), total (-T)."""
    out = []
    for offset in range(0, 1000, 100):
        try:
            with urllib.request.urlopen(GEMINI.format(offset), timeout=30) as r:
                page = json.load(r).get('data', [])
        except Exception as e:                      # the builder still works from the books alone
            print('odds: Gemini unavailable:', e)
            break
        out += [e for e in page if e.get('ticker', '').rsplit('-', 1)[-1] in ('M', 'S', 'T')]
        if len(page) < 100:
            break
    return out


def add_gemini(games):
    """Add Gemini's ask prices (decimal = 1 / ask, before fees) to legs the books already price.
    'Dallas wins by more than 3.5 points' is Dallas -3.5 (yes) and the other team +3.5 (no);
    'More than 43.5 points scored' is Over 43.5 (yes) and Under 43.5 (no)."""
    import re
    by_teams = {}
    for g in games:
        by_teams[(g['home'], g['away'])] = g
        for l in g['legs']:
            l['books'].pop('Gemini', None)
    added = 0
    for ev in gemini_events():
        teams = {p['side']: p['name'] for p in ev.get('participants', [])}
        g = by_teams.get((teams.get('home'), teams.get('away')))
        if not g:
            continue
        city = {t.rsplit(' ', 1)[0]: t for t in (g['home'], g['away'])}   # 'Dallas' -> 'Dallas Cowboys'
        name_of = lambda label: next((full for c, full in city.items() if label.startswith(c) or full.startswith(label)), None)
        other = lambda team: g['away'] if team == g['home'] else g['home']
        kind = ev['ticker'].rsplit('-', 1)[-1]
        for c in ev.get('contracts', []):
            buy = (c.get('prices') or {}).get('buy') or {}
            prices = []
            if kind == 'M':
                team = name_of(c['label'])
                prices = [(f'ML|{team}', buy.get('yes'))] if team else []
            elif kind == 'S':
                m = re.match(r'(.+?) wins by more than ([\d.]+) points', c['label'])
                team = m and name_of(m.group(1))
                if team:
                    pts = float(m.group(2))
                    prices = [(f'spread|{team} {-pts:+g}', buy.get('yes')), (f'spread|{other(team)} {pts:+g}', buy.get('no'))]
            elif kind == 'T':
                m = re.match(r'More than ([\d.]+) points', c['label'])
                if m:
                    prices = [(f'total|Over {float(m.group(1)):g}', buy.get('yes')), (f'total|Under {float(m.group(1)):g}', buy.get('no'))]
            for suffix, ask in prices:
                leg = next((l for l in g['legs'] if l['key'] == f"{g['id']}|{suffix}"), None)
                if leg and ask and 0.01 < float(ask) < 0.99:
                    leg['books']['Gemini'] = round(1 / float(ask), 3)
                    added += 1
    return added


if __name__ == '__main__':
    main('--force' in sys.argv)
