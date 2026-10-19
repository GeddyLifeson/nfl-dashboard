"""Search for factors the betting line misses. Run: python edge.py

The target is the line's error: home margin minus the closing spread. Each factor's direction is learned
on 2021-2024 only, then it is graded against the spread on 2025-2026. A bet at standard -110 odds needs
52.4% to break even, and with many factors tried at once, one will look good by luck, so a factor only
counts as an edge if it clears 52.4% out of sample with a p-value under 0.05 divided by the number of factors."""
import csv, json, math
import numpy as np
from model import Engine, load_games, load_params

TRAIN, TEST = (2021, 2022, 2023, 2024), (2025, 2026)
BREAK_EVEN = 110 / 210
WEST = {'SF', 'SEA', 'LA', 'LAC', 'LV', 'ARI', 'DEN'}
EAST = {'BUF', 'MIA', 'NE', 'NYJ', 'NYG', 'PHI', 'WAS', 'BAL', 'PIT', 'CLE', 'CIN', 'DET', 'ATL', 'CAR', 'TB', 'JAX', 'IND'}


def rows():
    """Played games with a line, plus model prediction and QB-change flags, in date order."""
    games = load_games()
    pred = {}
    e = Engine(games, load_params())
    for s in sorted({g['season'] for g in games}):
        for p, g in e.walk(s):
            pred[g['id']] = p
    last_qb, out = {}, []
    for r in sorted(csv.DictReader(open('data/games.csv')), key=lambda r: (r['gameday'], r['game_id'])):
        h, a = r['home_team'], r['away_team']
        change = {t: int(t in last_qb and r[side + '_qb_id'] != last_qb[t]) for t, side in ((h, 'home'), (a, 'away'))}
        if r['home_score'] == '' or r['spread_line'] == '':
            continue
        last_qb[h], last_qb[a] = r['home_qb_id'], r['away_qb_id']
        line = float(r['spread_line'])
        out.append(dict(season=int(r['season']), id=r['game_id'], line=line,
                        err=int(r['home_score']) - int(r['away_score']) - line,
                        model=pred.get(r['game_id']), rest=int(r['home_rest']) - int(r['away_rest']),
                        div=int(r['div_game']), wind=float(r['wind'] or 0), outdoors=r['roof'] in ('outdoors', 'open'),
                        qb=change[h] - change[a],
                        body_clock=int(a in WEST and h in EAST and r['gametime'] < '14:00') - int(a in EAST and h in WEST and r['gametime'] >= '20:00'),
                        thursday=r['weekday'] == 'Thursday'))
    return out


def fav(g):
    return 1 if g['line'] > 0 else -1 if g['line'] < 0 else 0


# Each factor gives a number per game: positive leans home against the line, negative leans away, 0 means no bet.
FACTORS = {
    'model disagrees with line by 3+': lambda g: 0 if g['model'] is None or abs(g['model'] - g['line']) < 3 else np.sign(g['model'] - g['line']),
    'rest advantage (days)': lambda g: max(-7, min(7, g['rest'])),
    'starting QB changed since last game': lambda g: g['qb'],
    'west team at east 1pm (body clock)': lambda g: g['body_clock'],
    'home underdog': lambda g: 1 if g['line'] < 0 else 0,
    'big favorite (7+)': lambda g: fav(g) if abs(g['line']) >= 7 else 0,
    'division game, underdog': lambda g: -fav(g) if g['div'] else 0,
    'wind 15+ mph, underdog': lambda g: -fav(g) if g['outdoors'] and g['wind'] >= 15 else 0,
    'Thursday game, underdog': lambda g: -fav(g) if g['thursday'] else 0,
}


def ats(games, f, sign):
    """Against-the-spread wins and bets for the side the factor picks (pushes skipped)."""
    win = n = 0
    for g in games:
        v = f(g) * sign
        if v == 0 or g['err'] == 0:
            continue
        n += 1
        win += (v > 0) == (g['err'] > 0)
    return win, n


def p_value(win, n):
    """One-sided chance of doing at least this well if the true rate were the break-even rate."""
    if n == 0:
        return 1.0
    z = (win - n * BREAK_EVEN) / math.sqrt(n * BREAK_EVEN * (1 - BREAK_EVEN))
    return 0.5 * math.erfc(z / math.sqrt(2))


def main():
    data = rows()
    train = [g for g in data if g['season'] in TRAIN]
    test = [g for g in data if g['season'] in TEST]
    bar = 0.05 / len(FACTORS)
    results = []
    for name, f in FACTORS.items():
        tw, tn = ats(train, f, 1)
        sign = 1 if tw * 2 >= tn else -1
        if sign < 0:
            tw = tn - tw
        w, n = ats(test, f, sign)
        p = p_value(w, n)
        results.append(dict(factor=name, direction='with' if sign > 0 else 'against', train_ats=round(tw / max(tn, 1), 3), train_bets=tn,
                            test_ats=round(w / max(n, 1), 3), test_bets=n, p=round(p, 4), edge=bool(n and w / n > BREAK_EVEN and p < bar)))
    print(f'{len(train)} training games, {len(test)} test games; edge needs > {BREAK_EVEN:.1%} with p < {bar:.4f}')
    for r in sorted(results, key=lambda r: r['p']):
        print(f"{r['factor']:38s} {r['direction']:7s} train {r['train_ats']:.1%} ({r['train_bets']})  "
              f"test {r['test_ats']:.1%} ({r['test_bets']})  p={r['p']:.3f}  {'EDGE' if r['edge'] else ''}")
    json.dump(dict(break_even=BREAK_EVEN, p_bar=bar, train=list(TRAIN), test=list(TEST), results=results), open('edge.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
