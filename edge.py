"""Search for bets that beat the sportsbooks. Run: python edge.py

Every idea is a betting rule on the spread, the total or the moneyline, graded by actual profit at the
odds nflverse recorded (so the bookmaker's cut is included). Which side of each rule to bet is learned on
2021-2024 only; the rule is then graded on 2025-2026. With many rules tried, one will look good by luck,
so a rule only counts as an edge if its test profit is positive with a p-value under 0.05 / number of rules."""
import csv, json, math
from typing import Any
from model import Engine, load_games, load_params

TRAIN, TEST = (2021, 2022, 2023, 2024), (2025, 2026)
WEST = {'SF', 'SEA', 'LA', 'LAC', 'LV', 'ARI', 'DEN'}
EAST = {'BUF', 'MIA', 'NE', 'NYJ', 'NYG', 'PHI', 'WAS', 'BAL', 'PIT', 'CLE', 'CIN', 'DET', 'ATL', 'CAR', 'TB', 'JAX', 'IND'}


def rows():
    """Played games in date order, with the model's walk-forward prediction and each team's previous game."""
    games = load_games()
    e = Engine(games, load_params())
    pred = {g['id']: p for s in sorted({g['season'] for g in games}) for p, g in e.walk(s)}
    last_qb, last_cover, out = {}, {}, []
    for r in sorted(csv.DictReader(open('data/games.csv')), key=lambda r: (r['gameday'], r['game_id'])):
        if r['home_score'] == '':
            continue
        h, a = r['home_team'], r['away_team']
        f = lambda k: float(r[k])
        margin = int(r['home_score']) - int(r['away_score'])
        g = dict(season=int(r['season']), margin=margin, points=int(r['home_score']) + int(r['away_score']),
                 line=f('spread_line'), total=f('total_line'), model=pred.get(r['game_id']),
                 odds=dict(spread_home=f('home_spread_odds'), spread_away=f('away_spread_odds'), over=f('over_odds'),
                           under=f('under_odds'), ml_home=f('home_moneyline'), ml_away=f('away_moneyline')),
                 rest=int(r['home_rest']) - int(r['away_rest']), div=r['div_game'] == '1', wind=float(r['wind'] or 0),
                 temp=float(r['temp'] or 60), outdoors=r['roof'] in ('outdoors', 'open'), dome=r['roof'] in ('dome', 'closed'),
                 qb_home=int(h in last_qb and r['home_qb_id'] != last_qb[h]), qb_away=int(a in last_qb and r['away_qb_id'] != last_qb[a]),
                 body_clock=a in WEST and h in EAST and r['gametime'] < '14:00', thursday=r['weekday'] == 'Thursday',
                 cover_home=last_cover.get(h), cover_away=last_cover.get(a))
        err = margin - g['line']
        last_qb[h], last_qb[a] = r['home_qb_id'], r['away_qb_id']
        last_cover[h], last_cover[a] = err, -err
        out.append(g)
    return out


def payout(odds):
    return odds / 100 if odds > 0 else 100 / -odds


def profit(g, bet):
    """Units won or lost on a 1-unit bet; pushes return 0."""
    market, side = bet
    if market == 'spread':
        res = (g['margin'] - g['line']) * (1 if side == 'home' else -1)
    elif market == 'total':
        res = (g['points'] - g['total']) * (1 if side == 'over' else -1)
    else:
        res = g['margin'] * (1 if side == 'home' else -1)
    if res == 0:
        return 0.0
    return payout(g['odds'][side if market == 'total' else f'{market}_{side}']) if res > 0 else -1.0


OPP = {'home': 'away', 'away': 'home', 'over': 'under', 'under': 'over'}


def fav(g):
    return 'home' if g['line'] > 0 else 'away'


def dog(g):
    return OPP[fav(g)]


# Each rule returns (market, side) to bet, or None to skip the game.
RULES = {
    'spread: model disagrees with line by 3+': lambda g: None if g['model'] is None or abs(g['model'] - g['line']) < 3 else ('spread', 'home' if g['model'] > g['line'] else 'away'),
    'spread: rest edge of 3+ days': lambda g: None if abs(g['rest']) < 3 else ('spread', 'home' if g['rest'] > 0 else 'away'),
    'spread: team with a new starting QB': lambda g: None if g['qb_home'] == g['qb_away'] else ('spread', 'home' if g['qb_home'] else 'away'),
    'spread: home team vs west team at 1pm east': lambda g: ('spread', 'home') if g['body_clock'] else None,
    'spread: division underdog': lambda g: ('spread', dog(g)) if g['div'] and g['line'] else None,
    'spread: Thursday underdog': lambda g: ('spread', dog(g)) if g['thursday'] and g['line'] else None,
    'spread: favorite of 7+': lambda g: ('spread', fav(g)) if abs(g['line']) >= 7 else None,
    'spread: fade team that covered by 14+ last week': lambda g: (
        ('spread', 'away') if (g['cover_home'] or 0) >= 14 and (g['cover_away'] or 0) < 14 else
        ('spread', 'home') if (g['cover_away'] or 0) >= 14 and (g['cover_home'] or 0) < 14 else None),
    'spread: home underdog': lambda g: ('spread', 'home') if g['line'] < 0 else None,
    'total: under in wind 15+ mph': lambda g: ('total', 'under') if g['outdoors'] and g['wind'] >= 15 else None,
    'total: under at 32F or colder': lambda g: ('total', 'under') if g['outdoors'] and g['temp'] <= 32 else None,
    'total: over in a dome': lambda g: ('total', 'over') if g['dome'] else None,
    'total: under in division games': lambda g: ('total', 'under') if g['div'] else None,
    'total: under when total is 50+': lambda g: ('total', 'under') if g['total'] >= 50 else None,
    'total: over when total is 40 or less': lambda g: ('total', 'over') if g['total'] <= 40 else None,
    'moneyline: underdog at +250 or longer': lambda g: next((('ml', s) for s in ('home', 'away') if g['odds']['ml_' + s] >= 250), None),
    'moneyline: favorite at -300 or shorter': lambda g: next((('ml', s) for s in ('home', 'away') if g['odds']['ml_' + s] <= -300), None),
    'moneyline: home underdog': lambda g: ('ml', 'home') if g['odds']['ml_home'] > 0 and g['odds']['ml_away'] < 0 else None,
}


def grade(games, rule, flip):
    p = []
    for g in games:
        bet = rule(g)
        if bet:
            p.append(profit(g, (bet[0], OPP[bet[1]]) if flip else bet))
    return p


def p_value(p):
    """One-sided chance of at least this much profit per bet if the true average were zero."""
    n = len(p)
    if n < 2:
        return 1.0
    m = sum(p) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in p) / (n - 1)) or 1e-9
    return 0.5 * math.erfc(m / (sd / math.sqrt(n)) / math.sqrt(2))


def summary(p) -> dict[str, Any]:
    return dict(bets=len(p), roi=round(sum(p) / len(p), 3) if p else 0.0, p=round(p_value(p), 4))


def main():
    data = rows()
    train = [g for g in data if g['season'] in TRAIN]
    test = [g for g in data if g['season'] in TEST]
    bar = 0.05 / len(RULES)
    results, kept = [], []
    for name, rule in RULES.items():
        flip = sum(grade(train, rule, True)) > sum(grade(train, rule, False))
        tr, te = grade(train, rule, flip), grade(test, rule, flip)
        r: dict[str, Any] = dict(rule=name, bet='opposite side' if flip else 'as named', train=summary(tr), test=summary(te))
        r['edge'] = r['test']['roi'] > 0 and r['test']['p'] < bar
        results.append(r)
        if r['train']['roi'] > 0:
            kept.append(te)
    portfolio = summary([x for te in kept for x in te])
    print(f'{len(train)} training games, {len(test)} test games. Edge needs test ROI > 0 with p < {bar:.4f}.')
    print(f"{'rule':50s} {'bet':13s} {'train ROI (bets)':>18s} {'test ROI (bets)':>17s}      p")
    for r in sorted(results, key=lambda r: r['test']['p']):
        t, s = r['train'], r['test']
        print(f"{r['rule']:50s} {r['bet']:13s} {t['roi']:+8.1%} ({t['bets']:4d})  {s['roi']:+8.1%} ({s['bets']:4d})  {s['p']:.3f} {'EDGE' if r['edge'] else ''}")
    print(f"all rules that profited in training, bet together on test: ROI {portfolio['roi']:+.1%} on {portfolio['bets']} bets, p={portfolio['p']:.3f}")
    json.dump(dict(train=list(TRAIN), test=list(TEST), p_bar=bar, results=results, portfolio=portfolio), open('edge.json', 'w'), indent=1)


if __name__ == '__main__':
    main()
