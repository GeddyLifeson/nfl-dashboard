"""How same-game legs move together, and how often games land on each margin, from every NFL
regular-season game since 2012 with a closing line. Writes site/correlations.json for the
parlay builder:

  phi:     correlation between favorite wins / favorite covers / game goes over, so a same-game
           ticket is priced as P(A and B) = pA*pB + phi*sqrt(pA(1-pA)pB(1-pB)) instead of pA*pB.
  margins: by closing spread (favorite's points, rounded to the half point, 0-14), the share of games
           the favorite won by exactly k points (k from -30 to 40), for pricing middles.
Run: python correlations.py   (rerun once a season; the numbers move slowly)
"""
import csv, io, json, math, urllib.request
from collections import Counter, defaultdict
from fetch import URL

FIRST = 2012


def phi(pairs):
    n = len(pairs)
    a = sum(x for x, _ in pairs) / n; b = sum(y for _, y in pairs) / n
    ab = sum(x * y for x, y in pairs) / n
    return (ab - a * b) / math.sqrt(a * (1 - a) * b * (1 - b))


def main():
    rows = [r for r in csv.DictReader(io.StringIO(urllib.request.urlopen(URL, timeout=120).read().decode('utf-8')))
            if r['game_type'] == 'REG' and int(r['season']) >= FIRST and r['home_score'] and r['spread_line'] and r['total_line']]
    win_over, cover_over, win_cover = [], [], []
    margins = defaultdict(Counter)
    for r in rows:
        sp, tot = float(r['spread_line']), float(r['total_line'])
        m = int(r['home_score']) - int(r['away_score'])
        fav_m = m if sp >= 0 else -m                  # favorite's margin (home on a pick'em)
        line = abs(sp)
        margins[min(round(line * 2) / 2, 14.0)][fav_m] += 1
        pts = int(r['home_score']) + int(r['away_score'])
        if fav_m == 0 or fav_m == line or pts == tot:
            continue                                  # ties and pushes grade neither way
        w, c, o = int(fav_m > 0), int(fav_m > line), int(pts > tot)
        win_over.append((w, o)); cover_over.append((c, o)); win_cover.append((w, c))
    out = dict(games=len(rows), seasons=[FIRST, max(int(r['season']) for r in rows)],
               phi=dict(fav_win_over=round(phi(win_over), 4), fav_cover_over=round(phi(cover_over), 4),
                        fav_win_cover=round(phi(win_cover), 4)),
               margins={f'{k:g}': {str(m): round(c / sum(cnt.values()), 5) for m, c in sorted(cnt.items()) if -30 <= m <= 40}
                        for k, cnt in sorted(margins.items()) if sum(cnt.values()) >= 20})
    json.dump(out, open('site/correlations.json', 'w'), separators=(',', ':'))
    print(f"{out['games']} games; phi {out['phi']}; {len(out['margins'])} spread buckets")


if __name__ == '__main__':
    main()
