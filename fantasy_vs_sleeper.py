"""Compare SAM's fantasy projections with Sleeper's own (Rotowire) weekly projections on the same player-games.

Needs Sleeper projection files in cache/sleeper/ (proj_<season>_w<week>.json, pulled from api.sleeper.app on the
user's PC) and players_compact.csv for the Sleeper id -> nflverse gsis id map. Caveat: the files were last saved
by Sleeper after each week ended, so they may include late pre-game news; they cannot include game results only if
Sleeper froze them at kickoff, which can't be verified from the files.

Usage: python fantasy_vs_sleeper.py   -> prints the table and writes fantasy_vs_sleeper.json
"""
import csv, json, os
from collections import defaultdict
import numpy as np
import fantasy
from lineup import norm

HERE = fantasy.HERE
SL = os.path.join(HERE, 'cache', 'sleeper')


def sleeper_ids():
    m = {}
    for r in csv.DictReader(open(os.path.join(SL, 'players_compact.csv'), encoding='utf-8')):
        g = (r['gsis_id'] or '').strip()
        if g:
            m[r['player_id']] = g
    return m


def sleeper_proj(season, week, ids, names=None):
    """Sleeper projected PPR points by nflverse gsis id; falls back to name + team when the id map has no gsis id."""
    path = os.path.join(SL, f'proj_{season}_w{week}.json')
    if not os.path.exists(path):
        return {}
    out = {}
    for r in json.load(open(path, encoding='utf-8')):
        g = ids.get(r['player_id'])
        if not g and names is not None:
            pl = r.get('player') or {}
            g = names.get((norm(f"{pl.get('first_name', '')} {pl.get('last_name', '')}"), r.get('team') or pl.get('team')))
        if g and 'pts_ppr' in (r.get('stats') or {}):
            out[g] = float(r['stats']['pts_ppr'])
    return out


def main(tests=(2025, 2026)):
    params = json.load(open(os.path.join(HERE, 'fantasy_params.json')))
    d = fantasy.Data(range(min(tests) - 1, max(tests) + 1), fantasy.SCORING['league'])
    ids = sleeper_ids()
    report = {}
    for s in tests:
        rows = d.instances(s)
        names = {}
        for pid, h in d.hist.items():
            for gm in h:
                if gm['season'] == s:
                    names[(norm(d.meta[pid][0]), gm['team'])] = pid
        sp = {}
        for w in range(1, 19):
            for g, p in sleeper_proj(s, w, ids, names).items():
                sp[(g, w)] = p
        by = defaultdict(list)
        for pid, w, x, y in rows:
            if (pid, w) not in sp:
                continue
            pos = d.meta[pid][1]
            by[pos].append((w, fantasy.predict(params, pos, x), sp[(pid, w)], y, x['season_avg']))
        for pos, rs in by.items():
            w = np.array([r[0] for r in rs])
            sam, slp, y, avg = (np.array([r[i] for r in rs]) for i in (1, 2, 3, 4))
            rel = slp >= 8   # fantasy-relevant by Sleeper's own projection; same pairs for every method
            res = {}
            for name, p in (('SAM', sam), ('Sleeper', slp), ('avg of both', (sam + slp) / 2)):
                right = total = 0
                for wk in set(w.tolist()):
                    idx = np.where((w == wk) & rel)[0]
                    for i in range(len(idx)):
                        for j in range(i + 1, len(idx)):
                            a, b = idx[i], idx[j]
                            if y[a] == y[b] or p[a] == p[b]:
                                continue
                            total += 1
                            right += (p[a] > p[b]) == (y[a] > y[b])
                res[name] = {'n': int(len(y)), 'mae': round(float(np.abs(p - y).mean()), 3),
                             'mae_8plus': round(float(np.abs(p - y)[rel].mean()), 3),
                             'corr': round(float(np.corrcoef(p, y)[0, 1]), 3),
                             'startsit': round(right / total, 4) if total else None, 'pairs': total}
            report.setdefault(str(s), {})[pos] = res
    json.dump(report, open(os.path.join(HERE, 'fantasy_vs_sleeper.json'), 'w'), indent=1)
    for s, by in report.items():
        print(f'== {s} ==')
        for pos in fantasy.POS:
            for name, m in by.get(pos, {}).items():
                print(f"{pos:3s} {name:12s} n={m['n']:5d} MAE {m['mae']:5.2f} MAE(8+) {m['mae_8plus']:5.2f} "
                      f"corr {m['corr']:.3f} start/sit {m['startsit']:.1%} ({m['pairs']} pairs)")
    return report


if __name__ == '__main__':
    main()
