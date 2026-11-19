"""Do the play-style tags mean anything? Each tag is given from games through week 8, then checked against the
same rule on weeks 9+ alone, 2014-2025. A tag is kept when it repeats at least twice as often as it would by chance
and at least 30% of the time. The same rules on raw, unsteadied numbers are shown for comparison.

  python research/traits_eval.py     # writes trait_rules.json and research/traits_results.txt
"""
import json, os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
import playerrate as pr, teamrate, traits

HERE = os.path.dirname(os.path.abspath(__file__))
SEASONS = range(2014, 2026)


def zdict(q, pos):
    return {pid: {l: float(r['a_' + l]) for l, *_ in pr.SPEC[pos]['attrs']} for pid, r in q.iterrows()}


def main():
    lines = []
    def say(*a):
        s = ' '.join(str(x) for x in a); print(s, flush=True); lines.append(s)
    P = pr.load(); df = pr.all_weeks(pr.FIRST, 2026)
    keep = {}
    say(f"{'':5s}{'tag':24s} | steadied: tagged repeat chance | raw: tagged repeat chance | kept")
    for pos in pr.POSITIONS:
        pp = P['pos'][pos]; raw_pp = dict(pp, attr_m={k: 0.0 for k in pp['attr_m']})
        cnt = {n: dict(new=[0, 0], raw=[0, 0], base=[0, 0]) for n, _, _ in traits.PLAYER[pos]}
        for s in SEASONS:
            new = zdict(pr.estimate(pr.totals(df, s, 9, pp['prior']), pos, pp), pos)
            raw = zdict(pr.estimate(pr.totals(df, s, 9, 0.0), pos, raw_pp), pos)
            fut = zdict(pr.estimate(pr.totals(df, s, 99, 0.0, from_week=9), pos, raw_pp), pos)
            for name, _, rule in traits.PLAYER[pos]:
                c = cnt[name]
                for pid, f in fut.items():
                    if pid not in new:
                        continue
                    hit = bool(rule(f))
                    c['base'][0] += hit; c['base'][1] += 1
                    if rule(new[pid]):
                        c['new'][0] += hit; c['new'][1] += 1
                    if pid in raw and rule(raw[pid]):
                        c['raw'][0] += hit; c['raw'][1] += 1
        keep[pos] = []
        for name, c in cnt.items():
            r = lambda x: x[0] / x[1] if x[1] else 0.0
            ok = c['new'][1] >= 15 and r(c['new']) >= 0.30 and r(c['new']) >= 2 * r(c['base'])
            if ok:
                keep[pos].append(name)
            say(f"{pos:5s}{name:24s} | {c['new'][1]:5d} {100 * r(c['new']):5.0f}% {100 * r(c['base']):5.0f}% | {c['raw'][1]:5d} {100 * r(c['raw']):5.0f}% | {'yes' if ok else 'NO'}")
    # teams
    sp = teamrate.load()
    R = teamrate.Ratings(range(2013, 2026), sp['k'], sp['carry'], sp['decay'])
    cnt = {n: dict(new=[0, 0], base=[0, 0]) for n, _, _ in traits.TEAM}
    for s in SEASONS:
        now = teamrate.team_traits_input(R.features(s, 9), R.tg[s][R.tg[s].week < 9])
        tg = R.tg[s][R.tg[s].week >= 9]
        o, d = teamrate.solve(tg, np.ones(len(tg)), k=2.0)
        fut_f = {t: dict({'o_' + c: o[teamrate.IDX[t], j] for j, c in enumerate(teamrate.CATS)}, **{'d_' + c: -d[teamrate.IDX[t], j] for j, c in enumerate(teamrate.CATS)}) for t in teamrate.ABBRS}
        fut = teamrate.team_traits_input(fut_f, tg)
        for name, _, rule in traits.TEAM:
            for t in teamrate.ABBRS:
                hit = bool(rule(fut[t])); c = cnt[name]
                c['base'][0] += hit; c['base'][1] += 1
                if rule(now[t]):
                    c['new'][0] += hit; c['new'][1] += 1
    keep['TEAM'] = []
    for name, c in cnt.items():
        r = lambda x: x[0] / x[1] if x[1] else 0.0
        ok = c['new'][1] >= 15 and r(c['new']) >= 0.30 and r(c['new']) >= 2 * r(c['base'])
        if ok:
            keep['TEAM'].append(name)
        say(f"TEAM {name:24s} | {c['new'][1]:5d} {100 * r(c['new']):5.0f}% {100 * r(c['base']):5.0f}% | {'':17s} | {'yes' if ok else 'NO'}")
    json.dump(keep, open(traits.KEEP_PATH, 'w'), indent=1)
    open(os.path.join(HERE, 'traits_results.txt'), 'w').write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
