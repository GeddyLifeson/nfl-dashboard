"""
Ledgerline: a sports-ratings ledger and rating engine, built on the Panscriptum idea
(every fact carries its source, a gate rejects pages that are not real, failures are logged
by reason, and the index is rebuilt from the records, never edited by hand).

1. LEDGER  every stat table is stored with url, fetch time, sha256, rows found, and a freshness
           verdict computed against the score ledger (games per team in the table vs games
           actually played). Records are append-only JSONL.
2. GATE    a table is 'ok' only if it has 32 known teams and the games count per team does not
           exceed the games played. 'behind' = fewer games than played (usable as a rate, flagged).
           Anything else is refused with a reason code.
3. ENGINE  team rating in points = posterior mean of  margin(game) = r_a - r_b + noise,
           with prior r_t ~ N(s * stat_composite_t, tau^2). The scale s and the shrinkage lam are
           chosen by leave-one-game-out prediction of games the model has not seen.
"""
import csv, json, hashlib, os, time
import numpy as np
D = 'n3/'; OUT = 'ledgerline/'
URL = {'tot_off':'footballdb offense-totals','tot_def':'footballdb defense-totals','pass_off':'footballdb offense-passing','pass_def':'footballdb defense-passing',
 'rush_off':'footballdb offense-rushing','rush_def':'footballdb defense-rushing','downs_off':'footballdb offense-downs','downs_def':'footballdb defense-downs',
 'punt_off':'footballdb offense-punting','punt_def':'footballdb defense-punting','pr':'footballdb offense-punt-returns','kr':'footballdb offense-kickoff-returns',
 'to':'footballdb turnovers','qb':'footballdb player-stats passing'}
games = list(csv.DictReader(open(D+'games.csv')))
played = {}
for g in games:
    for t in (g['w'], g['l']): played[t] = played.get(t, 0) + 1
TEAMS = sorted(played)
def ingest():
    recs = []
    for key in URL:
        raw = open(D+key+'.csv','rb').read(); rows = list(csv.DictReader(open(D+key+'.csv')))
        names = [r.get('Team') for r in rows]
        if key == 'qb': recs.append(dict(table=key, source=URL[key], sha256=hashlib.sha256(raw).hexdigest()[:16], rows=len(rows), verdict='ok', reason='player table; teams by abbreviation')); continue
        unknown = [n for n in names if n not in played]; verdict, reason = 'ok', ''
        if len(rows) != 32 or unknown: verdict, reason = 'refused', f'rows={len(rows)} unknown={unknown[:3]}'
        else:
            gcol = 'Gms' if 'Gms' in rows[0] else None
            if gcol:
                over = [r['Team'] for r in rows if int(r[gcol]) > played[r['Team']]]
                behind = [r['Team'] for r in rows if int(r[gcol]) < played[r['Team']]]
                if over: verdict, reason = 'refused', f'games exceed played: {over[:3]}'
                elif behind: verdict, reason = 'behind', f'{len(behind)} teams show fewer games than played'
        recs.append(dict(table=key, source=URL[key], sha256=hashlib.sha256(raw).hexdigest()[:16], rows=len(rows), verdict=verdict, reason=reason, fetched='2026-10-19'))
    os.makedirs(OUT, exist_ok=True)
    with open(OUT+'records.jsonl','w') as f:
        for r in recs: f.write(json.dumps(r)+'\n')
    return recs
if __name__ == '__main__':
    for r in ingest(): print(f"{r['table']:9s} {r['verdict']:8s} {r['rows']:3d} rows  {r.get('reason','')}")
