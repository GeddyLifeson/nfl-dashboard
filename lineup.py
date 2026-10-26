"""Lineup features: who is starting at quarterback and who is missing, known before kickoff.

Shared by the backtest (lineup_backtest.py, which fits how many points each feature is worth) and
the live predictions (build.py), so both measure absences the same way.

  qb:     (starter's EPA per dropback - team's EPA per dropback so far) x 35 dropbacks a game.
          The starter's EPA mixes this season's prior games with last season's, shrunk toward
          backup level (-0.10) by 150 dropbacks; little-used QBs count as backups.
  groups: for OL, WR, TE, RB, EDGE, DL, LB, CB, S: the sum over missing players of
          snap share x share of the team's games he played x chance he sits
          (Out 1, Doubtful .85, Questionable .25, regular who missed the last game and is not
          on the report .9). Only games before this one are used.
"""
import csv, gzip, re
from collections import defaultdict
import nflv

GROUP = {'T': 'OL', 'OT': 'OL', 'G': 'OL', 'OG': 'OL', 'C': 'OL', 'OL': 'OL', 'WR': 'WR', 'TE': 'TE',
         'RB': 'RB', 'FB': 'RB', 'DE': 'EDGE', 'OLB': 'EDGE', 'EDGE': 'EDGE', 'DT': 'DL', 'NT': 'DL', 'DL': 'DL',
         'LB': 'LB', 'ILB': 'LB', 'MLB': 'LB', 'CB': 'CB', 'S': 'S', 'FS': 'S', 'SS': 'S', 'DB': 'S'}
GROUPS = ['OL', 'WR', 'TE', 'RB', 'EDGE', 'DL', 'LB', 'CB', 'S']
P_STATUS = {'Out': 1.0, 'Doubtful': 0.85, 'Questionable': 0.25}
P_MISSED_LAST = 0.9
PRIOR_EPA, PRIOR_DB, DB_PER_GAME = -0.10, 150, 35.0
FIX = {'LAR': 'LA', 'OAK': 'LV', 'SD': 'LAC', 'STL': 'LA', 'WSH': 'WAS', 'JAC': 'JAX'}


def norm(n):
    n = re.sub(r"[^a-z ]", "", (n or '').lower().replace('-', ' '))
    return re.sub(r"\b(jr|sr|ii|iii|iv|v)\b", "", n).strip()


def short(n):
    p = norm(n).split()
    return (p[-1], p[0][0]) if p else ('', '')


class Season:
    """Everything needed to build lineup features for one season's games."""

    def __init__(self, season):
        self.season = season
        self.passing = defaultdict(lambda: [0, 0.0])        # (week, team, passer) -> dropbacks, epa
        self.prev_qb = defaultdict(lambda: [0, 0.0])         # passer -> last season's dropbacks, epa
        self.prev_passing_by_team = defaultdict(lambda: [0, 0.0])
        self._load_passing(season, self.passing)
        if season - 1 >= 2021:
            self._load_passing(season - 1, self.prev_passing_by_team)
            for (_, _, qb), (n, e) in self.prev_passing_by_team.items():
                self.prev_qb[qb][0] += n; self.prev_qb[qb][1] += e
        self.snaps = defaultdict(dict)                      # (team, week) -> {player key: (pos, share, name)}
        for r in csv.DictReader(open(nflv.get('snaps', season), encoding='utf-8')):
            if r['game_type'] != 'REG':
                continue
            pct = max(float(r['offense_pct'] or 0), float(r['defense_pct'] or 0))
            if pct > 0:
                team = FIX.get(r['team'], r['team'])
                self.snaps[(team, int(r['week']))][short(r['player'])] = (r['position'], pct, r['player'])
        self.status = defaultdict(dict)                     # (team, week) -> {player key: status}
        for r in csv.DictReader(open(nflv.get('injuries', season), encoding='utf-8')):
            if r['game_type'] == 'REG' and r['report_status'] in P_STATUS:
                self.status[(FIX.get(r['team'], r['team']), int(r['week']))][short(r['full_name'])] = r['report_status']

    @staticmethod
    def _load_passing(season, into):
        with gzip.open(nflv.get('pbp', season), 'rt', encoding='utf-8') as f:
            for r in csv.DictReader(f):
                if r['season_type'] == 'REG' and r['qb_dropback'] == '1' and r['passer_player_id'] and r['qb_epa'] not in ('', 'NA'):
                    x = into[(int(r['week']), FIX.get(r['posteam'], r['posteam']), r['passer_player_id'])]
                    x[0] += 1; x[1] += float(r['qb_epa'])

    def _est(self, qb, week):
        """A quarterback's EPA per dropback from last season and this season's earlier games, shrunk toward a backup."""
        n, e = self.prev_qb.get(qb, [0, 0.0])
        for (w, _, q), (dn, de) in self.passing.items():
            if w < week and q == qb:
                n += dn; e += de
        return (e + PRIOR_EPA * PRIOR_DB) / (n + PRIOR_DB)

    def main_qb(self, team, week):
        """The passer with the most dropbacks for this team so far (week 1: last season's)."""
        count = defaultdict(int)
        for (w, t, q), (n, _) in self.passing.items():
            if t == team and w < week:
                count[q] += n
        if not count:
            for (_, t, q), (n, _) in self.prev_passing_by_team.items():
                if t == team:
                    count[q] += n
        return max(count, key=count.get) if count else None

    def qb(self, team, starter, week):
        """Points per game the starter adds over the team's usual quarterback (0 when he is that QB)."""
        main = self.main_qb(team, week)
        if not starter or not main or starter == main:
            return 0.0
        return (self._est(starter, week) - self._est(main, week)) * DB_PER_GAME

    def missing(self, team, week):
        """{group: sum of snap share x share of games played x chance he sits} for this game."""
        weeks = sorted(w for (t, w) in self.snaps if t == team and w < week)
        out = dict.fromkeys(GROUPS, 0.0)
        if not weeks:
            return out
        role = defaultdict(list)
        for w in weeks:
            for key, (pos, pct, _) in self.snaps[(team, w)].items():
                role[key].append((pos, pct))
        last = self.snaps[(team, weeks[-1])]
        report = self.status.get((team, week), {})
        for key, games in role.items():
            pos = games[-1][0]
            g = GROUP.get(pos)
            if not g:
                continue
            share = sum(p for _, p in games) / len(games)
            st = report.get(key)
            if st:
                p = P_STATUS[st]
            elif key not in last and share >= 0.5 and len(games) >= 2:
                p = P_MISSED_LAST
            else:
                continue
            out[g] += share * len(games) / len(weeks) * p
        return out

    def missing_players(self, team, week):
        """Readable list of the absences behind missing(): (name, position, why, weight)."""
        weeks = sorted(w for (t, w) in self.snaps if t == team and w < week)
        if not weeks:
            return []
        role = defaultdict(list)
        names = {}
        for w in weeks:
            for key, (pos, pct, name) in self.snaps[(team, w)].items():
                role[key].append((pos, pct)); names[key] = name
        last = self.snaps[(team, weeks[-1])]
        report = self.status.get((team, week), {})
        rows = []
        for key, games in role.items():
            pos = games[-1][0]
            if not GROUP.get(pos):
                continue
            share = sum(p for _, p in games) / len(games)
            st = report.get(key)
            p = P_STATUS.get(st, 0.0) if st else (P_MISSED_LAST if key not in last and share >= 0.5 and len(games) >= 2 else 0.0)
            if p:
                rows.append((names[key], pos, st or 'missed last game', round(share * len(games) / len(weeks) * p, 2)))
        return sorted(rows, key=lambda r: -r[3])
