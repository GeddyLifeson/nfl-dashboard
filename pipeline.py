"""Refresh everything the app shows and write site/data.json.
Run: python pipeline.py   (the app's Refresh button and the scheduled tasks call this)"""
import contextlib, datetime, glob, io, json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
os.chdir(HERE)
sys.path.insert(0, HERE)

import build, fetch, lineup, nflv, players, qb_adjust, teamstats, units  # noqa: E402
from teams import display_abbr  # noqa: E402


def step(label, fn):
    """Run one stage quietly; a failed stage is reported in the data instead of stopping the refresh."""
    out = io.StringIO()
    try:
        with contextlib.redirect_stdout(out):
            fn()
        return None
    except Exception as e:
        return f"{label}: {type(e).__name__}: {e}"


def recent_results(season):
    """Last completed week: each game's score and whether the model's archived pick was right."""
    import csv
    games = [r for r in csv.DictReader(open('data/games.csv')) if r['season'] == str(season) and r['home_score']]
    if not games:
        return []
    week = max(int(r['week']) for r in games)
    picks = {}
    for path in sorted(glob.glob(f'history/{season}-w*.json')):
        for p in json.load(open(path)).get('upcoming', []):
            picks.setdefault(p['id'], p)
    out = []
    for r in games:
        if int(r['week']) != week:
            continue
        hs, as_ = int(r['home_score']), int(r['away_score'])
        p = picks.get(r['game_id'])
        right = None if not p or hs == as_ else (p['home_margin'] > 0) == (hs > as_)
        out.append(dict(week=week, away=display_abbr(r['away_team']), home=display_abbr(r['home_team']), away_score=as_,
                        home_score=hs, pick_margin=p['home_margin'] if p else None, right=right))
    return out


def main():
    errors = [e for e in (step('scores', fetch.main), step('rankings', build.main), step('units', units.main)) if e]
    season = nflv.current_season()
    league = json.load(open('league_data.json'))
    unit = json.load(open('units.json'))
    off = {u['abbr']: u for u in unit['offense']}
    de = {u['abbr']: u for u in unit['defense']}
    stats = {display_abbr(t): rows for t, rows in teamstats.table(season).items()}
    power = []
    for t in league['teams']:
        a = t['abbr']
        power.append(dict(pos=t['pos'], team=t['team'], abbr=a, record=t['record'], rating=t['rating'], move=t['move'],
                          margin=t['margin'], sos=t['sos'], sos_rank=t['sos_rank'], off_rank=off[a]['rank'],
                          off_epa=off[a]['epa_per_play'], def_rank=de[a]['rank'], def_epa=de[a]['epa_per_play_allowed']))
    # next game per team, and who will not play, for the projections
    nxt = {}
    for g in sorted(league['upcoming'], key=lambda g: g['day']):
        for team, opp in ((g['home'], g['away']), (g['away'], g['home'])):
            nxt.setdefault(team, (opp, g['week'], g['day']))
    raw = {('LA' if t == 'LAR' else t): (('LA' if o == 'LAR' else o), w, d) for t, (o, w, d) in nxt.items()}
    out_players = set()
    try:
        ss = lineup.Season(season)
        for team, (_, week, _) in raw.items():
            for n, _, why, _ in ss.missing_players(team, week):
                if why in ('Out', 'Doubtful', 'missed last game'):
                    out_players.add((team, n))
        qb1 = {t: n for t, (n, _, _) in qb_adjust.starters(season).items()}
    except Exception as e:
        errors.append(f"injuries: {e}")
        qb1 = {}
    proj = []
    for p in players.projections(season, raw, frozenset(out_players)):
        if p['pos'] == 'QB' and qb1 and p['team'] in qb1 and lineup.short(p['name']) != lineup.short(qb1[p['team']]):
            continue                                  # only the current starting QB gets a passing projection
        p['team'], p['opp'] = display_abbr(p['team']), display_abbr(p['opp'])
        proj.append(p)
    ranks = players.rankings(season)
    for rows in ranks.values():
        for r in rows:
            r['team'] = display_abbr(r['team'])
    params = json.load(open('params.json'))
    lw = json.load(open('lineup_weights.json')) if os.path.exists('lineup_weights.json') else {}
    pp = json.load(open('projection_params.json')) if os.path.exists('projection_params.json') else {}
    data = dict(
        updated=datetime.datetime.now().strftime('%a %b %d, %I:%M %p'), season=season, week=league['week'],
        through=league['through'], power=power, team_stats=stats, games=league['upcoming'], results=recent_results(season),
        quarterbacks=unit['quarterbacks'], players=ranks, projections=proj,
        model=dict(game=params.get('tested', {}), lineup=dict(train=lw.get('train'), test=lw.get('test'), weights=lw.get('weights')),
                   projections=pp.get('test', {}), pick_record=league.get('pick_record')),
        errors=errors)
    os.makedirs('site', exist_ok=True)
    with open('site/data.json.tmp', 'w', encoding='utf-8') as f:
        json.dump(data, f)
    os.replace('site/data.json.tmp', 'site/data.json')
    print(f"site/data.json written: {len(power)} teams, {len(data['games'])} games, {len(proj)} projections, "
          f"{sum(len(v) for v in ranks.values())} ranked players; errors: {errors or 'none'}")


if __name__ == '__main__':
    main()
