# NFL Power Rankings

GeddyLifeson's Power Rankings for the 2026 NFL season: the data, the model, and the weekly method.

## Weekly routine

```
pip install -r requirements.txt
python fetch.py       # scores, schedule and betting lines from nflverse -> data/games.csv
python build.py       # rankings, movement, next week's picks -> league_data.json, history/
python gen_md.py head # markdown overview and picks (python gen_md.py 0 8 for team pages)
python tiktok.py      # 1080x1920 image -> tiktok/
```

Run `python backtest.py` to re-tune the settings (writes `params.json`), `python test_model.py` to check the model, and `python edge.py` to test ideas against the betting line.

## How the rating works

A team's rating is how many points per game better it is than an average team on a neutral field. It comes from every score this season, adjusted for the opponent and for home field (2 points; none at neutral sites such as London), with each team starting from 60% of last season's final rating. That head start counts as 8 games, so it fades as the season goes on. The gap between two ratings, plus home field, is the predicted margin.

Movement arrows compare this week's ranking with the ranking the same model gave through last week. Each build saves the rankings and picks to `history/`, and later builds grade those picks once the games are played.

The stat tables in `stats/` (FootballDB, collected by hand) are shown for context only. Earlier walk-forward tests found box score stats added nothing beyond game results. A stat is marked when its source table is missing a game the team has played.

## How well it predicts

Settings were tuned on 2022 to 2024 and then frozen. Every prediction below uses only games from earlier weeks.

| Season | Games | Model picks winner | Always pick home team | Betting line picks winner |
|---|---|---|---|---|
| 2025 | 272 | 62.0% | 53.9% | 65.3% |
| 2026 (weeks 1-6) | 92 | 60.4% | 50.5% | 62.6% |

Strong leans are more reliable: across 2025 and 2026 from week 3, the half of games where the model leaned hardest went 71.1%, against 60.7% overall.

The betting line is still better. `edge.py` tests ideas the line might miss, learning each from 2021 to 2024 and grading it on 2025 and 2026 against the spread. Nothing has passed so far: rest, starting QB changes, body clock trips, wind, division and Thursday underdogs, big favorites, and the model's own disagreements with the line all fall short of the bar for real (52.4% to beat the bookmaker's cut, after allowing for testing many ideas at once). Results are in `edge.json`.

## Files

- `teams.py`: one table for every way the sources spell a team.
- `fetch.py`, `data/games.csv`: nflverse schedule from 2021 on. `data/ledger.jsonl` logs each download and stat table with its hash.
- `model.py`: the rating engine. `backtest.py` tunes and tests it, and `params.json` holds the frozen settings and test results.
- `build.py`, `stats.py`: weekly output. `gen_md.py` and `tiktok.py` turn it into the document and the image.
- `edge.py`: search for factors the betting line misses.
- `design/Main.dc.html`: hand-made design export for the Oct 18 graphic. It has its content baked in and is not updated by the scripts.

The earlier 145-stat composite and its experiments are in the first commit (`cd0d14a`).

## Sources

Scores, schedule, betting lines, rest days, weather and starting QBs: [nflverse](https://github.com/nflverse/nfldata). Season stat tables: FootballDB. 2025 honors: Pro Football Reference. Logos: ESPN, unless a `logos.json` is provided.
