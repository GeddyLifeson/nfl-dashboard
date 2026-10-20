# NFL Power Rankings

GeddyLifeson's Power Rankings for the 2026 NFL season: the data, the model, and the weekly method.

## Weekly routine

```
pip install -r requirements.txt
python fetch.py       # scores, schedule and betting lines from nflverse -> data/games.csv
python build.py       # rankings, movement, next week's picks -> league_data.json, history/
python gen_md.py head # markdown overview and picks (python gen_md.py 0 8 for team pages)
python tiktok.py      # 1080x1920 image -> tiktok/
python units.py       # offense, defense and QB rankings (opponent-adjusted EPA per play) -> units.json
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

## Against the sportsbooks

The closing line is still better than the model: 64.6% of winners and a mean squared error of 149 on the 2025-26 test games, against the model's 61.6% and 164. Adding play-by-play EPA (opponent-adjusted, pass, rush and special teams) did not close the gap: it scored 168 out of sample, and mixing any share of it into the line made the line worse.

`edge.py` tests 18 betting rules on spreads, totals and moneylines, graded by real profit at the recorded odds, so the bookmaker's cut is included. Which side to bet is learned on 2021 to 2024 and graded on 2025 and 2026. None has passed the bar (profit with p < 0.05 / 18). Betting every rule that made money in training lost 5.8% on 654 test bets, and the best-looking training rules reversed (under in high wind: +21.4% in training, -0.2% in test). Results are in `edge.json`.

Public data that arrives before kickoff is already in the closing line. The realistic ways to beat it are price, not rating: betting early numbers before they move to where the line closes, and taking the best price across several books. Both need odds from many books captured over time, which nflverse does not have.

## Files

- `teams.py`: one table for every way the sources spell a team.
- `fetch.py`, `data/games.csv`: nflverse schedule from 2021 on. `data/ledger.jsonl` logs each download and stat table with its hash.
- `model.py`: the rating engine. `backtest.py` tunes and tests it, and `params.json` holds the frozen settings and test results.
- `build.py`, `stats.py`: weekly output. `gen_md.py` and `tiktok.py` turn it into the document and the image.
- `edge.py`: betting rules graded by profit against the sportsbooks.
- `design/Main.dc.html`: hand-made design export for the Oct 18 graphic. It has its content baked in and is not updated by the scripts.

The earlier 145-stat composite and its experiments are in the first commit (`cd0d14a`).

## Sources

Scores, schedule, closing spreads, totals, moneylines and their odds, rest days, weather and starting QBs: [nflverse](https://github.com/nflverse/nfldata). Season stat tables: FootballDB. 2025 honors: Pro Football Reference. Logos: ESPN, unless a `logos.json` is provided.
