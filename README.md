# NFL Dashboard

GeddyLifeson's NFL power rankings, game predictions, team and player stats with league ranks, player
power rankings by position and next-game player projections, all rebuilt automatically from free
nflverse data.

**Online:** https://geddylifeson.github.io/nfl-dashboard/ refreshes itself on GitHub several times a day
(every 2 hours on Sundays and Mondays). "Run workflow" on the repo's Actions tab refreshes it by hand.

**On your PC:** `pip install -r requirements.txt`, then `python app.py` opens the same dashboard at
http://127.0.0.1:8030 with a Refresh button. Data older than 6 hours refreshes when it starts.

## What it shows

- **Rankings:** power rating (points better than an average team), plus offense and defense ranks by
  opponent-adjusted EPA per play. Sort by any column.
- **Teams:** 27 offense, defense and turnover stats from play-by-play, each with its league rank, and the next game.
- **Games:** next week's picks with win chance, the market line, and who is missing; last week's picks graded.
- **Player rankings:** QB, RB, WR, TE, EDGE, DL, LB, CB, S and K, rated 0-100 within each position.
- **Player projections:** next-game passing, rushing and receiving stats for QBs, RBs, WRs and TEs.
- **Model accuracy:** every backtest below, so each number can be checked.

## Scripts

```
python pipeline.py          # everything below, then site/data.json for the app
python fetch.py             # scores, schedule and betting lines -> data/games.csv
python build.py             # ratings, picks with lineup correction -> league_data.json, history/
python units.py             # offense, defense, QB EPA rankings -> units.json
python teamstats.py         # team stats with ranks from play-by-play
python backtest.py          # re-tune the game model (params.json)
python lineup_backtest.py   # re-fit the QB and injury correction (lineup_weights.json)
python players.py           # re-tune player projections (projection_params.json)
python test_model.py        # model checks
python gen_md.py head       # markdown overview; python tiktok.py makes the 1080x1920 image
```

Downloads are cached in `cache/` (finished seasons once, the current season every 6 hours).

## How the rating works

A team's rating is how many points per game better it is than an average team on a neutral field. It comes from every score this season, adjusted for the opponent and for home field (2 points; none at neutral sites such as London), with each team starting from 60% of last season's final rating. That head start counts as 8 games, so it fades as the season goes on. The gap between two ratings, plus home field, is the predicted margin.

Predictions also adjust for the current lineup (`build.py` -> `lineup_adjustments`, features in `lineup.py`):
the starting quarterback from today's nflverse depth chart (`qb_adjust.py`) and missing players from the injury
report, with weights fitted and tested as described under "Lineup correction" below. `results_margin` in
league_data.json keeps the pure results prediction.

Movement arrows compare this week's ranking with the ranking the same model gave through last week. Each build saves the rankings and picks to `history/`, and later builds grade those picks once the games are played.

The stat tables in `stats/` (FootballDB, collected by hand) are shown for context only. Earlier walk-forward tests found box score stats added nothing beyond game results. A stat is marked when its source table is missing a game the team has played.

## How well it predicts

Settings were tuned on 2022 to 2024 and then frozen. Every prediction below uses only games from earlier weeks.

| Season | Games | Model picks winner | Always pick home team | Betting line picks winner |
|---|---|---|---|---|
| 2025 | 272 | 62.0% | 53.9% | 65.3% |
| 2026 (weeks 1-6) | 92 | 60.4% | 50.5% | 62.6% |

Strong leans are more reliable: across 2025 and 2026 from week 3, the half of games where the model leaned hardest went 71.1%, against 60.7% overall.

## Lineup correction and player projections

Fitted on 2022-2024 and tested, frozen, on 2025-2026 (`lineup_backtest.py`): adding the starting QB and
injuries to the results model cut the error from 165.1 to 161.0 and raised winners picked from 61.7% to
62.8%. Most of it is the quarterback: when the depth chart's QB1 is not the team's usual passer, the
margin moves by the gap in EPA per dropback (both shrunk toward backup level) times 35 dropbacks,
scaled by the fitted weight. Missing corners, edge rushers, linebackers, receivers and linemen also
count; tight ends, interior linemen and safeties showed no measurable effect. Absences come from the
official injury report and from regulars who missed the previous game.

Player projections (`players.py`) blend recent games, last season and the opponent's allowance to the
position, with per-stat settings tuned on 2023-2024. On 2025 they beat a plain season average on 11 of
14 stats, by small margins: single-game player stats are noisy.

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
