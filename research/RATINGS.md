# Ratings overhaul, November 2026

Goal: power rankings, team ratings, player ratings and play-style tags that are as accurate as the public data
allows, with no sportsbook number anywhere in them. Game picks are a separate model and were left alone.

Everything below can be re-run: `python research/ratings_eval.py`, `python research/players_eval.py`,
`python research/traits_eval.py`, `python research/predictions_check.py`, `python alltime.py`, `python test_ratings.py`.
Full outputs are in the `*_results.txt` files next to this one.

## What was wrong before

| Area | Problem |
|---|---|
| Power rating | It was the team part of the game-picks regression, and two of its inputs were preseason ratings made from sportsbook season win totals. It also depended on two files from someone else's project. |
| Power rating | 34 overlapping inputs with some weights pointing the wrong way (a better defensive success rate lowered the rating). It had never been tested as a rating. |
| Team card | OVR, OFF and DEF were ranks stretched to 60-99, so a tiny gap and a huge gap between neighbours looked the same. |
| Player ratings | Hand-picked weights (60/20/20 and so on) with no test behind them. |
| Player ratings | No allowance for sample size: 100 plays counted like 500. The best player was always 100 and the worst 0, so one outlier set the scale. |
| Player ratings | This season only. Defenders rated on raw tackle and sack counts. Kickers on raw field-goal %. |
| Tags | Cut-offs on raw percentiles, never checked for whether they hold up. "Deep threat" used yards per catch although target depth is in the data. |

## Team rating (teamrate.py)

Every play since 1999 goes into a bucket: dropbacks that did not end in a turnover, designed runs, interceptions,
lost fumbles, penalties, field goals, punts, kickoffs, plus successful plays above average, points, and the capped
final margin (blowouts past 21 count as 21). Each bucket is in expected points per game, adjusted for opponents
and home field, weighted toward recent games (3% less per week back), and started from half of last season with
the weight of 9 games. A ridge regression fitted on 2000-2024 gives each bucket the weight that best predicted
future margins. The starting quarterback's value (public 538 formula from box-score stats, adjusted for defenses,
new QBs start at backup level) is added. Rating = Offense + Defense + Special teams + Penalties + Quarterback,
in points per game against an average team.

Test: the rating gap plus home field, on games the weights never saw.

| Rating | 2014-24, each season predicted from the others (2,879 games) | 2025-26 frozen (422 games) |
|---|---|---|
| New SAM rating, no book input | error 171.19, winners 65.1% | error 160.68, winners 63.1% |
| Old SAM rating (with book win totals) | error 173.68, winners 64.9% | error 161.76, winners 61.9% |
| Old SAM rating, win totals removed | error 173.97, winners 64.4% | error 161.69, winners 61.2% |
| Scores only | error 176.33, winners 64.0% | error 165.27, winners 62.6% |

Error is mean squared error of the margin (lower is better). New minus old on 2014-24: -2.49, 95% interval -4.28
to -0.73, so the gain there is real. On 2025-26 it is -1.08 with an interval of -5.36 to +3.41: 422 games is too
few to tell. Over all of 2000-2024 (6,447 games) the new rating beats scores-only by 3.97 (interval 2.63 to 5.36).
Rating after 4, 8 and 12 weeks against the rest of the season: correlation 0.56 / 0.55 / 0.56 for the new rating
and 0.54 for scores-only on 2002-2024; on 2025 alone the two are about level (0.34-0.38 new, 0.37-0.46 scores).
For scale only, never used in the fit: the closing line's error on 2014-24 is 167.36.

## Player ratings (playerrate.py)

- Each stat is pulled toward the position average by an amount measured from odd weeks against even weeks,
  2006-2022. Examples for QBs: sack rate is half-trusted after 260 dropbacks, EPA per play after 340 plays,
  interception rate after 851 attempts. Kicker accuracy above expected: 822 attempts, which is to say almost never.
- Last season counts where it helped predict the rest of the season: QB 0.3, RB 0.5, defenders 0.3-0.5, WR and
  TE 0 (it made them worse), kickers the last two seasons.
- QB, RB, WR, TE: efficiency is a fitted mix of stats, value = (efficiency - bench level) x workload, in expected
  points per game.
- Defenders: each sack (1.45 points), QB hit (0.2), tackle for loss (1.0), stop (0.55), pass broken up (0.75),
  interception (3.5) and forced fumble (2.0) at its measured value, each steadied separately.
- Linebackers with at least one sack or QB hit per five tackles are rated as edge rushers (the stat file labels
  both kinds "LB").
- OVR = 72 for an average qualifier, 9 per standard deviation.

Test on 2023-2025 (not used above): rate after week 8, compare with weeks 9+. Rank correlation, new vs old:
ahead on 13 of 21 measures. Clearly better on defenders' plays made (DL 0.62 vs 0.43, CB 0.59 vs 0.45, EDGE 0.62
vs 0.57), RB yards (0.52 vs 0.44) and WR yards and EPA per game (0.67 vs 0.61, 0.41 vs 0.39). About level for QBs
(0.463 vs 0.462). Behind on sacks per game for EDGE and DL (the old score was mostly sacks) and on raw passes
defended for CB and S. Neither rating predicts kickers (0.04-0.12).

Limits that no formula fixes with public data: no coverage or blocking grades, so corners who are never thrown
at and all offensive linemen cannot be rated properly; pressures are not recorded, only sacks and hits.

## Tags (traits.py)

A tag is shown only if, across 2014-2025, players (or teams) who earned it through week 8 met the same rule on
weeks 9+ alone at least 30% of the time and at least twice as often as chance. 30 of 69 rules passed. Dropped as
noise include: Accurate, Ball-safe and Game manager (QB), Big-play back and Goal-line back, Red-zone threat,
Efficient, Strip specialist, Ball hawk, every kicker tag, and for teams every defensive tag plus Turnover-prone,
Disciplined and Penalty-prone. Kept for teams: Pass-first, Run-first, Strong passing game, Strong run game,
Explosive offense. Deep threat now uses target depth.

## Game picks

Swapping the new team rating into the picks model, keeping its other factors: error 171.34 vs 172.44 on 2014-24
(interval -2.68 to +0.36) and 158.15 vs 158.43 on 2025-26. Not a measurable gain, so the picks model is unchanged.
It still uses book win totals as one input; removing them costs it nothing measurable either.

## All-time (alltime.py)

1999 is the first season with play-by-play. Team seasons use that season's games only. Player seasons are compared
with the same position that same year. Careers before 1999 are cut off at 1999. Regular season only.
