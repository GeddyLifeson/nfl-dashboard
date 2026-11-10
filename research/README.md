# Multi-factor model test (2026-11-10)

`features.py` builds, for every regular-season game since 2013, features known before kickoff (home minus away):
the current scores-based prediction; opponent-adjusted EPA and success rate per play (overall, pass, rush, offense
and defense, last season as a prior); special-teams EPA; turnover margin; starting-QB edge and missing starters by
position (lineup.py); rest, byes, time-zone travel, West-Coast early kickoffs, wind x passing strength; new or
mid-season head coach. Game list from nflverse/nfldata (`cache/games_2012.csv`).

`mf_backtest.py` fits ridge regressions of the final margin on several combinations (lambda by leave-one-season-out),
frozen, then tested on every 2025-26 game with a closing line (408 games). Results: `mf_results_*.txt`.

Run: `python research/features.py` then `TRAIN_FROM=2014 python research/mf_backtest.py`.

## Ten changes from how books set lines (2026-11-10)

`mf_tests.py` (results in `mf_tests_results.txt`) and `early_lines.py` (`early_lines_results.txt`) test the changes in
`how_books_set_nfl_odds.md` against the all-factor model, on the same frozen 2025-26 games. A change is kept only if it
lowers out-of-sample log loss. None of 1, 2, 3, 7 did by more than noise, so the production model (`multifactor.py`,
`sam_params.json`) is the all-factor model E. 4 and 9 need data nflverse does not keep (daily injury reports, opening
lines); 9 was run on the 44 games the dashboard archived an early line for.
