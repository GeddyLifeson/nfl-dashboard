# Multi-factor model test (2026-11-10)

`features.py` builds, for every regular-season game since 2013, features known before kickoff (home minus away):
the current scores-based prediction; opponent-adjusted EPA and success rate per play (overall, pass, rush, offense
and defense, last season as a prior); special-teams EPA; turnover margin; starting-QB edge and missing starters by
position (lineup.py); rest, byes, time-zone travel, West-Coast early kickoffs, wind x passing strength; new or
mid-season head coach. Game list from nflverse/nfldata (`cache/games_2012.csv`).

`mf_backtest.py` fits ridge regressions of the final margin on several combinations (lambda by leave-one-season-out),
frozen, then tested on every 2025-26 game with a closing line (408 games). Results: `mf_results_*.txt`.

Run: `python research/features.py` then `TRAIN_FROM=2014 python research/mf_backtest.py`.
