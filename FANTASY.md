# Fantasy football (season-long, Sleeper)

Built for Marlon's Sleeper league "Premier Bois Roundtable": 8-team dynasty, full PPR, 4-point passing TDs,
-1 per interception, lineup QB/RB/RB/WR/WR/TE/FLEX/FLEX/K/DEF.

## Scripts

```
python fantasy.py backtest        # fit QB/RB/WR/TE projections on 2017-2024, test 2025-2026 -> fantasy_params.json
python fantasy_kdst.py backtest   # kicker and defense projections -> fantasy_kdst_params.json
python fantasy_vs_sleeper.py      # SAM vs Sleeper's own projections on the same player-games
python fantasy_league.py 10       # best lineup, bench, waiver options for week 10
python test_fantasy.py            # scoring checks, prints ok
```

Sleeper's API is blocked from the cloud sandbox, so the league, roster and Sleeper projection files are pulled on
the PC into `cache/sleeper/` (not committed).

## What the projection uses

Recency-weighted fantasy points, targets, carries, pass attempts, target share, air-yards share, snap share and EPA
(last season counts as 3 games); the team's implied points from the betting total and spread; the opponent's points
allowed to the position; the starting QB's EPA per attempt and whether he changed; the player's Questionable tag
and the targets/carries left by teammates listed Out or Doubtful; wind, cold and domes. Ridge regression per position.

## Backtest (historical only)

SAM vs simple averages, all player-games (start/sit = how often the higher-projected of two players at the same
position in the same week scored more, among players averaging 8+):

| Season | Pos | SAM MAE | Season-avg MAE | SAM start/sit | Season-avg start/sit |
|---|---|---|---|---|---|
| 2025 | QB | 6.30 | 6.73 | 60.7% | 56.8% |
| 2025 | RB | 4.23 | 4.24 | 65.8% | 65.0% |
| 2025 | WR | 4.31 | 4.43 | 62.6% | 58.3% |
| 2025 | TE | 3.50 | 3.60 | 61.3% | 56.5% |
| 2026 wk 1-10 | QB | 6.32 | 6.90 | 63.5% | 60.6% |
| 2026 wk 1-10 | RB | 4.24 | 4.53 | 64.7% | 62.4% |
| 2026 wk 1-10 | WR | 4.35 | 4.55 | 64.5% | 61.5% |
| 2026 wk 1-10 | TE | 3.77 | 4.13 | 65.9% | 59.7% |

SAM vs Sleeper's own projections (Rotowire), same player-games, start/sit among players Sleeper projected 8+:

| Season | Pos | SAM MAE | Sleeper MAE | Avg-of-both MAE | SAM start/sit | Sleeper start/sit | Avg start/sit |
|---|---|---|---|---|---|---|---|
| 2025 | QB | 6.16 | 6.54 | 6.12 | 59.4% | 59.3% | 59.9% |
| 2025 | RB | 4.46 | 4.26 | 4.27 | 63.5% | 65.4% | 65.0% |
| 2025 | WR | 4.50 | 4.30 | 4.33 | 59.8% | 61.2% | 61.1% |
| 2025 | TE | 3.69 | 3.62 | 3.55 | 58.9% | 60.1% | 60.4% |
| 2026 | QB | 6.13 | 6.00 | 6.00 | 62.0% | 62.8% | 63.0% |
| 2026 | RB | 4.37 | 4.22 | 4.20 | 63.8% | 65.5% | 65.2% |
| 2026 | WR | 4.45 | 4.22 | 4.28 | 61.3% | 62.2% | 62.5% |
| 2026 | TE | 3.88 | 3.75 | 3.75 | 61.6% | 59.9% | 60.5% |

Sleeper is a little better for RB and WR; SAM is about even at QB and TE. The average of the two is about as good
as the better one, so `fantasy_league.py` uses the average. Sleeper's files were last saved after each week ended,
so they may include late pre-game news that SAM's backtest did not have.

Kickers and defenses: SAM MAE 3.78 (K) and 4.65/4.26 (DEF, 2025/2026) vs 3.94 and 4.90/4.52 for a recency
average. Kicker scoring is close to unpredictable (correlation 0.06-0.20).
