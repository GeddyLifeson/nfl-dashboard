# NFL Power Rankings

GeddyLifeson's Power Rankings for the 2026 NFL season: the data, the model code, and the weekly method.

Last updated Monday, Oct 19, 2026, with all games through Sunday, Oct 18.

## What is here

- `n3/` data files.
  - `games.csv`: 2026 scores, weeks 1 to 6.
  - `box_0.csv` to `box_3.csv`: 2026 per-game box score team stats (88 of 92 games; four Sunday games had no box score yet).
  - `y25_1.csv` to `y25_8.csv`: 2025 regular season box scores, 272 games.
  - `y23_scores.csv`, `y24_scores.csv`: 2023 and 2024 game scores.
  - `tot_*`, `pass_*`, `rush_*`, `downs_*`, `score_*`, `to.csv`, `pen.csv`, `punt_*`, `pr.csv`, `kr.csv`, `qb.csv`: 2026 team and QB season tables from FootballDB.
- `build3.py` builds the published rankings: about 145 inputs, each weighted by sign(rho) times rho squared (Spearman correlation with win%), composite regressed on scoring margin, plus strength of schedule. Output is `league_data.json` and `weights4_out.json`.
- `wf/` walk-forward tests. Each week is predicted using only earlier games.
  - `lab.py`, `tune.py`, `frozen.py`: results-based ratings tuned on 2023 and 2024, then tested on 2025 and 2026.
  - `walk.py`, `walk2.py`, `run25.py`, `run26.py`, `final.py`: earlier experiments with box score stats.
- `ledgerline/` ingest log that records the source, hash, row count and freshness of each stat table.
- `bayes.py`, `bayes2.py` experimental Bayesian engine.
- `design/Main.dc.html` source of the published graphic. `tiktok/` is the 1080x1920 photo. `tt1.py` makes it.
- `gen_md3.py` generates the markdown for the stats document.

Scripts use relative paths (`n3/`, `wf/`, `ext/`), so run them from the repo root, for example `python3 -I wf/frozen.py`.

## What the tests found

Out-of-sample, picking game winners:

| Test | Model | Home-team-only baseline |
|---|---|---|
| 2025, weeks 4 to 18 (224 games) | 58.7% | 52.7% |
| 2025, weeks 2 to 18 (256 games) | 61.2% | 53.5% |
| 2026, weeks 2 to 6 (76 games) | 61.3% | 47.4% |

Settings tuned on 2023 and 2024: rating shrunk toward average (k=6), 60% of last year's final rating carried over, home field worth 2 points, no cap on blowouts.

Games where the model leans hardest are picked better: about 71% for the top quarter and 77% for the top tenth (298 games, 2025 and 2026 from week 3). Box score stats added nothing beyond game results in these tests.

## Sources

Scores and stats from FootballDB. Box score team stats were read from each game's page.
