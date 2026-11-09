# NBA rebounds / assists / threes model

The NHL shots model's approach applied to the NBA, with box scores from ESPN's
public API (`site.api.espn.com`).

| Piece | Method |
|---|---|
| Per-minute rate | Recency-weighted (half-life 40 games), shrunk toward the league rate for the position group (G/F/C) with 150 min of prior |
| Minutes | Recency-weighted (half-life 5 games), every game counted |
| New season | Last season's rates keep half their weight and minutes a fifth at a player's first game; league averages fade with a ~140-day half-life |
| Opponent | Opponent's allowed per-game total of the stat vs league, shrunk (10-game prior) |
| Distribution | Negative binomial, spread per stat fitted on 2024-25 |

## Backtest (out of sample: 2025-26 + 2026-27 to date)

Proxy line = floor(last-10 average) + 0.5. Model **overs** hit 2-20 points under their
stated probability and **points** unders 4-7 points under, so the rule bets only
**unders on rebounds, assists and threes**. With a 1.5-point haircut on assists
and threes, unders the model rates at 62%+ hit **69.4% vs 69.7% predicted**
(~21,800 bets). Projections run a few percent high in October and low in April,
when rotations shift.

## Usage

```bash
python -m sports.nba_props.data              # download / refresh box scores (~1 min first time)
python -m sports.nba_props.backtest          # calibration tables
python -m sports.nba_props.paper pick odds.csv
python -m sports.nba_props.paper grade
```

`odds.csv`: `date, game (Away @ Home), start_time_ET, team, player, market, line, over_odds, under_odds`,
with `market` one of Rebounds / Assists / Threes / Points.

## Results page

"NBA Paper Picks" dashboard: https://claude.ai/artifact/RnYgenf7hcVmuwkPYrovfW
After each `grade` (and `pick`), upload `paper_bets.csv` to it as a new asset and point the `picks` dataset at the new file.
