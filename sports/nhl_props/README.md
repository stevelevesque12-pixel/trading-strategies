# NHL shots-on-goal & goalie-saves model

Prices NHL player **shots on goal** props, including Sports Interaction's two-way
over/unders and one-way "N+ shots" milestones. It also projects **goalie saves**.
Data comes from the free public NHL API (`api-web.nhle.com`).

## How it works

| Piece | Method |
|---|---|
| Shot rate | Shots per 60 min, recency-weighted (half-life 60 games), shrunk toward the league F/D rate with 150 min of prior |
| Ice time | Recency-weighted TOI (half-life 3 games); games under half of normal TOI (injury exit, ejection) are ignored |
| Opponent | Opponent's shots allowed per game vs league, shrunk (8-game prior) |
| Venue | Home teams take ~4% more shots |
| Distribution | Negative binomial (size fitted on 2024-25) gives P(over/under) for any line |
| Goalie saves | Expected shots against (opponent shots-for x own shots-against) x shrunk save % x 0.96 starter share |

Every projection uses only games played before that game's date.

## Backtest (out of sample: 2025-26 + 2026-27 to date)

Tuned on 2024-25 only. Proxy line = floor(last-10 average) + 0.5. Real odds history isn't freely available.

| Model's side probability | Bets | Hit rate |
|---|---|---|
| 55–60% | 10,992 | 58.1% |
| 60–65% | 10,934 | 62.5% |
| 65–70% | 9,116 | 67.2% |
| 70–80% | 8,820 | 74.0% |

Log loss on shots is 0.645 vs 0.676 for a naive last-10-games Poisson; on saves it is 0.643 vs 0.693.
The model is well calibrated. That still doesn't prove it beats the bookmaker's line.
Only logging real bets can show that.

## Usage

```bash
python -m sports.nhl_props.data            # first run: download 2024-25 onward (~5 min, cached)
python -m sports.nhl_props.slate odds.csv --update --min-edge 0.03 --bankroll 500 --out priced.csv
python -m sports.nhl_props.backtest        # re-run the backtest (--tune for the parameter grid)
```

`odds.csv` columns: `game, market, selection, decimal_odds, status`.

## Before betting a pick

- **Confirm the player is in tonight's lineup.** The model assumes normal ice time.
- **Be suspicious of edges over ~10%.** Usually the book knows something the model doesn't (injury, line change, back-to-back).
- **Goalie saves:** bet only once the starter is confirmed.
- **Line movement:** prices move; re-run close to puck drop.

## Daily paper trading

`paper.py` logs every bet the rule recommends (see its docstring: win-rate
filters, max 2 bets per game, players who missed their last game skipped) and
grades it the next morning at a flat $10 and at a quarter-Kelly stake on a
notional $1,000 bankroll, so the system's record builds up without real money.

```bash
python -m sports.nhl_props.paper pick odds.csv   # log tonight's recommended bets
python -m sports.nhl_props.paper grade           # grade finished games, print daily results
```

The log is `paper_bets.csv`.
