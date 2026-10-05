# MCL Trend Lab — 24h research loop

Goal: a trend-following MCL strategy (1m–15m) that can pass and survive a
**Lucid 50K Flex** account ($3,000 target, $2,000 EOD trailing drawdown,
50% consistency in eval, optional $1,200 daily loss limit).

Loop window: **2026-10-05 15:20 UTC → 2026-10-06 15:20 UTC.**

## Each iteration

1. Read `trend_lab/results/registry.json` (or the dashboard) — what's working, what isn't.
2. Build 2–4 **new** ideas as a new `trend_lab/strategies/families_vN.py`
   (new families, or refinements of the best families with new filters/exits),
   register it in `trend_lab/strategies/__init__.py`.
3. Optimize: `python -m trend_lab.optimize --families <new> --tfs 5min,10min,15min --n 400 --iteration N --note "..."`
4. Rebuild dashboard: `python -m trend_lab.dashboard` → `dashboard/index.html`.
5. `pytest -q tests/test_trend_lab.py`, commit, push to `claude/stoic-pascal-acfwf9`.

## Rules (anti-overfitting)

- Select configs on **in-sample only** (first 60% of days). Never pick by OOS.
- A strategy is only a "candidate" if OOS PF ≥ 1.25 with ≥ 25 OOS trades AND
  the median OOS PF of its top-10 in-sample configs ≥ 1.05 (robust region, not a lucky point).
- Prefer the 15m dataset (≈11 months) for conclusions; 5m/10m only cover ≈3.5 months,
  1m only ≈3.5 weeks.
- Costs always on: $1.24 RT commission + 1 tick slippage per side.
- Late in the loop: re-validate the best candidates with a walk-forward
  (rolling re-optimization) and parameter-neighborhood checks before calling one "the" strategy,
  then port the winner to Pine (`tradingview/`) with TradersPost alerts.

## Finding at iteration 6 (2026-10-05 ~16:00 UTC) — method change

The MCL Trend Dip winner (walk-forward PF ~1.5 on MCL) does **not** generalize:
with frozen settings it is negative on 10-year GC/SI/ES/NQ and on most
micros (see "Cross-market robustness" on the dashboard). So from here on, new
families are screened with `python -m trend_lab.xm_screen` (add `--with-mcl` to make MCL's first 60% a 5th training market): optimize in R-terms on
the 10-year GC/ES/NQ/SI histories (2016-2022), test on 2023-2026 for those
markets AND on all of MCL. A family only counts if it holds up on both.

## Iteration 9 (Engine A refinements)

Anchored walk-forward of trend_dip_atr_plus: best-score selection PF 1.70 / +$3,119 / DD $815
(baseline Engine A: PF 1.57 / +$2,353 / DD $915); plateau selection PF 1.51 / +$1,859.
Both modes chose a 12-bar time stop + 3R target + no breakeven in every window; no ADX/ER gate
was ever chosen. Added as an opt-in Pine input (A: time stop), defaults unchanged.

## Iterations 10-11

- Engine B widened for frequency (trend_dip_rsi_plus): walk-forward PF 1.05 (best) / 1.99 on 16 trades
  (robust) -- fewer trades than the original. Original Engine B kept.
- Timeframe transfer, NO refitting (bar-count params scaled to the same clock time), May 11 - Aug 25:
  15m PF 1.84 (73 trades) | 10m PF 1.58 (101 trades) | 5m PF 0.97 (237 trades).
  Holds down to 10m, breaks at 5m (costs are ~3x larger in R, dips are noise). A 10m chart variant
  is a forward-test option for more trades/week, but has only ~3.5 months of history behind it.

## Iteration 12-13: diagnostics, 30m transfer, conservative Monte Carlo

- OOS trade breakdown: both sides profitable (long PF 1.72, short 1.39), most entry hours positive,
  4/5 weekdays positive (Wed 0.83), every month since April positive. No filters added (would be fitting).
- 30m transfer (params x0.5): PF 0.66 / 1.09 -- fails. Works on 10m-15m only.
- Oct-Mar (fit period) at $300 risk: PF 1.28, max DD $2,208 (> Lucid's $2,000). Full-year Monte Carlo is
  therefore the conservative case. Pine default sizing changed to 20% of cushion, $150-$500
  (full-year: pass 63% / fail 11%, median 48 days; unseen-only: 66% / 4%).
