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
