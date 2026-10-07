# MCL Trend Lab — 24h research loop

Goal: a trend-following MCL strategy (1m–15m) that can pass and survive a
**Lucid 50K Flex** account ($3,000 target, $2,000 EOD trailing drawdown,
50% consistency in eval, optional $1,200 daily loss limit).

Loop windows: 2026-10-05 15:20 → 10-06 15:20 UTC (ended after it17, session idle); 2026-10-07 14:35 → **2026-10-08 14:35 UTC**.

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

## Iteration 14: equity-curve kill switch (untuned, single rule tested)

Live orders only while closed-trade equity >= mean of the last 20 closed-trade equity values
(shadow-trade otherwise). Frozen MCL Trend Dip, $300 risk:
full year PF 1.39 -> 1.48, DD $2,209 -> $1,831 | Oct-Mar PF 1.28 -> 1.33, DD $2,208 -> $1,645 |
Mar-Aug PF 1.54 -> 1.71. Costs ~15% of net and ~30% of trades.
Full-year Monte Carlo with the switch: 25% cushion $200-$600 -> pass 63.5% / fail 9.9% / median 47d,
funded breach 8.6% (vs 63.1 / 11.4 / 48 / 10.1 without it at 20% $150-$500).
Pine defaults now: kill switch on, cushion 25% $200-$600; cushion tracks live-trade P&L only.

## Iteration 15: Pine-parity simulator (trend_lab/combined.py)

One position, engine A priority, per-engine exits, shared daily counters, kill switch with shadow trades,
cushion sizing on live P&L -- mirrors tradingview/mcl_trend_dip.pine. Fixed $300: 221 trades, PF 1.39
(merged-engine portfolio: 222, PF 1.39). Kill switch: identical to iteration 14. Parity test added.

## Iteration 16: trend-definition sensitivity (Engine A, all else frozen)

Mar20-Aug: all 8 variants (EMA / Supertrend / Donchian mid / linreg slope, with/without base EMA100)
profitable, PF 1.04-1.72. Oct-Mar: only the fitted EMA definition is clearly positive (1.44); others
0.85-1.18. => dip-buying in trend worked broadly in spring-summer 2026 crude; Oct-Mar result is partly fit.
Regime-dependent edge -> the kill switch matters more than the exact trend indicator. Base EMA100 filter
helps in 7/8 comparisons.

## Iteration 17: regime diagnostics (descriptive, nothing fitted)

Full-year trades by prior 20-day avg daily range (terciles): low vol PF 1.96, mid 1.60, high 1.17.
By prior 20-day daily return: against it PF 1.99 (76 trades), with it PF 1.31 (115).
=> edge is a short-horizon (1h/4h) dip-in-trend effect that degrades in high volatility; consistent
with the 1.5-pt max-stop filter carrying part of the edge. Not adding a vol threshold (would be fit on
the same data); a daily-trend filter would hurt.

## Iteration 18 (2026-10-07): Engine A scale-out (untuned: half off at +1R)

Engine A alone, identical entries: max DD $1,645 -> $785 (Oct-Mar) and $1,540 -> $821 (Mar-Aug), win rate
~40% -> ~58%, PF 1.44 -> 1.59 / 1.53 -> 1.51. Full A+B with kill switch (Pine-parity sim):
Oct-Mar PF 1.33 -> 1.58, DD $1,645 -> $838; Mar-Aug PF 1.71 -> 1.52, DD $1,314 -> $1,144.
Full-year Monte Carlo, 25% cushion $200-$600: eval pass 63.5% -> 71.1%, fail 9.9% -> 2.2%,
funded breach 8.6% -> 2.1%, median days 47 -> 53. Now the Pine default (input "A: scale out half at R").
Entry webhooks now carry only a protective stop (no broker TP); partial exit sent as exit+quantity.

## TradingView parity check (2026-10-07, user's trade-list export)

User's TradingView run (365d to Oct 7): PF 1.10, +$2,092, DD $3,402 vs simulator PF 1.38.
`python -m trend_lab.tv_compare <csv>`: entry prices identical on all 164 matched trades (data is the
same). Two Pine bugs, both fixed:
1. Default 100% margin -> TradingView silently rejected every order above $50k notional: 41/41 such
   sim trades missing (the tight-stop, highest-qty trades; +$4.7k). Fix: margin_long/short = 5.
2. Scale-out never filled: exit orders get the position in creation order and the full exit (XL/XS)
   was created on the signal bar before the partial. Fix: create PL/PS first; cancel stale partials when flat.
Aug 26 - Oct 7 (never seen by the research): 26 trades, PF 1.32, +$939 in TradingView.

## TradingView re-run after the margin fix (2026-10-07)

365d to Oct 7: +$7,700, PF 1.355, max DD $2,491 (was +$2,092 / 1.10 / $3,402). tv_compare: all 213 overlap
trades matched (entry time+price identical); per-trade P&L matches the NO-scale-out simulation on 209/213
-> margin fix confirmed, but the +1R partial still never filled (creating PL before strategy.entry did not
help). Fixed with explicit quantities on both exits (PL = half, XL = rest until PL fills, detected vs the
entered qty). Aug 26 - Oct 7: 30 trades, PF 1.27, +$1,005.

## TradingView run 3 (explicit-qty scale-out) -- parity reached

PL/PS exits now fill (103 + 95). Per-trade P&L matches the scale-out simulation on 208/213 overlap trades.
365d: +$7,258, PF 1.336, max DD $2,033 (vs no-scale run: +$7,700, 1.355, $2,491). TradingView's tester
counts shadow trades (kill switch only gates webhooks) and uses fixed $300 before "Account start".
By period (TradingView positions): Oct-Mar PF 1.26 (+$2,707), Mar 20-Aug 25 PF 1.53 (+$3,916),
Aug 26-Oct 7 (new data) PF 1.17 (+$630, 30 trades).

## Iteration 19 (2026-10-07): prop-firm yearly sim, second market, funded sizing

- `python -m trend_lab.propsim`: Tradeify Select Flex 50K, one account, 1 year, 10k paths. Edge as last year:
  median +$3,046 (mean +$3,579), P(profit) 79%, ~4.9 payouts. Half edge: median -$721, P(profit) 39%.
- MGC (micro gold) with the FROZEN MCL settings (never fitted on gold): PF 1.15 (Oct-Mar) / 1.34 (Mar-Aug),
  daily correlation with MCL 0.04, days >= $150: 22% -> 34%. Tradeify sim MCL+MGC at $200 base each: median
  +$4,068, P(profit) 83%, 6.6 payouts, but funded failures 0.31/yr vs 0.05. Scaled down to $140 each: worse
  than MCL alone. => optional aggressive variant, not the default.
- Funded-phase sizing grid: more aggressive (30-40%, up to $800-$1,200) raises the mean year but not
  P(profit), multiplies funded failures, and lowers the median if the edge halves. Keep 25% / $200-$600.

## Iteration 20 (2026-10-07): stop-order entry for Engine A -- rejected

Buy/sell-stop at the prior bar's high/low (fills on the break) vs current close-confirmed market entry,
same exits incl. +1R half, $300 fixed (trend_lab/experiments/stop_entry_vs_market.py):
market: 200 trades, PF 1.47, +$7,411, DD $1,639 | stop: 272 trades, PF 1.29, +$7,297, DD $1,954.
Close confirmation filters false breaks; keep the market entry.

Also: Tradeify yearly sim at FIXED $500 risk: median +$4,281, P(profit) 85%, ~2.1 evals bought, 0.63 funded
accounts blown/yr (cheap there: $159/month, no activation). Half edge: median +$318, 54%.

## Iteration 21 (2026-10-07 19:50 UTC): time stop re-test with the scale-out on

Pine-parity sim, half at +1R, fixed $300: default Oct-Mar PF 1.27 / Mar-Aug 1.53; time stop 12 bars:
1.37 / 1.45; 16: 1.30 / 1.54; 24: 1.28 / 1.53. No consistent gain -> stays opt-in, off.
