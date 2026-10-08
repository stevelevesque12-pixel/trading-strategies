# Strat Failed 2 + 8/21 EMA ribbon — research notes

Handoff from a Claude chat session (2026-10-07). Everything here is research on the
data in `sample_data/real_multi_instrument/`; nothing has been traded live yet.

## Rules (TheStrat "failed 2", not Al Brooks)

- **Failed 2D (long):** bar breaks the prior bar's low, does NOT break its high
  (outside bars / "3"s excluded), and closes green.
- **Failed 2U (short):** mirror image — breaks prior high, not prior low, closes red.
- **Entry:** market order at the close of the failed 2 bar (fills next bar open).
- **Stop:** 1 tick beyond the failed 2 bar's extreme (the wick).
- **Ribbon filter (strict):** longs only if EMA8 > EMA21 and the failed 2 candle's
  *entire range incl. wick* stays above both EMAs; shorts mirror image.
- **Target:** originally Strat "magnitude" (prior bar's high/low); **0.5R fixed is now preferred.**
- **Session:** entries 09:30–15:45 ET, flatten 15:55 ET, no overnight.

Pine: `tradingview/strat_failed2_ribbon.pine` (set Target = "Fixed R", R multiple = 0.5).

## Files

| File | What it does |
|---|---|
| `bt_failed2.py` | Python replica of the Pine strategy. `run(df, tmode="mag"|float, ...)`, `resample(df, n)` |
| `mc_tradeify.py` | Monte Carlo of Tradeify Select 50K eval + Select Flex funded (day bootstrap) |
| `year.py` | 1-year simulation per account slot (restart on fail/blow, fees, 90% split, copy-trading) |
| `multi.py` | Instrument sweep (MNQ, MES, MYM, M2K, MGC, MCL, SIL) on 5m/15m |
| `combo.py`, `tf_year.py`, `rr_year.py`, `opt.py`, `stress.py` | Sizing / target / timeframe / combination studies |

Run from inside `strat_failed2/` (e.g. `cd strat_failed2 && python bt_failed2.py`). Needs `pyarrow`.

Assumptions: 1 tick slippage per side, $0.62/side MNQ commission, $4 RT for NQ.
Fill model mimics TradingView (signal on close, fill next open; ambiguous bars resolved
by nearer extreme first).

## Key results (1 NQ unless noted)

**Ribbon matters.** Without it every timeframe loses. Allowing the 8 EMA to be touched
(only 21 must be clear) or swapping the ribbon for VWAP (ETH or RTH anchor) both made
results worse — strict rule kept.

**EMA lengths.** Slow EMA 13–50 barely matters. Fast EMA is fragile: 8 works, 9 and 10
break on the 5m May–Aug window. 8/21 kept (standard, not optimized).

**Target.** 0.5R beat magnitude on every longer test:

| Test | Magnitude PF | 0.5R PF |
|---|---|---|
| 4m Aug 3–25 (22 trades) | 1.68 | 2.52 |
| 5m May 11–Aug 25 (64 trades), halves | 1.17 (1.29/1.07) | 1.98 (2.52/1.61) |
| 15m Sep 2025–Aug 2026 (94 trades), halves | 1.29 | 1.68 (1.53/1.84) |

0.5R needs >~67% win rate to break even — monitor live WR.

**Timeframes 1m–15m.** Only 17 sessions of 1m data, so 1m–15m short-window results are
noisy (4m best: PF 2.52 at 0.5R). 1m and 9–12m lose everywhere. Best-evidenced: 5m and 15m.

**Instruments (0.5R).** Only MNQ is profitable on both 5m and 15m. MGC passes 5m but fails
15m. MES, MYM, M2K, MCL, SIL lose.

**10-year check (NQ 15m, 2016–2026, 0.5R):** 993 trades, PF 1.15 overall, but PF 0.44–0.81
every year 2016–2020 (0.77 even with zero costs), PF 1.2–1.7 in 2021–2026 except 2023 (0.90).
**The edge is regime-dependent.** ES 15m loses over 10 years (PF 0.88).

## Tradeify Select 50K → Select Flex simulation

Rules used: $3,000 target, $2,000 EOD trailing DD (real-time), 40% consistency in eval,
min 3 days; funded DD locks at $50,100 once EOD ≥ $52,100 or on first payout; payout every
5 winning days ≥ $150, 50% of profit capped at $2,500, 90% split; $159/month eval fee.
Verify rules with Tradeify before relying on them.

Best sizing found: **3 NQ (30 MNQ) in eval, 2 NQ (20 MNQ) funded, withdraw max ASAP,
restart immediately on fail/blow.**

Yearly money in pocket per account slot, 0.5R target:

| Strategy | As tested | Half edge | Losing-year chance (half edge) |
|---|---|---|---|
| 4m (17 sessions only) | ~$54k | ~$24k | 0% |
| 5m | ~$13–14k | ~$3k | ~50% |
| 15m | ~$10k | ~$5k | ~24% |
| MNQ 5m + 15m + MGC 5m combined | ~$9k | ~$2.6k | ~33% |

The 4m figure is almost certainly optimistic (tiny sample, favourable regime). Plan around
half-edge numbers, ~$2–3k/year per account.

## Next steps

1. Get 6–12 months of **1m MNQ** (NinjaTrader 8 Historical Data export is free; Databento
   gives $125 free credit, dataset GLBX.MDP3, schema ohlcv-1m). Build 4m from 1m.
2. Re-run 4m (and 2m) with split-period durability test; check EMA 7/8/9 all hold.
3. Re-run the Tradeify year simulation on that data.
4. Live kill switch: stop if rolling 30-trade PF < 1.0.
5. Consider using these filters on the ORB Baseline instead of as a standalone setup.

## 1m MNQ test (2026-10-08) — `onemin.py`

Still only the 17 RTH sessions of 1m MNQ in the repo (Aug 3–25 2026). Yahoo is blocked
from the cloud environment and there is no Databento key, so **next step 1 (6–12 months of
1m data) is still open**. Everything below runs on 1 MNQ and is small-sample.

`bt_failed2.run()` now takes `fast=`/`slow=` EMA lengths; `resample()` takes `offset=` (minutes)
to shift bar boundaries.

**TF 1–5m, 0.5R, with split halves (H1 = Aug 3–13, H2 = Aug 14–25):**

| TF | Trades | Net | WR | PF | PF H1 | PF H2 |
|---|---|---|---|---|---|---|
| 1m | 81 | -$256 | 59% | 0.62 | 0.65 | 0.59 |
| 2m | 33 | +$67 | 70% | 1.25 | 2.67 | 0.78 |
| 3m | 27 | -$17 | 63% | 0.95 | 0.73 | 1.11 |
| 4m | 22 | +$217 | 82% | 2.37 | 6.87 | 1.31 |
| 5m | 15 | -$82 | 67% | 0.64 | 2.83 | 0.23 |

(4m PF here is 2.37 vs 2.52 in the earlier table above; the cause isn't confirmed.)

**Bar-alignment check (shifting where the bars start; 0.5R):** 4m is positive at all 4
offsets (PF 2.37 / 1.68 / 1.18 / 2.18). 2m flips to PF 0.39 at offset 1, so 2m is not robust.
3m and 5m are mixed.

**Fast EMA 6–10 on 4m (0.5R):** PF 3.95 / 1.59 / 2.37 / 2.41 / 1.32, so EMA 7/8/9 all hold.
On 2m: 1.07 / 1.54 / 1.25 / 1.19 / 1.26.

**Target on 4m:** 0.3R 1.83, **0.5R 2.37**, 0.75R 1.79, 1R 1.17, 1.5R 1.32. 0.5R stays the best.

**Takeaway:** 4m passes every robustness check available (alignment, EMA, target), but the
sample is 22 trades and H2 is much weaker than H1 (1.31 vs 6.87). 1m loses. 2m is fragile.
More 1m data is still needed before trusting 4m.

Note: `sample_data/daily/nasdaq_index_daily_2026-09-08_2026-10-07.csv` (user upload) is
**daily** index OHLC with no volume (22 days). It can't be used for the 1m/4m tests.

## 3 years of 1m NQ (2026-10-08) — `nq1m.py`

Data: user-supplied `Dataset_NQ_1min_2022_2025.csv`, saved as
`sample_data/real_multi_instrument/real_nq_1m_2022-12-26_2025-12-11.parquet`. It covers
764 RTH sessions with no gaps beyond weekends and holidays. The source timestamps mark the
bar *close*, so they were shifted back 1 minute to match the repo's bar-open convention.
Prices look back-adjusted (continuous contract); point P&L is unaffected. The CSV has exactly
1,048,575 data rows, Excel's row limit, so it may have been cut off at Dec 11 2025.
Assumptions: 1 NQ, $4 RT, 1 tick slippage per side.

**The 4m edge does not hold up.** 4m 0.5R: 900 trades, PF 0.95, −$4.5k
(2023 0.81, 2024 1.08, 2025 0.95). By half-year it ranges from 0.55 to 1.40. The 17-session
PF 2.37 was noise.

| TF | 0.5R PF (trades) | 2023 | 2024 | 2025 |
|---|---|---|---|---|
| 1m | 0.95 (2960) | 0.91 | 0.84 | 1.08 |
| 2m | 0.95 (1654) | 0.84 | 0.93 | 1.05 |
| 3m | 0.88 (1125) | 0.96 | 0.91 | 0.81 |
| 4m | 0.95 (900) | 0.81 | 1.08 | 0.95 |
| 5m | 0.78 (664) | 0.88 | 0.80 | 0.70 |
| 6m | 1.15 (575) | 1.01 | 1.11 | 1.31 |
| 8m | 1.21 (486) | 1.00 | 1.03 | 1.51 |
| 10m | 0.81 (386) | 1.44 | 0.75 | 0.60 |
| 15m | 1.11 (301) | 0.89 | 1.19 | 1.20 |

The 15m yearly PFs match the earlier 10-year NQ 15m study (2023 ≈ 0.90), which
cross-checks the dataset.

**Bar-alignment median PF (0.5R):** 2m ~0.93, 3m ~0.89, 4m ~0.96, 5m 1.05 (range 0.78–1.27),
6m 1.06 (all 6 offsets ≥ 1.04), 8m 1.07 (2/8 offsets lose), 15m 1.11 (3/15 lose).
Shifting where the bars start moves PF by ±0.2, which is as large as any "edge" here.

**EMA and target sweeps at 4m:** PF stays 0.94–1.06 for EMA 6–10 and 0.88–1.05 for targets
0.3R–2R. No setting rescues it.

**Conclusion:** across 1–5m there is no edge after costs. 6m and 15m show a thin PF of about
1.05–1.1 at median alignment. That is too thin for 0.5R in a $2k-drawdown eval, because the
break-even win rate is about 67% and live slippage alone eats it. The Tradeify sim was not
re-run, since the input edge is gone. Don't trade 4m. If anything continues, use 15m (with
the regime caveat above) or use these filters on the ORB Baseline (next step 5).

## Failed 2 scalper search (2026-10-08) — `f2lab.py`, `f2search.py`, `f2final.py`

Goal: find a filter (or filters) that makes failed 2s profitable on 1m–15m, at any R.

**Method (to avoid fooling ourselves):**
- 3 years of 1m NQ. Exits are resolved on 1m bars, and if a stop and target fall in the same
  minute, the stop is assumed hit first.
- Filters are selected on 2023–2024 and judged on 2025. The final check is MNQ Aug 2026,
  which no search ever touched.
- Costs: $4 RT per NQ, or $1.24 RT per MNQ.
- Harsh fills: 2-tick entry slippage, 1-tick stop/EOD slippage, and targets must trade 1 tick
  through the limit to fill.

**What failed:**
1. *Filters on the market-at-close entry.* An exhaustive search of 1–3 filter combos across 12
   timeframes and 6 targets found ~20k combos profitable in both 2023 and 2024, but only 44%
   stayed profitable in 2025, and the strongest training combos generalized *worse*. That is
   pure curve-fitting. Even before costs, the raw failed 2 is about break-even (PF 0.80–1.05).
2. *Single filters across all TFs.* The best ones add only about +0.03–0.05 PF: trend
   alignment (close beyond the ribbon, FTFC with the day/week open, 2h trend), larger bars,
   strong closes. Not enough to fix the entry.
3. *Retest/mid limit entries* are worse than market.

**What worked: entry by trigger, on 1m, on quiet failed 2 bars.**
- The TheStrat-style trigger (a stop order at the break of the failed 2 bar's other end,
  valid 1 bar) is positive unfiltered on 1m (PF 1.10, ~30k trades) but negative on 2m–15m.
- Adding **failed 2 bar range < x · ATR(14)** turns it into the only filter set that
  generalizes: 97–100% of the top training combos were also profitable in 2025.
- The effect is monotonic in the threshold, every year (harsh fills, 0.5R, PF 2023/2024/2025):
  range/ATR 0.4–0.6 → 1.29/1.56/1.68 · 0.6–0.8 → 1.08/1.16/1.26 · 0.8–1.0 → 0.98/0.95/1.05 ·
  above 1.0 → 0.82–0.98.
- A minimum stop of 6 pts removes trades where costs eat the edge (stops of 0–4 pts lose).
- The same rule loses on 2m–15m, so this is a very short-horizon 1m effect.

**Final rule** (`tradingview/failed2_1m_trigger.pine`):
- **Chart:** 1m NQ/MNQ.
- **Setup:** failed 2D (long) or failed 2U (short), where the bar's range is less than
  0.6 × ATR(14) and the stop distance is at least 6 pts.
- **Entry:** buy-stop at the failed 2 bar's high (sell-stop at its low for shorts), cancelled
  if not filled on the next bar.
- **Stop:** 1 tick beyond the wick.
- **Target:** 0.5R.
- **Session:** entries 09:30–15:44 ET, flatten 15:55 ET.
- No ribbon. Adding the strict ribbon improves PF slightly but cuts trades to about 0.5/day.

Results with harsh fills, one position at a time (`python f2final.py`):

| Period | Trades | WR | PF | Net / 10 MNQ | Max DD |
|---|---|---|---|---|---|
| 2023 | 316 | 67% | 1.25 | $4.5k | −$2.1k |
| 2024 | 520 | 69% | 1.51 | $15.3k | −$1.6k |
| 2025 (test year) | 644 | 71% | 1.64 | $26.1k | −$1.5k |
| MNQ Aug 2026 (untouched) | 46 | 67% | 1.57 | $2.0k | −$0.7k |

That's about 2 trades per day, profitable in 11 of 12 quarters. Raising the minimum stop to
8 pts is profitable in all 12 quarters at about 1 trade per day.

**Tradeify Select 50K → Flex** (`year.py` model, 2,000 simulated years), 20 MNQ in eval and
20 MNQ funded:
- As tested: mean ~$15k/yr, P(losing year) 2%.
- At **half edge: ~$4k/yr, P(loss) 29%.** Plan around the half-edge number.

**Risks:**
- Live stop-order slippage on fast 1m breaks is the biggest unknown. At 3 ticks of entry
  slippage, 2023 is close to break-even.
- Webhook latency: the stop order must be live within seconds of the 1m close.
- Confirm that TradersPost accepts `orderType: "stop"` entries with brackets.
- Paper trade first, and track realized entry slippage against the 2-tick assumption.
- Kill switch: stop if the rolling 100-trade PF falls below 1.0.
