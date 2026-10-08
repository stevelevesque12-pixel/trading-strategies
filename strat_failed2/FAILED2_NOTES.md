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
