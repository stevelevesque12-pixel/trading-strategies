# Failed 2s (TheStrat) — intraday automation for Tradovate

Codifies the "Failed 2s" multi-timeframe reversal system (Failed-2 bias +
Market Structure Shift/FVG entry trigger) popularized in TheStrat community,
with a backtester and a Tradovate live-execution layer.

**This was built from research, not from watching the source video** — the
X/Twitter video this was requested from couldn't be accessed from this
environment (egress to x.com is blocked here). The rules below are a
best-effort reconstruction from public TheStrat/Trader-Mike material (see
research summary earlier in this conversation). Validate against your own
understanding of the setup before trusting it with money.

**A second, higher-frequency strategy also lives in this repo**:
`structure_scalp/` -- 1-minute structure direction as bias, then on the 5s
chart: wait for a pullback against that bias, wait for the 5s to break back
in the bias direction (the reversal leg), anchor a VWAP to that leg's start,
and enter on the retest touch of that VWAP. Stop = the leg's low/high,
target = a fixed R multiple. See `structure_scalp/strategy.py` and
`tradingview/structure_scalp.pine` (works for MES or MNQ -- same tick size,
just set the point-value input per symbol). It reuses the swing/MSS
primitives below but is a different strategy from Failed-2s, built because
Failed-2s trades too infrequently to be practical for a prop-firm payout
timeline. An earlier version of this strategy entered immediately on every
5s structure break while 15m+1m aligned; that traded more often but mostly
on 5s noise, so it was replaced with the pullback+VWAP-retest sequence
above for better trade quality. Not backtested against real data -- no free
source available provides sub-1-minute history, so it's logic-tested
against synthetic bars only (`tests/test_structure_scalp.py`); forward-test
carefully before trusting it.

**A third strategy, `fair_value/`, codifies "JJ Simon's Fair Value Theory
NQ Strategy"** -- see the dedicated section below.

## Strategy logic (Failed-2s)

1. **Bias timeframe** — a Failed-2 (`F2U`/`F2D`) completes: a directional (2)
   candle that reverses and closes back against its own break (a failed
   liquidity sweep). This sets a directional bias and the swept level.
2. **Entry timeframe** — while that bias is active and we're inside the
   intraday entry window, wait for a Market Structure Shift (a strong-bodied
   close through the last confirmed opposite swing point) in the bias
   direction, confirmed by a Fair Value Gap. That's the entry trigger.
3. **Stop** — beyond the tighter of the broken swing point or the bias
   timeframe's swept level, plus a small tick buffer.
4. **Target** — `target_r * risk` (default 1:1).
5. **Intraday only** — all state (bias, swing structure) resets every session
   day. No new entries after `no_entry_after` (default 15:45 ET); any open
   position is force-flattened at `flatten_at` (default 15:55 ET).

Timeframe pairs implemented (entry timeframe, bias timeframe):

| Pair | Entry TF | Bias TF |
|---|---|---|
| `1m-15m` | 1 minute | 15 minute |
| `5m-1h` | 5 minute | 1 hour |
| `15m-4h` | 15 minute | 4 hour |

Code layout:
- `failed2s/` — pure strategy logic (bar classification, Failed-2, swing/MSS/FVG, the signal engine, risk manager, instrument specs). Shared by backtest, live, and the webhook path.
- `backtest/` — event-driven backtester over OHLCV CSV data.
- `live/` — Tradovate REST/WebSocket client + a Python live runner (polls/streams Tradovate directly).
- `tradingview/` — the current recommended way to go live: a Pine Script port runs on TradingView (which has the market data and computes risk-based position size), fires a webhook formatted for **TradersPost** (a hosted bridge that connects to Tradovate with a regular login — no paid API Access Add-On needed, which matters on a prop-firm sim account). See **TRADINGVIEW_WEBHOOK.md** for setup.
- `webhook/` — a self-hosted alternative to TradersPost (`webhook/server.py` talks to Tradovate directly), kept for if/when direct Tradovate API credentials become available. See the "Alternative" section at the bottom of TRADINGVIEW_WEBHOOK.md.
- `tests/` — unit tests for the strategy logic, backtest smoke tests, and webhook sizing/routing tests.

## Setup

```bash
pip install -r requirements.txt
```

## Backtesting

Data format: a CSV with `timestamp,open,high,low,close[,volume]` at
**1-minute** resolution (all three pairs are resampled from this single base
resolution).

**Real historical data.** `sample_data/fetch_real_data.py` pulls genuine
historical 1-minute bars (OANDA S&P 500 / Nasdaq-100 index CFD data,
republished under GPL-3.0 by the `FutureSharks/financial-data` GitHub repo)
and converts them to this repo's CSV schema:

```bash
python sample_data/fetch_real_data.py --instrument SPX500_USD --start-year 2018 --end-year 2019 --out sample_data/real_spx500_2018_2019.csv
```

Read the caveats at the top of that script before trusting results: this is
an index CFD proxy (not literal CME ES/MES tick data), coverage only goes
through mid-2020, and volume is OANDA's tick count, not real exchange
volume. It's genuine market data though — good enough to see whether the
pattern has any real edge, not a substitute for validating against recent
CME futures data before going live. For that, see Databento or FirstRate
Data (both require a paid/API-key account not set up in this environment).

A synthetic-data generator is also included, but only for smoke-testing the
pipeline (pure random walk, no real edge signal to find):

```bash
python sample_data/generate_sample.py --days 20 --out sample_data/sample_1min.csv
```

**Single pair:**

```bash
python -m backtest.run --data sample_data/sample_1min.csv --pair 5m-1h --symbol MES
```

**Compare all three pairs on the same data** (this is what you want to
figure out which pair performs best):

```bash
python -m backtest.compare --data sample_data/sample_1min.csv --symbol MES
```

Prints a side-by-side metrics table (num trades, win rate, total P&L,
profit factor, avg R, max drawdown, expectancy) and writes:
- `trades_<pair>.csv` — a full trade log per pair
- `pair_comparison.csv` — the summary table

Both CLIs support `--symbol` (MES, ES, MNQ, NQ, MCL, CL, MGC, GC),
`--contracts`, `--daily-loss-limit`, `--max-daily-trades`, `--target-r`.

**Backtest assumptions/limitations** (so you know what you're looking at):
fills are simulated at the triggering bar's close and stop/target hits are
checked on subsequent entry-timeframe bars' high/low (one bar of latency, no
slippage/commission modeled); if a bar's range hits both stop and target the
stop is assumed to fill first (conservative). 4-hour resampling floors to
UTC-clock-hour boundaries via pandas, which may not perfectly match how your
data provider aligns 4H candles. The engine force-closes any open position
the instant it sees a bar dated after the entry day (in addition to the
normal same-day flatten-at cutoff), so a position can never silently ride
across multiple sessions -- but if the data feed itself has no bars between
the cutoff and a holiday reopen (Thanksgiving, Christmas, etc.), the
position closes on the first bar that exists, which can be a day or two
later purely because there's no earlier price to close it at. This showed
up in the real SPX500 backtest (2 of 1330 trades, both on US holiday
weekends) -- worth checking your trade log for exit_reason=session_flatten
rows with a multi-day gap before relying on this unattended over holidays.

## Live trading on Tradovate

Three ways to go live:

- **TradingView + TradersPost** (`tradingview/failed2s_mes.pine`) — the
  current recommended path. TradingView runs the signal logic and computes
  risk-based position size directly in Pine, fires a webhook formatted for
  TradersPost, which connects to Tradovate with a regular login (no paid
  API Access Add-On needed). See **TRADINGVIEW_WEBHOOK.md**.
- **TradingView + self-hosted webhook** (`webhook/`) — same idea, but you
  host the bridge yourself instead of paying for TradersPost. Needs direct
  Tradovate API credentials (CID/Secret), which require the paid API
  Access Add-On on a live funded account — not available on most prop-firm
  sim accounts. See the "Alternative" section of TRADINGVIEW_WEBHOOK.md.
- **Python runner** (`live/runner.py`, below) — runs the same
  `failed2s/strategy.py` logic directly against Tradovate's own market data
  feed, no TradingView involved, fixed 1-contract sizing. Same API access
  requirement as the self-hosted webhook path.

**Read this before pointing any of these at a funded account.** The Tradovate client
(`live/tradovate_client.py`) follows Tradovate's public API docs but has
**not been tested end-to-end against a real Tradovate account** from this
environment (no credentials, no network access to tradovateapi.com here).
Validate every call against a **demo** account first.

Environment variables:

```bash
export TRADOVATE_USERNAME=...
export TRADOVATE_PASSWORD=...
export TRADOVATE_APP_ID=...
export TRADOVATE_CID=...        # from Tradovate API settings
export TRADOVATE_SEC=...        # from Tradovate API settings
export TRADOVATE_APP_VERSION=1.0   # optional
export TRADOVATE_DEVICE_ID=failed2s-bot  # optional
```

Dry run (default — logs signals, sends no orders):

```bash
python -m live.runner --pair 5m-1h --symbol MES --account-spec <your_username>
```

Live (demo environment, explicit opt-in):

```bash
python -m live.runner --pair 5m-1h --symbol MES --account-spec <your_username> --live --env demo
```

Only after demo validation, live/real money:

```bash
python -m live.runner --pair 5m-1h --symbol MES --account-spec <your_username> --live --env live
```

Known gap: the runner logs each signal and places a bracket order but does
not yet wire Tradovate's order-fill/user-sync WebSocket back into the risk
manager, so the daily-loss-limit lockout won't see live fills until you add
that callback (`LiveRunner._on_finished_bar` has a note where to hook it
in). Don't rely on the daily loss limit unattended until that's wired up.

## Fair Value Theory (FVT) strategy — NQ

Codifies "JJ Simon's Fair Value Theory NQ Strategy" (source: a PDF slide
deck). Lives in `fair_value/` (strategy logic) and
`backtest/fair_value_engine.py` + `backtest/run_fair_value.py` (backtester).
Reuses the same `Bar`/`SwingTracker`/`detect_mss` primitives as Failed-2s
(`failed2s/bars.py`, `failed2s/structure.py`) rather than re-implementing
swing/structure detection.

**Rules, as given in the source material:**

1. Two intraday windows, NY time: **9:30-11:00** and **14:00-15:00**. Each
   window's "fair value" anchor is the open price of its first 1-minute bar
   (9:30 open / 2pm price) -- the idea being that absent new information,
   price tends to revert to that anchor.
2. First **~10-15 minutes** of a window ("continuation" phase): look for a
   displacement candle + BOS/MSB *away* from fair value. Rest of the window
   ("reversion" phase): look for a displacement candle + BOS/MSB *back
   toward* fair value.
3. **Entry**: market order on the signal bar's close, targeting 1.5R, no
   trade management (no breakeven/trailing -- it's a fixed stop and target).
4. **Stop/target distance** comes from an ATR bucket: ATR > 20 -> 50pt
   SL / 75pt TP; 7-20 ATR -> 25pt SL / 37.5pt TP; ATR < 7 -> 16.5pt SL /
   24.75pt TP. Contracts are sized (1-3) so `stop_points * $20/pt *
   contracts` lands near $1,000 risk/trade, per the source material.
5. Avoid the first 3 minutes after the 9:30 open.

**Two interpretation calls the source leaves implicit, made explicit in
code** (see the docstring in `fair_value/strategy.py`):
- **Displacement candle** uses the source's own "more mechanical
  definition": counter-wick <= 20% of the candle's high-low range.
- **ATR** is computed on the entry timeframe itself (1-minute bars, the
  strategy's stated general parameter), not a daily ATR -- the given bucket
  thresholds (7 / 20 points) match a 1-minute NQ true range, not a daily one
  (which runs into the hundreds of points for NQ).

**Deliberately not implemented** -- the source material itself flags these
as discretionary / unconfirmed rather than core mechanical rules:
2nd-attempt re-entries, session VWAP as a discretionary confluence filter,
8:30am red-folder-news reversions, and tagging extra continuation trades
after price returns to fair value. The PDF's suggestion to also test first
90m after Asia/London opens isn't implemented either (NY-hours windows
only) but `fair_value/strategy.py`'s `SessionWindow` list is generic, so
more windows are a small addition if you want to test that.

One optional filter *is* wired up: `--restrict-to-first-hour` skips entries
in a window's 2nd hour, which the source flags as a possible optimization
worth testing (mixed results below -- helped on one dataset, hurt on the
other). `--windows` selects which session window(s) to trade (default
`AM,PM`; see the AM-only comparison below for why you'd want just `AM`).

**Run it:**

```bash
python -m backtest.run_fair_value --data sample_data/real_nas100_2016_2020.csv --symbol NQ
python -m backtest.run_fair_value --data sample_data/real_nq_1min_2022_2025.csv --symbol NQ --windows AM
```

`--symbol` supports the same instruments as Failed-2s (NQ's point value is
$20/point). Writes `trades_fair_value.csv` (now includes `window` and
`phase` columns, useful for exactly this kind of breakdown) and prints
summary metrics.

**Primary results: real NQ futures data.** A private 1-minute NQ dataset
(2022-12-26 through 2025-12-11, ~1.05M bars, real CME exchange volume) was
supplied directly for this backtest. It is **not included in this repo** --
likely paid-vendor data, kept out for licensing/privacy reasons the same way
no other real historical CSV is committed here. It's gitignored at
`sample_data/real_nq_1min_2022_2025.csv` (matches the existing
`sample_data/real_*.csv` ignore rule); supply your own file at that path (or
pass `--data <path>`) to reproduce these numbers or extend the range. One
thing worth knowing: the file is exactly 1,048,576 lines -- Excel's row cap
-- which is a common fingerprint of a CSV that got opened/saved in Excel and
silently truncated. The internal gaps all line up with real market closures
(weekends, Christmas, Thanksgiving half-days, etc.) with nothing that looks
like a truncation artifact mid-file, so it reads as genuinely continuous --
but it's possible data past 2025-12-11 existed upstream and got cut off when
this file was prepared.

**Results** (`--symbol NQ`, default $1,000 target risk/trade,
2022-12-26 through 2025-12-11):

| | num_trades | win_rate | profit_factor | avg_r | expectancy/trade | total_pnl | max_dd |
|---|---|---|---|---|---|---|---|
| All trades | 1,372 | 43.3% | 1.10 | 0.051 | $51.03 | $70,020 | $26,665 |
| `--restrict-to-first-hour` | 1,310 | 42.8% | 1.09 | 0.048 | $47.64 | $62,410 | $33,340 |

The first-hour-only filter (the source PDF's own suggested optimization)
made things slightly *worse* here, the opposite of what it did on the CFD
proxy data below -- treat that filter as unconfirmed either way rather than
a reliable improvement.

Breaking the unrestricted run down by window/phase:

| Segment | n | win_rate | profit_factor | expectancy/trade |
|---|---|---|---|---|
| AM window (9:30-11:00) | 1,178 | 43.7% | 1.12 | $66.10 |
| PM window (14:00-15:00) | 194 | 40.7% | 0.92 | **-$40.30** |
| Continuation phase | 895 | 42.7% | 1.10 | $58.10 |
| Reversion phase | 477 | 44.4% | 1.08 | $37.80 |
| Long trades | 684 | 43.4% | 1.08 | $45.30 |
| Short trades | 688 | 43.2% | 1.11 | $56.80 |

**Running `--windows AM` explicitly confirms it**: dropping the PM window
improves every metric, not just the ones that exclude PM by construction
(max drawdown, in particular, isn't a simple sum, so this wasn't guaranteed
in advance):

| | trades | win_rate | profit_factor | expectancy/trade | total_pnl | max_dd |
|---|---|---|---|---|---|---|
| AM+PM (both windows) | 1,372 | 43.3% | 1.10 | $51.03 | $70,020 | $26,665 |
| **AM-only** (`--windows AM`) | 1,178 | 43.7% | **1.12** | **$66.08** | **$77,840** | **$21,165** |

+11.2% total P&L on 194 fewer trades, +29.5% expectancy/trade, -20.6% max
drawdown. PM entries can never affect AM trades (AM always closes by 11:00,
hours before the 14:00 PM window opens each day), so the AM-only run is
exactly the AM subset of the combined run -- but the max-drawdown
improvement wasn't guaranteed by that alone and shows up in practice too.
**If trading this live, AM-only is the reasonable starting scope**, not the
full two-window version.

By year: 2023 $33.7/trade (444 trades), 2024 $59.8/trade (475 trades), 2025
$55.4/trade (445 trades) -- positive in every full year covered, including
2023's choppier regime, which is a better robustness signal than a single
bull run. Long and short were both solidly profitable here too (unlike the
CFD-proxy run below), consistent with 2022-2025 covering both the 2022 bear
tail and the 2023-2025 rally rather than one-directional drift.

**The AM-vs-PM split is now corroborated by two independent datasets**: the
PM window (14:00-15:00) was a net loser both here (-$40.30/trade) and on
the earlier NAS100 CFD proxy (-$13.60/trade, see below) -- across different
instruments, different data sources, and non-overlapping time periods. A PM
window that consistently costs money even with the strategy's own
1.5R/no-management structure is one of the stronger findings from this
backtest, not just a market-regime artifact. If you were to run this live,
**AM-only would be the reasonable starting scope**, not the full two-window
version.

Also notable: `window_flatten` exits (window ended before hitting stop or
target) were again the most profitable exit category (57.0% win rate,
$139.20 expectancy/trade on 151 trades) -- more profitable than trades that
ran to the full 1.5R target. That's the window cutoff acting as de facto
trade management even though the source says "no trade management"; worth
keeping in mind if you relax the window-flatten rule.

**Earlier cross-check: NAS100 CFD proxy data (2016-2020).** Before real NQ
data was available, this was backtested against
`sample_data/fetch_real_data.py --instrument NAS100_USD`, OANDA's Nasdaq-100
CFD 1-minute series (2016-01 through 2020-05, ~1.49M bars, republished under
GPL-3.0 by `FutureSharks/financial-data`). It's an index CFD proxy, not
literal CME NQ tick data, and its volume is OANDA's tick count, not real
exchange volume -- kept here as a secondary, non-overlapping-period
cross-check now that real NQ data has confirmed the same AM/PM pattern.

| | num_trades | win_rate | profit_factor | avg_r | expectancy/trade | total_pnl | max_dd |
|---|---|---|---|---|---|---|---|
| All trades | 2,444 | 48.0% | 1.09 | 0.033 | $32.69 | $79,902 | $33,471 |
| `--restrict-to-first-hour` | 2,309 | 48.2% | 1.11 | 0.040 | $39.82 | $91,944 | $33,198 |

| Segment | n | win_rate | profit_factor | expectancy/trade |
|---|---|---|---|---|
| AM window (9:30-11:00) | 1,643 | 48.0% | 1.14 | $55.30 |
| PM window (14:00-15:00) | 801 | 48.1% | 0.95 | **-$13.60** |
| Long trades | 1,248 | 51.3% | 1.17 | $57.40 |
| Short trades | 1,196 | 44.6% | 1.02 | $6.90 |

On this dataset the edge leaned long (2016-2020 was a persistent Nasdaq-100
uptrend), which the real-NQ run above doesn't repeat -- shorts were fine
there. That's a useful reminder that the long-bias finding was likely a
regime artifact of that specific period, while the AM/PM split held up
across both.

## Running the tests

```bash
pytest
```
