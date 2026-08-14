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

**A third strategy, `asian_sweep/`**, is built for NQ specifically: map the
20:00-00:00 ET Asian box, watch the 00:00-09:30 ET pre-market window for a
*closing* break of the box high/low (not just a wick), then don't trade the
break itself -- wait for a 15-minute Market Structure Shift (optionally
confirmed by a Fair Value Gap) in the break direction, then enter on the
retest of that gap or the broken box boundary. See `asian_sweep/strategy.py`
and `tradingview/asian_sweep_nq.pine`. Unlike the other two strategies it
runs off a single bar stream gated by time-of-day session windows rather
than a second higher timeframe. Backtested against real NAS100 CFD data
(see below) -- results and the long/short-symmetry question are discussed
in that section.

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

## Strategy logic (Asian Sweep — NQ)

Mechanical, single-bar-stream (15-minute recommended), driven by time-of-day
session windows rather than a second higher timeframe:

1. **Box** — map the Asian box high/low from bars between 20:00 and 00:00 ET.
2. **Sweep/break** — during the London/pre-market window (00:00-09:30 ET by
   default), watch for a bar to *close* beyond the box high or low (a break,
   not just a wick). Sets a directional bias and remembers the broken
   boundary (the swept level).
3. **Confirmation** — don't trade the break itself. Wait for a Market
   Structure Shift (MSS) in the break direction on the same bar stream. The
   source rule ("MSS or FVG confirms") is read as MSS-required, with an
   accompanying 3-bar Fair Value Gap (FVG) — if the same window that
   produced the MSS also qualifies as one — used only to sharpen the retest
   level, since the displacement leg that causes an MSS is usually the same
   leg that leaves the gap. `require_fvg=True` switches to a stricter mode
   that refuses to arm without one (mirrors Failed-2s).
4. **Retest entry** — arms a pending retest at the FVG's near edge if one
   qualified, else at the broken box boundary itself. Entry fires the first
   time a later bar's range touches that level. Invalidated if price closes
   back across the broken boundary before that touch ever happens.
5. **Stop** — `stop_mode="structural"` (default) places it beyond the
   further of the MSS's confirming swing point or the swept boundary, plus
   a tick buffer. `stop_mode="box_extreme"` instead anchors it beyond the
   *opposite* side of the whole Asian box (wider, but immune to intraday
   noise inside the range).
6. **Target** — a fixed R multiple (default 2R), or `target_mode="liquidity"`
   to use the previous trading day's high/low instead whenever that's at
   least as far out (falls back to fixed R otherwise).
7. **Intraday only** — all state resets every trading day (at 20:00 ET, when
   the next box opens). No new retests armed after `no_entry_after` (default
   11:00 ET); any open position is force-flattened at `flatten_at` (default
   15:55 ET).

**Are longs and shorts symmetric, or should this be traded one-sided?**
Tested against the real NAS100 CFD data described above (2018-01 through
2020-05, `--daily-loss-limit` effectively off, `--max-daily-trades 1`):

| Config | Side | Trades | Win % | Total P&L | Avg R |
|---|---|---|---|---|---|
| default (`require_fvg=False`) | long | 155 | 25.8% | +$680 | -0.25 |
| default | short | 121 | 29.8% | -$148 | -0.14 |
| `require_fvg=True` | long | 97 | 33.0% | +$3,638 | -0.10 |
| `require_fvg=True` | short | 80 | 35.0% | +$1,206 | +0.02 |

No consistent long/short edge shows up — it flips year to year (shorts won
big in the Dec-2018 selloff, longs won big in 2019's rally, longs even beat
shorts through the Feb-Mar 2020 COVID crash) rather than holding as a stable
property of the setup across regimes. **Nothing here supports running this
long-only or short-only** — the direction with the edge in any given
backtest window looks like it's tracking which regime the window happened to
cover, not something structural to the Asian-sweep mechanism itself. What
does clearly help is signal quality: requiring the FVG (`require_fvg=True`)
roughly doubles total P&L and lifts win rate ~7-8pp on both sides at once,
symmetrically — further evidence the asymmetry above is noise, not signal.
Caveats apply: this is CFD data (not CME NQ futures ticks), 276/177 trades
over ~2.5 years is a modest sample for a year-by-year breakdown, and coverage
stops at mid-2020 (see the real-data caveats above). Re-run
`python -m backtest.run_asian_sweep` yourself before trusting either side.

Code layout:
- `failed2s/` — pure strategy logic (bar classification, Failed-2, swing/MSS/FVG, the signal engine, risk manager, instrument specs). Shared by backtest, live, and the webhook path.
- `asian_sweep/` — the Asian-sweep strategy logic (box/session windows, break+retest signal engine). Reuses failed2s' swing/MSS/FVG primitives and risk manager.
- `backtest/` — event-driven backtesters over OHLCV CSV data (`engine.py`+`run.py`/`compare.py` for Failed-2s, `asian_sweep_engine.py`+`run_asian_sweep.py` for Asian Sweep).
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

**Asian Sweep (needs overnight/pre-market coverage, not RTH-only data):**

```bash
python -m backtest.run_asian_sweep --data sample_data/real_nas100_2018_2020.csv --symbol NQ
```

`sample_data/generate_sample.py --full-day` produces a synthetic 24-hour
clock (vs. the default RTH-only 9:30-16:00) for smoke-testing this
strategy's pipeline; for real data, fetch `NAS100_USD` with
`fetch_real_data.py` as above (it's OANDA CFD data and, unlike
`SPX500_USD`, trades a near-24h clock like NQ does, so it actually covers
the box/pre-market windows this strategy needs). Supports the same
`--symbol`/`--contracts`/`--daily-loss-limit`/`--max-daily-trades` flags
plus `--target-r`, `--target-mode` (`fixed_r`/`liquidity`), `--stop-mode`
(`structural`/`box_extreme`), and `--require-fvg`. Writes `trades_asian_sweep.csv`.

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

- **TradingView + TradersPost** (`tradingview/failed2s_mes.pine`,
  `tradingview/structure_scalp.pine`, or `tradingview/asian_sweep_nq.pine`)
  — the current recommended path. TradingView runs the signal logic and
  computes risk-based position size directly in Pine, fires a webhook
  formatted for TradersPost, which connects to Tradovate with a regular
  login (no paid API Access Add-On needed). See **TRADINGVIEW_WEBHOOK.md**
  (written around the Failed-2s script; the same TradersPost/webhook setup
  applies to the other two, just point the alert at a different `.pine` file).
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

## Running the tests

```bash
pytest
```
