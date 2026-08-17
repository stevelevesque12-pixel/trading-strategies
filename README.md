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

**A third strategy lives in this repo**: `overextension/` -- a VWAP
mean-reversion fade for high-beta tech futures (NQ/MNQ), trading a single
lower timeframe (1m or 5m) directly rather than a bias+entry pair. Tracks
the session VWAP and a volume-weighted stdev band around it, turning
"distance from fair value" into a z-score; when price pushes several
standard deviations away and then shows the first sign of exhaustion (a
confirming-direction bar with the z-score already recovering off its
extreme), it fades back toward VWAP. Stop sits beyond the extension's
extreme, target is a (by default full) partial reversion back to the
current VWAP. See `overextension/strategy.py` and
`tradingview/overextension.pine`. Unlike structure_scalp, this one trades
at 1m/5m resolution, so it's fully backtestable against the same
1-minute data as Failed-2s -- see the Backtesting section below.

**A fourth strategy lives in this repo**: `orb/` -- a classic Opening
Range Breakout with an ATR "velocity" filter. Tracks the high/low of the
first N minutes of the session (the Opening Range, N configurable -- 5, 15,
60, whatever), then watches for the first bar whose range crosses OR High +
`0.2 * ATR` (long) or OR Low - `0.2 * ATR` (short); the ATR buffer is what
filters a real breakout from a tick-through-and-fade. Stop sits at the
opposite side of the range, one breakout attempt per session day, unfilled
orders are cancelled after a time cutoff (default 10:15 ET) and any open
position is force-flattened later in the day (default 15:45 ET). See
`orb/strategy.py` and `tradingview/orb.pine`. Trades at 1m/5m resolution
like overextension, so it's backtestable against the same data.

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

## Strategy logic (VWAP Overextension)

Single timeframe (no bias tier) -- run directly on 1-minute or 5-minute bars:

1. **Session VWAP + band** — every bar accumulates a volume-weighted VWAP and
   a volume-weighted standard deviation around it, both resetting at the
   start of each session day. `z = (close - vwap) / stdev` (stdev floored at
   `min_stdev_ticks` to avoid blowups early in a session).
2. **Extension** — `z <= -entry_z` (or `>= entry_z` for the short side)
   starts tracking an "extension" episode: the extreme price and the most
   extreme z reached. It keeps extending on new extremes without firing —
   firing on the threshold cross alone means fading a move still in
   progress.
3. **Exhaustion trigger** — fires the first bar that's both (a) in the
   opposite direction (bullish to fade a downside extension, bearish for
   upside) and (b) has recovered at least `confirm_z` off the episode's most
   extreme z reading. That's the entry.
4. **Stop** — the episode's extreme, plus a small tick buffer.
5. **Target** — `entry + reversion_target_pct * (vwap - entry)`, i.e. a
   (default full) reversion back toward fair value rather than a fixed R
   multiple, since the thesis is literally "reverts to VWAP." Set
   `reversion_target_pct` below 1.0 for a more conservative partial fade.
6. **Trend-day guard** — an episode is abandoned (no trade) if its extreme z
   ever exceeds `max_z`: a move that extended can still be a real trend day,
   not a range-bound overextension, and this strategy shouldn't fade those.
7. **Intraday only** — same session-reset/entry-window/flatten-cutoff
   behavior as Failed-2s (`SessionConfig`, shared).

Primarily meant for high-beta tech futures (NQ/MNQ), but works on any
instrument in `failed2s/instruments.py`.

## Strategy logic (ORB + ATR Velocity Filter)

Single timeframe (no bias tier), same as Overextension:

1. **Opening Range (OR)** — the high/low of the first `or_minutes` of the
   session (default 15: 9:30-9:45 ET, but this is the one knob explicitly
   meant to be turned — 5, 15, 60, whatever fits your instrument/timeframe).
   Resets every session day.
2. **Trailing ATR** — a rolling simple-moving-average of True Range over
   `atr_period` bars (default 14). Deliberately **not** session-reset (see
   the module docstring in `orb/strategy.py` for why: at the default 14
   bars on a 5-minute chart, resetting it at 9:30 wouldn't warm it up again
   until 70 minutes in — past the default 10:15 breakout cutoff). It
   carries over from the prior session's trailing bars instead, same as it
   would behave as an indicator on a real chart.
3. **Breakout triggers** (resting-stop-style — they fire the instant a
   bar's range crosses the level, at that exact level, not at the bar's
   close): long on `bar.high >= OR High + atr_mult * ATR`, short on
   `bar.low <= OR Low - atr_mult * ATR`. The ATR term is the "velocity"
   filter — a breakout has to clear the range by a volatility-scaled
   amount, not just tick through it.
4. **One trade per day** — once a breakout fires (either side), no further
   entries are taken that session, win or lose.
5. **Time filter** — no entries once `no_entry_after` passes (default
   10:15 ET), even if the OR breakout never triggered; any open position is
   force-flattened at `flatten_at` (default 15:45 ET).
6. **Stop / target** — stop at the opposite side of the Opening Range
   (a breakout that fully round-trips back through the range has
   invalidated its own thesis), plus an optional tick buffer (default 0).
   Target = `target_r * risk` (default 1:1 — tune upward, ORB breakouts
   with real velocity are commonly traded for >1R since the range itself is
   often a tight stop).

Picking `or_minutes` too large for the default 10:15 cutoff (e.g. a
60-minute OR ending 10:30, after the cutoff) raises a `ValueError` at
construction rather than silently producing zero trades — widen
`no_entry_after` to match if you lengthen the OR.

Code layout:
- `failed2s/` — pure strategy logic (bar classification, Failed-2, swing/MSS/FVG, the signal engine, risk manager, instrument specs). Shared by backtest, live, and the webhook path.
- `overextension/` — the VWAP overextension strategy's logic (`strategy.py`), independent of failed2s aside from reusing `Bar`/`SessionConfig`.
- `orb/` — the ORB strategy's logic (`strategy.py`), same reuse pattern as overextension.
- `backtest/` — event-driven backtester over OHLCV CSV data (Failed-2s, Overextension, and ORB).
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

**VWAP Overextension** (same 1-minute CSV, resampled to your chosen
timeframe -- try real NASDAQ-100 data via `fetch_real_data.py
--instrument NAS100_USD`, since this strategy is aimed at high-beta tech):

```bash
python -m backtest.run_overextension --data sample_data/sample_1min.csv --timeframe 5min --symbol MNQ
```

Supports `--timeframe` (`1min`/`5min`), `--symbol`, `--contracts`,
`--daily-loss-limit`, `--max-daily-trades`, `--entry-z`, `--confirm-z`,
`--max-z`, `--reversion-target-pct`, `--warmup-bars`. Same trade-log CSV
format and metrics as `backtest.run`/`backtest.compare` above (uses the same
`backtest.metrics`/`backtest.report` helpers) -- just a single timeframe, no
`compare`-style multi-pair CLI since there's only one timeframe per run.

**ORB + ATR Velocity Filter** (same 1-minute CSV, resampled to your chosen timeframe):

```bash
python -m backtest.run_orb --data sample_data/sample_1min.csv --or-minutes 15 --timeframe 5min --symbol MES
```

Supports `--timeframe` (`1min`/`5min`), `--or-minutes` (the OR length --
try 5, 15, 60, ...), `--atr-period`, `--atr-mult`, `--target-r`,
`--stop-buffer-ticks`, `--no-breakout-after`/`--flatten-at` (`HH:MM` ET),
`--symbol`, `--contracts`, `--daily-loss-limit`, `--max-daily-trades`. If
you widen `--or-minutes` past `--no-breakout-after`, the strategy raises a
clear error instead of silently trading zero times -- push
`--no-breakout-after` out to match. Same trade-log/metrics plumbing as the
other CLIs.

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

## Running the tests

```bash
pytest
```
